"""
The macroscopic measurement the prediction is compared against.

The minimum a config has to supply is a number and its uncertainty. Whether it
came from bending, tension, dynamic mechanical analysis or a datasheet is the
user's business, and requiring a particular test would rule out most of the
people this tool is for.

The chord route below is one supported way to obtain that number, from
load-displacement curves, because it is the one the worked example uses. It is
not privileged: `macro.measured` alone is a complete configuration.
"""

import numpy as np
import pandas as pd

from . import io
from .config import ConfigError, Measured


def chord_modulus(strain, stress, lo, hi):
    """
    Chord modulus over a strain interval, as a least-squares slope.

    Least squares through every recorded point inside the interval, not a
    secant between the two endpoints. A curve with any concavity reads low
    through its endpoints, and regression also stops one noisy endpoint from
    setting the answer.

    The unit is whatever stress/strain gives. With stress in MPa and strain in
    per cent the slope is MPa per per-cent, so the caller divides by 10 to
    reach GPa; that conversion is left to the caller because only the caller
    knows what units the file is in.
    """
    # Two points define a line, so fewer than two is not a thin fit but no fit
    # at all. Usually it means the chord interval was written for a different
    # strain unit than the file is in.
    inside = (strain >= lo) & (strain <= hi)
    if inside.sum() < 2:
        raise ValueError(f"only {int(inside.sum())} points fall between "
                         f"{lo} and {hi}; the chord interval is outside the "
                         f"range of this curve")
    return np.polyfit(strain[inside], stress[inside], 1)[0]


def toe_intercept(strain, stress, width):
    """
    Strain at which the steepest linear region extrapolates to zero stress.

    A seating preload leaves a toe: the crosshead takes up slack before the
    specimen carries load, and a chord measured through that region reads low.
    The correction is to take the strain origin from where the steepest part of
    the curve crosses zero stress.

    "Steepest linear part" needs a definition and the choice matters. A window
    of fixed strain width is used rather than a fixed number of points, so the
    answer does not depend on how densely the instrument happened to sample.
    """
    steepest, intercept = -np.inf, 0.0
    for start in strain:
        inside = (strain >= start) & (strain <= start + width)
        # Three points minimum, so a window that catches the sparse tail of a
        # curve cannot win on the strength of a two-point line through noise.
        if inside.sum() < 3:
            continue
        slope, offset = np.polyfit(strain[inside], stress[inside], 1)
        if slope > steepest:
            steepest, intercept = slope, -offset / slope
    return intercept


def to_stress_strain(displacement, force, geometry):
    """
    Force and displacement to stress and strain, for three-point bending.

    The standard formulae for a rectangular bar in three-point bending, which
    is the same standard the chord interval and the toe correction come from:

        stress = 3 P L / (2 b d^2)
        strain = 6 D d / L^2

    With force in newtons and every length in millimetres the stress comes out
    in MPa. The strain is returned in per cent, so that the chord interval
    means the same thing whichever of the two input shapes was given.
    """
    span, width, thickness = (geometry["span"], geometry["width"],
                              geometry["thickness"])
    stress = 3.0 * force * span / (2.0 * width * thickness ** 2)
    strain = 100.0 * 6.0 * displacement * thickness / span ** 2
    return strain, stress


def _curve(group, spec):
    """One specimen's curve as strain in per cent against stress."""
    cols = spec.columns
    if spec.is_raw:
        group = group.sort_values(cols["displacement"])
        return to_stress_strain(group[cols["displacement"]].to_numpy(float),
                                group[cols["force"]].to_numpy(float),
                                spec.geometry)
    group = group.sort_values(cols["strain"])
    return (group[cols["strain"]].to_numpy(float),
            group[cols["stress"]].to_numpy(float))


def moduli_from_curves(curves, spec):
    """
    One row per specimen: the toe intercept and the chord before and after it.

    Values are returned unrounded, as a slope in stress-unit per per-cent
    strain. Scaling to a modulus unit and rounding are the caller's, because
    those are reporting conventions and belong with whoever reports.
    """
    cols = spec.columns
    pair = ("displacement", "force") if spec.is_raw else ("strain", "stress")
    io.require(curves, [cols[k] for k in ("specimen",) + pair],
               "the chord-modulus calculation")
    specimen = cols["specimen"]
    group_c = cols.get("group")

    rows = []
    for name, group in curves.groupby(specimen):
        strain, stress = _curve(group, spec)

        toe = toe_intercept(strain, stress, spec.toe_window) \
            if spec.toe_window else 0.0
        row = {"specimen": name,
               "toe_intercept": toe,
               "chord_raw": chord_modulus(strain, stress, *spec.chord),
               "chord_toe_compensated": chord_modulus(strain - toe, stress,
                                                      *spec.chord)}
        if group_c:
            row["group"] = group[group_c].iloc[0]
        if spec.extra_window:
            # Measured on the raw strain axis on purpose. A second window is
            # usually placed relative to a feature located on the raw curve, so
            # shifting it by the toe correction would move it away from the
            # thing that defined it.
            row["chord_extra_window"] = chord_modulus(strain, stress,
                                                      *spec.extra_window)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("specimen").reset_index(drop=True)


def summarise(table, scale=1.0, column="chord_toe_compensated"):
    """Mean, standard deviation and count per group, in the caller's unit."""
    if "group" not in table:
        table = table.assign(group="all")
    out = table.groupby("group")[column].agg(["mean", "std", "count"])
    out[["mean", "std"]] *= scale
    return out.rename(columns={"count": "n"})


def contrast(table, scale=1.0, column="chord_toe_compensated"):
    """
    Compare exactly two groups: Welch's t, its p, and Hedges g.

    Welch rather than Student because the two groups need not have the same
    variance and there is no reason to assume they do. Hedges g rather than
    Cohen d because the correction matters at these sample sizes: five
    specimens per group is where the uncorrected d is visibly optimistic.

    Returns None unless there are exactly two groups, because a difference
    between two things is the only question this answers.
    """
    if "group" not in table:
        return None
    groups = list(dict.fromkeys(table.group))
    if len(groups) != 2:
        return None

    a = table.loc[table.group == groups[0], column].to_numpy(float) * scale
    b = table.loc[table.group == groups[1], column].to_numpy(float) * scale
    if len(a) < 2 or len(b) < 2:
        return None

    from scipy import stats

    t, p = stats.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
    g = (1 - 3 / (4 * (na + nb) - 9)) * (a.mean() - b.mean()) / pooled
    return {"groups": groups, "welch_t": float(t), "p": float(p),
            "hedges_g": float(g)}


def measurements(macro, curves_table=None, scale=1.0,
                 column="chord_toe_compensated"):
    """
    Everything to compare the prediction against.

    The values a config states outright, plus one per group reduced from the
    curves when they are given. Before this, a user who configured curves had
    to run a second command and type the answers back into the config, in a
    unit the tool never told them about.
    """
    if macro is None:
        return []
    out = list(macro.measured)
    if curves_table is None:
        return out

    summary = summarise(curves_table, scale, column)
    for name, row in summary.iterrows():
        out.append(Measured(label=str(name), value=float(row["mean"]),
                            sd=None if pd.isna(row["std"]) else float(row["std"]),
                            n=int(row["n"])))
    return out


def compare(bounds, measured, names=("Voigt", "Hill", "Reuss")):
    """
    Predicted bounds against each measured value.

    The gap is reported both absolutely and as a fraction of the bound. The
    percentage is of the prediction rather than of the measurement, so that a
    reader comparing two predictions against the same specimen is comparing
    like with like.
    """
    if not measured:
        raise ConfigError("no macroscopic measurement to compare against; add "
                          "a macro section with a measured value")
    rows = []
    for m in measured:
        for name, predicted in zip(names, bounds):
            rows.append({"measurement": m.label, "measured": m.value,
                         "sd": m.sd, "n": m.n, "bound": name,
                         "predicted": predicted,
                         "gap": predicted - m.value,
                         "gap_pct_of_bound": 100 * (predicted - m.value) / predicted})
    return pd.DataFrame(rows)
