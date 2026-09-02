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
from .config import ConfigError


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
        if inside.sum() < 3:
            continue
        slope, offset = np.polyfit(strain[inside], stress[inside], 1)
        if slope > steepest:
            steepest, intercept = slope, -offset / slope
    return intercept


def moduli_from_curves(curves, spec):
    """
    One row per specimen: the toe intercept and the chord before and after it.

    Values are returned unrounded and in the units the file is in. Rounding and
    naming are the caller's, because those are reporting conventions and belong
    with whoever is doing the reporting.
    """
    cols = spec.columns
    needed = [cols[k] for k in ("specimen", "strain", "stress") if k in cols]
    io.require(curves, needed, "the chord-modulus calculation")
    specimen, strain_c, stress_c = cols["specimen"], cols["strain"], cols["stress"]
    group_c = cols.get("group")

    rows = []
    for name, group in curves.groupby(specimen):
        group = group.sort_values(strain_c)
        strain = group[strain_c].to_numpy(float)
        stress = group[stress_c].to_numpy(float)

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


def measurements(macro):
    """The measured comparisons a config declares, as a list."""
    if macro is None:
        return []
    return list(macro.measured)


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
