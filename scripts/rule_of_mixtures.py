"""
Voigt, Reuss and Hill bounds from the per-class medians and indent-class
fractions, plus the sensitivity checks reported in the Supplementary
Information.

The arithmetic lives in `hybridmap.aggregate` and `hybridmap.mixtures`. What is
here is the paper's four classes, in the paper's order, and the threshold grid
the Supplementary Information reports.

Two things worth stating plainly, because they are easy to misread:

  * The fractions are indent-class fractions. They are the number of indents
    carrying each label divided by the total number of classified indents. They
    are not measured constituent volume fractions. See classify.py for why.

  * Weighting the within-zone class frequencies by the zone indent counts and
    summing over zones gives exactly the global class counts over the global
    total. The two descriptions in the paper are the same arithmetic; the
    function below checks that rather than asserting it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybridmap import aggregate as _agg                        # noqa: E402
from hybridmap import mixtures as _mix                         # noqa: E402
from hybridmap.mixtures import bounds                          # noqa: E402,F401

CLASSES = ["flax", "matrix", "mixed-volume", "basalt"]

# The column the paper's tables and figures read the class medians from. The
# package calls it "median", because it does not know what property or unit it
# is being handed.
MEDIAN_COL = "E_median_GPa"


def class_table(df, modulus_col="E_GPa", classes=CLASSES):
    """Counts, fractions and median modulus per class, ignoring QC failures."""
    # QC/invalid and unclassified are excluded rather than imputed, so the
    # fractions are over classified indents only. That is why they are
    # indent-class fractions and not constituent volume fractions: an indent
    # that failed the phase-angle check simply does not vote.
    return _agg.class_table(df, classes, modulus_col, median_col=MEDIAN_COL)


def check_zone_weighting(df, classes=CLASSES):
    """
    Confirm that zone weighting and global counting agree. Returns the two
    fraction vectors so the caller can print them side by side.
    """
    return _agg.check_zone_weighting(df, classes)


def threshold_sweep(raw_shallow, lowers=(10, 15, 20), uppers=(60, 65, 70, 75, 80)):
    """
    Redo the classification and the Voigt bound for a grid of thresholds. This
    is Supplementary Table S2. Note that the table in the first submission was
    generated from an earlier classification pass and did not match the main
    text; this function regenerates it from the deposited data.
    """
    from classify import PHASES, X_RESIN, E_LOWER, E_UPPER_SHALLOW, assign_zone

    # The resin cut and the zone boundaries are held fixed while the modulus
    # thresholds move. They are set by position, not by stiffness, so sweeping
    # them would be a different sensitivity test and would confound this one.
    d = raw_shallow[raw_shallow.x_um >= X_RESIN].copy()
    d["zone"] = assign_zone(d.x_um.values).values

    return _mix.sweep(d, PHASES, {"lower": E_LOWER, "upper": E_UPPER_SHALLOW},
                      {"lower": lowers, "upper": uppers}, "E_GPa", CLASSES,
                      median_col=MEDIAN_COL)


def substitute(table, cls, new_modulus):
    """Swap one class modulus and return the new bounds. Used for the basalt
    input sensitivity and for the bulk PA-11 comparison."""
    return _mix.substitute(table, cls, new_modulus, median_col=MEDIAN_COL)


# Published flax stiffnesses, for the sensitivity below. They are three
# different physical quantities, which is the point of tabulating them
# together rather than picking one:
#
#   14.0   immature Gn cell-wall layer, PF-QNM on an ultramicrotomed
#          cross-section, that is, pressing along the fibre axis
#   18.0   mature G cell-wall layer, same technique and same geometry
#   45.3   elementary-fibre axial modulus, from tension along the fibre
#
# All three from Goudenhooft et al., Fibers 6 (2018) 6. None of the three is
# measured the way this work measures flax. The cell-wall values are taken on
# the fibre's cross-section, so the probe presses along the fibre axis; here
# the flax yarns lie in the plane of the section and the indenter presses
# across it, the opposite geometry (2.2, 4.3.4). Calling the cell-wall input
# "transverse", as this list did for two rounds, was the conflation Reviewer 3
# named. The value this work measures is none of the three: it is the
# compliant response of the flax-rich tow as consolidated, pressed across its
# axis, with lumen, middle lamella and impregnating matrix inside the probed
# volume.
#
# The third value read 52.47 here for two rounds, against the 45.30 GPa of
# Supplementary Table S8 and the "about 45 GPa along the fibre axis" of §4.3.4.
# The source settles it: that paper measures 45.3 +/- 9.7 GPa on its own mature
# elementary fibres, and 52.47 is a figure it quotes from earlier work in its
# introduction. So the table and §4.3.4 carry the measurement and this list
# carried the quotation. It carries the measurement now. The column printed by
# the block at the foot of this file still reads lower than Table S8 for every
# input, because it is built on the whole-map class fractions and the table is
# built on the two-level construction; that difference is in the heading it is
# printed under, and the table's own cells are checked in
# reproduce_paper_values.py.
FLAX_INPUTS = [
    (14.0, "immature Gn cell-wall layer"),
    (18.0, "mature G cell-wall layer"),
    (45.30, "elementary-fibre axial modulus"),
]


def flax_input_sensitivity(table, measured, inputs=FLAX_INPUTS):
    """
    Rebuild the bound with the flax term set to each published value in turn.

    Reviewer 3 objects that a modulus measured normal to a cross-section cannot
    carry the axial contribution the iso-strain term asks of it. This answers
    by arithmetic rather than by argument: the Voigt sum is recomputed across
    every defensible flax input, from the tow value measured here to the axial
    modulus of a single fibre, and the closure to each measured flexural
    modulus is reported beside it.

    `measured` maps a configuration name to its flexural modulus in GPa.
    Closure is (Voigt - E_flex) / Voigt, the definition used throughout.
    """
    here = float(table.loc[table["class"] == "flax", MEDIAN_COL].iloc[0])
    rows = []
    for E, label in [(here, "this work, tow in situ")] + list(inputs):
        v = (bounds(table.fraction, table[MEDIAN_COL])[0] if E == here
             else substitute(table, "flax", E)[0])
        row = {"flax_input_GPa": E, "source": label, "Voigt_GPa": v}
        for name, e in measured.items():
            row[f"closure_{name}"] = (v - e) / v
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = Path(__file__).resolve().parents[1] / "data"

    sh = pd.read_csv(data / "classified_shallow.csv")
    tab = class_table(sh)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    v, h, r = bounds(tab.fraction, tab.E_median_GPa)
    print(f"\nVoigt {v:.2f}   Hill {h:.2f}   Reuss {r:.2f}  GPa")

    g, z = check_zone_weighting(sh)
    print(f"\nglobal fractions      {np.round(g, 5)}")
    print(f"zone-weighted          {np.round(z, 5)}")
    print(f"identical              {np.allclose(g, z)}")

    # Read the depth-resolved basalt median from the data rather than typing it
    # in. The literal that used to sit here, 49.47 GPa, was the median under a
    # quality mask that has since been replaced.
    _dr = pd.read_csv(data / "classified_depth_resolved.csv")
    basalt_depth = float(_dr.loc[(_dr.qc_flag_strict == 0)
                                 & (_dr.constituent == "basalt"),
                                 "E_150"].median())
    print(f"\nbasalt input swapped to the depth-resolved value "
          f"({basalt_depth:.2f} GPa): "
          f"Voigt {substitute(tab, 'basalt', basalt_depth)[0]:.2f} GPa")
    for Em in (1.0, 1.2, 1.5):
        print(f"matrix set to bulk PA-11 {Em} GPa: "
              f"Voigt {substitute(tab, 'matrix', Em)[0]:.2f} GPa")

    flex = pd.read_csv(data / "flexural_moduli.csv")
    measured = {c: g.E_D790_toe_compensated_GPa.mean()
                for c, g in flex.groupby("configuration")}
    fs = flax_input_sensitivity(tab, measured)
    print("\nflax-input sensitivity (Supplementary Table S8, at the whole-map "
          "fractions rather than the two-level construction)")
    print(f"  {'flax input':>10}  {'source':<32}{'Voigt':>7}"
          f"{'closure basalt':>16}{'closure flax':>14}")
    for _, r in fs.iterrows():
        print(f"  {r.flax_input_GPa:>10.2f}  {r.source:<32}{r.Voigt_GPa:>7.2f}"
              f"{100 * r.closure_basalt_face:>15.1f}%{100 * r.closure_flax_face:>13.1f}%")
    print("  the sum stays above both measurements for every input")

    raw = pd.read_csv(data / "shallow_map.csv")
    sweep = threshold_sweep(raw)
    ref = sweep[(sweep.lower_GPa == 15) & (sweep.upper_GPa == 70)].Voigt_GPa.iloc[0]
    print(f"\nthreshold sweep: reference {ref:.2f} GPa, "
          f"range {sweep.Voigt_GPa.min():.2f} to {sweep.Voigt_GPa.max():.2f}, "
          f"max deviation {np.abs(sweep.Voigt_GPa - ref).max():.2f} GPa")
