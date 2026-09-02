"""
Build flexural_moduli.csv from flexural_curves.csv.

The fitting is in `hybridmap.macro`. What is here is our bending standard's
chord interval, our toe-correction convention and the column names of the
deposited file.

ASTM D790 gives the flexural modulus as the chord between 0.05 % and 0.25 %
strain. Every specimen here carries a seating preload, so the raw curve starts
with a toe: the crosshead takes up slack before the specimen begins to carry
load, and a chord measured through that region underestimates the modulus.
Annex A1.1 corrects it by extrapolating the steepest linear part of the curve
back to zero stress and taking that intercept as the true strain origin.

"Steepest linear part" needs a definition, and the choice matters more than it
looks: fitting over a fixed number of points instead of a fixed strain width
moves the intercept by up to 0.006 % strain, which is 0.31 GPa on one specimen
here. That is a third of the 0.81 GPa difference the paper's central comparison
turns on, so the convention is written down rather than left implicit.

    The window is 0.15 % of strain wide. It slides along the curve, a straight
    line is fitted inside it at each position, and the fit with the greatest
    slope supplies the intercept.

Verified against the values used in the paper: the intercept matches all ten
specimens to better than 0.0001 % strain.

Usage
    python make_flexural_moduli.py [--data ../data] [--out ../data]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybridmap import macro as _macro                          # noqa: E402
from hybridmap.config import Curves                            # noqa: E402

# ASTM D790 chord limits, in per cent strain.
CHORD_LO, CHORD_HI = 0.05, 0.25

# Width of the sliding fit window used for the toe correction, in per cent
# strain. Wide enough to average out digitisation noise, narrow enough to sit
# inside the linear region of the stiffest specimen.
TOE_WINDOW = 0.15

# A second chord, above the tangent-modulus crossover at 0.28 % strain. It is
# not an ASTM quantity; it is reported because the two configurations rank the
# other way round above the crossover, and the paper says so.
WINDOW_LO, WINDOW_HI = 0.28, 0.43

# Stress in MPa against strain in per cent gives a slope in MPa per per-cent.
MPA_PER_PCT_TO_GPA = 10.0


def chord_modulus(strain_pct, stress_MPa, lo=CHORD_LO, hi=CHORD_HI):
    """
    Chord modulus in GPa over a strain interval.

    Least squares through every recorded point inside the interval, not a
    secant between the two endpoints. The distinction is not cosmetic: the
    curve is slightly concave here, so an endpoint secant reads low, by
    0.05 GPa on a typical specimen and 0.39 GPa on the one with the largest
    toe. Regression also stops one noisy endpoint from setting the answer.
    """
    return _macro.chord_modulus(strain_pct, stress_MPa, lo, hi) / MPA_PER_PCT_TO_GPA


def toe_intercept(strain_pct, stress_MPa, width=TOE_WINDOW):
    """
    Strain at which the steepest linear region extrapolates to zero stress.

    Returns a strain in per cent, to be subtracted from the whole curve.
    """
    return _macro.toe_intercept(strain_pct, stress_MPa, width)


def main(data, out):
    curves = pd.read_csv(Path(data) / "flexural_curves.csv")

    spec = Curves(path=Path(data) / "flexural_curves.csv",
                  columns={"specimen": "specimen_id",
                           "group": "configuration",
                           "strain": "strain_pct",
                           "stress": "stress_MPa"},
                  chord=(CHORD_LO, CHORD_HI),
                  toe_window=TOE_WINDOW,
                  # Measured on the raw strain axis, deliberately, unlike the
                  # two chords above. The 0.28 % crossover was located on the
                  # raw axis, so the window that reports the ranking above it
                  # is placed on the same axis. Applying the toe correction
                  # here would shift the window relative to the crossover that
                  # defines it, and would change this column by up to 0.88 GPa.
                  extra_window=(WINDOW_LO, WINDOW_HI))
    fitted = _macro.moduli_from_curves(curves, spec)

    table = pd.DataFrame({
        "specimen_id": fitted.specimen,
        "configuration": fitted.group,
        "toe_intercept_pct": fitted.toe_intercept.round(4),
        "E_D790_raw_GPa": (fitted.chord_raw / MPA_PER_PCT_TO_GPA).round(2),
        "E_D790_toe_compensated_GPa":
            (fitted.chord_toe_compensated / MPA_PER_PCT_TO_GPA).round(2),
        "E_window_0p28_0p43_GPa":
            (fitted.chord_extra_window / MPA_PER_PCT_TO_GPA).round(2),
    }).sort_values("specimen_id")

    target = Path(out) / "flexural_moduli.csv"
    # Line endings pinned to LF. Pandas otherwise writes whatever the platform
    # uses, so regenerating the deposited file on Windows produced a file that
    # differed from it in every line while agreeing in every value. The point
    # of this script is that its output can be compared against the deposited
    # copy, and that comparison should not depend on who runs it.
    table.to_csv(target, index=False, lineterminator="\n")
    print(f"{len(table)} specimens -> {target.name}")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=here.parent / "data")
    p.add_argument("--out", default=here.parent / "data")
    a = p.parse_args()
    main(a.data, a.out)
