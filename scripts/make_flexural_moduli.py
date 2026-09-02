"""
Build flexural_moduli.csv from flexural_curves.csv.

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
from pathlib import Path

import numpy as np
import pandas as pd

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


def chord_modulus(strain_pct, stress_MPa, lo=CHORD_LO, hi=CHORD_HI):
    """
    Chord modulus in GPa over a strain interval.

    Least squares through every recorded point inside the interval, not a
    secant between the two endpoints. The distinction is not cosmetic: the
    curve is slightly concave here, so an endpoint secant reads low, by
    0.05 GPa on a typical specimen and 0.39 GPa on the one with the largest
    toe. Regression also stops one noisy endpoint from setting the answer.
    """
    inside = (strain_pct >= lo) & (strain_pct <= hi)
    slope = np.polyfit(strain_pct[inside], stress_MPa[inside], 1)[0]
    return slope / 10.0          # MPa per % strain -> GPa


def toe_intercept(strain_pct, stress_MPa, width=TOE_WINDOW):
    """
    Strain at which the steepest linear region extrapolates to zero stress.

    Returns a strain in per cent, to be subtracted from the whole curve.
    """
    steepest, intercept = -np.inf, 0.0
    for start in strain_pct:
        inside = (strain_pct >= start) & (strain_pct <= start + width)
        if inside.sum() < 3:
            continue
        slope, offset = np.polyfit(strain_pct[inside], stress_MPa[inside], 1)
        if slope > steepest:
            steepest, intercept = slope, -offset / slope
    return intercept


def main(data, out):
    curves = pd.read_csv(Path(data) / "flexural_curves.csv")

    rows = []
    for specimen, group in curves.groupby("specimen_id"):
        group = group.sort_values("strain_pct")
        strain = group.strain_pct.to_numpy(float)
        stress = group.stress_MPa.to_numpy(float)

        toe = toe_intercept(strain, stress)
        rows.append({
            "specimen_id": specimen,
            "configuration": group.configuration.iloc[0],
            "toe_intercept_pct": round(toe, 4),
            "E_D790_raw_GPa": round(chord_modulus(strain, stress), 2),
            "E_D790_toe_compensated_GPa":
                round(chord_modulus(strain - toe, stress), 2),
            # Measured on the raw strain axis, deliberately, unlike the two
            # columns above. The 0.28 % crossover was located on the raw axis,
            # so the window that reports the ranking above it is placed on the
            # same axis. Applying the toe correction here would shift the
            # window relative to the crossover that defines it, and would
            # change this column by up to 0.88 GPa.
            "E_window_0p28_0p43_GPa":
                round(chord_modulus(strain, stress,
                                    WINDOW_LO, WINDOW_HI), 2),
        })

    table = pd.DataFrame(rows).sort_values("specimen_id")
    target = Path(out) / "flexural_moduli.csv"
    table.to_csv(target, index=False)
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
