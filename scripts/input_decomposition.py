"""Which input drives the difference between a tabulated and a measured prediction.

Reviewer 3 asked for this: the comparison between the two constructions changes
the constituent moduli and the volume fractions at the same time, so it cannot be
read as evidence about either. The decomposition holds one set fixed while the
other changes.

Both constructions are put on the same three-constituent basis, basalt, flax and
matrix, because that is the only basis on which the two are comparable: the
tabulated route has no counterpart for the mixed-volume contact class. The
laminate bound reported in the paper is the four-class construction and is not
what this script computes; what it computes is the sensitivity of a
three-constituent Voigt sum to each of its two input sets.

    tabulated moduli   basalt 80, flax 60, matrix 1.5 GPa, the upper midpoints
                       of the ranges cited in the introduction
    measured moduli    the per-class medians of the shallow acquisition
    nominal fractions  0.25 / 0.25 / 0.50, a 50 % fibre volume fraction with the
                       fibres split equally, which is what the tabulated route
                       assumes in the absence of a measurement
    measured fractions the backscattered-electron segmentation of the same
                       section, 24.4 / 39.5 / 36.1 %

Run: python input_decomposition.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

TABULATED_E = {"basalt": 80.0, "flax": 60.0, "matrix": 1.5}
NOMINAL_F = {"basalt": 0.25, "flax": 0.25, "matrix": 0.50}
MEASURED_F = {"basalt": 0.244, "flax": 0.395, "matrix": 0.361}


def measured_moduli():
    sh = pd.read_csv(DATA / "classified_shallow.csv")
    sh = sh[sh.constituent != "QC/invalid"]
    med = sh.groupby("constituent")["E_GPa"].median()
    return {"basalt": float(med["basalt"]),
            "flax": float(med["flax"]),
            "matrix": float(med["matrix"])}


def voigt(E, f):
    return sum(f[k] * E[k] for k in f)


def main():
    E_meas = measured_moduli()
    print("measured per-constituent medians, shallow acquisition")
    for k in ("basalt", "flax", "matrix"):
        print(f"    {k:8s} {E_meas[k]:6.2f} GPa   tabulated {TABULATED_E[k]:5.1f} GPa")

    both_tab = voigt(TABULATED_E, NOMINAL_F)
    moduli_only = voigt(E_meas, NOMINAL_F)
    fractions_only = voigt(TABULATED_E, MEASURED_F)
    both_meas = voigt(E_meas, MEASURED_F)

    print("\nthree-constituent Voigt sum, GPa")
    print(f"    tabulated moduli, nominal fractions     {both_tab:6.2f}")
    print(f"    measured moduli,  nominal fractions     {moduli_only:6.2f}"
          f"   ({moduli_only - both_tab:+.2f})")
    print(f"    tabulated moduli, measured fractions    {fractions_only:6.2f}"
          f"   ({fractions_only - both_tab:+.2f})")
    print(f"    measured moduli,  measured fractions    {both_meas:6.2f}"
          f"   ({both_meas - both_tab:+.2f})")

    dm = moduli_only - both_tab
    df = fractions_only - both_tab
    print(f"\n    moduli alone move the sum by   {dm:+.1f} GPa")
    print(f"    fractions alone move it by     {df:+.1f} GPa")
    print(f"    the two together                {both_meas - both_tab:+.1f} GPa")
    print(f"\n    |moduli| / |fractions| = {abs(dm) / abs(df):.1f}")

    # Which constituent carries the moduli term.
    print("\n    contribution of each constituent to the moduli term, GPa")
    for k in ("basalt", "flax", "matrix"):
        print(f"      {k:8s} {NOMINAL_F[k] * (E_meas[k] - TABULATED_E[k]):+7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
