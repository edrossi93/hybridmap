"""Supplementary Table S1: the per-class moduli under phase-specific Poisson ratios.

Table S1 reports each class median twice, once under the uniform nu = 0.30 the
paper adopts and once under a value chosen for the phase, and then reports what
the laminate estimate does. Nothing in the deposit produced those numbers. The
censusing of every printed number against everything the deposit can compute
found 89.05 GPa, the basalt median under nu = 0.24, with no route to it.

There is a route, and it is short. The instrument measures a reduced modulus,
which does not depend on the Poisson ratio assumed for the specimen; the
specimen modulus does. `Er_GPa` is deposited per indent, so both columns of
Table S1 follow from the same measurement under two conventions:

    1/Er = (1 - nu^2)/E + (1 - nu_i^2)/E_i

with the indenter constants for diamond. Inverting it for E at each nu gives
the table.

The Poisson ratios are inputs rather than measurements and are named in the
table's own caption: basalt 0.24, flax 0.30, PA-11 0.42, and 0.34 for the
mixed-volume class.

The 0.34 is an assumed intermediate value between the basalt and the matrix
figures, and nothing more. It is not a Voigt-Reuss-Hill average of 0.24 and
0.42: at a basalt fraction of 0.50 the Voigt mean is 0.330, the Reuss mean
0.305 and the Hill average 0.318, and at the fractions the deconvolution of 4.1
puts on the mixed-volume contact, 0.54 and 0.67, the Hill average is 0.311 and
0.289. `hill_average_nu` computes
those and the JSON carries them, so that the description can be checked rather
than believed. The value itself is not changed: the table tests the sensitivity
of the estimate to a plausible phase-specific set, and 0.34 is inside the
interval either end member bounds. The term it contributes to the budget is
0.056 GPa.

Run: python poisson_sensitivity.py            writes data/poisson_sensitivity.json
     python poisson_sensitivity.py --no-write  prints the table only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

# Diamond, the values the Oliver-Pharr analysis uses throughout. They are
# properties of the indenter, not of this specimen.
E_INDENTER = 1140.0
NU_INDENTER = 0.07

# The uniform convention the paper adopts, and the phase-specific alternative
# Table S1 tests against it.
NU_UNIFORM = 0.30
# The mixed-volume value is an assumed intermediate between the basalt and
# the matrix figures; see the module docstring for what a Hill average would
# give instead.
NU_PHASE = {"flax": 0.30, "matrix": 0.42, "mixed-volume": 0.34, "basalt": 0.24}

# The basalt fractions at which the Hill average of the two end-member Poisson
# ratios is reported: 0.50, an equal split, and the two fractions the
# deconvolution of 4.1 puts on the mixed-volume contact.
HILL_FRACTIONS = (0.50, 0.54, 0.67)

CLASSES = ["flax", "matrix", "mixed-volume", "basalt"]


def modulus(er, nu):
    """Specimen modulus from the reduced modulus, at an assumed Poisson ratio."""
    return (1 - nu ** 2) / (1.0 / er - (1 - NU_INDENTER ** 2) / E_INDENTER)


def hill_average_nu(f_basalt, nu_basalt=NU_PHASE["basalt"],
                    nu_matrix=NU_PHASE["matrix"]):
    """Voigt, Reuss and Hill means of two Poisson ratios at a basalt fraction.

    Voigt is the arithmetic mean by fraction, Reuss the harmonic one, Hill the
    average of the two. Returned as a dict so that all three travel together.
    """
    voigt = f_basalt * nu_basalt + (1 - f_basalt) * nu_matrix
    reuss = 1.0 / (f_basalt / nu_basalt + (1 - f_basalt) / nu_matrix)
    return {"voigt": voigt, "reuss": reuss, "hill": 0.5 * (voigt + reuss)}


def table():
    """One row per class: the median under each convention, and the shift."""
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    rows = []
    for cls in CLASSES:
        er = shallow.loc[shallow.constituent == cls, "Er_GPa"]
        uniform = float(np.median(modulus(er.values, NU_UNIFORM)))
        phase = float(np.median(modulus(er.values, NU_PHASE[cls])))
        rows.append({"class": cls,
                     "median_Er_GPa": float(er.median()),
                     "E_uniform_nu_GPa": uniform,
                     "E_phase_specific_GPa": phase,
                     "nu_phase_specific": NU_PHASE[cls],
                     "shift_pct": 100 * (phase - uniform) / uniform})
    return pd.DataFrame(rows)


def estimates(rows):
    """The laminate estimate under each convention, at the Eq. (1) fractions."""
    import zone_share as zs
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    per_zone, medians, _ = zs.zone_fractions(shallow)
    fractions, _ = zs.at_share(per_zone, medians, zs.tile_share())
    by_class = dict(zip(zs.CLASSES, fractions))
    out = {}
    for column, name in (("E_uniform_nu_GPa", "uniform"),
                         ("E_phase_specific_GPa", "phase_specific")):
        out[name] = float(sum(by_class[r["class"]] * r[column]
                              for _, r in rows.iterrows()))
    out["shift_pct"] = 100 * (out["uniform"] - out["phase_specific"]) \
        / out["uniform"]
    return out


def main(out=None):
    rows = table()
    est = estimates(rows)

    print(f"  {'class':<14}{'median Er':>11}{'E at 0.30':>11}"
          f"{'nu':>7}{'E at nu':>10}{'shift':>9}")
    for _, r in rows.iterrows():
        print(f"  {r['class']:<14}{r.median_Er_GPa:11.3f}"
              f"{r.E_uniform_nu_GPa:11.2f}{r.nu_phase_specific:7.2f}"
              f"{r.E_phase_specific_GPa:10.2f}{r.shift_pct:8.2f} %")

    print(f"\n  laminate estimate, uniform nu = 0.30   {est['uniform']:.2f} GPa")
    print(f"  laminate estimate, phase-specific nu   "
          f"{est['phase_specific']:.2f} GPa")
    print(f"  shift                                  {est['shift_pct']:.1f} %")
    print("\n  Both estimates here are computed from the unrounded medians. "
          "Recomputing\n  them from the two-decimal values printed in Table S1 "
          "gives 18.50 and 18.30.")

    hill = {f"{f:.2f}": hill_average_nu(f) for f in HILL_FRACTIONS}
    print(f"\n  the mixed-volume nu of {NU_PHASE['mixed-volume']} is an assumed "
          f"intermediate value; a Hill average of\n  {NU_PHASE['basalt']} and "
          f"{NU_PHASE['matrix']} would give")
    for f, v in hill.items():
        print(f"    basalt fraction {f}: Voigt {v['voigt']:.3f}  Reuss "
              f"{v['reuss']:.3f}  Hill {v['hill']:.3f}")

    if out:
        Path(out).write_text(json.dumps(
            {"indenter": {"E_GPa": E_INDENTER, "nu": NU_INDENTER},
             "nu_uniform": NU_UNIFORM,
             "per_class": rows.to_dict(orient="records"),
             "laminate_estimate_GPa": est,
             "mixed_volume_nu": {
                 "assumed": NU_PHASE["mixed-volume"],
                 "note": ("an assumed intermediate value between the basalt "
                          "and matrix ratios, not a Voigt-Reuss-Hill average "
                          "of them; the averages at each basalt fraction are "
                          "given so the description can be checked"),
                 "averages_by_basalt_fraction": hill}},
            indent=2) + "\n", encoding="utf8", newline="\n")
        print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    # Writes the product by default, like every other generator in the
    # deposit. `provenance_check.py --rebuild` runs each generator with no
    # arguments, so a generator that wrote only when --out was given would
    # leave the product untouched and be reported as reproduced without ever
    # having regenerated it. --out writes a copy elsewhere; `--no-write`
    # prints the table and leaves the deposit alone.
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "poisson_sensitivity.json"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    sys.exit(main(None if args.no_write else args.out))
