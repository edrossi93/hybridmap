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
mixed-volume class as a Voigt-Reuss-Hill average at a basalt fraction of 0.50.

Run: python poisson_sensitivity.py [--out ../data/poisson_sensitivity.json]
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
E_INDENTER = 1141.0
NU_INDENTER = 0.07

# The uniform convention the paper adopts, and the phase-specific alternative
# Table S1 tests against it.
NU_UNIFORM = 0.30
NU_PHASE = {"flax": 0.30, "matrix": 0.42, "mixed-volume": 0.34, "basalt": 0.24}

CLASSES = ["flax", "matrix", "mixed-volume", "basalt"]


def modulus(er, nu):
    """Specimen modulus from the reduced modulus, at an assumed Poisson ratio."""
    return (1 - nu ** 2) / (1.0 / er - (1 - NU_INDENTER ** 2) / E_INDENTER)


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

    if out:
        Path(out).write_text(json.dumps(
            {"indenter": {"E_GPa": E_INDENTER, "nu": NU_INDENTER},
             "nu_uniform": NU_UNIFORM,
             "per_class": rows.to_dict(orient="records"),
             "laminate_estimate_GPa": est},
            indent=2) + "\n", encoding="utf8")
        print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", nargs="?", const=str(DATA /
                                                  "poisson_sensitivity.json"))
    sys.exit(main(ap.parse_args().out))
