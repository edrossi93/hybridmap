"""The areal-mass route to the constituent volume fractions quoted in 3.4.

3.4 reports a second, independent route to the constituent volume fractions: not
the segmented section, but the fabric specification. Sixteen plies of a 360 g/m2
fabric at a 50/50 mass split, spread through a 4.60 mm laminate, fix the basalt
and flax volumes from their densities; the matrix follows by difference and so
absorbs whatever porosity there is. Until this script existed, nothing in the
deposit computed it, and the only densities anywhere in here were the pair in
`reply_figures.py`, which belongs to a different calculation and does not
reproduce the printed fractions.

What the route needs, and where each input comes from:

    sixteen plies            2.1, "a sixteen-ply laminate"
    360 g/m2 per ply         2.1, the fabric's nominal areal weight
    50 wt % basalt           2.1, "equal mass fractions of flax and basalt"
    4.60 mm laminate         2.5, the bending specimen section, 80 x 15 x 4.6 mm
    basalt fibre density     NOT STATED IN THE MANUSCRIPT
    flax fibre density       NOT STATED IN THE MANUSCRIPT

The two densities are the gap, and it is a gap in the manuscript rather than in
this script. 3.4 says "constituent densities" and prints none. The pair used here
is the pair the printed numbers require: 2.70 and 1.50 g/cm3 return 23.2, 41.7
and 35.1 % and a fibre volume fraction of 0.649, which are the four figures 3.4
and Figure 8 print, all four from one pair of inputs. Two free parameters fixed
by four published values is not a fit, and both are the round datasheet figures
for basalt and flax fibre rather than adjusted decimals, but they are inferred
and this file says so. The manuscript has to print them where it states the
route.

The pair in `reply_figures.py`, 2.67 and 1.45 g/cm3, is also within the range
either fibre is quoted at, and it does not reproduce the printed fractions: it
gives 23.4, 43.2 and 33.4 % with a fibre volume fraction of 0.666, and the worst
disagreement with the imaging becomes 3.6 percentage points rather than 2.2. That
second set is computed and printed below rather than left for a reader to
discover, because the deposit carries both pairs and a reader who finds only the
other one cannot reproduce the paper. Nothing here changes a published number.

Run: python areal_mass_fractions.py
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

N_PLIES = 16
AREAL_MASS_PER_PLY_G_PER_M2 = 360.0
BASALT_MASS_FRACTION = 0.50
LAMINATE_THICKNESS_MM = 4.60

# Inferred from the published fractions, not stated in the manuscript. See above.
RHO_BASALT_G_PER_CM3 = 2.70
RHO_FLAX_G_PER_CM3 = 1.50

# The pair reply_figures.py carries, for the fibre-modulus volume split of the
# lamination-theory comparison. Reported here as the sensitivity it is.
RHO_ELSEWHERE_IN_DEPOSIT = (2.67, 1.45)

PHASES = ("basalt", "flax", "matrix")


def fractions(rho_basalt=RHO_BASALT_G_PER_CM3, rho_flax=RHO_FLAX_G_PER_CM3):
    """Constituent volume fractions of the laminate, from the fabric and a thickness.

    Per square metre of laminate: the fabric contributes a known mass of each
    fibre, each mass divided by its density is a volume, and the laminate volume
    is the thickness. The matrix is the remainder, which is why it carries the
    porosity: nothing here measures a void.
    """
    fabric_g_per_m2 = N_PLIES * AREAL_MASS_PER_PLY_G_PER_M2
    basalt_g_per_m2 = fabric_g_per_m2 * BASALT_MASS_FRACTION
    flax_g_per_m2 = fabric_g_per_m2 - basalt_g_per_m2
    # 1 m2 of laminate is 1e4 cm2 of face by the thickness in cm.
    laminate_cm3_per_m2 = 1e4 * LAMINATE_THICKNESS_MM / 10.0

    basalt = basalt_g_per_m2 / rho_basalt / laminate_cm3_per_m2
    flax = flax_g_per_m2 / rho_flax / laminate_cm3_per_m2
    return {"basalt": basalt, "flax": flax, "matrix": 1.0 - basalt - flax,
            "fibre": basalt + flax}


def fibre_volume_fraction(**kwargs):
    """The fibre volume fraction of the same route, which Figure 8 also uses."""
    return fractions(**kwargs)["fibre"]


def imaged():
    """The segmented fractions of the same section, for the comparison.

    Read at full precision from the deposited summary rather than from the one
    decimal the paper prints, so that the agreement below is a comparison of two
    calculations and not of two roundings.
    """
    seg = json.loads((DATA / "bse_segmentation_summary.json").read_text())
    vf = {phase: float(seg["volume_fraction"][phase]) for phase in PHASES}
    vf["fibre"] = vf["basalt"] + vf["flax"]
    return vf


def agreement(calculated, measured):
    """Differences in percentage points, phase by phase, and the worst of them.

    The worst of the three is the figure 3.4, the Figure 2e caption, 4.3.7 and
    the Conclusions all quote as the agreement between the two routes.
    """
    diffs = {phase: 100.0 * (calculated[phase] - measured[phase])
             for phase in PHASES}
    return diffs, max(abs(d) for d in diffs.values())


def report(rho_basalt, rho_flax, label):
    """One density pair, against the imaging, printed."""
    calc = fractions(rho_basalt, rho_flax)
    meas = imaged()
    diffs, worst = agreement(calc, meas)

    print(f"\n{label}: basalt {rho_basalt:.2f}, flax {rho_flax:.2f} g/cm3")
    print(f"  {'phase':8s} {'areal mass':>11s} {'imaging':>9s} {'difference':>12s}")
    for phase in PHASES:
        print(f"  {phase:8s} {100 * calc[phase]:10.2f} % {100 * meas[phase]:8.2f} %"
              f" {diffs[phase]:+9.2f} pts")
    print(f"  fibre volume fraction {calc['fibre']:.4f} calculated against "
          f"{meas['fibre']:.4f} measured, "
          f"{100 * (calc['fibre'] - meas['fibre']):+.2f} points apart")
    print(f"  worst disagreement over the three phases: "
          f"{worst:.2f} percentage points")
    return calc, worst


def main():
    fabric = N_PLIES * AREAL_MASS_PER_PLY_G_PER_M2
    print("inputs")
    print(f"  {N_PLIES} plies at {AREAL_MASS_PER_PLY_G_PER_M2:.0f} g/m2, "
          f"{100 * BASALT_MASS_FRACTION:.0f} wt % basalt, so {fabric:.0f} g/m2 "
          f"of fabric, {fabric * BASALT_MASS_FRACTION:.0f} g/m2 of each fibre")
    print(f"  laminate thickness {LAMINATE_THICKNESS_MM:.2f} mm, so "
          f"{1e4 * LAMINATE_THICKNESS_MM / 10.0:.0f} cm3 of laminate per m2")
    print("  the matrix fraction is taken by difference, so it absorbs the "
          "porosity; the")
    print("  imaging excludes voids from its denominator instead")

    report(RHO_BASALT_G_PER_CM3, RHO_FLAX_G_PER_CM3,
           "densities the published fractions require")
    report(*RHO_ELSEWHERE_IN_DEPOSIT,
           label="densities carried in reply_figures.py")

    print("\nneither density is stated in the manuscript. The first pair is "
          "inferred from")
    print("the four published values it reproduces; the manuscript has to print "
          "the pair")
    print("it used where it states the route.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
