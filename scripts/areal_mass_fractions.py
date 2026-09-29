"""The areal-mass route to the constituent volume fractions quoted in 3.4.

3.4 reports a second, independent route to the constituent volume fractions: not
the segmented section, but the fabric specification. Sixteen plies of a 360 g/m2
fabric at a 50/50 mass split, spread through a 4.60 mm laminate, fix the basalt
and flax volumes from their densities; the matrix follows by difference and so
absorbs whatever porosity there is. This script is where the deposit computes
it.

Two things about the imaging it is compared against also live here, because
the comparison is where they are needed: `imaged()` reads the segmented
fractions, and `imaged_intervals()` computes the +/- printed beside them in
Figure 2e and 3.4, so that their recipe is in the deposit.

What the route needs, and where each input comes from:

    sixteen plies            2.1, "a sixteen-ply laminate"
    360 g/m2 per ply         2.1, the fabric's nominal areal weight
    50 wt % basalt           2.1, "equal mass fractions of flax and basalt"
    4.60 mm laminate         2.5, the bending specimen section, 80 x 15 x 4.6 mm
    2.70 g/cm3 basalt        3.4, the nominal datasheet density
    1.50 g/cm3 flax          3.4, the nominal datasheet density

The manuscript states the two densities in 3.4, so they are read from the page
like the other inputs. They return
23.2, 41.7 and 35.1 % and a fibre volume fraction of 0.649, which are the four
figures 3.4 and Figure 8 print, all four from one pair of inputs.

The same pair also fixes the volume split between the two fibres, which the
lamination-theory comparison of 3.4, Figure 8 and 4.2 needs: at a 50/50 mass
split the basalt takes 1/2.70 of the fibre volume for every 1/1.50 the flax
takes. `basalt_share_of_fibre_volume()` is that number, and it is the one place
in the deposit it is computed. Every script that needs the split takes it from
here, directly or through `fig08_rom`, so there is one density pair and one
lamination-theory prediction.

Run: python areal_mass_fractions.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

N_PLIES = 16
AREAL_MASS_PER_PLY_G_PER_M2 = 360.0
BASALT_MASS_FRACTION = 0.50
LAMINATE_THICKNESS_MM = 4.60

# The nominal datasheet densities 3.4 states.
RHO_BASALT_G_PER_CM3 = 2.70
RHO_FLAX_G_PER_CM3 = 1.50

PHASES = ("basalt", "flax", "matrix")


def basalt_share_of_fibre_volume(rho_basalt=RHO_BASALT_G_PER_CM3,
                                 rho_flax=RHO_FLAX_G_PER_CM3,
                                 basalt_mass_fraction=BASALT_MASS_FRACTION):
    """The basalt share of the fibre volume, from the mass split and the densities.

    The fabric specification gives the split by mass. Each mass over its density
    is a volume, and the basalt share of the fibre volume is its volume over the
    two. This is what a lamination-theory calculation needs to combine the two
    tabulated fibre moduli into one fibre modulus by volume.
    """
    basalt = basalt_mass_fraction / rho_basalt
    flax = (1.0 - basalt_mass_fraction) / rho_flax
    return basalt / (basalt + flax)


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


# A tile counts towards a zone's field-to-field scatter when that zone covers
# at least this fraction of the tile. Below it the in-zone fibre fraction is
# read off a sliver of a few thousand pixels and is noise rather than a field.
# Any threshold from 4.9 to 5.2 % selects the same 72 basalt-zone tiles, and
# any from 4.3 to 7.8 % the same 88 flax-zone tiles.
MIN_ZONE_COVERAGE = 0.05


def imaged_intervals(min_coverage=MIN_ZONE_COVERAGE):
    """The +/- on the imaged fibre fractions, Figure 2e and 3.4, in percentage points.

    The recipe behind the 1.1 and 3.2 printed there, which the summary file
    carries as recorded values and which nothing in the deposit computed until
    this function. For each fibre zone: the in-zone fibre fraction of every tile
    the zone covers by at least `min_coverage` of its pixels, one value per
    tile, unweighted; its standard deviation (n - 1) over the square root of the
    tile count is the standard error of the mean; 1.96 of those is the 95 %
    half-width; and that half-width times the zone's share of the marked mosaic
    area, from the stitched mask, is the interval on the volume fraction. The
    matrix takes the remainder and carries no interval of its own.

    The zone share is the mask's, 0.367 basalt, and not the tile count's, 0.362,
    because the volume fractions themselves are built on the mask share, and the
    interval scales the quantity it is an interval on. The tiles are unweighted
    because the interval is field-to-field scatter, one field one vote, which is
    what the Figure 2 caption says it is. Reproduces both recorded values to the
    last digit; the tolerance in `reproduce_paper_values.py` is a rounding one
    because the page prints one decimal.

    Returns, per fibre phase, the half-width and the tile count it rests on.
    """
    tiles = pd.read_csv(DATA / "bse_segmentation_tiles.csv")
    seg = json.loads((DATA / "bse_segmentation_summary.json").read_text())
    zone = seg["zone_area_from_mosaic_mask"]
    marked = zone["basalt"] + zone["flax"]
    out = {}
    for phase in ("basalt", "flax"):
        covers = tiles[f"n_{phase}zone"] >= min_coverage * tiles["n_px"]
        fibre = tiles.loc[covers, f"{phase}zone_fib"].to_numpy(float)
        n = len(fibre)
        half_width = 1.96 * fibre.std(ddof=1) / np.sqrt(n)
        out[phase] = {"half_width_pct": 100.0 * half_width * zone[phase] / marked,
                      "n_tiles": int(n)}
    return out


def agreement(calculated, measured):
    """Differences in percentage points, phase by phase, and the worst of them.

    The worst of the three is the figure 3.4, the Figure 2e caption, 4.3.3 and
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
           "the nominal datasheet densities stated in 3.4")

    vb = basalt_share_of_fibre_volume()
    print(f"\n  basalt share of the fibre volume at that pair: {vb:.4f}, so a "
          f"fibre modulus by volume of\n  {vb:.4f} x 80 + {1 - vb:.4f} x 60 = "
          f"{vb * 80 + (1 - vb) * 60:.2f} GPa for the lamination-theory "
          f"comparison")

    ci = imaged_intervals()
    print("\n95 % intervals on the imaged fibre fractions, from field-to-field "
          "scatter")
    print(f"  tiles count when the zone covers at least "
          f"{100 * MIN_ZONE_COVERAGE:.0f} % of the tile")
    for phase in ("basalt", "flax"):
        print(f"  {phase:8s} +/- {ci[phase]['half_width_pct']:.2f} percentage "
              f"points over {ci[phase]['n_tiles']} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
