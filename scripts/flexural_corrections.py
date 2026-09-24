"""The corrections and departures quoted around the bending test, from the geometry.

Three passages in the paper carry numbers that follow from the specimen geometry
and the standard rather than from a table, and none of them had a script. They
are collected here because they share their inputs: one span, one thickness, one
crosshead rate.

  2.5   the two departures from ASTM D790. The overhang the specimen leaves at
        each support against the overhang the standard asks for, and the crosshead
        rate against the rate the standard computes. The rate departure is then
        turned into a bound on the reported modulus through the strain-rate
        sensitivity measured on the same laminate.

  4.3.5 the Timoshenko bracket. At a span-to-depth ratio of 16 the measured
        flexural modulus still carries a transverse-shear contribution, so it
        understates the true one; correcting for it widens the gap to the
        estimate rather than closing it.

What is measured and what is assumed, because the two are not the same evidence.
The span, the thickness, the specimen length and the crosshead rate are settings.
The strain-rate sensitivity is measured, per class, in the depth-resolved
acquisition, and the bound uses the largest of the four class medians, which is
the flax one: a bound wants the most sensitive constituent, not the average.
The through-thickness shear modulus is neither: 1.5 to 3.0 GPa is a range the
paper assumes, nothing in the deposit measures it, and every figure below that
depends on it inherits that.

Two things a reader should know about the Timoshenko arithmetic. The relation is
inverted exactly here, E_true = E_app / (1 - c E_app), not truncated at first
order; the first-order form would give 6.2 % where the exact one gives 6.6, and
the paper prints 7. And the paper's own bracket is printed to the unit, so the
gaps it quotes next, 5 % and 14 %, follow from the rounded 3 and 7 % rather than
from the exact 3.0 and 6.6: carrying the exact correction through gives 5.0 and
13.8 %. Both chains are printed below. The shear coefficient, 6/5 for a
rectangular section, and the exact inversion are both stated in 4.3.5 now; for
two rounds the paragraph gave neither, and the bracket could not be reproduced
from the page.

Run: python flexural_corrections.py
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

# Specimen and test settings, from 2.5.
SPECIMEN_LENGTH_MM = 80.0
THICKNESS_MM = 4.60
SPAN_TO_THICKNESS = 16.0
CROSSHEAD_MM_PER_MIN = 2.5

# ASTM D790: the rate of straining of the outer fibre is 0.01 per minute, and the
# specimen must overhang each support by at least 10 % of the span.
D790_OUTER_FIBRE_STRAIN_RATE_PER_MIN = 0.01
D790_MIN_OVERHANG_FRACTION = 0.10

# Assumed in 4.3.5, measured nowhere in this deposit.
SHEAR_MODULUS_RANGE_GPA = (1.5, 3.0)
# Rectangular section, Timoshenko; 4.3.5 states it.
SHEAR_COEFFICIENT = 6.0 / 5.0

# The bracket as 4.3.5 prints it, to the unit. Used only to reproduce the gaps
# that paragraph quotes next, which are computed from it and not from the exact
# correction.
PRINTED_BRACKET_PCT = (3.0, 7.0)


def span_mm():
    return SPAN_TO_THICKNESS * THICKNESS_MM


def overhang_mm():
    """What the specimen leaves beyond each support, from its length and the span."""
    return 0.5 * (SPECIMEN_LENGTH_MM - span_mm())


def required_overhang_mm():
    return D790_MIN_OVERHANG_FRACTION * span_mm()


def d790_rate_mm_per_min():
    """Crosshead rate for the standard's outer-fibre strain rate, R = Z L^2 / 6 d."""
    return (D790_OUTER_FIBRE_STRAIN_RATE_PER_MIN * span_mm() ** 2
            / (6.0 * THICKNESS_MM))


def rate_excess_pct():
    """By how much the rate used exceeds the one the standard computes."""
    return 100.0 * (CROSSHEAD_MM_PER_MIN / d790_rate_mm_per_min() - 1.0)


def strain_rate_sensitivity():
    """Median strain-rate sensitivity index per class, depth-resolved acquisition."""
    depth = pd.read_csv(DATA / "classified_depth_resolved.csv")
    depth = depth[depth.constituent != "QC/invalid"]
    return {cls: float(g.SRS.median())
            for cls, g in depth.groupby("constituent")}


def rate_bound_pct(m=None):
    """Bound on the reported modulus from the rate departure, with E ~ rate^m.

    The exponent defaults to the largest class median, which is what makes this a
    bound rather than an estimate.
    """
    if m is None:
        m = max(strain_rate_sensitivity().values())
    ratio = CROSSHEAD_MM_PER_MIN / d790_rate_mm_per_min()
    return 100.0 * (ratio ** m - 1.0)


def shear_correction_pct(e_apparent_GPa, g_GPa):
    """How much a Timoshenko correction raises an apparent flexural modulus.

    Exact inversion of 1/E_app = 1/E_true + k / (S^2 G), with S the span-to-depth
    ratio and k the shear coefficient of the section.
    """
    c = SHEAR_COEFFICIENT / (SPAN_TO_THICKNESS ** 2 * g_GPa)
    x = c * e_apparent_GPa
    return 100.0 * x / (1.0 - x)


def gap_pct(e_measured_GPa, estimate_GPa, raise_pct=0.0):
    """Relative gap of a measurement above the estimate, after raising it."""
    corrected = e_measured_GPa * (1.0 + raise_pct / 100.0)
    return 100.0 * (corrected - estimate_GPa) / estimate_GPa


def measured_means():
    """Mean toe-compensated chord modulus per configuration, from the replicates."""
    flex = pd.read_csv(DATA / "flexural_moduli.csv")
    return {cfg: float(g.E_D790_toe_compensated_GPa.mean())
            for cfg, g in flex.groupby("configuration")}


def bracket(means=None):
    """The correction bracket over both configurations and both shear moduli.

    Low end: the softer measurement at the stiffer assumed shear modulus. High
    end: the stiffer measurement at the softer one.
    """
    means = means or measured_means()
    g_low, g_high = min(SHEAR_MODULUS_RANGE_GPA), max(SHEAR_MODULUS_RANGE_GPA)
    lowest = shear_correction_pct(min(means.values()), g_high)
    highest = shear_correction_pct(max(means.values()), g_low)
    return lowest, highest


def main():
    print("ASTM D790 departures (2.5)")
    print(f"  span {span_mm():.1f} mm at a span-to-thickness ratio of "
          f"{SPAN_TO_THICKNESS:.0f}, thickness {THICKNESS_MM:.2f} mm")
    print(f"  overhang per end {overhang_mm():.2f} mm against "
          f"{required_overhang_mm():.2f} mm required")
    print(f"  crosshead {CROSSHEAD_MM_PER_MIN:.1f} mm/min against "
          f"{d790_rate_mm_per_min():.2f} mm/min computed from the standard, "
          f"{rate_excess_pct():.0f} % higher")

    srs = strain_rate_sensitivity()
    print("\n  strain-rate sensitivity index, class medians")
    for cls in sorted(srs, key=srs.get):
        print(f"    {cls:13s} {srs[cls]:.3f}")
    worst = max(srs, key=srs.get)
    print(f"  the bound takes the largest, {worst} at {srs[worst]:.3f}, and the "
          f"rate ratio above:")
    print(f"  the reported modulus is bounded at {rate_bound_pct():.1f} % by the "
          f"rate departure")

    means = measured_means()
    print("\nTimoshenko shear correction (4.3.5)")
    print(f"  assumed through-thickness shear modulus "
          f"{min(SHEAR_MODULUS_RANGE_GPA):.1f} to "
          f"{max(SHEAR_MODULUS_RANGE_GPA):.1f} GPa, assumed and not measured; "
          f"shear coefficient {SHEAR_COEFFICIENT:.1f}")
    for cfg in sorted(means):
        for g in sorted(SHEAR_MODULUS_RANGE_GPA):
            print(f"  {cfg:12s} {means[cfg]:6.2f} GPa at G = {g:.1f} GPa: "
                  f"+{shear_correction_pct(means[cfg], g):.2f} %")
    lowest, highest = bracket(means)
    print(f"  bracket over both configurations and both shear moduli: "
          f"{lowest:.2f} to {highest:.2f} %")

    import zone_share as zs                                    # noqa: E402
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    per_zone, medians, _ = zs.zone_fractions(shallow)
    estimate = float(zs.at_share(per_zone, medians, zs.tile_share())[1][0])
    small, large = min(means, key=means.get), max(means, key=means.get)
    print(f"\n  gap to the {estimate:.2f} GPa estimate, before and after "
          f"correcting the measurement")
    print(f"    uncorrected           {gap_pct(means[small], estimate):5.2f} % "
          f"and {gap_pct(means[large], estimate):5.2f} %")
    print(f"    at the exact bracket  "
          f"{gap_pct(means[small], estimate, lowest):5.2f} % and "
          f"{gap_pct(means[large], estimate, highest):5.2f} %")
    print(f"    at the printed {PRINTED_BRACKET_PCT[0]:.0f} and "
          f"{PRINTED_BRACKET_PCT[1]:.0f} %  "
          f"{gap_pct(means[small], estimate, PRINTED_BRACKET_PCT[0]):5.2f} % and "
          f"{gap_pct(means[large], estimate, PRINTED_BRACKET_PCT[1]):5.2f} %")
    print("  4.3.5 prints 5 and 14 %, which is the last line rounded. The exact "
          "bracket now")
    print("  rounds to the same pair, so the two chains no longer disagree on "
          "the page,")
    print("  but they still differ by half a point at the upper end and both "
          "are printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
