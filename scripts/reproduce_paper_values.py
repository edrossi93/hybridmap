"""
Recompute every number the paper prints, from the deposited tables, and put each
recomputed value beside the published one.

Run this first if you want to know whether the deposit is intact. It takes a few
seconds, needs nothing but the files in `data/`, and writes nothing.

What it answers. Whether each printed value follows from the deposited
measurements under the construction the paper describes: the two-level
construction of Eq. (1), with the class fractions and class medians measured
inside each architectural zone and the share of the two zones measured on the
imaged tiles. Every value is computed here or read from a deposited product whose
hash `provenance_check.py` holds to its own script. Nothing is a literal copied
from the manuscript except the published value each line is compared against.

What it cannot answer. It does not test whether the construction is the right
one; the sensitivity analyses do that, and they are checked here as published
numbers rather than as arguments. It does not check the acquisition either: the
per-indent tables are where this deposit starts.

What the count means. Every line marked as published compares a value printed
in the paper or its supplement with the value recomputed here, at the precision
it is printed. Lines marked as internal hold the deposit against
itself: a stored product against the live computation of the same quantity, or
an invariant of printed numbers. A check against a value no document carries is
not kept, because it would report success on nothing.

A few printed statements cannot be recomputed from what is deposited: the
dark-feature figures of §3.1, which are measured on the raw tile images; the
zone boundary at x = 730 µm, a setting read from a moving-median profile; and
the loading-rate ratios of Table 1, which are acquisition settings. They are
named at the end of the run, so that a reader does not have to find the
boundary of the deposit by looking for a check that is not there.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

from classify import classify_shallow                      # noqa: E402
from rule_of_mixtures import class_table                   # noqa: E402
from clustering_check import run as cluster_run            # noqa: E402
from class_intervals import intervals as _intervals       # noqa: E402
import zone_share as zs                                    # noqa: E402
import phase_angle_sensitivity as _pas                     # noqa: E402
import border_rejection as _border                         # noqa: E402
import flexural_comparisons as _fx                         # noqa: E402
import input_decomposition as _dec                         # noqa: E402
import areal_mass_fractions as _am                         # noqa: E402
import flexural_corrections as _flex                       # noqa: E402
import spacing_compliance as _spacing                      # noqa: E402

# Two lists, because they answer two different questions and only the first is
# what this deposit promises. `PUBLISHED` holds one entry per value printed in the
# paper or its supplement. `INTERNAL` holds checks
# that hold the deposit against itself: a stored product against the live
# computation of the same quantity, or an invariant of numbers that are printed.
# An internal check earns its place by catching a stale file, but none of them is
# evidence about a published number, so they are counted apart from them and the
# headline count is the first list alone.
PUBLISHED = []
INTERNAL = []

# Half a unit in a figure the paper prints to one decimal, and a hair more. An
# exact decimal such as 21.65 is stored just below itself, so a half-unit
# tolerance written as 0.05 rejects the value the table rounds it to. The same
# applies to the low end of the tabulated range in 3.4, which is exactly 28.25
# and is printed 28.3.
#
# One tolerance covers every cell: the modulus term of 3.4, 10.746, is printed
# 10.7, and the last gap cell of Supplementary Table S8 is printed 50.4 on the
# 45.30 GPa flax input, which the recomputation reproduces.
ROUNDED = 0.051


def report(label, got, printed, tol):
    """A recomputation against a value printed in the paper or its supplement.

    `printed` is a literal read off the page, at the precision the page carries.
    Nothing else belongs in this function: a comparison against a quantity the
    paper does not print is not a check on the paper, whatever else it is.
    """
    ok = abs(got - printed) <= tol
    PUBLISHED.append(ok)
    print(f"  [{'ok ' if ok else 'DIFF'}] {label:<52} {got:>9.3f}   paper {printed}")


def report_same(label, got, expected):
    """A deposited product against the live computation of the same quantity.

    Not a published value. These lines exist so that a stale product cannot pass
    the checks that read it.
    """
    ok = abs(got - expected) <= 1e-9
    INTERNAL.append(ok)
    print(f"  [{'ok ' if ok else 'DIFF'}] {label:<52} {got:>9.3f}   "
          f"recomputed here")


def report_internal(label, got, expected, tol, against):
    """A check on the deposit that is not a check on a printed value.

    `against` says what the number on the right is, so that a reader cannot mistake
    it for something the paper prints.
    """
    ok = abs(got - expected) <= tol
    INTERNAL.append(ok)
    print(f"  [{'ok ' if ok else 'DIFF'}] {label:<52} {got:>9.3f}   "
          f"{against}")


def report_below(label, got, limit):
    """For a cell the paper prints as an inequality rather than as a value."""
    PUBLISHED.append(got < limit)
    print(f"  [{'ok ' if got < limit else 'DIFF'}] {label:<52} {got:>9.2e}   "
          f"paper < {limit}")


def report_above(label, got, limit):
    """For a value the paper prints as a lower bound, "over N"."""
    PUBLISHED.append(got > limit)
    print(f"  [{'ok ' if got > limit else 'DIFF'}] {label:<52} {got:>9.3f}   "
          f"paper over {limit}")


def report_sci(label, got, mantissa, exponent):
    """For a value the paper prints in scientific form, m x 10^e.

    Compared on the mantissa at the printed exponent, to half a unit in the
    last printed digit and a hair more, so that 2.25e-12 passes against the
    printed 2.3 x 10^-12 and 6.52e-8 against 6.5 x 10^-8.
    """
    scaled = got / 10.0 ** exponent
    ok = abs(scaled - mantissa) <= ROUNDED
    PUBLISHED.append(ok)
    print(f"  [{'ok ' if ok else 'DIFF'}] {label:<52} {got:>9.2e}   "
          f"paper {mantissa} x 10^{exponent}")


print("classification, load-controlled map")
raw = pd.read_csv(DATA / "shallow_map.csv")
lab = classify_shallow(raw)
report("indents acquired", len(raw), 8553, 0)
report("indents after the resin cut", len(lab), 5553, 0)
counts = lab.constituent.value_counts()
for cls, n in [("flax", 2300), ("matrix", 894), ("mixed-volume", 1977),
               ("basalt", 380)]:
    report(f"n({cls})", int(counts.get(cls, 0)), n, 0)
classified = int(sum(counts.get(c, 0) for c in zs.CLASSES))
report("indents classified", classified, 5551, 0)
report("indents neither classified nor on resin", len(lab) - classified, 2, 0)
# The per-zone counts of 3.3. They are printed as a chain, and the chain is only
# worth printing if it closes, so the totals are checked with the parts.
for zone, n in (("basalt", 2760), ("flax", 2793)):
    report(f"n(indents in the {zone}-rich zone)", int((lab.zone == zone).sum()),
           n, 0)
cross = pd.crosstab(lab.zone, lab.constituent)
for zone, cls, n in [("basalt", "matrix", 894), ("basalt", "mixed-volume", 1525),
                     ("basalt", "basalt", 340), ("flax", "flax", 2300),
                     ("flax", "mixed-volume", 452), ("flax", "basalt", 40)]:
    report(f"n({cls}) in the {zone}-rich zone", int(cross.loc[zone, cls]), n, 0)
for cls, pct in [("flax", 41.4), ("matrix", 16.1), ("mixed-volume", 35.6),
                 ("basalt", 6.8)]:
    report(f"{cls}, per cent of the classified indents",
           100 * counts.get(cls, 0) / classified, pct, 0.05)

print("\nper-class moduli and intervals, load-controlled map (Table 3)")
sh = pd.read_csv(DATA / "classified_shallow.csv")
tab = class_table(sh)
_sh_ci = _intervals(sh, "E_GPa").set_index("class")
for cls, E, lo, hi in [("flax", 4.70, 4.56, 4.84), ("matrix", 4.66, 4.31, 4.99),
                       ("mixed-volume", 35.79, 34.4, 37.0),
                       ("basalt", 85.99, 83.7, 87.8)]:
    report(f"median E({cls})",
           float(tab.loc[tab["class"] == cls, "E_median_GPa"].iloc[0]), E, 0.02)
    report(f"CI low  E({cls})", float(_sh_ci.ci_low_GPa[cls]), lo, 0.05)
    report(f"CI high E({cls})", float(_sh_ci.ci_high_GPa[cls]), hi, 0.05)

# The depth-resolved column of Table 3, and the quality mask it depends on.
#
# Without this block nothing would compare the printed column against the data,
# and the manuscript and the supplement could print different counts for the
# same mask. Every cell of that column is checked here, the interval included,
# because an interval is as much a published number as a median.
print("\nper-class values, depth-resolved map at z = 150 nm (Table 3)")
_dr_all = pd.read_csv(DATA / "classified_depth_resolved.csv")
report("depth-resolved indents acquired", len(_dr_all), 2000, 0)
report("rejected by the phase-angle criterion",
       int(_dr_all.qc_flag_strict.sum()), 304, 0)
report("rejected, per cent", 100 * _dr_all.qc_flag_strict.mean(), 15.2, 0.05)
# The interval comes from the script that owns the caption's recipe, not from a
# second bootstrap written here. Two implementations of the same recipe drift,
# and then the table and the check disagree with each other rather than with the
# data.
_dr_ci = _intervals(_dr_all, "E_150").set_index("class")
# Every figure below is the one Table 3 prints, at the precision it prints it.
# The modulus column of this table does not print to one precision: the two compliant
# classes are given to two decimals and the two stiff ones to one, because the
# table follows the significance of the value and not a format string. The
# tolerance follows the cell, so `tol` is carried per row. Half a unit in the last
# printed digit, and a hair more, exactly as ROUNDED is for a one-decimal cell.
for cls, n, E, lo, hi, H, tol in [
        ("flax", 1118, 4.10, 3.97, 4.27, 0.20, 0.0051),
        ("matrix", 115, 5.50, 4.78, 6.55, 0.21, 0.0051),
        ("mixed-volume", 221, 29.1, 27.8, 31.3, 1.58, ROUNDED),
        ("basalt", 242, 54.7, 53.4, 57.1, 3.98, ROUNDED)]:
    g = _dr_all[_dr_all.constituent == cls]
    report(f"n({cls}), depth-resolved", len(g), n, 0)
    report(f"median E({cls}) @ 150 nm", float(g.E_150.median()), E, tol)
    report(f"median H({cls}) @ 150 nm", float(g.H_150.median()), H, 0.0051)
    report(f"CI low  E({cls}) @ 150 nm", float(_dr_ci.ci_low_GPa[cls]), lo, tol)
    report(f"CI high E({cls}) @ 150 nm", float(_dr_ci.ci_high_GPa[cls]), hi, tol)
# 2.3 and Table 2: why the depth-resolved cut is at 40 GPa and not 70. The
# basalt class is read at 150 nm, so the count is of its members below 70 GPa
# there.
_dr_basalt = _dr_all[_dr_all.constituent == "basalt"]
report("basalt contacts of the depth-resolved class a 70 GPa cut would reject",
       int((_dr_basalt.E_150 <= 70).sum()), 197, 0)

# The four strain-rate sensitivity medians of 3.2, of which 3.2 and 4.3.4 quote the
# flax and matrix ones as the evidence that the compliant classes are separable
# where the measurement can resolve them. The flax one is also what bounds the
# rate departure of 2.5 below, so it is checked where it is measured.
_srs = _flex.strain_rate_sensitivity()
for cls, printed in (("basalt", 0.021), ("mixed-volume", 0.028),
                     ("matrix", 0.062), ("flax", 0.073)):
    report(f"strain-rate sensitivity, {cls}", _srs[cls], printed, 0.0005)

# The rest of what 3.2 reads off the depth-resolved curves. The peak load per class is the median over its indents of
# the largest load each trace records, which on a depth-controlled acquisition
# is the load at the 300 nm target; the two shares reaching 2 mN are counted on
# the same per-indent maximum. The decline of the two stiff medians between 100
# and 300 nm is read on the route Figure 5b draws, so the sentence and the panel
# are held to one computation: `fig05_curves_per_class.resample` puts every
# curve on the figure's 5 nm grid and the class median is taken at each depth.
print("\ndepth-resolved loading curves and depth signatures (3.2, Figure 5)")
from fig05_curves_per_class import resample as _fig05_resample, GRID as _fig05_grid  # noqa: E402
_curves = pd.read_csv(DATA / "depth_resolved_curves.csv")
_peak = _curves.groupby("indent_id").load_mN.max()
for cls, printed in (("flax", 0.37), ("matrix", 0.47), ("mixed-volume", 2.39),
                     ("basalt", 5.18)):
    ids = _dr_all.loc[_dr_all.constituent == cls, "indent_id"]
    report(f"median peak load, {cls}, mN", float(_peak.reindex(ids).median()),
           printed, 0.0051)
for cls, printed in (("flax", 0.1), ("matrix", 4)):
    ids = _dr_all.loc[_dr_all.constituent == cls, "indent_id"]
    report(f"{cls} indents reaching 2 mN, per cent",
           100 * float((_peak.reindex(ids) >= 2.0).mean()), printed,
           ROUNDED if printed < 1 else 0.5)
for cls, at_100, at_300 in (("basalt", 61, 40), ("mixed-volume", 31, 22)):
    ids = _dr_all.loc[_dr_all.constituent == cls, "indent_id"]
    stack = np.nanmedian(_fig05_resample(_curves, ids, "E_GPa"), axis=0)
    grid = list(_fig05_grid)
    report(f"median E({cls}) at 100 nm, Figure 5b route",
           float(stack[grid.index(100.0)]), at_100, 0.5)
    report(f"median E({cls}) at 300 nm, Figure 5b route",
           float(stack[grid.index(300.0)]), at_300, 0.5)

# The per-band counts of 3.3: the basalt-centre band's classified contacts and
# the three classes inside it, with their shares of the band, and the stiff
# contacts the two flax-rich bands hold above the 40 GPa cut. Printed as a
# chain, like the per-zone counts of the load-controlled map above, and checked
# as one.
print("\nper-band counts of the depth-resolved map (3.3)")
_bands = _dr_all[_dr_all.constituent != "QC/invalid"]
_centre = _bands[_bands.zone == "basalt-centre"]
report("classified contacts in the basalt-centre band", len(_centre), 534, 0)
for cls, n, pct in (("matrix", 115, 22), ("mixed-volume", 197, 37),
                    ("basalt", 222, 42)):
    report(f"n({cls}) in the basalt-centre band",
           int((_centre.constituent == cls).sum()), n, 0)
    report(f"{cls}, per cent of the band",
           100 * float((_centre.constituent == cls).mean()), pct, 0.5)
for zone, above, of in (("flax-top", 19, 731), ("flax-bottom", 1, 431)):
    g = _bands[_bands.zone == zone]
    report(f"classified contacts in the {zone} band", len(g), of, 0)
    report(f"of them above 40 GPa at 150 nm", int((g.E_150 > 40).sum()), above, 0)

# Supplementary Table S1 has a script behind it, so every figure it prints,
# 89.05 GPa included, has a route from the data.
# Where the rejections fall, which 2.2 reports in two sentences. The border
# definition is imported rather than repeated: `border_rejection.py` owns it, and
# `border_effect.py` uses the same one on the other map.
# The spacing test of 2.2 and the depths of Table 1. The contact depths are the
# instrument's own, read from the deposited column; the maximum depths are the
# deposited column the spacing criterion is applied to, indent by indent.
print("\nindent spacing and contact depth (2.2, Table 1)")
_sp = json.loads((DATA / "spacing_compliance.json").read_text())
_sp_rows = {r["class"]: r for r in _sp["rows"]}
for cls, depth, hmax, hardness in (("basalt", 41, 56, 6.98),
                                   ("mixed-volume", 78, 97, 1.94),
                                   ("flax", 308, 350, 0.21),
                                   ("matrix", 325, 366, 0.19)):
    g = sh[sh.constituent == cls]
    report(f"median contact depth, {cls}, nm",
           float(g.contact_depth_nm.median()), depth, 0.51)
    report(f"median maximum depth, {cls}, nm",
           float(g.max_depth_nm.median()), hmax, 0.51)
    report(f"median hardness, {cls}, GPa", float(g.H_GPa.median()), hardness,
           0.0051)
for cls, short, long_ in (("flax", 84, 19), ("matrix", 82, 16)):
    report(f"{cls} indents breaching the criterion, short axis, per cent",
           _spacing.breaching_pct(sh[sh.constituent == cls], cls, "short"),
           short, 0.5)
    report(f"{cls} indents breaching the criterion, long axis, per cent",
           _spacing.breaching_pct(sh[sh.constituent == cls], cls, "long"),
           long_, 0.5)
# The two stiff classes are printed as meeting the rule: every basalt contact
# and 98 % of mixed-volume contacts on the short axis, every one of both on the
# long axis.
for cls, short in (("basalt", 100), ("mixed-volume", 98)):
    report(f"{cls} indents meeting the criterion, short axis, per cent",
           100 - _spacing.breaching_pct(sh[sh.constituent == cls], cls, "short"),
           short, 0.5 if short < 100 else 0)
    report(f"{cls} indents meeting the criterion, long axis, per cent",
           100 - _spacing.breaching_pct(sh[sh.constituent == cls], cls, "long"),
           100, 0)
report("short-axis pitch of the load-controlled map, um",
       _sp["load_controlled_pitch_nm"]["short"] / 1000, 2.48, 0.005)
report("long-axis pitch, um", _sp["load_controlled_pitch_nm"]["long"] / 1000,
       5.10, 0.005)
report("pitch the flax median maximum depth calls for, um",
       _sp_rows["flax"]["median_max_depth_nm"] * 10 / 1000, 3.5, ROUNDED)
report("pitch the matrix median maximum depth calls for, um",
       _sp_rows["matrix"]["median_max_depth_nm"] * 10 / 1000, 3.7, ROUNDED)
report_same("spacing product, flax short-axis share against the live count",
            _sp_rows["flax"]["breaching_short_axis_pct"],
            round(_spacing.breaching_pct(sh[sh.constituent == "flax"], "flax",
                                         "short"), 1))
report_internal("depth-resolved map, pitch over maximum depth",
                _sp["depth_resolved"]["pitch_over_max_depth"], 10, 0,
                "behind 'sits exactly at the rule'")

print("\nwhere the rejected indents fall (2.2)")
_edge = _border.rejection_rates(_dr_all, "x_um")
report("rejected on the two outer rows of the short axis",
       _edge["border"]["pct"], 15.5, ROUNDED)
report("rejected inside them", _edge["interior"]["pct"], 15.2, ROUNDED)
report("indents on the two outer rows of the short axis", _edge["border"]["n"],
       200, 0)
report("indents inside them", _edge["interior"]["n"], 1800, 0)
for zone, published in (("basalt-centre", 11.0), ("flax-bottom", 20.2),
                        ("flax-top", 15.0)):
    g = _dr_all[_dr_all.zone == zone]
    report(f"rejected in the {zone} band", 100 * g.qc_flag_strict.mean(),
           published, ROUNDED)

print("\nPoisson-ratio sensitivity (Supplementary Table S1)")
import poisson_sensitivity as _poi                          # noqa: E402
_s1 = _poi.table().set_index("class")
for cls, uniform, phase in (("flax", 4.70, 4.70), ("matrix", 4.66, 4.22),
                            ("mixed-volume", 35.79, 34.78),
                            ("basalt", 85.99, 89.05)):
    report(f"median E({cls}), uniform nu",
           float(_s1.E_uniform_nu_GPa[cls]), uniform, 0.005)
    report(f"median E({cls}), phase-specific nu",
           float(_s1.E_phase_specific_GPa[cls]), phase, 0.005)
_est = _poi.estimates(_poi.table())
report("estimate at phase-specific Poisson ratios", _est["phase_specific"],
       18.31, 0.005)

# Eq. (1) as the paper builds it. The fractions are mixed inside each zone and
# the two zones are weighted by the share they occupy on the imaged section, not
# by how many indents landed in each. Weighting by indent counts is the
# alternative construction; it is kept below because Table 2, 3.4 and the
# Figure 8 caption all print what it gives, as the comparison that motivates
# the section weighting.
print("\nEq. (1): the two-level construction (Table 2, Table 3, 2.4, 3.4)")
per_zone, medians, w_indents = zs.zone_fractions(sh)
w_tiles = zs.tile_share()
report("basalt-rich share of the section, imaged tiles", 100 * w_tiles, 36.2, 0.05)
report("flax-rich share of the section, imaged tiles",
       100 * (1 - w_tiles), 63.8, 0.05)
report("basalt-rich share implied by the indent counts",
       100 * w_indents, 49.7, 0.05)
fractions, (voigt, hill, reuss) = zs.at_share(per_zone, medians, w_tiles)
for cls, f in zip(zs.CLASSES, fractions):
    report(f"Eq. (1) fraction, {cls}", float(f),
           {"flax": 0.526, "matrix": 0.117, "mixed-volume": 0.303,
            "basalt": 0.054}[cls], 0.0005)
report_internal("Eq. (1) fractions, sum", float(fractions.sum()), 1.000, 1e-9,
                "an invariant of the four printed fractions")
report("laminate estimate", voigt, 18.50, 0.005)
report("Hill average", hill, 12.7, 0.05)
report("Reuss value", reuss, 6.8, 0.05)
report("estimate with the zones weighted by indent counts",
       zs.at_share(per_zone, medians, w_indents)[1][0], 21.33, 0.005)


# Where the sum comes from, class by class. 4.3.4 and 4.3.6 quote these as the
# reason the flax input matters less than it looks.
E_class = np.array([medians[c] for c in zs.CLASSES])
terms = dict(zip(zs.CLASSES, fractions * E_class))
compliant = terms["flax"] + terms["matrix"]
report_internal("compliant classes, contribution to the estimate", compliant,
                3.02, 0.005, "behind the printed 16 %; 3.019 is printed in 4.3.2")
report("compliant classes, per cent of the estimate",
       100 * compliant / voigt, 16, 0.5)
report_internal("mixed-volume, contribution to the estimate",
                terms["mixed-volume"], 10.9, 0.05,
                "behind the printed 59 %; the 10.9 GPa of 3.4 is the "
                "flax-rich zone sum")
report("mixed-volume, per cent of the estimate",
       100 * terms["mixed-volume"] / voigt, 59, 0.5)
report("pure basalt, per cent of the estimate",
       100 * terms["basalt"] / voigt, 25, 0.5)
report_internal("the two stiff classes, contribution to the estimate",
                terms["mixed-volume"] + terms["basalt"], 15.48, 0.005,
                "behind the printed 84 %; not printed")
report("the two stiff classes, per cent of the estimate",
       100 * (terms["mixed-volume"] + terms["basalt"]) / voigt, 84, 0.5)
# Interchanging the two compliant labels, which is the residual ambiguity of
# 4.3.5. Only the two medians swap; the fractions are unchanged.
swapped = dict(medians)
swapped["flax"], swapped["matrix"] = medians["matrix"], medians["flax"]
compliant_swapped = float(fractions[0] * swapped["flax"]
                          + fractions[1] * swapped["matrix"])
report("compliant contribution, labels interchanged", compliant_swapped,
       3.001, 0.005)
report("what interchanging the two labels moves",
       compliant - compliant_swapped, 0.018, 0.001)

# The exclusion test in the Supplementary Figure S3 caption.
# Fifty-nine contacts in the flax-rich zone read below 1 GPa; the caption argues
# they are runaway indents rather than soft material, and reports what removing
# them would do. The whole caption is checked here, the evidence for the argument
# included, because the argument is what licenses leaving them in.
print("\nthe exclusion test of the Supplementary Figure S3 caption")
# The definition of a runaway contact is the budget's, imported so that the
# caption's exclusion test and the budget's sixth term cannot come to describe
# two different sets of indents.
from uncertainty_budget import runaway_contacts as _runaway_mask    # noqa: E402
_flax_zone = sh[sh.zone == "flax"]
_runaway = sh[_runaway_mask(sh)]
report("flax-rich indents above 15 GPa", int((_flax_zone.E_GPa > 15).sum()),
       492, 0)
report("flax-rich contacts below 1 GPa", len(_runaway), 59, 0)
report("their median contact depth, um",
       float(_runaway.contact_depth_nm.median()) / 1000, 3.2, 0.051)
report("median contact depth of the rest of the zone, nm",
       float(_flax_zone[_flax_zone.E_GPa >= 1.0].contact_depth_nm.median()),
       274, 0.51)
report("their median hardness, GPa", float(_runaway.H_GPa.median()), 0.002,
       0.0005)
report("median hardness of the rest of the zone, GPa",
       float(_flax_zone[_flax_zone.E_GPa >= 1.0].H_GPa.median()), 0.252, 0.0005)
# Both the flax median and the class fractions move when 59 indents leave, so the
# estimate is rebuilt from the surviving indents rather than by substituting one
# median into the construction.
_kept = sh[~_runaway_mask(sh)]
report("flax median with them excluded",
       float(_kept.loc[_kept.constituent == "flax", "E_GPa"].median()), 4.78,
       0.005)
_pz_kept, _med_kept, _ = zs.zone_fractions(_kept)
report("estimate with them excluded",
       float(zs.at_share(_pz_kept, _med_kept, w_tiles)[1][0]), 18.63, 0.005)

print("\nuncertainty budget (Supplementary Table S5, 2.4)")
# Read from the deposited product rather than recomputed here: the budget is
# tens of thousands of resamplings and `provenance_check.py` holds the file to
# `uncertainty_budget.py`. The point estimate and the zone share are checked
# against the live computation above, so a stale budget cannot pass quietly.
budget = json.loads((DATA / "uncertainty_budget.json").read_text())
report_same("budget, point estimate", budget["voigt_point_estimate"], voigt)
report_same("budget, basalt-rich share used", budget["w_basalt_mosaic"], w_tiles)
report("u, class fractions within each zone",
       budget["type_a_sd_GPa"]["fractions"], 1.080, 0.001)
report("u, class medians", budget["type_a_sd_GPa"]["medians"], 0.345, 0.001)
report("u, share of the two zones", budget["type_a_sd_GPa"]["zone_share"],
       0.611, 0.001)
report("u, classification thresholds",
       budget["type_b_sd_GPa"]["classification_thresholds"], 0.243, 0.001)
report("u, Poisson ratio", budget["type_b_sd_GPa"]["poisson_ratio"], 0.056,
       0.001)
# The sixth row: the fifty-nine runaway contacts of the
# Supplementary Figure S3 caption, as a rectangular term over the estimate with
# them in and with them out. The range is held to the two estimates computed
# live above, so the budget cannot carry a stale pair; the standard uncertainty
# and its share are held to what Table S5 prints, 0.037 GPa and 0.1 %. The
# combined value does not move at three decimals, which the line below it
# confirms rather than assumes.
_run_lo, _run_hi = budget["type_b_ranges"]["runaway_contacts_in_and_out_GPa"]
report_same("runaway term, estimate with the contacts in", _run_lo, voigt)
report_same("runaway term, estimate with them out", _run_hi,
            float(zs.at_share(_pz_kept, _med_kept, w_tiles)[1][0]))
report("u, runaway contacts left in the flax class",
       budget["type_b_sd_GPa"]["runaway_contacts"], 0.037, 0.001)
report("the three sampling terms together",
       budget["type_a_combined_sd_GPa"], 1.42, 0.005)
report("combined standard uncertainty", budget["combined_sd_GPa"], 1.441, 0.001)
report("combined uncertainty, per cent of the estimate",
       100 * budget["combined_sd_GPa"] / voigt, 7.8, 0.05)
# The share-of-variance column of Table S5, one term at a time over the
# combined value squared, which is why the shares do not sum to one. The
# quadrature sum of the three frozen terms is the 1.29 GPa the caption names as
# what they are not to be added to.
for key, printed, tol in (("fractions", 56, 0.5), ("medians", 6, 0.5),
                          ("zone_share", 18, 0.5), ("type_a_combined", 97, 0.5),
                          ("classification_thresholds", 3, 0.5),
                          ("poisson_ratio", 0.1, ROUNDED),
                          ("runaway_contacts", 0.1, ROUNDED)):
    report(f"share of variance, {key}, per cent",
           budget["share_of_variance_pct"][key], printed, tol)
report("quadrature sum of the three frozen sampling terms",
       sum(v ** 2 for v in budget["type_a_sd_GPa"].values()) ** 0.5, 1.29,
       0.005)
report("95 % interval from the draws, low",
       budget["percentile_95_interval_GPa"][0], 15.78, 0.005)
report("95 % interval from the draws, high",
       budget["percentile_95_interval_GPa"][1], 21.34, 0.005)
# The block sweep is 1000 draws per width and its values are the joint sampling
# term alone. The combined form is the sweep value in quadrature with the three
# Type B terms. Both forms are checked, so that either can be printed correctly
# named.
_type_b_sq = sum(v ** 2 for v in budget["type_b_sd_GPa"].values())
for cols, um, alone, combined in (("2", 10, 1.21, 1.23), ("4", 20, 1.38, 1.41),
                                  ("8", 41, 1.68, 1.70)):
    sampled = budget["block_size_sensitivity_sd_GPa"][cols]
    report(f"sampling term alone at a {um} um block, 1000 draws", sampled,
           alone, 0.005)
    report(f"with the Type B terms in quadrature, {um} um block",
           (sampled ** 2 + _type_b_sq) ** 0.5, combined, 0.005)

# Where the map could have sat, which is the other half of the zone-share term:
# a window of the load-controlled map's own footprint slid over the mosaic.
# Written by zone_share.py, read here.
#
# The sliding windows of 3.4: the full footprint of the load-controlled map
# and the 704 um of laminate its retained indents cover, slid over the
# downsampled mosaic mask, whose own whole-mosaic share and estimate the means
# sit at.
window = json.loads((DATA / "zone_share_sensitivity.json").read_text())
report("basalt-rich share of the downsampled mask",
       window["w_basalt_downsampled_mask"], 0.367, 0.0005)
report("estimate at that share",
       float(zs.at_share(per_zone, medians,
                         window["w_basalt_downsampled_mask"])[1][0]),
       18.60, 0.005)
report("placements of the span at or below the mapped share, per cent",
       window["windows"]["load-controlled map, laminate span after the resin cut"]
       ["placements_at_or_below_mapped_share_pct"], 85, 0.5)
placements = window["windows"]["load-controlled map"]
report("placements of the load-controlled footprint on the mosaic",
       placements["n_placements"], 29880, 0)
report("basalt-rich share over those placements, mean",
       placements["share_mean"], 0.369, 0.0005)
report("basalt-rich share over those placements, SD",
       placements["share_sd"], 0.070, 0.0005)
report("estimate over those placements, mean", placements["voigt_mean"], 18.65,
       0.005)
report("estimate over those placements, SD", placements["voigt_sd"], 1.46, 0.005)
span = window["windows"]["load-controlled map, laminate span after the resin cut"]
report("placements of the retained 704 um span on the mosaic",
       span["n_placements"], 52290, 0)
report("basalt-rich share over those placements, mean", span["share_mean"],
       0.360, 0.0005)
report("basalt-rich share over those placements, SD", span["share_sd"], 0.141,
       0.0005)
report("estimate over those placements, mean", span["voigt_mean"], 18.47, 0.005)
report("estimate over those placements, SD", span["voigt_sd"], 2.96, 0.005)
report_same("the share those placements are ranked against",
            span["mapped_window_share"], w_indents)

print("\nthreshold sensitivity (Supplementary Table S2)")
grid = zs.threshold_grid(raw, w_tiles)
published_S2 = {(10, 60): 18.84, (10, 65): 18.79, (10, 70): 18.74,
                (10, 75): 18.72, (10, 80): 18.59, (15, 60): 18.53,
                (15, 65): 18.42, (15, 70): 18.50, (15, 75): 18.48,
                (15, 80): 18.41, (20, 60): 18.24, (20, 65): 18.24,
                (20, 70): 18.21, (20, 75): 18.12, (20, 80): 18.00}
for (lower, upper), published in published_S2.items():
    row = grid[(grid.lower_GPa == lower) & (grid.upper_GPa == upper)]
    report(f"estimate at lo = {lower}, hi = {upper}",
           float(row.estimate_GPa.iloc[0]), published, 0.005)
reference = float(grid[(grid.lower_GPa == 15)
                       & (grid.upper_GPa == 70)].estimate_GPa.iloc[0])
report("largest deviation across the grid",
       float((grid.estimate_GPa - reference).abs().max()), 0.50, 0.005)
# The budget converts the extremes of this grid into its Type B term. Two
# implementations of one sweep are two chances to be wrong, so they are held
# against each other here.
report_same("grid low, against the budget's recorded range",
            float(grid.estimate_GPa.min()),
            budget["type_b_ranges"]["threshold_sweep_GPa"][0])
report_same("grid high, against the budget's recorded range",
            float(grid.estimate_GPa.max()),
            budget["type_b_ranges"]["threshold_sweep_GPa"][1])

print("\nphase-angle quality sensitivity (Supplementary Table S3)")
# Every cell of the table. The first row applies the instrument's own flag,
# deposited as `qc_flag_instrument`; the rest apply the stated rule.
rows_S3 = _pas.sweep()
# Three of these four fractions are exact halves at one decimal: 433, 473 and
# 217 over 2000 give 21.65, 23.65 and 10.85. The supplement rounds all three
# half-up and is self-consistent with the "11 % to 24 %" in the body, so the
# checks carry the half-up figures.
published_S3 = [(433, 21.7, 51.98, 16.67), (473, 23.7, 54.70, 16.82),
                (304, 15.2, 54.67, 16.81), (217, 10.9, 54.59, 16.81)]
for row, (n, pct, basalt, estimate) in zip(rows_S3, published_S3):
    name = str(row["phase_angle_limit_deg"]).split(" ")[0]
    report(f"rejected at {name} deg", row["rejected"], n, 0)
    report(f"rejected at {name} deg, per cent", row["rejected_pct"], pct, ROUNDED)
    report(f"basalt input at {name} deg",
           row["alternative_basalt_input_GPa"], basalt, 0.005)
    report(f"estimate on that input at {name} deg",
           row["Voigt_with_that_input_GPa"], estimate, 0.005)
swept = [r["Voigt_with_that_input_GPa"] for r in rows_S3[1:]]
report("what the 10 to 20 deg sweep moves", max(swept) - min(swept), 0.01, 0.005)
# Table 2 prints this move. It is the difference of the two estimates before
# they are rounded into the table.
report("the estimate between the instrument's flag and the stated rule",
       _pas.instrument_flag_move(), 0.14, 0.005)
_flags = pd.read_csv(DATA / "depth_resolved_indents.csv")[
    ["indent_id", "qc_flag_instrument"]].merge(
    _dr_all[["indent_id", "qc_flag_strict"]], on="indent_id")
report_internal("indents the stated rule readmits against the instrument's flag",
                int(((_flags.qc_flag_instrument == 1)
                     & (_flags.qc_flag_strict == 0)).sum()), 132, 0,
                "433 - 132 + 3 = 304; not printed")
report_internal("indents it newly rejects",
                int(((_flags.qc_flag_instrument == 0)
                     & (_flags.qc_flag_strict == 1)).sum()), 3, 0,
                "not printed")

print("\ninput substitutions (Supplementary Tables S4 and S8, 3.4)")
flex = pd.read_csv(DATA / "flexural_moduli.csv")
measured = {c: float(g.E_D790_toe_compensated_GPa.mean())
            for c, g in flex.groupby("configuration")}


def with_input(cls, value):
    """The estimate with one class modulus replaced, everything else held."""
    return float(zs.at_share(per_zone, {**medians, cls: value}, w_tiles)[1][0])


def gap(estimate, configuration):
    """The relative gap the paper reports: (estimate - measured) / estimate."""
    return 100 * (estimate - measured[configuration]) / estimate


report("relative gap, basalt-longitudinal", gap(voigt, "basalt_face"), -1.9, 0.05)
report("relative gap, flax-longitudinal", gap(voigt, "flax_face"), -6.8, 0.05)
basalt_depth = float(_dr_all.loc[(_dr_all.qc_flag_strict == 0)
                                 & (_dr_all.constituent == "basalt"),
                                 "E_150"].median())
report("depth-resolved basalt input", basalt_depth, 54.67, 0.005)
estimate_depth = with_input("basalt", basalt_depth)
report("estimate with the depth-resolved basalt input", estimate_depth, 16.81,
       0.005)
report("relative gap on that input, basalt-longitudinal",
       gap(estimate_depth, "basalt_face"), -12.1, 0.05)
report("relative gap on that input, flax-longitudinal",
       gap(estimate_depth, "flax_face"), -17.5, 0.051)
report("estimate with the bulk PA-11 matrix, 1.5 GPa", with_input("matrix", 1.5),
       18.13, 0.005)
report("estimate with the matrix at 1.0 GPa", with_input("matrix", 1.0), 18.07,
       0.005)
# The reading depth as a sensitivity, beside the cut: the basalt input read
# from the selected basalt-centre traces of Supplementary Figure S7 at 80 nm,
# where the two acquisitions agree, and at 50 nm, the shallowest deposited
# point, each substituted as the depth-resolved median is above. The medians
# are held to the live computation in `depth_softening.py`, and the two rows
# to the estimates Table S4 prints for them, 18.43 and 19.14 GPa, with their
# gaps on the table's convention.
print("\nthe basalt input read at 80 and at 50 nm (Supplementary Table S4)")
import depth_softening as _soft                                     # noqa: E402
_readings = _soft.basalt_readings()["median_GPa"]
_sweep = json.loads((DATA / "basalt_input_sweep.json").read_text())
for row, (printed_input, printed, g_basalt, g_flax) in zip(
        _sweep["reading_depth"],
        ((84.7, 18.43, -2.3, -7.2), (97.9, 19.14, 1.5, -3.2))):
    z = row["read_at_nm"]
    report_same(f"basalt input read at {z:g} nm, product against the traces",
                row["basalt_input_GPa"], _readings[z])
    report(f"basalt input read at {z:g} nm on the selected traces, GPa",
           row["basalt_input_GPa"], printed_input, ROUNDED)
    report_same(f"estimate on that input, product against Eq. (1)",
                row["voigt_mosaic_GPa"], with_input("basalt", row["basalt_input_GPa"]))
    report(f"estimate with the basalt input read at {z:g} nm",
           row["voigt_mosaic_GPa"], printed, 0.005)
    report("relative gap on that input, basalt-longitudinal",
           gap(row["voigt_mosaic_GPa"], "basalt_face"), g_basalt, ROUNDED)
    report("relative gap on that input, flax-longitudinal",
           gap(row["voigt_mosaic_GPa"], "flax_face"), g_flax, ROUNDED)
# Supplementary Table S8. The three inputs are the ones printed in the table's
# own first column; they are three different physical quantities and none of
# them is what this work measures.
for value, estimate, g_basalt, g_flax in [(14.00, 23.38, 19.4, 15.5),
                                          (18.00, 25.49, 26.0, 22.5),
                                          (45.30, 39.84, 52.7, 50.4)]:
    got = with_input("flax", value)
    report(f"estimate with the flax input at {value:.2f} GPa", got, estimate,
           0.005)
    report("relative gap at that input, basalt-longitudinal",
           gap(got, "basalt_face"), g_basalt, ROUNDED)
    report("relative gap at that input, flax-longitudinal",
           gap(got, "flax_face"), g_flax, ROUNDED)

print("\ntabulated-input comparison (3.4)")
E_tab, f_nominal = _dec.TABULATED_E, _dec.NOMINAL_F
E_measured, f_measured = _dec.measured_moduli(), _dec.MEASURED_F
tabulated = _dec.voigt(E_tab, f_nominal)
report("Voigt sum, tabulated moduli and nominal fractions", tabulated, 35.8, 0.05)
# 3.4 gives the 35.8 GPa beside the two measurements rather than as a gap,
# because a gap taken on the prediction reads, to anyone who does not know the
# convention, as a gap on the measurement. The gaps are held internally only.
for cfg, published in (("basalt_face", 47.3), ("flax_face", 44.7)):
    report_internal(f"that sum overshoots {cfg} by", gap(tabulated, cfg),
                    published, 0.05, "the 35.8 GPa is printed; not the gap")
# What the tabulated route leaves open: the same fractions with the constituent
# moduli anywhere inside the ranges the Introduction cites.
for label, basalt, flax, published in (("low", 70.0, 40.0, 28.3),
                                       ("high", 95.0, 80.0, 44.5)):
    report(f"tabulated sum at the {label} end of the cited ranges",
           _dec.voigt({"basalt": basalt, "flax": flax, "matrix": 1.5},
                      f_nominal), published, ROUNDED)
moduli_only = _dec.voigt(E_measured, f_nominal)
fractions_only = _dec.voigt(E_tab, f_measured)
both = _dec.voigt(E_measured, f_measured)
report("measured moduli at the nominal fractions", moduli_only, 25.0, 0.05)
report("tabulated moduli at the measured fractions", fractions_only, 43.8, 0.05)
report("measured moduli at the measured fractions", both, 24.5, 0.05)
report_internal("what the moduli alone move", moduli_only - tabulated, -10.7,
                ROUNDED, "35.8 to 25.0 GPa is printed")
report_internal("what the fractions alone move", fractions_only - tabulated,
                8.0, 0.05, "35.8 to 43.8 GPa is printed")
for cfg, published in (("basalt_face", 23.1), ("flax_face", 19.4)):
    report_internal(f"the measured-fraction construction overshoots {cfg} by",
                    gap(both, cfg), published, 0.05,
                    "24.5 GPa is printed; not the gap")
# The Halpin-Tsai baseline of the same paragraph and of the Figure 8 panel. The
# function is imported from the figure script rather than rewritten here so that
# the bar and this line cannot come to disagree. The two fibre moduli are
# combined by volume at the densities 3.1 states, so that 3.4 and 4.2 give one
# prediction for the panel; this line holds the deposit to that value.
from fig08_rom import halpin_tsai_balanced as _ht, fibre_modulus_by_volume  # noqa: E402
report_internal("fibre modulus by volume at the stated densities, GPa",
                fibre_modulus_by_volume(), 67.1, ROUNDED,
                "behind the 26.3 GPa below; not printed")
report("Halpin-Tsai lamina, balanced 0/90 average", _ht()[2], 26.3, ROUNDED)

# 4.2: each laminate normalised by its own lamination-theory prediction, and
# the decomposition of the ratio against the prior hybrids. All six figures are
# checked here, and `fig08_rom.prior_laminate_table` calls the same
# lamination-theory route as the line above, so there is one prediction behind them.
print("\nthe normalised comparison against the prior laminates (4.2)")
from fig08_rom import prior_laminate_table, consolidation_decomposition  # noqa: E402
_rows = prior_laminate_table()
_prior = [r[4] for r in _rows if not r[5]]
report("prior laminates, lowest share of their own prediction", min(_prior),
       0.37, 0.005)
report("prior laminates, highest share", max(_prior), 0.58, 0.005)
for r in _rows:
    if r[5]:
        report(f"{r[0]}, share of its own prediction", r[4],
               {"this work, basalt-long": 0.72, "this work, flax-long": 0.75}[r[0]],
               0.005)
_d = consolidation_decomposition(_rows)
report("stiffer than the prior hybrids by a factor of", _d["observed"], 2.07,
       0.005)
report("of which, from the fibre volume fraction", _d["fibre_volume_fraction"],
       1.44, 0.005)
report("and from the share of the prediction realised",
       _d["consolidation_quality"], 1.41, 0.005)
report("their product recovers the ratio to within, per cent",
       _d["residual_pct"], 2, 0.5)

print("\nsegmentation of the mosaic (Figure 2, 3.1, Supplementary Table S6)")
# Both files here are primary measurements: the tile statistics and the
# leave-one-field-out validation need the tile images, which are not deposited.
# The lines below are therefore a check on the transcription into the paper, which
# is the failure this block exists to catch, and not on the segmentation itself.
seg = json.loads((DATA / "bse_segmentation_summary.json").read_text())
report("mosaic area, mm2", seg["area_mm2"], 6.65, 0.005)
report("imaged tiles", seg["n_tiles"], 91, 0)
# No document prints a figure for the unmarked interface band. The 4.0 % is stated
# in the README and in this file's own method note, so comparing against it holds
# the note to the array beside it, which is worth doing and is not a check on the
# paper, so it is reported as an internal check.
report_internal("unmarked interface band, per cent",
                100 * seg["zone_area_from_mosaic_mask"]["unmarked"], 4.0, 0.05,
                "the figure the file's own method note states")
report_internal("the three mosaic zone shares, sum",
                sum(seg["zone_area_from_mosaic_mask"].values()), 1.0, 1e-9,
                "they partition the mosaic")
# The intervals on the two fibre fractions are recorded in the summary file and
# computed live here. `imaged_intervals()`
# in `areal_mass_fractions.py` is the recipe: per zone, the tiles that zone
# covers by at least 5 % of their pixels, the (n - 1) standard deviation of the
# in-zone fibre fraction over them, 1.96 standard errors, times the zone's share
# of the marked mosaic from the mask. The recorded value is held to the live one
# first, then the page to the recorded one.
_ci_live = _am.imaged_intervals()
for cls, published, ci in (("basalt", 24.4, 1.1), ("flax", 39.5, 3.2),
                           ("matrix", 36.1, None)):
    report(f"volume fraction, {cls}", 100 * seg["volume_fraction"][cls],
           published, 0.05)
    if ci is not None:
        report_same(f"recorded interval against the recipe, {cls}",
                    seg["ci_95_on_the_mean_pct"][cls],
                    _ci_live[cls]["half_width_pct"])
        report(f"its 95 % interval on the mean, {cls}",
               seg["ci_95_on_the_mean_pct"][cls], ci, 0.05)
report_internal("tiles behind the basalt interval", _ci_live["basalt"]["n_tiles"],
                72, 0, "the tiles the basalt zone covers by 5 % or more")
report_internal("tiles behind the flax interval", _ci_live["flax"]["n_tiles"],
                88, 0, "the tiles the flax zone covers by 5 % or more")
report("fibre volume fraction, measured",
       seg["volume_fraction"]["basalt"] + seg["volume_fraction"]["flax"],
       0.639, 0.0005)
report("basalt packing inside its own zone",
       seg["in_zone_fibre_fraction_pooled_over_tiles"]["basalt_fib"], 0.665,
       0.0005)

# The second route to the same fractions, from the fabric specification, which
# 3.1, the Figure 2e caption, 4.3.3 and the Conclusions all report, computed by
# `areal_mass_fractions.py`.
# The agreement, 2.2 percentage points, is quoted in all four places and is the
# worst of the three phase differences, so it is checked as the maximum it is
# rather than phase by phase.
#
# The densities that route needs, 2.70 and 1.50 g/cm3, are stated in 3.1 as
# the nominal datasheet values, and the script reads them as inputs.
print("\nthe areal-mass route to the same fractions (3.1, Figure 2e, 4.3.3)")
_calc = _am.fractions()
_diffs, _worst = _am.agreement(_calc, _am.imaged())
for cls, pct in (("basalt", 23.2), ("flax", 41.7), ("matrix", 35.1)):
    report(f"areal-mass volume fraction, {cls}", 100 * _calc[cls], pct, ROUNDED)
report("fibre volume fraction, calculated", _calc["fibre"], 0.649, 0.0005)
report_internal("the two routes, percentage points apart on the fibre fraction",
                100 * (_calc["fibre"] - _am.imaged()["fibre"]), 1.0, 0.05,
                "0.639 and 0.649 are printed")
report("worst disagreement with the imaging, percentage points", _worst, 2.2,
       ROUNDED)
# The Halpin-Tsai baseline of Figure 8 is drawn at the fibre volume fraction of
# this route, rounded as the paper prints it. Holding the figure's input to the
# route keeps the panel and the paragraph on one derivation.
from fig08_rom import HT_V_FIBRE as _ht_v                        # noqa: E402
report_internal("the fibre fraction Figure 8 draws its baseline at", _ht_v,
                round(_calc["fibre"], 3), 1e-12,
                "the route above, at the printed precision")

val = json.loads((DATA / "bse_validation_metrics.json").read_text())
for zone, phases_only, all_three in (("basalt zone", 96.0, 82.8),
                                     ("flax zone", 97.9, 88.1)):
    report(f"{zone}, accuracy over the two solid phases",
           val[zone]["accuracy_pct_phases_only"], phases_only, 0.05)
    report(f"{zone}, accuracy with void as a third class",
           val[zone]["accuracy_pct"], all_three, 0.05)
for zone, cls, precision, recall, pixels in (
        ("basalt zone", "basalt", 92.5, 90.6, 33834),
        ("basalt zone", "matrix", 77.6, 74.4, 18465),
        ("basalt zone", "void", 72.5, 77.7, 20636),
        ("flax zone", "flax", 94.7, 94.2, 44000),
        ("flax zone", "matrix", 79.3, 78.1, 32493),
        ("flax zone", "void", 87.9, 88.9, 67127)):
    entry = val[zone]["classes"][cls]
    report(f"precision, {cls} in the {zone}", entry["precision_pct"], precision,
           0.05)
    report(f"recall, {cls} in the {zone}", entry["recall_pct"], recall, 0.05)
    report(f"labelled pixels, {cls} in the {zone}", entry["labelled_pixels"],
           pixels, 0)
# The three confusion figures 3.4 quotes, which are the argument that the pair
# backscattered contrast separates least is not the pair the classifier confuses.
flax_zone = val["flax zone"]["confusion_true_by_predicted"]
report("flax pixels called matrix", flax_zone["flax"]["matrix"], 635, 0)
report("flax pixels called matrix, per cent",
       100 * flax_zone["flax"]["matrix"] / val["flax zone"]["classes"]["flax"]["labelled_pixels"],
       1.4, 0.05)
report("matrix pixels called flax", flax_zone["matrix"]["flax"], 825, 0)
report("matrix pixels called flax, per cent",
       100 * flax_zone["matrix"]["flax"] / val["flax zone"]["classes"]["matrix"]["labelled_pixels"],
       2.5, 0.05)
report("matrix pixels called void, per cent",
       100 * flax_zone["matrix"]["void"] / val["flax zone"]["classes"]["matrix"]["labelled_pixels"],
       19.3, 0.05)

print("\nmixed-volume deconvolution (4.1, 4.3.6)")
# Read from the deposited product: 4000 bootstrap resamplings per acquisition,
# written by mixed_class_deconvolution.py and hash-recorded.
dec = json.loads((DATA / "mixed_class_deconvolution.json").read_text())
acq = dec["acquisitions"]
report("basalt fraction on the tie line, load-controlled",
       acq["shallow"]["phi_compliance"], 0.69, 0.005)
report("its interval, low", acq["shallow"]["phi_compliance_ci95"][0], 0.68, 0.005)
report("its interval, high", acq["shallow"]["phi_compliance_ci95"][1], 0.71, 0.005)
report("basalt fraction on the tie line, depth-resolved",
       acq["depth"]["phi_compliance"], 0.78, 0.005)
report("its interval, low", acq["depth"]["phi_compliance_ci95"][0], 0.76, 0.005)
report("its interval, high", acq["depth"]["phi_compliance_ci95"][1], 0.81, 0.005)
report("the same under linear mixing, load-controlled",
       acq["shallow"]["phi_linear"], 0.36, 0.005)
report("the same under linear mixing, depth-resolved",
       acq["depth"]["phi_linear"], 0.57, 0.005)
report("basalt packing in the zone, from the imaging",
       dec["independent_comparison"]["in_zone_basalt_packing_from_imaging"],
       0.665, 0.001)
report("two-class sum, load-controlled",
       dec["two_class_sum_GPa"]["compliance"]["shallow"], 61.0, 0.05)
report("two-class sum, depth-resolved",
       dec["two_class_sum_GPa"]["compliance"]["depth"], 68.4, 0.05)
report("the two sums, per cent apart",
       dec["two_class_sum_GPa"]["compliance"]["spread_pct"], 11.4, 0.05)
bulk = [e for e in acq["shallow"]["endpoint_sensitivity"]
        if e["e_matrix"] == 1.0 and abs(e["e_basalt"] - acq["shallow"]["e_basalt_end_member_GPa"]) < 0.01][0]
report("load-controlled fraction at a bulk matrix end member",
       bulk["phi_compliance"], 0.91, 0.005)
bulk_depth = [e for e in acq["depth"]["endpoint_sensitivity"]
              if e["e_matrix"] == 1.0 and abs(e["e_basalt"] - acq["depth"]["e_basalt_end_member_GPa"]) < 0.01][0]
report("depth-resolved fraction at a bulk matrix end member",
       bulk_depth["phi_compliance"], 0.95, 0.005)
# 4.3.6 prints this span to the unit, so that is what the two lines compare
# against.
report("gap between the acquisitions across that sweep, low",
       dec["cross_acquisition_agreement"]["gap_span_pct"][0], 4, 0.5)
report("gap between the acquisitions across that sweep, high",
       dec["cross_acquisition_agreement"]["gap_span_pct"][1], 10, 0.5)
report("mixed-volume share of the basalt zone, load-controlled",
       100 * acq["shallow"]["mixed_class_fraction"], 55, 0.5)
report("mixed-volume share of the basalt zone, depth-resolved",
       100 * acq["depth"]["mixed_class_fraction"], 37, 0.5)
# The two fibre fractions 4.1, 4.2 and Table S1 print for the mixed-volume contact, 0.54
# and 0.67. They are the Hill inversion: the
# class median placed on the arithmetic mean of the Voigt and Reuss curves
# between the two shallow end members, 86.0 and 4.66 GPa, and solved for the
# basalt fraction. Both medians are inverted on the shallow end members. The
# same medians on the linear form give 0.31 and 0.38 and on the compliance form
# 0.89 and 0.92, so the printed pair holds under this convention alone.
from mixed_class_deconvolution import hill_fraction as _hill      # noqa: E402
_e_b = float(tab.loc[tab["class"] == "basalt", "E_median_GPa"].iloc[0])
_e_m = float(tab.loc[tab["class"] == "matrix", "E_median_GPa"].iloc[0])
_mixed_depth = float(_dr_all.loc[_dr_all.constituent == "mixed-volume",
                                 "E_150"].median())
_mixed_shallow = float(tab.loc[tab["class"] == "mixed-volume",
                               "E_median_GPa"].iloc[0])
report("fibre fraction in the mixed-volume contact, depth-resolved (Hill)",
       _hill(_mixed_depth, _e_b, _e_m), 0.54, 0.005)
report("fibre fraction in the mixed-volume contact, load-controlled (Hill)",
       _hill(_mixed_shallow, _e_b, _e_m), 0.67, 0.005)

# Where the mixed-volume contacts sit, which 4.1 states in five numbers and
# three words. Read from the deposited product, which `figs_supplementary.py`
# writes and `provenance_check.py` holds to it; the definitions are in the file
# beside the values. The three medians are printed as words, three eighths,
# five eighths and one eighth, and are checked as the
# fractions those words name.
print("\nspatial localisation of the mixed-volume class (4.1, Figure S6)")
_spatial = json.loads((DATA / "spatial_localisation.json").read_text())
for cls, printed, word in (("mixed-volume", 0.375, "three eighths"),
                           ("basalt", 0.625, "five eighths"),
                           ("matrix", 0.125, "one eighth")):
    report(f"basalt share of the eight neighbours, {cls} ({word})",
           _spatial["median_basalt_neighbour_fraction"][cls], printed, 0.0005)
report_sci("Mann-Whitney p, mixed-volume above matrix, k = 8",
           _spatial["p_depth_resolved"], 2.6, -15)
report("bootstrap interval on the median difference, low",
       _spatial["median_difference_ci95"][0], 0.125, 0.0005)
report("bootstrap interval on the median difference, high",
       _spatial["median_difference_ci95"][1], 0.375, 0.0005)
report_below("largest p over k = 3 to 20", _spatial["p_max_over_k_3_to_20"],
             1e-9)
report("indents left after excluding 6 um at each end of the band",
       _spatial["n_after_edge_exclusion"], 467, 0)
report_sci("p on those indents", _spatial["p_excluding_edge_bands"], 9.1, -11)
report_sci("p on the load-controlled map", _spatial["p_load_controlled_map"],
           1.1, -77)
report("indents in the load-controlled basalt-rich zone",
       _spatial["n_load_controlled_zone"], 2760, 0)

# Figure 8a and the sentence of 3.4 that reads it: the within-band construction
# of the basalt-centre band with the basalt input taken from the load-controlled
# map, against the modulus measured in the band. The two percentages are
# printed as "below the observed mean" and reproduce on the Hill value as the
# denominator, (observed - Hill) / Hill; on the observed mean as the
# denominator they would read 10 %, 15 % and 15 %. The check follows the
# printed figure and this comment records which ratio it is.
print("\nthe within-band construction (3.4, Figure 8a)")
from fig08_rom import zone_table as _zone_table                     # noqa: E402
_zt = _zone_table(_dr_all).set_index("zone")
report("basalt-centre, Voigt with the load-controlled basalt",
       float(_zt.voigt_load_controlled_basalt["basalt-centre"]), 47.8, ROUNDED)
report("basalt-centre, Hill with the load-controlled basalt",
       float(_zt.hill_load_controlled_basalt["basalt-centre"]), 32.8, ROUNDED)
report("basalt-centre, observed zone mean",
       float(_zt.observed["basalt-centre"]), 36.7, ROUNDED)
for zone, printed in (("basalt-centre", 12), ("flax-bottom", 16),
                      ("flax-top", 19)):
    hill_c, obs = (float(_zt.hill_load_controlled_basalt[zone]),
                   float(_zt.observed[zone]))
    report(f"{zone}, Hill below the observed mean, per cent of Hill",
           100 * (obs - hill_c) / hill_c, printed, 0.5)

print("\nunsupervised cross-check (2.3, 3.3, Figure S5)")
dr = pd.read_csv(DATA / "classified_depth_resolved.csv")
c = cluster_run(dr)
# 3.3 carries the first of these to four decimals where it reports the seed
# sweep, and the second to two everywhere it appears. ARI* merges the compliant
# classes on both sides, as 2.3 describes: the two most compliant clusters into
# one, and flax with matrix into one compliant rule class.
report("ARI, rule vs k-means (k = 4)", c["ARI"], 0.40, 0.005)
report("ARI*, compliant classes merged on both sides", c["ARI_star"], 0.81,
       0.0051)
# The rest of what 3.3 prints about the partition: the two variants, the seed
# sweep and the subsample stability. The stability is plain ARI between the
# full fit and each refit at k = 4, with nothing merged and the rule taking no
# part, so it is neither the ARI against the rule nor ARI*.
for label, kw, ari, ari_star in (("three principal components",
                                  dict(use_pca=True), 0.404, 0.81),
                                 ("full-covariance Gaussian mixture",
                                  dict(model="gmm"), 0.357, 0.74)):
    v = cluster_run(dr, **kw)
    report(f"ARI, {label}", v["ARI"], ari, 0.0005)
    report(f"ARI*, {label}", v["ARI_star"], ari_star, 0.0051)
_seed_aris = sorted(round(cluster_run(dr, seed=s)["ARI"], 4)
                    for s in (0, 1, 42, 123, 2024))
_seed_mode = max(set(_seed_aris), key=_seed_aris.count)
report("seeds reproducing the same partition",
       _seed_aris.count(_seed_mode), 4, 0)
report("the partition's ARI", _seed_mode, 0.4016, 0.00005)
report("the fifth seed, ARI", max(_seed_aris), 0.4028, 0.00005)
from clustering_check import bootstrap_stability as _stability      # noqa: E402
_stab_mean, _stab_sd = _stability(dr)
report("stability over twenty 90 % subsamples, mean ARI", _stab_mean, 0.979,
       0.0005)
report("stability over twenty 90 % subsamples, SD", _stab_sd, 0.016, 0.0005)

print("\nthree-point bending (3.4, Supplementary Table S7)")
# Read the replicates rather than restating summary statistics, so that the t,
# p and g below are computed from the nine specimens.
#
# The five quantities come from flexural_comparisons.py, which defines each one
# and is the only place any of them is defined. The two moduli of Table S7 and
# the strength are also recomputed from the curves there, so this block reduces
# nine specimens rather than restating a summary.
spec = _fx.per_specimen()
a_all = spec[spec.configuration == "basalt_face"]
f_all = spec[spec.configuration == "flax_face"]
# One line per configuration, because the two counts are not the same and
# one line checking one of them would leave the other unchecked. The flax
# configuration is four because the source workbook held one of its curves twice;
# see the data dictionary under `flexural_curves.csv`.
report("basalt-longitudinal specimens", len(a_all), 5, 0)
report("flax-longitudinal specimens", len(f_all), 4, 0)

# The Hedges g of the strain at break is None rather than a number, because
# Supplementary Table S7 prints n/a in that cell: the comparison was not
# pre-specified, so the table reports the two means and declines the effect size.
# A check against the 8.79 the data give would be a check against nothing on
# the page.
for label, column, ba, bs, fa, fsd, g_pub in [
        ("chord modulus", "chord_GPa", 18.85, 0.54, 19.76, 0.50, -1.54),
        ("peak tangent modulus", "peak_tangent_GPa", 20.01, 0.55, 20.78, 0.26,
         -1.51),
        ("flexural strength", "strength_MPa", 181.68, 3.22, 117.35, 4.89, 14.21),
        ("softening to 0.55 % strain", "softening_pct", 8.40, 0.29, 34.51, 2.75,
         -12.80),
        ("strain at break", "strain_at_break_pct", 1.29, 0.08, 0.71, 0.03,
         None)]:
    x, y = a_all[column], f_all[column]
    report(f"{label}, basalt-longitudinal mean", x.mean(), ba, 0.01)
    report(f"{label}, basalt-longitudinal SD", x.std(ddof=1), bs, 0.005)
    report(f"{label}, flax-longitudinal mean", y.mean(), fa, 0.01)
    report(f"{label}, flax-longitudinal SD", y.std(ddof=1), fsd, 0.005)
    if g_pub is not None:
        report(f"{label}, Hedges g", _fx.compare(x, y)["hedges_g"], g_pub, 0.005)

# No document prints a t statistic, here or anywhere. The p value below is the
# published quantity and it pins t at the two group sizes, so the line that
# compared t against -2.487 verified nothing a reader could look up.
chord = _fx.compare(a_all.chord_GPa, f_all.chord_GPa)
report("chord modulus, Welch p uncorrected", chord["p"], 0.0357, 0.0001)
report("the stiffness difference", -chord["difference"], 0.91, 0.005)
report("that difference, per cent of the flax-longitudinal mean",
       -100 * chord["difference"] / f_all.chord_GPa.mean(), 4.6, 0.05)
report("its 95 % interval, low", -chord["ci95"][1], 0.08, 0.005)
report("its 95 % interval, high", -chord["ci95"][0], 1.74, 0.005)
report("peak tangent modulus, Welch p uncorrected",
       _fx.compare(a_all.peak_tangent_GPa, f_all.peak_tangent_GPa)["p"],
       0.0337, 0.0001)

# The Holm correction under all three families. The stiffness result depends on
# which family it is counted in, and the paper says so, so all three are checked
# rather than the one that was pre-specified.
print("\nHolm correction, three families (Supplementary Table S7, 3.4)")
quantities = {"chord modulus (GPa)": "chord_GPa",
              "peak tangent modulus (GPa)": "peak_tangent_GPa",
              "flexural strength (MPa)": "strength_MPa",
              "softening to 0.55 % strain (%)": "softening_pct",
              "strain at break (%)": "strain_at_break_pct"}
raw_p = {name: _fx.compare(a_all[col], f_all[col])["p"]
         for name, col in quantities.items()}
families = {
    "four comparisons, Table S7":
        ["chord modulus (GPa)", "peak tangent modulus (GPa)",
         "flexural strength (MPa)", "softening to 0.55 % strain (%)"],
    "stiffness as one comparison: three":
        ["chord modulus (GPa)", "flexural strength (MPa)",
         "softening to 0.55 % strain (%)"],
    "strain at break in place of softening: three":
        ["chord modulus (GPa)", "flexural strength (MPa)",
         "strain at break (%)"]}
published_holm = {"four comparisons, Table S7": 0.0675,
                  "stiffness as one comparison: three": 0.0357,
                  "strain at break in place of softening: three": 0.0357}
for label, members in families.items():
    corrected = dict(zip(members,
                         _fx.holm(np.array([raw_p[n] for n in members]))))
    report(f"Holm p on the chord modulus, {label}",
           corrected["chord modulus (GPa)"], published_holm[label], 0.0001)
    if "peak tangent modulus (GPa)" in corrected:
        report("Holm p on the peak tangent modulus, same family",
               corrected["peak tangent modulus (GPa)"], 0.0675, 0.0001)
    report_below(f"Holm p on the strength, {label}",
                 corrected["flexural strength (MPa)"], 0.0001)
    if "softening to 0.55 % strain (%)" in corrected:
        # A decade looser than the strength cell beside it. At four flax
        # specimens the softening corrects to about 6e-04 and 9e-04, so the
        # table cannot print it as < 0.0001 and a check held to that limit
        # would be held to a cell that is not on the page.
        report_below("Holm p on the softening, same family",
                     corrected["softening to 0.55 % strain (%)"], 0.001)

# The crossover strain is printed and checked further down. Above it the
# basalt-longitudinal configuration is the stiffer, which the mean modulus over
# the window above the crossover shows; the window means themselves are not
# printed, so the ranking is held as an internal check.
_win_means = (pd.read_csv(DATA / "flexural_moduli.csv")
              .groupby("configuration")["E_window_0p28_0p43_GPa"].mean())
report_same("above the crossover the basalt-longitudinal one is the stiffer",
            bool(_win_means["basalt_face"] > _win_means["flax_face"]), True)

print("\nASTM D790 departures and the bound that follows (2.5)")
# All four follow from the specimen geometry and the standard, and all four are
# computed by a script. The rate bound also needs the strain-rate sensitivity, which is
# measured and is checked above where it is measured.
report("overhang per end, mm", _flex.overhang_mm(), 3.20, 0.005)
report("overhang the standard requires, mm", _flex.required_overhang_mm(), 7.36,
       0.005)
report("crosshead rate the standard computes, mm/min",
       _flex.d790_rate_mm_per_min(), 1.96, 0.005)
report("by how much the rate used exceeds it, per cent", _flex.rate_excess_pct(),
       27, 0.5)
report("bound on the reported modulus from that departure, per cent",
       _flex.rate_bound_pct(), 1.8, ROUNDED)

print("\nTimoshenko shear correction (4.3.5)")
# The assumed shear modulus is the paper's, 1.5 to 3.0 GPa, and nothing in this
# deposit measures it, so these four lines are checks on arithmetic over an
# assumption rather than on a measurement. They are reported as published values
# because the bracket and the widened gaps are printed, and the assumption they
# rest on is printed beside them, as are the shear coefficient, 6/5 for a
# rectangular section, and the exact inversion, both of which 4.3.5 states.
_means = _flex.measured_means()
_low, _high = _flex.bracket(_means)
report("correction at the stiff end of the assumed shear modulus", _low, 3, 0.5)
report("correction at the compliant end", _high, 7, 0.5)
# The gaps 4.3.5 prints next are computed from the bracket as that paragraph
# prints it, to the unit, and not from the exact 3.0 and 6.6 %: the exact
# correction gives 5.0 and 13.8 %, and 13.8 is not what the page says. Both
# chains are printed by `flexural_corrections.py`, and the label here says which
# one is being checked.
_printed_low, _printed_high = _flex.PRINTED_BRACKET_PCT
report("widened gap at the printed low end of the bracket",
       _flex.gap_pct(min(_means.values()), voigt, _printed_low), 5, 0.5)
report("widened gap at the printed high end",
       _flex.gap_pct(max(_means.values()), voigt, _printed_high), 14, 0.5)

# The depth dependence of the stiff contacts, which 4.4 argues from and the
# Figure S7 caption summarises. `depth_softening.py` computes them and writes
# `depth_softening.json`, which `provenance_check.py` holds to the script. The
# checks read the product, after holding its selection to a live recomputation
# so that a stale file cannot pass.
print("\nthe depth dependence of the stiff contacts (4.4, Figure S7)")
_soften = json.loads((DATA / "depth_softening.json").read_text())
_sel = _soften["selection"]
report_same("selected contacts, product against the traces", _sel["n"],
            _soft.basalt_readings()["n"])
report("basalt-centre contacts above 70 GPa at 80 nm", _sel["n"], 86, 0)
report("of them classified basalt at 150 nm",
       _sel["class_at_150_nm"].get("basalt", 0), 86, 0)
_ez = _soften["E_median_GPa_by_depth"]
for _z, _printed in (("50", 97.9), ("60", 91.0), ("70", 87.5), ("80", 84.7),
                     ("100", 78.1), ("150", 68.5), ("200", 61.6), ("300", 46.3)):
    report(f"their median E at {_z} nm, GPa", _ez[_z], _printed, ROUNDED)
_r = _soften["E300_over_E80"]
report("E(300)/E(80), every one softer at 300 nm, per cent",
       _r["softening_pct"], 100, 0)
report_internal("softer by half or more", _r["at_or_below_half"], 38, 0,
                "the median ratio is 0.52; the count is not printed")
report_internal("E(300)/E(80), lowest", _r["min"], 0.33, 0.0051,
                "the range behind '100 % softening'; not printed")
report_internal("E(300)/E(80), highest", _r["max"], 0.92, 0.0051,
                "as above; not printed")
report("E(300)/E(80), median", _r["median"], 0.52, 0.0051)
report_internal("the median falls by, per cent", _r["median_E_fall_pct"], 45,
                0.5, "84.7 to 46.3 GPa, printed as the two medians")
_bands_printed = ((86, 0.52, 100), (177, 0.55, 95), (141, 0.88, 62),
                  (130, 1.28, 42))
for band, (n, ratio, pct) in zip(_soften["bands_by_E80_over_classified"],
                                 _bands_printed):
    lo, hi = band["lower_GPa"], band["upper_GPa"]
    name = (f"above {lo:g}" if hi is None else f"below {hi:g}" if lo is None
            else f"{lo:g} to {hi:g}")
    report(f"band {name} GPa at 80 nm, n", band["n"], n, 0)
    report(f"band {name} GPa, median E(300)/E(80)", band["median_ratio"],
           ratio, 0.0051)
    report(f"band {name} GPa, softening, per cent", band["softening_pct"], pct,
           0.5)
_called = _soften["called_basalt_at_150_nm"]
report("contacts called basalt at 150 nm", _called["n"], 222, 0)
report("of them softer at 300 than at 80 nm, per cent",
       _called["E300_over_E80"]["softening_pct"], 87, 0.5)
report_internal("their median E(300)/E(150)",
                _called["E300_over_E150"]["median"], 0.70, 0.0051,
                "behind the 95 %; not printed")
report("of them softer at 300 than at 150 nm, per cent",
       _called["E300_over_E150"]["softening_pct"], 95, 0.5)
report_internal("the selected contacts, median E(300)/E(150)",
                _soften["E300_over_E150"]["median"], 0.66, 0.0051,
                "not printed")
_dist = _soften["distance_to_nearest_compliant_contact"]
report("grid pitch, um", _dist["pitch_um"], 3, ROUNDED)
report("distance to the nearest sub-15 GPa contact, shortest, um",
       _dist["distance_range_um"][0], 3, 0.5)
report("distance to the nearest sub-15 GPa contact, longest, um",
       _dist["distance_range_um"][1], 19, 0.5)
report("Spearman rho, ratio against that distance", _dist["spearman_rho"],
       0.01, 0.0051)
report("its p", _dist["p"], 0.91, 0.0051)
report("indents more than two pitches from any such contact",
       _dist["far"]["n"], 25, 0)
report("their median ratio", _dist["far"]["median_ratio"], 0.53, 0.0051)
report("indents closer to one", _dist["near"]["n"], 61, 0)
report("their median ratio", _dist["near"]["median_ratio"], 0.52, 0.0051)
_n8 = _soften["eight_neighbour_basalt_fraction"]
report("neighbourhoods at least six parts basalt", _n8["n_at_or_above"], 37, 0)
report("their median ratio", _n8["median_ratio_at_or_above"], 0.53, 0.0051)
report("the rest, median ratio", _n8["median_ratio_rest"], 0.52, 0.0051)
report("Mann-Whitney p between the two", _n8["mann_whitney_two_sided_p"], 0.95,
       0.0051)
_h = _soften["H300_over_H80"]
report("H(300)/H(80), median", _h["median"], 0.71, 0.0051)
report("contacts whose hardness is lower at 300 nm", _h["lower_at_300"], 71, 0)
# Held to the printed values, as the ARI* lines above are.
_s2p = _soften["S2_over_P"]
report("S^2/P, median at 50 nm, GPa", _s2p["median_GPa_at_50_nm"], 2372, 0.51)
report("S^2/P, median at 300 nm, GPa", _s2p["median_GPa_at_300_nm"], 747, 0.51)
report_internal("S^2/P, median ratio 300 over 50",
                _s2p["median_ratio_300_over_50"], 0.30, 0.0051,
                "printed as 'a factor of three'")
report_below("S^2/P, ratio of the medians, under a third",
             _s2p["ratio_of_medians"], 1 / 3)
report("contacts on which S^2/P falls", _s2p["falling"], 85, 0)
report("contacts with S^2/P at both depths", _s2p["n_with_both"], 85, 0)
# The contact stiffness 4.4 prints is read off the S^2/P channel, as
# sqrt(S^2/P * P), so that the sentence and the S^2/P sentence before it rest
# on one stiffness record. The exported stiffness channel is a second record
# that does not agree with it (DATA_DICTIONARY.md, depth_resolved_curves.csv);
# its figures are in the JSON under 'stiffness_channel' and are not printed.
_s = _soften["contact_stiffness_S"]
report("S, median at 250 nm, kN/m", _s["median_kN_per_m_at_250_nm"], 67.3,
       ROUNDED)
report("S, median at 300 nm, kN/m", _s["median_kN_per_m_at_300_nm"], 68.5,
       ROUNDED)
report_below("S, change of the medians 250 to 300 nm, under 2 per cent",
             abs(_s["change_of_medians_250_to_300_nm_pct"]), 2)
report("load, median at 250 nm, mN", _s["median_load_mN_at_250_nm"], 5.0,
       ROUNDED)
report("load, median at 300 nm, mN", _s["median_load_mN_at_300_nm"], 6.2,
       ROUNDED)
report("load, rise of the medians 250 to 300 nm, per cent",
       _s["load_change_of_medians_250_to_300_nm_pct"], 23, 0.5)
report_internal("S(300)/S(80), median on the selected contacts",
                _s["S300_over_S80_median_selected"], 1.64, 0.0051,
                "not printed")
report_internal("S(300)/S(80), median on the matrix contacts",
                _s["S300_over_S80_median_matrix"], 3.06, 0.0051,
                "not printed")
report_internal("the stiffness channel, contacts falling 250 to 300 nm",
                _soften["stiffness_channel"]["falling_250_to_300_nm"], 67, 0,
                "the other record; not printed")
_share = _soften["cut_free_shares"]
report_internal("share above 70 GPa at 80 nm, basalt-centre, per cent",
                _share["depth_resolved_above_70_GPa_at_80_nm"]["pct"], 16, 0.5,
                "not printed")
report("basalt class of the load-controlled basalt-rich zone, per cent",
       _share["load_controlled_basalt_class"]["pct"], 12, 0.5)
report_same("that share's numerator, against the count above",
            _share["load_controlled_basalt_class"]["n"],
            int(cross.loc["basalt", "basalt"]))

# Contact size in the two acquisitions (3.3, 4.1, 4.4), from
# `contact_geometry.py`: contact depths and equivalent radii by class, the
# basalt-centre band re-read at a depth where its stiff contacts have
# the contact depth of the load-controlled ones, and the selected stiff
# contacts read at the load-controlled contact depth on basalt.
print("\ncontact size in the two acquisitions (3.3, 4.1, 4.4)")
_cg = json.loads((DATA / "contact_geometry.json").read_text())
_lc = _cg["load_controlled"]
for cls, a in (("basalt", 0.12), ("mixed-volume", 0.22), ("flax", 0.86),
               ("matrix", 0.91)):
    report(f"equivalent contact radius, load-controlled, {cls}, um",
           _lc[cls]["a_um"], a, 0.0051)
_d150 = _cg["depth_resolved_by_depth"]["150"]
for cls, hc, a in (("basalt", 121, 0.34), ("mixed-volume", 126, 0.35),
                   ("flax", 129, 0.36), ("matrix", 130, 0.36)):
    # Table 1 prints the range, 121 to 130 nm, so its two ends are published
    # and the two inner values are held as internal checks.
    if cls in ("basalt", "matrix"):
        report(f"contact depth at z = 150 nm, {cls}, nm", _d150[cls]["h_c_nm"],
               hc, 0.51)
    else:
        report_internal(f"contact depth at z = 150 nm, {cls}, nm",
                        _d150[cls]["h_c_nm"], hc, 0.51, "inside 121 to 130 nm")
    report(f"equivalent contact radius at z = 150 nm, {cls}, um",
           _d150[cls]["a_um"], a, 0.0051)
_rr = _cg["basalt_centre_reread"]["80"]
report("stiff contacts of the band, contact depth at z = 80 nm, nm",
       _rr["h_c_stiff_nm"], 66, 0.51)
report("mixed-volume share of the band read at z = 80 nm, per cent",
       _rr["mixed_volume_pct"], 26, 0.5)
_lcz = _cg["load_controlled_basalt_zone"]
report("stiff contacts of the load-controlled basalt-rich zone, contact depth, nm",
       _lcz["h_c_stiff_nm"], 68, 0.51)
report("mixed-volume share of that zone on a 40 GPa cut, per cent",
       _lcz["mixed_volume_pct_at_40"], 31, 0.5)
_m = _cg["stiff_contacts_matched"]
report_same("selected contacts, against the depth-softening product",
            _m["n_selected"], _sel["n"])
report("their contact depth at z = 50 nm, nm", _m["h_c_nm"], 40, 0.51)
report("their median E there, GPa", _m["E_GPa"], 97.9, ROUNDED)
report("their median load there, mN", _m["load_mN"], 0.41, 0.0051)
report("load-controlled basalt, median peak load, mN",
       _m["load_controlled_basalt_load_mN"], 0.51, 0.0051)
report_internal("their equivalent contact radius at 300 nm, um",
                _m["a_um_at_300"], 0.6, ROUNDED, "not printed")

_rr300 = _cg["basalt_centre_reread"]["300"]
report("mixed-volume share of the band read at z = 300 nm, per cent",
       _rr300["mixed_volume_pct"], 56, 0.5)
_mat = [v["matrix_pct"] for k, v in _cg["basalt_centre_reread"].items()
        if float(k) >= 80]
report("matrix share of the band from 80 to 300 nm, lowest, per cent",
       min(_mat), 21, 0.5)
report("matrix share of the band from 80 to 300 nm, highest, per cent",
       max(_mat), 24, 0.5)
_gap = [v["z_minus_h_c_nm"] for v in _d150.values()]
report("z - h_c at z = 150 nm, smallest class median, nm", min(_gap), 20, 0.5)
report("z - h_c at z = 150 nm, largest class median, nm", max(_gap), 30, 0.5)

# The remaining printed values of 2.2, 3.4, 4.4 and Supplementary Table S2.
print("\nfurther printed values (2.2, 3.4, 4.4, Supplementary Table S2)")
_sweep_rows = {r["upper_GPa"]: r for r in _sweep["shallow"] if r["lower_GPa"] == 15}
for upper, median, n in ((60, 76.70, 575), (65, 81.47, 457), (70, 85.99, 380),
                         (75, 89.19, 313), (80, 93.07, 249)):
    report(f"pure-basalt median at a {upper} GPa cut", 
           _sweep_rows[upper]["basalt_median_GPa"], median, 0.005)
    report(f"pure-basalt count at a {upper} GPa cut",
           _sweep_rows[upper]["n_basalt"], n, 0)
_pct = window["windows"]["load-controlled map"]["voigt_percentiles_5_25_50_75_95"]
report("full footprint, 5th percentile of placements", _pct[0], 16.31, 0.005)
report("full footprint, 95th percentile of placements", _pct[-1], 21.16, 0.005)
_pct = window["windows"]["load-controlled map, laminate span after the resin "
                         "cut"]["voigt_percentiles_5_25_50_75_95"]
report("indented span, 5th percentile of placements", _pct[0], 12.82, 0.005)
report("indented span, 95th percentile of placements", _pct[-1], 22.62, 0.005)
_z1 = float(zs.at_share(per_zone, medians, 1.0)[1][0])
_z0 = float(zs.at_share(per_zone, medians, 0.0)[1][0])
report("Voigt sum of the basalt-rich zone alone", _z1, 31.9, ROUNDED)
report("Voigt sum of the flax-rich zone alone", _z0, 10.9, ROUNDED)
report("estimate per percentage point of basalt-rich share", (_z1 - _z0) / 100,
       0.21, 0.005)
import border_effect as _border_effect                              # noqa: E402
_bord = _border_effect.comparisons()[0]
report("border test, comparisons the populations support", len(_bord), 5, 0)
report("border test, smallest p before correction", _bord[0][6], 0.037, 0.0005)
report("border test, the same after Holm", _bord[0][7], 0.18, 0.005)
report_same("border test, the smallest runs the other way (border softer)",
            bool(_bord[0][4] < _bord[0][5]), True)
from fig09_flexural import find_crossover as _crossover, tangent_modulus as _tm  # noqa: E402
_curves_f = pd.read_csv(DATA / "flexural_curves.csv")
_mod_f = pd.read_csv(DATA / "flexural_moduli.csv").set_index("specimen_id")
_per = {"basalt_face": [], "flax_face": []}
for _spec, _g in _curves_f.groupby("specimen_id", sort=True):
    _e = _g.strain_pct.to_numpy() - _mod_f.loc[_spec, "toe_intercept_pct"]
    _s = _g.stress_MPa.to_numpy()
    _k = _e >= 0
    _per[_g.configuration.iloc[0]].append(_tm(_e[_k], _s[_k]))
report("strain at which the tangent moduli cross, per cent",
       _crossover(_per), 0.28, 0.005)
report("phase angle of the selected contacts at 50 to 80 nm, median, deg",
       _m["phase_angle_median_deg_50_to_80"], 1, 0.5)
report("of them, readings above the 15 degree limit at 50 to 80 nm",
       _m["phase_angle_above_15_deg_50_to_80"], 0, 0)

# ---------------------------------------------------------------------------
# Geometry, coverage and the remaining printed figures.
# ---------------------------------------------------------------------------
print("\nwhat the windows cover (2.2, 3.1, 4.3.3, Supplementary Figure S2)")
import figS2_map_coverage as _cov                                 # noqa: E402
_c = _cov.coverage()
_win = _c["windows"]
report("plies the backscattered-electron mosaic spans",
       _win["Backscattered-electron mosaic"]["plies"], 5.6, ROUNDED)
report("mosaic, through the thickness, mm",
       _win["Backscattered-electron mosaic"]["through_thickness_mm"], 1.60, 0.005)
report("plies the retained load-controlled indents span",
       _win["Load-controlled map"]["plies"], 2.4, ROUNDED)
report("load-controlled window, through the thickness, mm",
       _win["Load-controlled map"]["through_thickness_mm"], 0.70, 0.005)
report("mosaic bounding box, mm2", _c["mosaic_bounding_box_mm2"], 7.03, 0.005)
_lc = _win["Load-controlled map"]
report_above("mosaic area over the load-controlled footprint",
             seg["area_mm2"] / (_lc["through_thickness_mm"] * _lc["in_plane_mm"]),
             60)
report("mosaic width in yarn spacings", _c["mosaic_yarn_spacings"], 1.58, 0.005)
report("mosaic width in 2/2 twill repeats", _c["mosaic_twill_repeats"], 0.4,
       ROUNDED)

_val = json.loads((DATA / "bse_validation_metrics.json").read_text())
report("painted fields, basalt-rich zone", _val["basalt zone"]["n_fields"], 16, 0)
report("painted fields, flax-rich zone", _val["flax zone"]["n_fields"], 21, 0)
report("painted fields in all",
       _val["basalt zone"]["n_fields"] + _val["flax zone"]["n_fields"], 37, 0)

print("\nthe resin edge and the depth-resolved bands (2.2, 2.3)")
from classify import resin_profile as _resin, X_RESIN as _x_resin    # noqa: E402
_prof = _resin(raw)
report("median modulus of the resin rows, x below 300 um, GPa",
       float(_prof[_prof.index < 300].median()), 5.4, ROUNDED)
_rise = float(_prof[(_prof.index >= 300) & (_prof > 15)].index.min())
report_internal("first row above 15 GPa past the resin, um", _rise, 315, 15,
                "inside the printed 300 to 330 um")
report("resin cut, um", _x_resin, 380, 0)
for zone, lo, hi in (("flax-bottom", 0, 78), ("basalt-centre", 81, 168),
                     ("flax-top", 171, 297)):
    g = _dr_all[_dr_all.zone == zone]
    report(f"{zone} band, lowest y, um", float(g.y_um.min()), lo, 0.5)
    report(f"{zone} band, highest y, um", float(g.y_um.max()), hi, 0.5)

print("\nfurther values of 3.3, 4.3 and the supplement")
report("modulus range of the class medians, stiffest over most compliant",
       float(tab.E_median_GPa.max() / tab.E_median_GPa.min()), 18.5, ROUNDED)
report("flax-rich zone, stiffest indent, GPa",
       float(sh.loc[sh.zone == "flax", "E_GPa"].max()), 135, 0.5)
for cls, pct in (("matrix", 32), ("mixed-volume", 55)):
    report(f"{cls}, per cent of the load-controlled basalt-rich zone",
           100 * int(cross.loc["basalt", cls]) / int((lab.zone == "basalt").sum()),
           pct, 0.5)
_lc_classified = int(sum(counts.get(c, 0) for c in zs.CLASSES))
_dr_classified = _dr_all[_dr_all.constituent != "QC/invalid"]
report("depth-resolved indents classified", len(_dr_classified), 1696, 0)
for cls, lc_pct, dr_pct in (("flax", 41.4, 65.9), ("matrix", 16.1, 6.8),
                            ("mixed-volume", 35.6, 13.0),
                            ("basalt", 6.8, 14.3)):
    report(f"Table 3, {cls}, per cent of the classified load-controlled indents",
           100 * int(counts.get(cls, 0)) / _lc_classified, lc_pct, ROUNDED)
    report(f"Table 3, {cls}, per cent of the classified depth-resolved indents",
           100 * float((_dr_classified.constituent == cls).mean()), dr_pct,
           ROUNDED)
report("flax and matrix medians of the load-controlled map differ by, GPa",
       float(medians["flax"] - medians["matrix"]), 0.04, 0.005)
report("their strain-rate sensitivities differ by, per cent (about a fifth)",
       100 * (_srs["flax"] / _srs["matrix"] - 1), 20, 5)
_bias = float(zs.at_share(per_zone, {**medians,
                                     "flax": 1.1 * medians["flax"],
                                     "matrix": 1.1 * medians["matrix"]},
                          w_tiles)[1][0])
report("a 10 % bias on both compliant medians moves the estimate by, GPa",
       _bias - voigt, 0.30, 0.005)
report_sci("largest p over k = 3 to 20, at the supplement's precision",
           _spatial["p_max_over_k_3_to_20"], 9.4, -13)
_ids = _soften["selection"]["indent_ids"]
_cv = pd.read_csv(DATA / "depth_resolved_curves.csv")
_first = (_cv[_cv.indent_id.isin(_ids)].dropna(subset=["E_GPa"])
          .groupby("indent_id").depth_nm.min())
report("selected traces starting at 55 nm", int((_first == 55.0).sum()), 1, 0)

print("\nthe Poisson ratio (2.2, Supplementary Table S1)")
_pois = json.loads((DATA / "poisson_sensitivity.json").read_text())
_shift = {r["class"]: r["shift_pct"] for r in _pois["per_class"]}
for cls, printed in (("flax", 0.00), ("matrix", -9.49),
                     ("mixed-volume", -2.81), ("basalt", 3.56)):
    report(f"shift of the {cls} median, per cent", _shift[cls], printed, 0.0051)
report("largest shift of a class median, per cent",
       max(abs(v) for v in _shift.values()), 9.5, ROUNDED)
report("change of the laminate estimate, per cent",
       abs(_pois["laminate_estimate_GPa"]["shift_pct"]), 1.0, ROUNDED)
_vrh = _pois["mixed_volume_nu"]["averages_by_basalt_fraction"]
for frac in ("0.54", "0.67"):
    report(f"Hill average of the two ratios at basalt fraction {frac}",
           _vrh[frac]["hill"], 0.30, 0.015)
report("basalt-longitudinal measurement above the phase-specific estimate, %",
       _flex.gap_pct(_means["basalt_face"],
                     _pois["laminate_estimate_GPa"]["phase_specific"]), 3.0,
       ROUNDED)

print("\nthe threshold grid, deviation columns (2.3, Supplementary Table S2)")
published_S2_delta = {
    (10, 60): (0.34, 1.84), (10, 65): (0.29, 1.59), (10, 70): (0.25, 1.33),
    (10, 75): (0.22, 1.18), (10, 80): (0.09, 0.50), (15, 60): (0.03, 0.17),
    (15, 65): (-0.08, -0.42), (15, 70): (0.00, 0.00), (15, 75): (-0.02, -0.12),
    (15, 80): (-0.09, -0.50), (20, 60): (-0.26, -1.41), (20, 65): (-0.26, -1.39),
    (20, 70): (-0.29, -1.58), (20, 75): (-0.38, -2.06), (20, 80): (-0.50, -2.70)}
for (lower, upper), (d_gpa, d_pct) in published_S2_delta.items():
    row = grid[(grid.lower_GPa == lower) & (grid.upper_GPa == upper)]
    delta = float(row.estimate_GPa.iloc[0]) - reference
    report(f"deviation at lo = {lower}, hi = {upper}, GPa", delta, d_gpa, 0.0051)
    report(f"deviation at lo = {lower}, hi = {upper}, per cent",
           100 * delta / reference, d_pct, 0.0051)
report("largest deviation, per cent of the reference",
       100 * float((grid.estimate_GPa - reference).abs().max()) / reference, 2.7,
       ROUNDED)
_up = grid[grid.lower_GPa == 15]
report("upper threshold over 60 to 80 GPa, largest change, GPa",
       float((_up.estimate_GPa - reference).abs().max()), 0.09, 0.005)
_lo = grid[grid.upper_GPa == 70]
report("lower threshold over 10 to 20 GPa, largest change, GPa",
       float((_lo.estimate_GPa - reference).abs().max()), 0.29, 0.005)

print("\nthe flexural comparisons, remaining cells (2.5, Supplementary Table S7)")
report_below("raw p, flexural strength", raw_p["flexural strength (MPa)"], 0.0001)
report_below("raw p, softening", raw_p["softening to 0.55 % strain (%)"], 0.001)
report_below("raw p, strain at break", raw_p["strain at break (%)"], 0.0001)
_sab = dict(zip(families["strain at break in place of softening: three"],
                _fx.holm(np.array([raw_p[n] for n in
                         families["strain at break in place of softening: three"]]))))
report_below("Holm p on the strain at break, its family",
             _sab["strain at break (%)"], 0.0001)
from scipy.stats import mannwhitneyu as _mwu                    # noqa: E402
report("smallest two-sided rank-test p at five and four specimens",
       _mwu([1, 2, 3, 4, 5], [6, 7, 8, 9], alternative="two-sided",
            method="exact").pvalue, 0.0159, 0.00005)

print("\nthe Timoshenko correction against the interval (4.2, 4.3.5)")
_u = 1.44
_need = {cfg: 100 * ((voigt + _u) / e - 1) for cfg, e in _means.items()}
report_internal("correction that takes basalt-longitudinal outside, per cent",
                _need["basalt_face"], 5.8, 0.05,
                "behind 'only towards 7 %'; not printed")
report_same("flax-longitudinal is outside already at the low end of 3 to 7 %",
            bool(_means["flax_face"] * 1.03 > voigt + _u), True)

print("\nk-means internal validity (3.3, Supplementary Figure S4)")
_diag = json.loads((DATA / "clustering_diagnostics.json").read_text())
_ks, _sil, _bic = _diag["k"], _diag["silhouette"], _diag["bic"]
report("silhouette largest at k", _diag["silhouette_max_at_k"], 2, 0)
report_same("silhouette at k = 4 and 5 within 0.005 of each other",
            bool(abs(_sil[_ks.index(4)] - _sil[_ks.index(5)]) < 0.005), True)
report_same("the k = 4-5 plateau stands above k = 3 and k = 6",
            bool(min(_sil[_ks.index(4)], _sil[_ks.index(5)])
                 > max(_sil[_ks.index(3)], _sil[_ks.index(6)])), True)
report("mixture BIC smallest at k", _diag["bic_min_at_k"], 7, 0)
report_same("BIC turns up at k = 8", bool(_bic[_ks.index(8)] > _bic[_ks.index(7)]),
            True)
report_same("indents in the diagnostics, against the classified count",
            _diag["n_indents"], len(_dr_classified))

# Printed, and not reproducible from what is deposited. Named rather than left
# out, because the absence of a check is invisible. Each line says what is
# missing and where the missing thing is.
print("\nprinted statements this deposit cannot reproduce")
for where, what, why in [
    ("3.1", "dark-feature area from 12.9 % to 2.7 % across the threshold range; "
     "about 2-3 % at the strict threshold, mostly debonds 1-2 um wide, "
     "macroporosity above 5 um rare",
     "measured on the backscattered-electron pixels, which are several "
     "gigabytes of tile images and are not deposited; the deposit carries the "
     "segmentation at the threshold used, not the sweep"),
    ("2.3", "zone boundary at x = 730 um",
     "a setting, read from a moving-median profile of the modulus along the long "
     "axis; it is carried as X_ZONE_SPLIT in classify.py and resin_profile() "
     "returns the profile"),
    ("Table 1", "loading-rate ratios of about 3 and 0.05 per second, and "
     "'about sixty times faster' in 4.4, which is their ratio",
     "acquisition settings of the two methods, not results"),
    ]:
    print(f"  [--  ] {where}: {what}")
    print(f"         {why}")

print("\n" + "-" * 72)
print(f"{sum(INTERNAL)} of {len(INTERNAL)} internal consistency checks agree. "
      f"They hold the deposit")
print("against itself, a stored product against the live computation of the same "
      "quantity or")
print("an invariant of numbers that are printed, and none of them is a published "
      "value.")
print(f"{sum(PUBLISHED)} of {len(PUBLISHED)} values reproduce")
if not (all(PUBLISHED) and all(INTERNAL)):
    print("Values marked DIFF disagree with the paper.")
    sys.exit(1)
