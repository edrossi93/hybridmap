"""How much of the laminate figure is decided by where the map was placed.

Three things are computed here: the Voigt sum of each architectural zone on its
own, the sensitivity of the laminate figure to the share of the two zones, and
that share as a term in the uncertainty budget. The mosaic and the indentation
window disagree about how much of the section is basalt-rich, and the
disagreement matters more than anything else in the budget.

The construction is two-level, and the two levels come from different
measurements:

    within a zone   class fractions and class median moduli, from the indents
    between zones   the share of each zone, which the indent counts only
                    report for the window that was mapped

The indent counts put the window at 49.7 % basalt-rich. The operator's tow
marking of the backscattered-electron mosaic, over sixty times the area, puts
the section at 36.2 % of the marked area.

That share is read from `bse_segmentation_tiles.csv`, the tile counts at the
imaging resolution, and not from the run-length mask, which is the same marking
downsampled for the sliding window below. The two disagree by half a point, 36.2
against 36.7, purely because downsampling redraws the zone boundary. The
headline of 18.50 GPa rests on the tile count. One quantity, one source: the
tiles are the measurement, the mask is a rendering of it, and the mask is used
here only for the geometric question of where a window could have sat.

The indent-count share and the section share are not the same quantity: one
describes where the map sat, the other describes the section. This script
computes the laminate figure under both, and then asks how much the figure
would move if the same window had been placed elsewhere, by sliding a window of
the map's own footprint over the mosaic.

The class medians are held fixed throughout. Only the zone share moves, which
is what isolates the term.

The sliding window's footprint is written as (along the thickness, in the
plane) and slid that way. The map's long axis, 1086 um, runs through the
laminate thickness (2.2), which in the mosaic is the vertical direction, the
rows of the mask. Read the other way round, as width x height, the 1086 um
would lie along the plane of the section and the 98 um through the thickness: a
thin strip lying along one tow instead of a column crossing nearly four of the
sixteen plies, which would see one zone or the other far more often than a real
placement can and would overstate the spread. A third window is also swept: the map's own 704 um of laminate after the resin cut, since that is the
span the retained indents cover and the indent-count share of 0.497 belongs to
it. For each window the share of placements at or below the share that window
saw is recorded, so that the mapped window can be placed on the distribution
rather than only compared with its mean.

Run: python zone_share.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_of_mixtures as rom                                 # noqa: E402
from hybridmap.mixtures import bounds                          # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
CLASSES = ["flax", "matrix", "mixed-volume", "basalt"]

# Footprints in micrometres, written (along the thickness, in the plane). Both
# maps have their long axis through the thickness (2.2, Table 1), and in the
# mosaic the thickness is vertical, so the first number is slid down the rows
# of the mask and the second across its columns. The third entry is the part of
# the load-controlled map that lies on the laminate once the resin is cut away,
# about 704 x 98 um (3.1); it is the span whose indent counts give the 0.497.
WINDOWS = {"load-controlled map": (1086, 98),
           "load-controlled map, laminate span after the resin cut": (704, 98),
           "depth-resolved map": (297, 57)}

STRIDE_PX = 8          # a sliding step below the fibre diameter


def zone_mask():
    """The operator's tow marking, decoded to a pixel array.

    Returns the mask, the micrometres per pixel, and the basalt share of the
    marked area. 1 is the flax zone, 2 the basalt zone, 0 the unmarked
    interface band, which is excluded from every share computed here because
    the classification cannot assign it either.
    """
    d = json.loads((DATA / "zone_mask_rle.json").read_text())
    rle = d["mask_rle"]
    m = np.repeat(np.array(rle[0::2]), np.array(rle[1::2]))
    if m.size != d["width"] * d["height"]:
        raise ValueError("the run lengths do not fill the mosaic")
    m = m.reshape(d["height"], d["width"]).astype(np.uint8)
    marked = (m > 0).sum()
    return m, d["um_per_px"], (m == 2).sum() / marked


def zone_fractions(shallow):
    """Class fractions inside each zone, and the class medians, from the map."""
    table = rom.class_table(shallow)
    medians = dict(zip(table["class"], table[rom.MEDIAN_COL]))
    out = {}
    for zone, sub in shallow.groupby("zone"):
        counts = sub[sub.constituent.isin(medians)].constituent.value_counts()
        out[zone] = {c: float(counts.get(c, 0)) / counts.sum() for c in CLASSES}
    n = shallow[shallow.constituent.isin(medians)].groupby("zone").size()
    return out, medians, float(n["basalt"]) / float(n.sum())


def at_share(per_zone, medians, w_basalt):
    """Voigt, Hill and Reuss with the two zones mixed in the given proportion.

    The mixing is done on the class fractions rather than on the three moduli,
    because the Reuss and Hill terms are not linear in the zone share and a
    weighted average of two zone-level Reuss values is not the Reuss value of
    the mixture.
    """
    f = np.array([w_basalt * per_zone["basalt"][c]
                  + (1 - w_basalt) * per_zone["flax"][c] for c in CLASSES])
    E = np.array([medians[c] for c in CLASSES])
    return f, bounds(pd.Series(f), pd.Series(E))


def sliding(mask, um_px, thickness_um, in_plane_um, stride=STRIDE_PX):
    """Basalt share of the marked area for every placement of one window.

    `thickness_um` is the extent down the rows of the mask, which is the
    through-thickness direction of the mosaic; `in_plane_um` is the extent
    across its columns. The two arguments are named for the direction rather
    than called width and height, because a width and a height are what let the
    footprint go in transposed.

    Summed-area tables rather than a loop over pixels: the mosaic is six
    million pixels and there are tens of thousands of placements.
    """
    h, w = int(round(thickness_um / um_px)), int(round(in_plane_um / um_px))
    tables = [np.pad((mask == v).astype(np.int64), ((1, 0), (1, 0)))
              .cumsum(0).cumsum(1) for v in (2, 1)]
    H, W = mask.shape
    out = []
    for y in range(0, H - h + 1, stride):
        for x in range(0, W - w + 1, stride):
            b, f = (t[y + h, x + w] - t[y, x + w] - t[y + h, x] + t[y, x]
                    for t in tables)
            if b + f:
                out.append(b / (b + f))
    return np.array(out)


def depth_resolved_share():
    """The basalt-rich share of the depth-resolved strip, by indent count.

    The strip crosses three bands, flax, basalt, flax, so its basalt-rich
    share is the width of the centre band: 600 of the 2000 indents. It is the
    value the depth-resolved window is placed at on its own distribution below,
    as the load-controlled window is placed at 0.497 on its.
    """
    d = pd.read_csv(DATA / "classified_depth_resolved.csv")
    return float((d.zone == "basalt-centre").mean())


def tile_share():
    """The basalt-zone share of the section, from the tile counts.

    This is the share Eq. (1) uses, and `uncertainty_budget.py` uses the same
    one.
    """
    t = pd.read_csv(DATA / "bse_segmentation_tiles.csv")
    b, f = float(t.n_basaltzone.sum()), float(t.n_flaxzone.sum())
    return b / (b + f)


def threshold_grid(raw_shallow, w_basalt, lowers=(10, 15, 20),
                   uppers=(60, 65, 70, 75, 80)):
    """The estimate over the classification-threshold grid of Table S2.

    The same sweep as `rule_of_mixtures.threshold_sweep`, but carried through
    the two-level construction instead of the whole-map indent fractions, which
    is what the published table reports. The resin cut and the zone boundary are
    held fixed while the modulus thresholds move: they are set by position
    rather than by stiffness, so sweeping them would confound this test with a
    different one.

    `uncertainty_budget.threshold_term` needs only the extremes of this grid and
    computes them its own way for its Type B term. That the two agree is checked
    in `reproduce_paper_values.py` rather than assumed here.
    """
    import classify as cls

    d = raw_shallow[raw_shallow.x_um >= cls.X_RESIN].copy()
    d["zone"] = cls.assign_zone(d.x_um.values).values

    rows = []
    for lower in lowers:
        for upper in uppers:
            label = cls.classify(d.E_GPa.values, d.zone.values,
                                 e_lower=lower, e_upper=upper).values
            f = d.assign(constituent=label)
            per_zone, medians, _ = zone_fractions(f)
            rows.append({"lower_GPa": lower, "upper_GPa": upper,
                         "estimate_GPa": at_share(per_zone, medians,
                                                  w_basalt)[1][0]})
    return pd.DataFrame(rows)


def main():
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    per_zone, medians, w_indents = zone_fractions(shallow)
    mask, um_px, w_mask = zone_mask()
    w_mosaic = tile_share()

    print(f"  zone share from the indent counts   {w_indents:.4f}")
    print(f"  zone share from the mosaic tiles    {w_mosaic:.4f}")
    print(f"  the same marking, downsampled       {w_mask:.4f}"
          f"   (used only for the sliding window)")
    print(f"  mosaic {mask.shape[1] * um_px:.0f} x {mask.shape[0] * um_px:.0f} "
          f"um, {100 * (mask == 0).mean():.1f} % unmarked\n")

    for name, w in (("each zone alone, basalt-rich", 1.0),
                    ("each zone alone, flax-rich", 0.0),
                    ("indent counts", w_indents),
                    ("mosaic tiles", w_mosaic)):
        f, (V, H, R) = at_share(per_zone, medians, w)
        print(f"  {name:<28} w_basalt {w:.4f}  "
              f"Voigt {V:7.4f}  Hill {H:7.4f}  Reuss {R:6.4f}")

    print()
    print("  windows of each acquisition's footprint slid over the mosaic, the "
          "long axis through the thickness")
    # The share each window actually saw, by indent count, so that the window
    # can be placed on the distribution of where it could have sat.
    w_depth = depth_resolved_share()
    mapped = {"load-controlled map": w_indents,
              "load-controlled map, laminate span after the resin cut":
                  w_indents,
              "depth-resolved map": w_depth}
    out = {"w_basalt_indent_counts": w_indents,
           "w_basalt_mosaic": w_mosaic,
           "w_basalt_downsampled_mask": w_mask,
           "w_basalt_depth_resolved_strip": w_depth,
           "windows": {}}
    for name, (thickness_um, in_plane_um) in WINDOWS.items():
        s = sliding(mask, um_px, thickness_um, in_plane_um)
        v = np.array([at_share(per_zone, medians, x)[1][0] for x in s])
        # Where the mapped window sits on the distribution of placements: the
        # share of placements at or below the share it saw, and the share at
        # or above it, both as percentages.
        seen = mapped[name]
        below = 100 * float((s <= seen).mean())
        above = 100 * float((s >= seen).mean())
        out["windows"][name] = {
            "footprint_um_along_thickness_by_in_plane":
                [thickness_um, in_plane_um],
            "n_placements": int(s.size),
            "share_mean": float(s.mean()), "share_sd": float(s.std()),
            "voigt_mean": float(v.mean()), "voigt_sd": float(v.std()),
            "voigt_percentiles_5_25_50_75_95":
                [float(x) for x in np.percentile(v, [5, 25, 50, 75, 95])],
            "mapped_window_share": seen,
            "placements_at_or_below_mapped_share_pct": below,
            "placements_at_or_above_mapped_share_pct": above}
        print(f"  {name}: {s.size} placements of {thickness_um} um through "
              f"the thickness x {in_plane_um} um in the plane")
        print(f"    basalt share  mean {s.mean():.3f}  sd {s.std():.3f}")
        print(f"    Voigt sum     mean {v.mean():.2f}  sd {v.std():.2f}"
              f"  GPa, 5-95 % {np.percentile(v, 5):.2f} to "
              f"{np.percentile(v, 95):.2f}")
        print(f"    the share this window saw, {seen:.3f}: "
              f"{below:.1f} % of placements at or below it, "
              f"{above:.1f} % at or above it")

    (DATA / "zone_share_sensitivity.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print("\n  written: data/zone_share_sensitivity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
