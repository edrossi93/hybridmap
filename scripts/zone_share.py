"""How much of the laminate figure is decided by where the map was placed.

Reviewer 3 asked for three things in one comment: the Voigt sum of each
architectural zone on its own, the sensitivity of the laminate figure to the
share of the two zones, and that share as a term in the uncertainty budget. He
also observed that the mosaic and the indentation window seem to disagree about
how much of the section is basalt-rich. They do, and the disagreement matters
more than anything already in the budget.

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
against 36.7, purely because downsampling redraws the zone boundary. Both used
to be in circulation: the headline of 18.50 GPa rests on the tile count, while
this script published the mask value and its own 18.60 GPa beside it. One
quantity, one source: the tiles are the measurement, the mask is a rendering of
it, and the mask is used here only for the geometric question of where a window
could have sat. Those are not the same quantity: one
describes where the map sat, the other describes the section. This script
computes the laminate figure under both, and then asks how much the figure
would move if the same window had been placed elsewhere, by sliding a window of
the map's own footprint over the mosaic.

The class medians are held fixed throughout. Only the zone share moves, which
is what isolates the term.

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

# Footprints of the two acquisitions, long x short in micrometres (Table 1).
WINDOWS = {"load-controlled map": (1086, 98),
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


def sliding(mask, um_px, width_um, height_um, stride=STRIDE_PX):
    """Basalt share of the marked area for every placement of one window.

    Summed-area tables rather than a loop over pixels: the mosaic is six
    million pixels and there are tens of thousands of placements.
    """
    h, w = int(round(height_um / um_px)), int(round(width_um / um_px))
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


def tile_share():
    """The basalt-zone share of the section, from the tile counts.

    This is the authoritative one, and it is the one `uncertainty_budget.py`
    uses, so the two scripts now answer with the same number.
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
    out = {"w_basalt_indent_counts": w_indents,
           "w_basalt_mosaic": w_mosaic,
           "w_basalt_downsampled_mask": w_mask,
           "windows": {}}
    for name, (width_um, height_um) in WINDOWS.items():
        s = sliding(mask, um_px, width_um, height_um)
        v = np.array([at_share(per_zone, medians, x)[1][0] for x in s])
        out["windows"][name] = {
            "footprint_um": [width_um, height_um],
            "n_placements": int(s.size),
            "share_mean": float(s.mean()), "share_sd": float(s.std()),
            "voigt_mean": float(v.mean()), "voigt_sd": float(v.std()),
            "voigt_percentiles_5_25_50_75_95":
                [float(x) for x in np.percentile(v, [5, 25, 50, 75, 95])]}
        print(f"  {name}: {s.size} placements of {width_um} x {height_um} um")
        print(f"    basalt share  mean {s.mean():.3f}  sd {s.std():.3f}")
        print(f"    Voigt sum     mean {v.mean():.2f}  sd {v.std():.2f}"
              f"  GPa, 5-95 % {np.percentile(v, 5):.2f} to {np.percentile(v, 95):.2f}")

    (DATA / "zone_share_sensitivity.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print("\n  written: data/zone_share_sensitivity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
