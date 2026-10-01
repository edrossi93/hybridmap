"""
Uncertainty on the laminate Voigt-type estimate, by resampling rather than by
first-order propagation.

The construction has two levels and they are measured by different instruments:

    within a zone   class fractions and class median moduli, from the indents
    between zones   the share of the two architectural zones, from the operator
                    tow marking of the backscattered-electron mosaic

The zone share is taken from the mosaic because indent counts would describe
the window that happened to be mapped rather than the section, and it appears in
this budget as a term of its own. As a term it is the second largest of the
three sampling terms, 0.61 GPa against 1.08 for the class fractions; what puts
the estimate on the mosaic share is not the size of the term but the 2.83 GPa
between the two weightings, which is a correction and not an uncertainty and is
reported as the alternative in Table 2. See `zone_share.py` for that
measurement and for how far the figure moves.

Three sources are propagated together, in one Monte Carlo rather than as
independent terms added afterwards, so that their correlations are carried:

  medians     class median moduli, from the resampled indents
  fractions   class fractions inside each zone, from the same resample
  zone share  resampled over the 91 mosaic tiles

The indent resampling is a **spatial block bootstrap**, not a draw over
individual indents, because a Dirichlet on the raw counts treats neighbouring
indents as independent observations, which at a 5.1 x 2.5 um pitch on 10 to
20 um fibres they are not: adjacent contacts
frequently sample the same fibre or the same matrix pocket. The blocks here are
contiguous groups of grid columns spanning the full short axis of the map, four
columns wide, about 20 um, or one fibre diameter. Blocks are resampled with
replacement within each zone. `BLOCK_COLS` is swept in the reported output so
that the choice can be seen rather than trusted.

Three further terms are not random quantities but are still uncertainty. They
are included as Type B contributions in the sense of the GUM, each evaluated as
a rectangular distribution over the range its sensitivity analysis spans, so
the standard uncertainty is the half-range divided by root three:

  thresholds  the classification thresholds, over the same 3 x 5 grid as
              Supplementary Table S2, recomputed at the mosaic zone share
  poisson     the uniform Poisson ratio of 0.30 against the phase-specific
              values of Supplementary Table S1
  runaway     the fifty-nine flax-zone contacts reading below 1 GPa, which the
              Supplementary Figure S3 caption argues are failed contacts and
              which the load-controlled map keeps because it carries no
              filter: the estimate with them in against the estimate with
              them out, 18.50 against 18.63 GPa. It is the smallest term in
              the budget and moves the combined value by less than a
              thousandth of a GPa

One term stays outside the budget on purpose. Which basalt input to use, the
shallow 86.0 GPa or the depth-resolved 54.7 GPa, is a choice rather than an
uncertainty, and folding it in would report a decision as noise. The choice
rests on the stiff depth-resolved contacts reading 86 GPa near the surface and
then falling with depth (Supplementary Figure S7), as the surroundings of the
fibre enter the measurement with load; the depth-resolved median is therefore
not a second estimate of the fibre modulus. It is reported as a sensitivity instead.

The block-size sweep at the end is printed in two forms, as the caption of
Supplementary Table S5 gives both. The sweep resamples 1000
draws at each block width and its standard deviation is the joint sampling
term alone, Type A; the caption's "combined value" is that term in quadrature
with the Type B terms. Both are printed with their draw counts. The
budget's own joint sampling term is the 6000-draw value, and at the four-column
block the 1000-draw sweep lands near it but not on it, which is what a
resampled quantity does at a sixth of the draws.

Run: python uncertainty_budget.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

import classify as _cls                                        # noqa: E402
import zone_share as _zs                                       # noqa: E402
from hybridmap.mixtures import bounds                           # noqa: E402
from rule_of_mixtures import CLASSES, MEDIAN_COL, class_table   # noqa: E402

SEED = 42
N_DRAWS = 6_000

# The per-term standard deviations only need three figures, and each costs a
# full resampling pass of its own, so they run shorter than the combined draw.
FROZEN_DRAWS = 2_000

# Four grid columns, 5.0959 um apart, so about 20 um: one fibre diameter, the
# length over which adjacent contacts stop being independent samples.
BLOCK_COLS = 4
BLOCK_SWEEP = (2, 4, 8)

# The per-class shift when the uniform Poisson ratio of 0.30 is replaced by
# phase-specific values is derived rather than copied from Supplementary
# Table S1: poisson_sensitivity.py inverts the Oliver-Pharr relation on the
# deposited reduced modulus at each nu. Two routes to one quantity can disagree
# in the fourth decimal, which is enough to move a printed value across a
# rounding boundary, and a hard-coded table cannot be checked against anything.

RECTANGULAR = 3 ** 0.5

# A flax-zone contact reading below this is a runaway indent rather than soft
# material: the Supplementary Figure S3 caption gives the evidence, a median
# contact depth of 3.2 um against 274 nm for the rest of the zone. The map is
# used as acquired, so they stay in every count the paper reports, and what
# leaving them in is worth enters the budget below as a bounded choice.
RUNAWAY_ZONE = "flax"
RUNAWAY_BELOW_GPA = 1.0


def runaway_contacts(shallow):
    """True on the flax-zone contacts reading below 1 GPa."""
    return (shallow.zone == RUNAWAY_ZONE) & (shallow.E_GPa < RUNAWAY_BELOW_GPA)


def rectangular(low, high):
    """Standard uncertainty of a value known only to lie in [low, high]."""
    return abs(high - low) / 2 / RECTANGULAR


def runaway_term(shallow, w_mosaic):
    """Type B: the estimate with the runaway contacts in against with them out.

    Both the flax median and the class fractions move when fifty-nine indents
    leave, so the alternative is rebuilt from the surviving indents rather than
    by substituting one median into the construction. This is the same
    arithmetic `reproduce_paper_values.py` holds to the caption's 18.63 GPa.
    """
    kept = shallow[~runaway_contacts(shallow)]
    per_zone, medians, _ = _zs.zone_fractions(kept)
    _, (excluded, _, _) = _zs.at_share(per_zone, medians, w_mosaic)
    per_zone, medians, _ = _zs.zone_fractions(shallow)
    _, (included, _, _) = _zs.at_share(per_zone, medians, w_mosaic)
    return rectangular(included, excluded), [included, excluded]


def blocks(shallow, block_cols):
    """Row indices grouped into contiguous column blocks, per zone."""
    out = {}
    for zone, sub in shallow.groupby("zone"):
        cols = np.sort(sub.col.unique())
        which = {c: i // block_cols for i, c in enumerate(cols)}
        key = sub.col.map(which).to_numpy()
        out[zone] = [np.flatnonzero(key == b) + sub.index[0] * 0
                     for b in np.unique(key)]
        # index positions within the original frame, not label offsets
        out[zone] = [sub.index.to_numpy()[key == b] for b in np.unique(key)]
    return out


def one_draw(shallow, blk, rng, tiles):
    """One resampled estimate: blocks within each zone, tiles for the share."""
    frames = {}
    for zone, bs in blk.items():
        pick = rng.integers(0, len(bs), len(bs))
        frames[zone] = shallow.loc[np.concatenate([bs[i] for i in pick])]
    pooled = pd.concat(frames.values())

    medians, fracs = {}, {}
    for c in CLASSES:
        v = pooled.loc[pooled.constituent == c, "E_GPa"].to_numpy()
        medians[c] = np.median(v) if v.size else np.nan
    for zone, f in frames.items():
        counts = f[f.constituent.isin(CLASSES)].constituent.value_counts()
        fracs[zone] = {c: float(counts.get(c, 0)) / counts.sum()
                       for c in CLASSES}

    i = rng.integers(0, len(tiles), len(tiles))
    b, fl = tiles.n_basaltzone.values[i].sum(), tiles.n_flaxzone.values[i].sum()
    w = b / (b + fl)

    f = np.array([w * fracs["basalt"][c] + (1 - w) * fracs["flax"][c]
                  for c in CLASSES])
    return float((f * np.array([medians[c] for c in CLASSES])).sum()), w


def frozen(shallow, blk, tiles, rng, vary, point):
    """The same draw with only one source allowed to move.

    Reporting a share of variance for each term means each term has to be
    evaluated on its own, which for a correlated Monte Carlo means freezing
    the other two at their measured values rather than subtracting variances.
    """
    per_zone, medians, _ = _zs.zone_fractions(shallow)
    w0 = tiles.n_basaltzone.sum() / (tiles.n_basaltzone.sum()
                                     + tiles.n_flaxzone.sum())
    out = []
    for _ in range(FROZEN_DRAWS):
        m, fr, w = dict(medians), {z: dict(v) for z, v in per_zone.items()}, w0
        if vary in ("medians", "fractions"):
            frames = {}
            for zone, bs in blk.items():
                pick = rng.integers(0, len(bs), len(bs))
                frames[zone] = shallow.loc[
                    np.concatenate([bs[i] for i in pick])]
            if vary == "medians":
                pooled = pd.concat(frames.values())
                for c in CLASSES:
                    v = pooled.loc[pooled.constituent == c,
                                   "E_GPa"].to_numpy()
                    if v.size:
                        m[c] = float(np.median(v))
            else:
                for zone, f in frames.items():
                    counts = f[f.constituent.isin(CLASSES)
                               ].constituent.value_counts()
                    fr[zone] = {c: float(counts.get(c, 0)) / counts.sum()
                                for c in CLASSES}
        else:
            i = rng.integers(0, len(tiles), len(tiles))
            b, fl = (tiles.n_basaltzone.values[i].sum(),
                     tiles.n_flaxzone.values[i].sum())
            w = b / (b + fl)
        f = np.array([w * fr["basalt"][c] + (1 - w) * fr["flax"][c]
                      for c in CLASSES])
        out.append(float((f * np.array([m[c] for c in CLASSES])).sum()))
    return float(np.std(out))


def threshold_term(shallow_raw, tiles, point):
    """Type B: the classification thresholds, at the mosaic zone share."""
    w = tiles.n_basaltzone.sum() / (tiles.n_basaltzone.sum()
                                    + tiles.n_flaxzone.sum())
    d = shallow_raw[shallow_raw.x_um >= _cls.X_RESIN].copy()
    d["zone"] = _cls.assign_zone(d.x_um.values).values
    vals = []
    for lower in (10, 15, 20):
        for upper in (60, 65, 70, 75, 80):
            lab = _cls.classify(d.E_GPa.values, d.zone.values,
                                e_lower=lower, e_upper=upper).values
            f = d.assign(constituent=lab)
            med = {c: f.loc[f.constituent == c, "E_GPa"].median()
                   for c in CLASSES}
            fr = {}
            for zone, sub in f.groupby("zone"):
                counts = sub[sub.constituent.isin(CLASSES)
                             ].constituent.value_counts()
                fr[zone] = {c: float(counts.get(c, 0)) / max(counts.sum(), 1)
                            for c in CLASSES}
            if any(np.isnan(med[c]) for c in CLASSES):
                continue
            frac = np.array([w * fr["basalt"][c] + (1 - w) * fr["flax"][c]
                             for c in CLASSES])
            vals.append(float((frac * np.array([med[c]
                                                for c in CLASSES])).sum()))
    return rectangular(min(vals), max(vals)), [min(vals), max(vals)]


def main():
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    raw = pd.read_csv(DATA / "shallow_map.csv")
    tiles = pd.read_csv(DATA / "bse_segmentation_tiles.csv")

    per_zone, medians, w_indents = _zs.zone_fractions(shallow)
    w_mosaic = tiles.n_basaltzone.sum() / (tiles.n_basaltzone.sum()
                                           + tiles.n_flaxzone.sum())
    _, (point, hill, reuss) = _zs.at_share(per_zone, medians, w_mosaic)
    _, (window_point, _, _) = _zs.at_share(per_zone, medians, w_indents)

    blk = blocks(shallow, BLOCK_COLS)
    print(f"  blocks per zone: "
          f"{ {z: len(b) for z, b in blk.items()} }, {BLOCK_COLS} columns each")

    rng = np.random.default_rng(SEED)
    draws = [one_draw(shallow, blk, rng, tiles)[0] for _ in range(N_DRAWS)]
    combined_a = float(np.std(draws))

    type_a = {}
    for vary in ("medians", "fractions", "zone_share"):
        type_a[vary] = frozen(shallow, blk, tiles,
                              np.random.default_rng(SEED), vary, point)

    thr, thr_range = threshold_term(raw, tiles, point)
    import poisson_sensitivity as _poi
    poisson_med = {r["class"]: r["E_phase_specific_GPa"]
                   for _, r in _poi.table().iterrows()}
    _, (poisson_point, _, _) = _zs.at_share(per_zone, poisson_med, w_mosaic)
    poi = rectangular(point, poisson_point)
    run, run_range = runaway_term(shallow, w_mosaic)
    type_b = {"classification_thresholds": thr, "poisson_ratio": poi,
              "runaway_contacts": run}

    combined = (combined_a ** 2 + sum(v ** 2 for v in type_b.values())) ** 0.5
    # The share of variance each term carries, as Supplementary Table S5
    # prints it: the term squared over the combined value squared. Evaluated
    # one term at a time with the others held, so the shares do not sum to
    # one; the joint sampling term is listed beside its three parts.
    shares = {k: 100 * v ** 2 / combined ** 2
              for k, v in {**type_a, "type_a_combined": combined_a,
                           **type_b}.items()}

    sweep = {}
    for bc in BLOCK_SWEEP:
        r = np.random.default_rng(SEED)
        b2 = blocks(shallow, bc)
        sweep[bc] = float(np.std([one_draw(shallow, b2, r, tiles)[0]
                                  for _ in range(1000)]))

    budget = {
        "construction": ("class fractions and medians from the indents, zone "
                         "share from the mosaic tow marking"),
        "voigt_point_estimate": point,
        "hill": hill, "reuss": reuss,
        "voigt_at_indent_count_zone_share": window_point,
        "w_basalt_mosaic": float(w_mosaic),
        "w_basalt_indent_counts": float(w_indents),
        "n_draws": N_DRAWS, "seed": SEED,
        "block_columns": BLOCK_COLS,
        "block_size_sensitivity_sd_GPa": sweep,
        "type_a_sd_GPa": type_a,
        "type_a_combined_sd_GPa": combined_a,
        "type_b_sd_GPa": type_b,
        "type_b_ranges": {"threshold_sweep_GPa": thr_range,
                          "poisson_specific_voigt_GPa": poisson_point,
                          "runaway_contacts_in_and_out_GPa": run_range},
        "combined_sd_GPa": combined,
        "share_of_variance_pct": shares,
        "percentile_95_interval_GPa": [float(np.percentile(draws, 2.5)),
                                       float(np.percentile(draws, 97.5))],
        "excluded": ("The choice of basalt input, a discriminated choice "
                     "reported as a sensitivity rather than as noise."),
    }
    out = DATA / "uncertainty_budget.json"
    out.write_text(json.dumps(budget, indent=2) + "\n",
                   encoding="utf8", newline="\n")

    print(f"\n  zone share  mosaic {w_mosaic:.4f}   indent counts "
          f"{w_indents:.4f}")
    print(f"  Voigt-type estimate            {point:.3f} GPa "
          f"(window-weighted {window_point:.3f})")
    print(f"  Hill {hill:.3f}   Reuss {reuss:.3f}\n")
    print("  Type A, one source at a time, spatial block bootstrap")
    for k, v in type_a.items():
        print(f"    {k:<28} ± {v:.3f}")
    print(f"    {'all three together':<28} ± {combined_a:.3f}\n")
    print("  Type B, rectangular over the sensitivity range")
    for k, v in type_b.items():
        print(f"    {k:<28} ± {v:.3f}")
    print(f"    the runaway range: {run_range[0]:.4f} with the {int(runaway_contacts(shallow).sum())} contacts in, "
          f"{run_range[1]:.4f} with them out")
    print(f"\n  combined                       ± {combined:.3f}")
    print(f"  reported as                      {point:.2f} ± {combined:.2f} GPa")
    print("\n  share of variance, one term at a time, as Supplementary Table S5 prints it")
    for k, v in shares.items():
        print(f"    {k:<28} {v:6.2f} %")
    type_b_sq = sum(v ** 2 for v in type_b.values())
    print(f"\n  block-size sensitivity, 1000 draws at each width "
          f"(grid columns 5.0959 um apart)")
    print(f"    {'block':<14}{'sampling term alone':>21}"
          f"{'with Type B in quadrature':>28}")
    for k, v in sweep.items():
        print(f"    {k} cols, {k * 5.0959:4.0f} um{v:>17.3f}"
              f"{(v ** 2 + type_b_sq) ** 0.5:>25.3f}")
    print(f"    the budget's own sampling term, {N_DRAWS} draws at "
          f"{BLOCK_COLS} cols: {combined_a:.3f}, combined {combined:.3f}")
    print(f"\nwritten to {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
