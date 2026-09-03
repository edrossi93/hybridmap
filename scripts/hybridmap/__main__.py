"""
Command line for the tool.

Run it from the scripts/ directory as `python -m hybridmap`, or install the
deposit with `pip install -e .` and call `hybridmap` from anywhere.

Starting from your own CSV:

    hybridmap init      --map your_map.csv --out yours.yaml
    hybridmap validate  --config yours.yaml
    hybridmap histogram --config yours.yaml --out figs
    hybridmap report    --config yours.yaml --out results

The other commands, once the config is settled:

    classify      the per-indent labels, as a CSV
    sweep         the bounds across a grid of cut values
    uncertainty   the resampled spread on the bound
    cluster       the unsupervised cross-check
    macro         bending curves reduced to per-specimen moduli

Every command except init takes --set to override one value without editing the
file, for example --set cuts.lower=12. Overrides are applied before validation,
so a typo in one is caught the same way a typo in the file is.

A configuration error exits 2 and prints what to fix. Nothing here writes to
the input file or to data/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from . import aggregate, classify, cluster, io, macro, mixtures, uncertainty
from .config import ConfigError, coverage_warnings, load

RULE = "-" * 72


def _load(args):
    cfg = load(args.config, args.set)
    frame = io.read_map(cfg.map)
    cols = io.resolve(frame, cfg.map)
    return cfg, frame, cols


def _labelled(cfg, frame, cols):
    return classify.apply(frame, cfg, cols)


def _warn(cfg):
    for w in coverage_warnings(cfg):
        print(f"  warning: {w}")


def cmd_validate(args):
    cfg, frame, cols = _load(args)
    print(f"{cfg.name}   ({cfg.source.name})")
    print(f"  map        {cfg.map.path.name}, {len(frame)} rows, "
          f"{cfg.map.kind.replace('_', ' ')}")
    print(f"  modulus    {cols['modulus']}")

    # What the tool actually sees in the file. A config can be perfectly valid
    # against a file it is reading wrongly, and the two questions a user has
    # when a run looks odd are whether the file was read as intended and
    # whether the cuts are anywhere near the data.
    data = io.describe_map(frame, cols, cfg)
    if data["usable"]:
        print(f"  values     {data['usable']} usable, {data['missing']} missing"
              f"; {data['min']:.3g} to {data['max']:.3g}, "
              f"median {data['median']:.3g}")
    if "position" in data:
        lo, hi = data["position"]
        print(f"  position   {cols['position']}, {lo:.4g} to {hi:.4g}")
    if cfg.zones:
        print(f"  zones      {', '.join(cfg.zones.names) or 'none kept'}")
        dropped = [b.name for b in cfg.zones.bands if b.discard]
        if dropped:
            print(f"  discarded  {', '.join(dropped)}")
    else:
        print("  zones      none, so the classification is modulus only")
    print(f"  cuts       " + ", ".join(f"{k} = {v:g}" for k, v in cfg.cuts.items()))
    print(f"  phases     {', '.join(cfg.phase_names)}")
    if cfg.clustering:
        print(f"  clustering k = {cfg.clustering.k} on "
              f"{len(cfg.clustering.features)} features")
    if cfg.macro:
        for m in cfg.macro.measured:
            sd = f" +/- {m.sd:g}" if m.sd is not None else ""
            print(f"  measured   {m.label}: {m.value:g}{sd}")
        if cfg.macro.curves:
            print(f"  curves     {cfg.macro.curves.path.name}")
    _warn(cfg)
    for w in data["warnings"]:
        print(f"  warning: {w}")

    # Loading is not proof it will run: the columns the clustering and the
    # quality check name are only touched later, and a config that fails then
    # has failed validation in every sense the user cares about.
    labelled = _labelled(cfg, frame, cols)
    counts = labelled[classify.LABEL_COL].value_counts()
    unlabelled = int(counts.get(classify.UNMATCHED, 0))
    print(f"  labelled   {len(labelled) - unlabelled} of {len(labelled)} kept "
          f"indents")
    if unlabelled:
        print(f"  warning: {unlabelled} indents matched no phase")
    print("config is usable")
    return 0


def cmd_classify(args):
    cfg, frame, cols = _load(args)
    labelled = _labelled(cfg, frame, cols)
    counts = labelled[classify.LABEL_COL].value_counts()
    print(f"{len(frame)} rows read, {len(labelled)} kept after the zone discard")
    print(counts.to_string())
    if args.out:
        labelled.to_csv(args.out, index=False)
        print(f"written to {args.out}")
    return 0


def _reduce_curves(cfg):
    """The per-specimen moduli, or None when no curves are configured."""
    if not cfg.macro or not cfg.macro.curves:
        return None
    spec = cfg.macro.curves
    curves = io.read_table(spec.path, spec.read, "macro.from_curves")
    return macro.moduli_from_curves(curves, spec)


def _sweep(cfg, labelled, cols):
    return mixtures.sweep(labelled, cfg.phases, cfg.cuts, cfg.sweep,
                          cols["modulus"], cfg.phase_names)


def _table_and_bounds(cfg, labelled, cols):
    table = aggregate.class_table(labelled, cfg.phase_names, cols["modulus"])
    return table, mixtures.bounds(table.fraction, table["median"])


def cmd_report(args):
    cfg, frame, cols = _load(args)
    labelled = _labelled(cfg, frame, cols)
    _warn(cfg)

    print(f"\n{cfg.name}")
    print(RULE)
    print(f"{len(frame)} rows read, {len(labelled)} kept, "
          f"modulus from {cols['modulus']}")

    table, (voigt, hill, reuss) = _table_and_bounds(cfg, labelled, cols)
    print("\nphases")
    print(table.to_string(index=False,
                          float_format=lambda v: f"{v:.4f}"))

    # The bound carries its resampled spread here, not in a separate command:
    # a bound quoted bare invites being read as exact.
    # It costs about a second on a map of several thousand indents.
    replicates = uncertainty.bootstrap_medians(
        labelled, cfg.phase_names, cols["modulus"],
        cfg.uncertainty["n_draws"], cfg.uncertainty["seed"])
    spread = uncertainty.budget(table, replicates, cfg.phase_names,
                                seed=cfg.uncertainty["seed"],
                                n_draws=cfg.uncertainty["n_draws"],
                                level=cfg.uncertainty["level"])

    print(f"\nVoigt {voigt:.2f} +/- {spread['combined_sd']:.2f}   "
          f"Hill {hill:.2f}   Reuss {reuss:.2f}")
    print("Fractions are indent-class fractions, not volume fractions: they "
          "count\nindents carrying each label over indents carrying any label.")

    if cfg.zones and len(cfg.zones.names) > 1:
        glob, zoned = aggregate.check_zone_weighting(labelled, cfg.phase_names)
        agree = "identical" if np.allclose(glob, zoned) else "NOT identical"
        print(f"\nzone-weighted fractions are {agree} to the global counts")

    sweep_table, reduced, comparison, measured = None, None, None, []
    if cfg.sweep:
        sweep_table = sw = _sweep(cfg, labelled, cols)
        s = mixtures.sweep_summary(sw, cfg.cuts)
        if s:
            print(f"\ncut sensitivity over {len(sw)} combinations: "
                  f"{s['min_GPa']:.2f} to {s['max_GPa']:.2f}, "
                  f"largest departure from the configured cuts "
                  f"{s['max_deviation_GPa']:.2f}")

    if cfg.sensitivity:
        # Reported beside the answer rather than folded into it. Which value a
        # phase should take is sometimes a choice between two defensible
        # measurements, and a choice is not noise.
        print("\nif a phase input were different")
        for s in cfg.sensitivity:
            v, _, _ = mixtures.substitute(table, s.phase, s.value,
                                          median_col="median")
            label = s.label or f"{s.phase} set to {s.value:g}"
            print(f"  {label:<48} Voigt {v:.2f}")

    if cfg.macro:
        # Reduced here. Before this the user was told to go and run a second
        # command, then type its answers back into the config, in a unit
        # nothing had mentioned.
        reduced = _reduce_curves(cfg)
        scale = cfg.macro.curves.modulus_scale if reduced is not None else 1.0
        measured = macro.measurements(cfg.macro, reduced, scale)

        if reduced is not None:
            print(f"\nfrom {cfg.macro.curves.path.name}, "
                  f"{len(reduced)} specimens")
            print(macro.summarise(reduced, scale)
                  .to_string(float_format=lambda v: f"{v:.3f}"))
            _contrast(reduced, scale)

        if measured:
            comparison = macro.compare((voigt, hill, reuss), measured)
            print("\nagainst the macroscopic measurement")
            print(comparison.to_string(index=False,
                                       float_format=lambda v: f"{v:.2f}"))

    if args.out:
        written = _write_report(args.out, cfg, table, (voigt, hill, reuss),
                                spread, replicates,
                                sweep_table if cfg.sweep else None,
                                reduced if cfg.macro else None,
                                comparison if cfg.macro and measured else None,
                                labelled, cols)
        print(f"\n{len(written)} files written to {args.out}")
        for path in written:
            print(f"  {path.name}")
    return 0


def _guess(columns, wanted):
    """First column whose name contains one of the words, or None."""
    for word in wanted:
        for column in columns:
            if word in str(column).lower():
                return column
    return None


def cmd_init(args):
    """
    Write a starter config for a CSV, with the user's own column names in it.

    The column guesses are guesses and are marked as such. The cuts are placed
    at even quantiles of the data, which is a stated rule and not a
    judgement: a cut is a decision about where the populations divide, and this
    has not looked at the populations. The comment in the file says so, and
    'hybridmap histogram' is how to replace them with a real choice.
    """
    frame = io.read_table(args.map, {}, "map")
    columns = list(frame.columns)

    modulus = _guess(columns, ("modulus", "e_", "_e", "er", "young"))
    if modulus is None:
        raise ConfigError(
            f"could not tell which column in {Path(args.map).name} holds the "
            f"modulus. Write the config by hand from the template in the "
            f"README; the columns are: {', '.join(map(str, columns))}")
    identifier = _guess(columns, ("id", "index", "point"))
    position = _guess(columns, ("x_", "x ", "pos", "distance"))

    values = io.numeric(frame, modulus, "the phases are cut on").dropna()
    n = max(int(args.phases), 2)
    quantiles = [float(values.quantile(i / n)) for i in range(1, n)]

    lines = [
        f"# Starting point for {Path(args.map).name}, written by",
        "# 'hybridmap init'. Two things in here are not decisions yet.",
        "#",
        "# The column roles below are guesses from the column names. Check them.",
        "#",
        "# The cuts are at even quantiles of the modulus, which spreads the",
        "# indents evenly between phases and means nothing physically. Run",
        "# 'hybridmap histogram --config this-file' and put them where your",
        "# distribution actually divides, then say here why each one is there.",
        "",
        f"name: {Path(args.map).stem}",
        "",
        "map:",
        f"  path: {Path(args.map).name}",
        "  kind: single_depth",
        "  columns:",
    ]
    if identifier:
        lines.append(f"    id: {identifier}")
    lines.append(f"    modulus: {modulus}")
    if position:
        lines.append(f"    position: {position}          # only needed for zones")
    lines += [
        "",
        "# Uncomment to split the map into bands along the position axis, for",
        "# example to discard a mounting medium. A modulus-only classification",
        "# is legitimate when there is no architecture to exploit.",
        "# zones:",
        "#   axis: position",
        "#   bands:",
        "#     - {name: mount, below: 0.0, discard: true}",
        "#     - {name: specimen}",
        "",
        "cuts:",
    ]
    for i, value in enumerate(quantiles, start=1):
        lines.append(f"  cut{i}: {value:.4g}        # placeholder, see above")

    lines += ["", "phases:"]
    edges = [None] + [f"cut{i}" for i in range(1, n)] + [None]
    for i in range(n):
        low, high = edges[i], edges[i + 1]
        if low is None:
            rule = f"below: {high}"
        elif high is None:
            rule = f"above: {low}"
        else:
            rule = f"between: [{low}, {high}]"
        lines.append(f"  - {{name: phase{i + 1}, {rule}}}")

    lines += [
        "",
        "# What you are comparing the prediction against. A value with an",
        "# uncertainty is enough; see the README for reducing bending curves",
        "# to one instead.",
        "# macro:",
        "#   measured:",
        "#     - {label: three-point bending, value: 0.0, sd: 0.0, n: 1}",
        "",
    ]

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf8")
        print(f"written to {args.out}")
        # No invocation prefix: this runs both as `python -m hybridmap` and
        # as an installed `hybridmap`.
        print(f"next: 'histogram --config {args.out} --out figs', to choose "
              f"the cuts, then 'report'")
    else:
        print(text)
    return 0


def _write_report(where, cfg, table, bounds, spread, replicates, sweep_table,
                  reduced, comparison, labelled, cols):
    """
    Everything the report computed, as files instead of scrollback.

    A user analysing their own laminate needs a table they can put in a paper
    and a figure they can look at, not terminal output to retype. The figures
    are the same three the histogram command writes, built from the same
    numbers as the tables beside them.
    """
    import json

    from . import plots

    out = Path(where)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def csv(frame, name, index=False):
        # Pinned, so a table written on Windows and one written on Linux are
        # the same file.
        path = out / name
        frame.to_csv(path, index=index, lineterminator="\n")
        written.append(path)

    # Named for the construction, not called a confidence interval.
    # There is more than one defensible interval for a median, they do not
    # agree, and a column headed "interval" invites being compared against one
    # built some other way.
    intervals = uncertainty.intervals(replicates, cfg.uncertainty["level"])
    phases = table.copy()
    phases["bootstrap_low"] = [intervals[c][0] for c in phases["class"]]
    phases["bootstrap_high"] = [intervals[c][1] for c in phases["class"]]
    csv(phases, "phases.csv")

    voigt, hill, reuss = bounds
    summary = {
        "config": str(cfg.source),
        "map": str(cfg.map.path),
        "modulus_column": cols["modulus"],
        "indents_kept": int(len(labelled)),
        "cuts": cfg.cuts,
        "bounds": {"Voigt": voigt, "Hill": hill, "Reuss": reuss},
        "voigt_combined_sd": spread["combined_sd"],
        "voigt_interval": spread["interval"],
        "interval_level_pct": cfg.uncertainty["level"],
        "resampling": {"n_draws": cfg.uncertainty["n_draws"],
                       "seed": cfg.uncertainty["seed"]},
        "components_sd": spread["components_sd"],
        "interval_method": (
            "Percentile interval of the bootstrap distribution of each phase "
            "median, resampling indents within the phase with replacement."),
        "excluded_from_the_budget": (
            "Where the cuts go and which value a phase takes are choices, not "
            "random quantities. They are reported by the sweep and the "
            "sensitivity section instead of being folded in here."),
    }
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf8")
    written.append(path)

    if sweep_table is not None:
        csv(sweep_table, "cut_sweep.csv")
    if reduced is not None:
        scale = cfg.macro.curves.modulus_scale
        shown = reduced.copy()
        for column in [c for c in shown.columns if c.startswith("chord")]:
            shown[column] *= scale
        csv(shown, "macro_specimens.csv")
    if comparison is not None:
        csv(comparison, "comparison.csv")

    written.append(plots.save(plots.modulus_histogram(labelled[cols["modulus"]],
                                                      cfg),
                              "modulus_histogram", out))
    if sweep_table is not None:
        s = mixtures.sweep_summary(sweep_table, cfg.cuts)
        written.append(plots.save(
            plots.sweep_plot(sweep_table, list(cfg.sweep),
                             s["reference_GPa"] if s else None),
            "cut_sweep", out))
    measured = macro.measurements(
        cfg.macro, reduced,
        cfg.macro.curves.modulus_scale if reduced is not None else 1.0)
    if measured:
        written.append(plots.save(
            plots.prediction_vs_measurement(bounds, measured),
            "prediction", out))
    return written


def cmd_uncertainty(args):
    cfg, frame, cols = _load(args)
    labelled = _labelled(cfg, frame, cols)
    table, (voigt, _, _) = _table_and_bounds(cfg, labelled, cols)
    settings = cfg.uncertainty

    replicates = uncertainty.bootstrap_medians(
        labelled, cfg.phase_names, cols["modulus"],
        settings["n_draws"], settings["seed"])
    result = uncertainty.budget(table, replicates, cfg.phase_names,
                                seed=settings["seed"],
                                n_draws=settings["n_draws"],
                                level=settings["level"])

    print(f"{settings['n_draws']} replicates, seed {settings['seed']}")
    print("\nper phase, median and its resampled interval")
    for name, (lo, hi) in uncertainty.intervals(replicates,
                                                settings["level"]).items():
        point = float(table.loc[table["class"] == name, "median"].iloc[0])
        print(f"  {name:<16} {point:8.3f}   [{lo:.3f}, {hi:.3f}]")

    print(f"\nVoigt bound               {result['point']:.3f}")
    print(f"phase medians alone       +/- {result['components_sd']['class_medians']:.3f}")
    print(f"phase fractions alone     +/- {result['components_sd']['class_fractions']:.3f}")
    print(f"combined                  +/- {result['combined_sd']:.3f}")
    lo, hi = result["interval"]
    print(f"{settings['level']:g} % interval             {lo:.2f} to {hi:.2f}")
    print("\nThe cuts and the choice of phase input are outside this budget. "
          "They are\nchoices rather than random quantities: see 'sweep' and "
          "the sensitivity section.")
    return 0


def cmd_cluster(args):
    cfg, frame, cols = _load(args)
    if not cfg.clustering:
        raise ConfigError(
            "this config has no clustering section. Add one naming the feature "
            "columns and k, for example  clustering: {features: [E_100, "
            "E_200], k: 3}")
    labelled = _labelled(cfg, frame, cols)
    c = cfg.clustering

    print("representation                       n      ARI    ARI*   silhouette")
    for label, kw in (("k-means", dict()),
                      ("k-means on principal components", dict(use_pca=True)),
                      ("Gaussian mixture", dict(model="gmm"))):
        r = cluster.run(labelled, c.features, cols["modulus"], k=c.k,
                        seed=c.seed, n_init=c.n_init,
                        merge_lowest=c.merge_lowest, **kw)
        print(f"{label:<34}{r['n']:>5}{r['ARI']:>9.3f}{r['ARI_star']:>8.3f}"
              f"{r['silhouette']:>13.3f}")

    mean, sd = cluster.bootstrap_stability(labelled, c.features, k=c.k,
                                           seed=c.seed, n_init=c.n_init)
    print(f"\nstability over 20 subsamples at k = {c.k}: ARI {mean:.3f} "
          f"+/- {sd:.3f}")
    print("\nARI is chance-corrected: the value expected for unrelated "
          "partitions is 0.\nThis does not choose how many phases there are. "
          "It says whether a partition\nbuilt with no knowledge of the "
          "architecture lands somewhere compatible with\nthe rule that used it.")
    return 0


def cmd_sweep(args):
    cfg, frame, cols = _load(args)
    labelled = _labelled(cfg, frame, cols)
    if not cfg.sweep:
        raise ConfigError("this config has no sweep section. Add one naming "
                          "the cuts to move and the values to try, for "
                          "example  sweep: {lower: [10, 15, 20]}")
    table = _sweep(cfg, labelled, cols)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    s = mixtures.sweep_summary(table, cfg.cuts)
    if s:
        print(f"\nas configured {s['reference_GPa']:.2f}, "
              f"range {s['min_GPa']:.2f} to {s['max_GPa']:.2f}, "
              f"largest departure {s['max_deviation_GPa']:.2f}")
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"written to {args.out}")
    return 0


def cmd_histogram(args):
    from . import plots      # imported here so the other commands need no backend

    cfg, frame, cols = _load(args)
    labelled = _labelled(cfg, frame, cols)
    fig = plots.modulus_histogram(labelled[cols["modulus"]], cfg, log=args.log)
    path = plots.save(fig, "modulus_histogram", args.out)
    print(f"written to {path}")

    if cfg.sweep:
        sw = _sweep(cfg, labelled, cols)
        s = mixtures.sweep_summary(sw, cfg.cuts)
        fig = plots.sweep_plot(sw, list(cfg.sweep),
                               s["reference_GPa"] if s else None)
        print(f"written to {plots.save(fig, 'cut_sweep', args.out)}")

    measured = macro.measurements(cfg.macro)
    if measured:
        table, bounds = _table_and_bounds(cfg, labelled, cols)
        fig = plots.prediction_vs_measurement(bounds, measured)
        print(f"written to {plots.save(fig, 'prediction', args.out)}")
    return 0


def _contrast(reduced, scale):
    """Print the two-group comparison, when there are exactly two groups."""
    c = macro.contrast(reduced, scale)
    if c is None:
        return
    print(f"{c['groups'][0]} against {c['groups'][1]}: "
          f"Welch t {c['welch_t']:.3f}, p {c['p']:.4f}, "
          f"Hedges g {c['hedges_g']:.3f}")


def cmd_macro(args):
    cfg, _, _ = _load(args)
    reduced = _reduce_curves(cfg)
    if reduced is None:
        raise ConfigError("this config has no macro.from_curves section")
    spec = cfg.macro.curves
    scale = spec.modulus_scale

    # Scaled on the way out, not inside the fit. The fit works in whatever
    # unit the file is in; the scale is the one place that records which.
    shown = reduced.copy()
    for column in [c for c in shown.columns if c.startswith("chord")]:
        shown[column] *= scale

    print(f"{'force and displacement' if spec.is_raw else 'stress and strain'} "
          f"from {spec.path.name}, moduli scaled by {scale:g}")
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nby group")
    print(macro.summarise(reduced, scale)
          .to_string(float_format=lambda v: f"{v:.4f}"))
    _contrast(reduced, scale)

    if args.out:
        shown.to_csv(args.out, index=False, lineterminator="\n")
        print(f"written to {args.out}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="hybridmap", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", required=True, help="path to the config file")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override one config value, e.g. cuts.lower=12")
        return sp

    sp = sub.add_parser("init", help="write a starter config for a CSV")
    sp.add_argument("--map", required=True, help="the map CSV to describe")
    sp.add_argument("--out", help="write the config here instead of printing it")
    sp.add_argument("--phases", default=3, type=int,
                    help="how many phases to lay out, default 3")
    sp.set_defaults(fn=cmd_init)

    common(sub.add_parser("validate", help="check a config against its data")
           ).set_defaults(fn=cmd_validate)

    sp = common(sub.add_parser("classify", help="label every indent"))
    sp.add_argument("--out", help="write the labelled table here")
    sp.set_defaults(fn=cmd_classify)

    sp = common(sub.add_parser("report", help="the whole chain, printed"))
    sp.add_argument("--out", help="directory for the tables and figures")
    sp.set_defaults(fn=cmd_report)

    common(sub.add_parser("uncertainty",
                          help="resampled spread on the bound")
           ).set_defaults(fn=cmd_uncertainty)

    common(sub.add_parser("cluster",
                          help="unsupervised cross-check of the rule")
           ).set_defaults(fn=cmd_cluster)

    sp = common(sub.add_parser("sweep", help="bounds across a grid of cuts"))
    sp.add_argument("--out", help="write the swept table here")
    sp.set_defaults(fn=cmd_sweep)

    sp = common(sub.add_parser("histogram",
                               help="the modulus histogram with the cuts on it"))
    sp.add_argument("--out", default=".", help="directory for the figures")
    sp.add_argument("--log", action="store_true", help="logarithmic counts")
    sp.set_defaults(fn=cmd_histogram)


    sp = common(sub.add_parser("macro", help="reduce bending curves to moduli"))
    sp.add_argument("--out", help="write the per-specimen table here")
    sp.set_defaults(fn=cmd_macro)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except ConfigError as exc:
        print(f"\nconfiguration problem: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
