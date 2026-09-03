"""
Command line for the tool. Run from the scripts/ directory:

    python -m hybridmap validate  --config examples/paper_shallow.yaml
    python -m hybridmap histogram --config examples/paper_shallow.yaml --out figs
    python -m hybridmap report    --config examples/paper_shallow.yaml
    python -m hybridmap sweep     --config examples/paper_shallow.yaml
    python -m hybridmap classify  --config examples/paper_shallow.yaml --out labelled.csv

Every command takes --set to override one value without editing the file, for
example --set cuts.lower=12. Overrides are applied before validation, so a
typo in one is caught the same way a typo in the file is.

A configuration error exits 2 and prints what to fix. Nothing here writes to
the input file or to data/.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from . import aggregate, classify, io, macro, mixtures
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

    print(f"\nVoigt {voigt:.2f}   Hill {hill:.2f}   Reuss {reuss:.2f}")
    print("Fractions are indent-class fractions, not volume fractions: they "
          "count\nindents carrying each label over indents carrying any label.")

    if cfg.zones and len(cfg.zones.names) > 1:
        glob, zoned = aggregate.check_zone_weighting(labelled, cfg.phase_names)
        agree = "identical" if np.allclose(glob, zoned) else "NOT identical"
        print(f"\nzone-weighted fractions are {agree} to the global counts")

    if cfg.sweep:
        sw = _sweep(cfg, labelled, cols)
        s = mixtures.sweep_summary(sw, cfg.cuts)
        if s:
            print(f"\ncut sensitivity over {len(sw)} combinations: "
                  f"{s['min_GPa']:.2f} to {s['max_GPa']:.2f}, "
                  f"largest departure from the configured cuts "
                  f"{s['max_deviation_GPa']:.2f}")

    if cfg.macro:
        measured = macro.measurements(cfg.macro)
        if cfg.macro.curves and not measured:
            print("\nmacro.from_curves is configured; run "
                  "'hybridmap macro' to reduce the curves first")
        if measured:
            print("\nagainst the macroscopic measurement")
            print(macro.compare((voigt, hill, reuss), measured)
                  .to_string(index=False,
                             float_format=lambda v: f"{v:.2f}"))
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
    path = plots.save(fig, f"{cfg.source.stem}_modulus_histogram", args.out)
    print(f"written to {path}")

    if cfg.sweep:
        sw = _sweep(cfg, labelled, cols)
        s = mixtures.sweep_summary(sw, cfg.cuts)
        fig = plots.sweep_plot(sw, list(cfg.sweep),
                               s["reference_GPa"] if s else None)
        print(f"written to {plots.save(fig, f'{cfg.source.stem}_cut_sweep', args.out)}")

    measured = macro.measurements(cfg.macro)
    if measured:
        table, bounds = _table_and_bounds(cfg, labelled, cols)
        fig = plots.prediction_vs_measurement(bounds, measured)
        print(f"written to {plots.save(fig, f'{cfg.source.stem}_prediction', args.out)}")
    return 0


def cmd_macro(args):
    cfg, _, _ = _load(args)
    if not cfg.macro or not cfg.macro.curves:
        raise ConfigError("this config has no macro.from_curves section")
    spec = cfg.macro.curves
    curves = pd.read_csv(spec.path)
    table = macro.moduli_from_curves(curves, spec)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nby group, in the units of the file")
    print(macro.summarise(table).to_string(float_format=lambda v: f"{v:.4f}"))
    if args.out:
        table.to_csv(args.out, index=False)
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

    common(sub.add_parser("validate", help="check a config against its data")
           ).set_defaults(fn=cmd_validate)

    sp = common(sub.add_parser("classify", help="label every indent"))
    sp.add_argument("--out", help="write the labelled table here")
    sp.set_defaults(fn=cmd_classify)

    common(sub.add_parser("report", help="the whole chain, printed")
           ).set_defaults(fn=cmd_report)

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
