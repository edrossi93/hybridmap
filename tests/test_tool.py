"""
Check that the reusable tool is reusable, and that it still is the paper.

    python tests/test_tool.py

Three things are tested, and the third is the one that matters most.

A synthetic map with different column names, three phases and one zone runs end
to end through the command line. The populations in it are planted, so the test
can ask whether the tool recovers what was put in rather than merely whether it
finishes without raising.

The paper's own config reproduces the published bounds. A worked example that
drifts from the analysis it claims to reproduce is worse than no example.

The paper's config and the paper's scripts agree with each other. classify.py
keeps the thresholds in Python and examples/paper_shallow.yaml keeps them in
YAML, which is two copies of the same decisions. Two copies drift unless
something compares them, so this compares them.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPTS))

failures = []
checks = 0


def check(label, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        failures.append(f"{label}{'  ' + detail if detail else ''}")
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


def run_cli(*args):
    """Invoke the tool the way a user does, from scripts/."""
    proc = subprocess.run([sys.executable, "-m", "hybridmap", *args],
                          cwd=SCRIPTS, capture_output=True, text=True,
                          timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def synthetic_dataset(generated):
    """
    The whole chain on a map that is not ours.

    Written to a temporary directory rather than into the deposit. A synthetic
    CSV sitting in a folder that later gets zipped and uploaded is exactly the
    file that must never be mistaken for a measurement, and the safest place
    for it is one that cannot outlive the test.
    """
    print("\nsynthetic map: a second dataset, generated here, not deposited")
    proc = subprocess.run([sys.executable, str(HERE / "make_synthetic_map.py"),
                           "--out", str(generated)],
                          capture_output=True, text=True, timeout=300)
    check("the generator runs", proc.returncode == 0,
          proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
    if proc.returncode:
        return

    config = str((HERE / "synthetic.yaml").resolve())
    # The config points where a hand-run of the generator writes; the test
    # writes somewhere temporary, so the path is overridden. That exercises
    # --set as a side effect, which is worth having covered.
    where = f"map.path={(generated / 'synthetic_map.csv').as_posix()}"

    for command in ("validate", "report", "sweep"):
        code, out = run_cli(command, "--config", config, "--set", where)
        check(f"'{command}' runs on the synthetic map", code == 0,
              out.strip().splitlines()[-1] if code else "")

    labelled = generated / "synthetic_labelled.csv"
    code, out = run_cli("classify", "--config", config, "--set", where,
                        "--out", str(labelled))
    check("'classify' writes a labelled table", code == 0 and labelled.exists())
    if not labelled.exists():
        return

    # The populations were planted, so the classifier can be scored rather than
    # eyeballed. Anything below near-perfect here means the cuts and the
    # populations have drifted apart, not that the classification is hard.
    got = pd.read_csv(labelled)
    planted = {"polymer": "polymer", "boundary": "boundary",
               "ceramic": "ceramic", "invalid": "QC/invalid"}
    agree = (got.constituent == got.true_phase.map(planted)).mean()
    check("the planted phases are recovered", agree > 0.98,
          f"{100 * agree:.1f} % of points carry the phase they were drawn from")

    counts = got.constituent.value_counts()
    check("all three phases are populated",
          all(counts.get(p, 0) > 0 for p in ("polymer", "boundary", "ceramic")),
          ", ".join(f"{p} {counts.get(p, 0)}" for p in
                    ("polymer", "boundary", "ceramic")))
    check("the points with no modulus are rejected, not labelled",
          counts.get("QC/invalid", 0) == 20, f"{counts.get('QC/invalid', 0)} of 20")


def paper_config():
    """The worked example must still be the paper."""
    print("\nthe worked example against the published values")
    from hybridmap import aggregate, classify, config, io, mixtures

    cfg = config.load(SCRIPTS / "examples" / "paper_shallow.yaml")
    frame = io.read_map(cfg.map)
    cols = io.resolve(frame, cfg.map)
    labelled = classify.apply(frame, cfg, cols)
    table = aggregate.class_table(labelled, cfg.phase_names, cols["modulus"])
    voigt, hill, reuss = mixtures.bounds(table.fraction, table["median"])

    check("indents after the resin cut", len(labelled) == 5553, str(len(labelled)))
    for name, published in (("flax", 2300), ("matrix", 894),
                            ("mixed-volume", 1977), ("basalt", 380)):
        got = int(table.loc[table["class"] == name, "n"].iloc[0])
        check(f"n({name})", got == published, f"{got}, paper {published}")
    for name, published in (("flax", 4.70), ("matrix", 4.66),
                            ("mixed-volume", 35.79), ("basalt", 85.99)):
        got = float(table.loc[table["class"] == name, "median"].iloc[0])
        check(f"median E({name})", abs(got - published) <= 0.02,
              f"{got:.2f}, paper {published}")
    for label, got, published in (("Voigt", voigt, 21.33), ("Hill", hill, 14.41),
                                  ("Reuss", reuss, 7.50)):
        check(f"{label} bound", abs(got - published) <= 0.02,
              f"{got:.2f}, paper {published}")

    sweep = mixtures.sweep(labelled, cfg.phases, cfg.cuts, cfg.sweep,
                           cols["modulus"], cfg.phase_names)
    summary = mixtures.sweep_summary(sweep, cfg.cuts)
    check("largest departure across the swept cuts",
          abs(summary["max_deviation_GPa"] - 0.50) <= 0.02,
          f"{summary['max_deviation_GPa']:.2f}, paper 0.50")


def config_matches_the_scripts():
    """
    The same decisions are written down twice. Compare the copies.

    classify.py holds the paper's thresholds in Python because the reproduction
    path must not need a YAML parser; examples/paper_shallow.yaml holds them
    because a user has to be able to read them. Neither copy is redundant, so
    the drift has to be caught rather than designed away.
    """
    print("\nthe paper's config against the paper's scripts")
    import classify as paper
    from hybridmap import config

    cfg = config.load(SCRIPTS / "examples" / "paper_shallow.yaml")
    edges = {b.name: b.below for b in cfg.zones.bands}

    check("resin cut", edges["resin"] == paper.X_RESIN,
          f"{edges['resin']} against X_RESIN {paper.X_RESIN}")
    check("zone split", edges["basalt"] == paper.X_ZONE_SPLIT,
          f"{edges['basalt']} against X_ZONE_SPLIT {paper.X_ZONE_SPLIT}")
    check("compliant/stiff cut", cfg.cuts["lower"] == paper.E_LOWER,
          f"{cfg.cuts['lower']} against E_LOWER {paper.E_LOWER}")
    check("pure-fibre cut", cfg.cuts["upper"] == paper.E_UPPER_SHALLOW,
          f"{cfg.cuts['upper']} against E_UPPER_SHALLOW {paper.E_UPPER_SHALLOW}")
    check("phase names and order",
          cfg.phase_names == [p.name for p in paper.PHASES],
          ", ".join(cfg.phase_names))

    depth = config.load(SCRIPTS / "examples" / "paper_depth_resolved.yaml")
    check("depth-resolved pure-fibre cut",
          depth.cuts["upper"] == paper.E_UPPER_DEPTH,
          f"{depth.cuts['upper']} against E_UPPER_DEPTH {paper.E_UPPER_DEPTH}")
    check("phase-angle limit", depth.qc.limit == paper.PA_QC,
          f"{depth.qc.limit} against PA_QC {paper.PA_QC}")

    import clustering_check as cluster_paper
    check("clustering features",
          list(depth.clustering.features) == cluster_paper.FEATURES,
          f"{len(depth.clustering.features)} features")
    check("clustering seed", depth.clustering.seed == cluster_paper.SEED,
          f"{depth.clustering.seed}")

    import make_flexural_moduli as flex
    curves = cfg.macro.curves
    check("chord interval",
          curves.chord == (flex.CHORD_LO, flex.CHORD_HI), str(curves.chord))
    check("toe window", curves.toe_window == flex.TOE_WINDOW,
          str(curves.toe_window))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="hybridmap_test_") as tmp:
        synthetic_dataset(Path(tmp))
    paper_config()
    config_matches_the_scripts()

    print("\n" + "-" * 72)
    if failures:
        print(f"{len(failures)} of {checks} checks failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"{checks} of {checks} checks passed")
