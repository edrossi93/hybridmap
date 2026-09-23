"""Every deposited file is either a measurement or a product of one. Nothing else.

This exists because of a specific failure. `classified_depth_resolved.csv` was
written once by hand-run code that was never kept, and then read by everything
downstream. Nobody could regenerate it, so three defects lived inside it
undetected across two rounds of review: a quality mask matching no stated rule,
slice values that had drifted away from the file they were supposed to equal,
and a cluster column that had already needed one repair. A reviewer found the
edge of it from the outside, which is the worst way to find it.

The rule this gate enforces is simple. Each file under `data/` is declared as
exactly one of:

  PRIMARY    a measurement. It cannot be regenerated from anything else in the
             deposit, and it says where it came from.
  DERIVED    a product. It names the script that writes it, and re-running that
             script has to leave the file byte-identical.

A file that is in neither list fails the gate, so a new file cannot be added
quietly. A derived file that changes when its own script is re-run fails too,
which catches a stale product before a reader does.

Run: python provenance_check.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

PRIMARY = {
    "shallow_map.csv":
        "load-controlled acquisition, one row per indent, exported from the "
        "instrument record",
    "depth_resolved_indents.csv":
        "depth-resolved acquisition, one row per indent: position, the three "
        "slice values and the strain-rate index, exported from the instrument "
        "record",
    "depth_resolved_curves.csv":
        "depth-resolved acquisition, the full traces on a common 50 to 300 nm "
        "grid",
    "flexural_curves.csv":
        "three-point bending records",
    "bse_segmentation_tiles.csv":
        "per-tile zone and phase pixel counts from the segmented mosaic",
    "bse_segmentation_summary.json":
        "pooled constituent volume fractions from the same segmentation",
    "zone_mask_rle.json":
        "operator tow marking of the mosaic, run-length encoded",
    "bse_validation_metrics.json":
        "leave-one-field-out validation of the segmentation against the "
        "operator labels: confusion matrix, precision and recall per class. "
        "Computed outside the deposit because it needs the tile images, which "
        "are several gigabytes and stay with the corresponding author",
    "DATA_DICTIONARY.md": "documentation",
    "LICENSE": "licence",
}

DERIVED = {
    "classified_shallow.csv": "classify.py",
    "classified_depth_resolved.csv": "make_classified_depth.py",
    "flexural_moduli.csv": "make_flexural_moduli.py",
    "uncertainty_budget.json": "uncertainty_budget.py",
    "phase_angle_sensitivity.json": "phase_angle_sensitivity.py",
    "spacing_compliance.json": "spacing_compliance.py",
    "mixed_class_deconvolution.json": "mixed_class_deconvolution.py",
    "spatial_localisation.json": "figs_supplementary.py",
    "zone_share_sensitivity.json": "zone_share.py",
    "poisson_sensitivity.json": "poisson_sensitivity.py",
    "flexural_comparisons.json": "flexural_comparisons.py",
    "basalt_input_sweep.json": "basalt_input_sweep.py",
}


MANIFEST = DATA / "PROVENANCE.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record():
    """Write the manifest from the files as they stand, running nothing.

    For when the generators have just been run by hand, one at a time. The
    slow path is `--rebuild`; this is the same bookkeeping without the wait.
    """
    manifest = {"derived": {n: digest(DATA / n) for n in DERIVED
                            if (DATA / n).exists()},
                "primary": {n: digest(DATA / n) for n in PRIMARY
                            if (DATA / n).exists()}}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")
    print(f"  recorded {len(manifest['derived'])} products and "
          f"{len(manifest['primary'])} measurements")
    return 0


def rebuild():
    """Run every generator, then record what each produced.

    Separate from the default check because it is slow: the budget alone
    resamples tens of thousands of times. Run it whenever an input or a script
    changes; the fast check then holds the result to account.
    """
    problems = {}
    for name, script in sorted(DERIVED.items()):
        run = subprocess.run([sys.executable, str(HERE / script)],
                             capture_output=True, text=True, cwd=HERE)
        if run.returncode != 0:
            tail = (run.stderr.strip().splitlines() or ["no message"])[-1]
            problems[name] = f"{script} exits {run.returncode}: {tail}"
            print(f"  FAILED  {script}: {tail}")
        else:
            print(f"  rebuilt {name}")
    manifest = {"derived": {n: digest(DATA / n) for n in DERIVED
                            if (DATA / n).exists() and n not in problems},
                "primary": {n: digest(DATA / n) for n in PRIMARY
                            if (DATA / n).exists()}}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")
    print(f"\n  manifest written for {len(manifest['derived'])} products and "
          f"{len(manifest['primary'])} measurements")
    return 1 if problems else 0


def main():
    if "--rebuild" in sys.argv:
        return rebuild()
    if "--record" in sys.argv:
        return record()

    problems = []
    present = {p.name for p in DATA.iterdir() if p.is_file()}
    declared = set(PRIMARY) | set(DERIVED) | {MANIFEST.name}

    for name in sorted(present - declared):
        problems.append(f"{name} is in data/ and is declared neither a "
                        f"measurement nor a product")
    for name in sorted(declared - present - {MANIFEST.name}):
        problems.append(f"{name} is declared but is not in data/")

    if not MANIFEST.exists():
        problems.append("no manifest: run with --rebuild once")
        print("\n".join("  " + p for p in problems))
        return 1

    manifest = json.loads(MANIFEST.read_text())
    for kind in ("derived", "primary"):
        for name, recorded in manifest[kind].items():
            path = DATA / name
            if not path.exists():
                problems.append(f"{name} was recorded and is now missing")
            elif digest(path) != recorded:
                who = (f"{DERIVED[name]} is the only thing that may write it"
                       if kind == "derived" else "it is a measurement record")
                problems.append(f"{name} has changed since it was recorded; "
                                f"{who}")

    n = len(manifest["derived"]) + len(manifest["primary"])
    print(f"  {n} files, {len(manifest['primary'])} measurements and "
          f"{len(manifest['derived'])} products")
    if problems:
        print("\n".join("  " + p for p in problems))
        return 1
    print("  every file in data/ is a declared measurement or a recorded "
          "product, unchanged since it was built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
