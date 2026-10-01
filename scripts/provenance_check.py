"""Every deposited file is a measurement, a product of one, or documentation.

It guards against a specific failure. A file written once by code that is not
kept, and then read by everything downstream, cannot be regenerated, so defects
can live inside it undetected: a quality mask matching no stated rule, slice
values that drift away from the file they are supposed to equal, a column that
needs repair. The only way to find them is then from the outside.

The rule this gate enforces is simple. Each file under `data/` is declared as
exactly one of:

  PRIMARY    a measurement. It cannot be regenerated from anything else in the
             deposit, and it says where it came from.
  DERIVED    a product. It names the script that writes it, and re-running that
             script has to leave the file byte-identical.
  DOCUMENTATION
             the data dictionary and the licence. Held to their hashes like a
             measurement, and counted apart from the measurements.

A file that is in none of the three lists fails the gate, so a new file cannot be added
quietly. A derived file that changes when its own script is re-run fails too,
which catches a stale product before a reader does.

For that last sentence to hold, the manifest must not follow the files. A
`rebuild` that ran every generator, looked only at the exit status and then
wrote the manifest from the files as they stood afterwards would re-record a
drifted product under its new hash, return 0, and let the fast check pass
against that new hash for ever, so the byte-identity promised above would be
tested nowhere. The same holds for `record`, which does not run the scripts.

Both are therefore held to one rule: a hash is written to the manifest only when
the script reproduced the file byte for byte and that is the hash the manifest
already carried, or when someone passed `--accept-changes` to record bytes that
were not proved. So a plain run of this tool cannot change what any file is
held to. It either confirms the manifest or fails.

Run: python provenance_check.py            the fast check, against the manifest
     python provenance_check.py --rebuild  re-run every generator and compare
     --accept-changes, with --rebuild or --record, records unproved bytes
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
        "depth-resolved acquisition, one row per indent: position, the slice "
        "values at 100, 150 and 200 nm as the 20 nm window means of "
        "Supplementary Note S1, the strain-rate index and the instrument's own "
        "quality flag. Computed from the "
        "instrument record at export; not regenerable exactly from the "
        "resampled curves, see the README under Slice extraction",
    "depth_resolved_curves.csv":
        "depth-resolved acquisition, the full traces on a common 50 to 300 nm "
        "grid",
    "flexural_curves.csv":
        "three-point bending records, five basalt-longitudinal specimens and "
        "four flax-longitudinal. One flax-longitudinal curve appeared twice in "
        "the source workbook under two names and the duplicate was removed "
        "before deposit; the data dictionary says which",
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
}

DOCUMENTATION = {
    "DATA_DICTIONARY.md": "the meaning, unit and range of every column and key",
    "LICENSE": "the licence of the data",
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
    "depth_softening.json": "depth_softening.py",
    "contact_geometry.json": "contact_geometry.py",
    "clustering_diagnostics.json": "figs_supplementary.py",
}

# The three kinds, in manifest order.
KINDS = (("derived", DERIVED), ("primary", PRIMARY),
         ("documentation", DOCUMENTATION))
NOUN = {"derived": "product", "primary": "measurement",
        "documentation": "document"}


MANIFEST = DATA / "PROVENANCE.json"

# A hash in the manifest asserts that these are the bytes the deposit means to
# publish. For a product the assertion is earned by re-running its script and
# finding the file unchanged. Where that evidence is missing, because the script
# was not run or because the file is a measurement with no script at all, this
# flag is how a person states that they are recording the bytes anyway. It is
# spelled out on the command line so that accepting drift is a decision someone
# takes and can be found afterwards in a shell history, rather than a silent
# side effect of running the tool.
ACCEPT = "--accept-changes"
FLAGS = {"--rebuild", "--record", ACCEPT}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def held():
    """The hashes the manifest carries at this moment, empty if there is none."""
    if not MANIFEST.exists():
        return {kind: {} for kind, _ in KINDS}
    manifest = json.loads(MANIFEST.read_text())
    return {kind: dict(manifest.get(kind, {})) for kind, _ in KINDS}


def write_manifest(hashes):
    """Serialise the given hashes, in declaration order.

    Declaration order rather than sorted order, because that is the order the
    published file has and a reordering would read as a change to every file in
    it for anyone diffing this deposit against the copy they already have.
    """
    manifest = {kind: {n: hashes[kind][n] for n in names if n in hashes[kind]}
                for kind, names in KINDS}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")
    return manifest


def record(accept):
    """Write the manifest from the files as they stand, running nothing.

    For when the generators have just been run by hand, one at a time. The
    slow path is `--rebuild`; this is the same bookkeeping without the wait.

    Running nothing means proving nothing. This path cannot tell a product that
    still reproduces from one that has drifted, so on its own it launders
    exactly what the gate exists to catch, and it is the shorter and more
    tempting command of the two. It therefore refuses unless the acceptance flag
    is given: the person passing it is the only evidence there is.
    """
    if not accept:
        print("  --record writes the manifest from whatever is on disk without "
              "running a single")
        print("  generator, so it cannot tell a product that still reproduces "
              "from one that has")
        print(f"  drifted. Use --rebuild to earn the hashes, or "
              f"--record {ACCEPT} to record")
        print("  them unproved on your own authority.")
        return 1

    hashes = {kind: {n: digest(DATA / n) for n in names if (DATA / n).exists()}
              for kind, names in KINDS}
    manifest = write_manifest(hashes)
    print(f"  recorded {len(manifest['derived'])} products, "
          f"{len(manifest['primary'])} measurements and "
          f"{len(manifest['documentation'])} documents")
    print(f"  no script was run: every hash above is unproved and recorded "
          f"because {ACCEPT} was given")
    return 0


def rebuild(accept):
    """Re-run every generator and hold its product to the bytes it had before.

    Separate from the default check because it is slow: the budget alone
    resamples tens of thousands of times. Run it whenever an input or a script
    changes; the fast check then holds the result to account.

    The before-and-after comparison here is the only place the deposit's central
    promise is tested. See the module docstring for the failure it guards
    against.

    One rule governs what reaches the manifest. A hash is written without the
    acceptance flag only when the script reproduced the file byte for byte and
    that is the hash the manifest already carried. Everything else is unproved:
    a product whose bytes moved when its script ran, a product that reproduces
    but does not match the record, a file the manifest has never held, a
    measurement whose bytes moved. An unproved file keeps whatever hash it
    already had, and minting a new one for it takes the flag.

    The consequence worth knowing is that a plain run cannot alter what any file
    is held to. It either confirms the manifest or fails. That is what stops the
    tool laundering by repetition: the second identical run of a red rebuild is
    just as red as the first, because the drifted bytes on disk are still not
    the bytes the record names.
    """
    hashes = {kind: {} for kind, _ in KINDS}
    problems = {}
    unproved = {}
    # Read before anything is written. With no manifest at all there is no
    # earlier claim about any file, so the first run on a deposit has to be
    # allowed to establish one; after that the record is the reference.
    bootstrap = not MANIFEST.exists()
    was = held()

    def keep(kind, name, why, label):
        """Refuse to mint a hash, and leave any earlier one standing.

        Leaving it standing rather than dropping it matters: the earlier hash is
        the evidence, a failing run should not destroy it, and the fast check
        then reports what the file was held to instead of only that it is no
        longer held to anything.
        """
        unproved[name] = why
        if name in was[kind]:
            hashes[kind][name] = was[kind][name]
        print(f"  {label}")

    for name, script in sorted(DERIVED.items()):
        path = DATA / name
        before = digest(path) if path.exists() else None
        run = subprocess.run([sys.executable, str(HERE / script)],
                             capture_output=True, text=True, cwd=HERE)
        if run.returncode != 0:
            tail = (run.stderr.strip().splitlines() or ["no message"])[-1]
            problems[name] = f"{script} exits {run.returncode}: {tail}"
            if name in was["derived"]:
                hashes["derived"][name] = was["derived"][name]
            print(f"  FAILED  {script}: {tail}")
            continue
        if not path.exists():
            problems[name] = f"{script} exits 0 and writes no {name}"
            print(f"  FAILED  {script}: exits 0 and writes no {name}")
            continue

        after = digest(path)
        recorded = was["derived"].get(name)
        if before is None:
            keep("derived", name,
                 f"{name} was not on disk beforehand, so there were no bytes "
                 f"for {script} to reproduce",
                 f"NEW     {name}: nothing to compare it against")
        elif after != before:
            keep("derived", name,
                 f"re-running {script} changed {name}: {before[:12]} became "
                 f"{after[:12]}",
                 f"CHANGED {name}: {before[:12]} became {after[:12]}")
        elif bootstrap or recorded == after:
            hashes["derived"][name] = after
            print(f"  same    {name}")
        elif recorded is None:
            keep("derived", name,
                 f"{script} reproduces {name}, but the manifest has never held "
                 f"a hash for it",
                 f"UNHELD  {name}: reproduces, but was never recorded")
        else:
            keep("derived", name,
                 f"{script} reproduces {name}, but at {after[:12]}, and the "
                 f"manifest names {recorded[:12]}",
                 f"MOVED   {name}: reproduces at {after[:12]}, recorded as "
                 f"{recorded[:12]}")

    # A measurement has no script, so re-running is not available as evidence
    # and the only thing to hold it to is the hash the manifest already carries.
    # Rewriting that hash unasked would launder a changed measurement, and a
    # measurement is the one kind of file in here that nothing can reconstruct.
    for kind, names in KINDS[1:]:
        for name in names:
            path = DATA / name
            if not path.exists():
                continue
            now = digest(path)
            recorded = was[kind].get(name)
            if bootstrap or recorded == now:
                hashes[kind][name] = now
            elif recorded is None:
                keep(kind, name,
                     f"{name} is a {NOUN[kind]} and the manifest has never held "
                     f"a hash for it, so there is nothing to check its bytes "
                     f"against",
                     f"UNHELD  {name}: no recorded hash to check it against")
            else:
                keep(kind, name,
                     f"{name} is a {NOUN[kind]} and its bytes do not match the "
                     f"manifest: {recorded[:12]} is now {now[:12]}",
                     f"CHANGED {name}: {recorded[:12]} is now {now[:12]}")

    if accept:
        for name in unproved:
            kind = next(k for k, names in KINDS if name in names)
            hashes[kind][name] = digest(DATA / name)

    manifest = write_manifest(hashes)
    print(f"\n  manifest written for {len(manifest['derived'])} products, "
          f"{len(manifest['primary'])} measurements and "
          f"{len(manifest['documentation'])} documents")
    landed = (f"recorded because {ACCEPT} was given" if accept
              else "not recorded")
    for name in sorted(unproved):
        print(f"  {unproved[name]}; {landed}")
    for name in sorted(problems):
        print(f"  {problems[name]}; its product is not re-recorded")
    if unproved and not accept:
        print(f"  re-run with {ACCEPT} to record the bytes above as the new "
              f"baseline; that is a decision, not a default")
    return 1 if problems or (unproved and not accept) else 0


def main():
    # An unrecognised flag is a usage error and not a pass. A mistyped
    # --accept-change would otherwise be dropped in silence and the person who
    # typed it would believe they had accepted something.
    unknown = sorted(a for a in sys.argv[1:] if a not in FLAGS)
    if unknown:
        print(f"  unrecognised argument(s): {', '.join(unknown)}")
        print(f"  usage: provenance_check.py [--rebuild | --record] [{ACCEPT}]")
        return 2

    accept = ACCEPT in sys.argv
    if "--rebuild" in sys.argv:
        return rebuild(accept)
    if "--record" in sys.argv:
        return record(accept)
    if accept:
        print(f"  {ACCEPT} does nothing on its own: the default check reads the "
              f"manifest and never writes it")
        return 2

    problems = []
    present = {p.name for p in DATA.iterdir() if p.is_file()}
    declared = set(PRIMARY) | set(DERIVED) | set(DOCUMENTATION) | {MANIFEST.name}

    for name in sorted(present - declared):
        problems.append(f"{name} is in data/ and is declared neither a "
                        f"measurement, a product nor documentation")
    for name in sorted(declared - present - {MANIFEST.name}):
        problems.append(f"{name} is declared but is not in data/")

    if not MANIFEST.exists():
        problems.append("no manifest: run with --rebuild once")
        print("\n".join("  " + p for p in problems))
        return 1

    # Walk the declared files, not the manifest entries. Iterating the manifest
    # meant the set being checked was whatever the last build had managed to
    # record, so a product left out of it, which is precisely the case that
    # needs reporting, was not checked at all: it was silently not in the loop.
    # A file declared, present, and unrecorded is therefore a failure here.
    manifest = held()
    for kind, declarations in KINDS:
        noun = NOUN[kind]
        for name in sorted(declarations):
            path = DATA / name
            if not path.exists():
                continue                      # reported by the loop above
            recorded = manifest[kind].get(name)
            if recorded is None:
                problems.append(f"{name} is in data/ and is declared a {noun}, "
                                f"but the manifest holds no hash for it: it is "
                                f"being published unchecked. Run --rebuild")
            elif digest(path) != recorded:
                who = (f"{DERIVED[name]} is the only thing that may write it"
                       if kind == "derived" else f"it is a {noun}")
                problems.append(f"{name} has changed since it was recorded; "
                                f"{who}")

    # A hash for something not declared, or recorded under the wrong kind.
    for kind, declarations in KINDS:
        for name in sorted(set(manifest[kind]) - set(declarations)):
            problems.append(f"the manifest records {name} as a {NOUN[kind]}, "
                            f"which is not declared as one")

    n = sum(len(manifest[kind]) for kind, _ in KINDS)
    print(f"  {n} files: {len(manifest['primary'])} measurements, "
          f"{len(manifest['derived'])} products and "
          f"{len(manifest['documentation'])} documents")
    if problems:
        print("\n".join("  " + p for p in problems))
        return 1
    print("  every file in data/ is declared and unchanged since it was "
          "recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
