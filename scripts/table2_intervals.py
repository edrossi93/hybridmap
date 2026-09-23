"""
The per-class confidence intervals of Table 3, recomputed from the deposited
tables.

The file keeps its name from the round in which that table was numbered 2. The
table it reproduces is the one carrying the per-class medians, counts and
intervals.

Table 3's caption declares the construction: "95 % bootstrap confidence
intervals on the medians (10^4 resamplings, seed 42)". That is a complete
recipe, so it should be runnable, and until this script existed it was the one
column of the paper that no deposited code produced. The medians beside it come
from `rule_of_mixtures.class_table`; the intervals had no such route.

The recipe is followed exactly as the caption states it: resample the indents
within a class with replacement, take the median of each resample, and report
the 2.5th and 97.5th percentiles of the ten thousand medians, drawn from a
generator seeded with 42.

    python table2_intervals.py [--out table2_intervals.csv]

The bootstrap is over indents within a class, so the interval says how well that
class median is pinned down by the indents that landed on it. It says nothing
about whether the class is the right partition, which is what the threshold
sweep in rule_of_mixtures.py is for, and nothing about how representative the
section is, which one cross-section cannot answer.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

from hybridmap import uncertainty as _unc                      # noqa: E402
from rule_of_mixtures import CLASSES                           # noqa: E402

# The two numbers the caption names. They are the caption's, not a choice made
# here; change them and the script stops reproducing the table it is for.
N_DRAWS = 10_000
SEED = 42

# Each acquisition, with the column its median is taken from. The shallow map is
# load-controlled and returns one modulus per indent, so its column is that
# modulus; the depth-resolved column is the 150 nm slice, which is the depth
# Table 3 reports.
ACQUISITIONS = [
    ("shallow", "classified_shallow.csv", "E_GPa"),
    ("depth-resolved", "classified_depth_resolved.csv", "E_150"),
]


def intervals(df, modulus_col, classes=CLASSES, n_draws=N_DRAWS, seed=SEED):
    """One row per class: n, median, and the interval on that median."""
    # Each class gets its own generator from the same seed, so a row does not
    # depend on how many classes were drawn before it. A caption that says
    # "seed 42" does not fix this, and the two readings differ in the second
    # decimal, so the convention has to be pinned deliberately.
    #
    # This reading is the one the paper already uses. The Methods sensitivity
    # note prints the shallow basalt interval as 83.7 to 87.8, which is what
    # per-class seeding gives; sharing one generator across the four classes
    # gives 87.64 and would put that sentence out of step with this table.
    replicates = _unc.bootstrap_medians(df, classes, modulus_col,
                                        n_draws=n_draws, seed=seed,
                                        seed_per_phase=True)
    bounds = _unc.intervals(replicates, level=95.0)

    rows = []
    for name in classes:
        values = df.loc[df.constituent == name, modulus_col].dropna()
        lo, hi = bounds[name]
        rows.append({"class": name, "n": len(values),
                     "median_GPa": values.median(),
                     "ci_low_GPa": lo, "ci_high_GPa": hi})
    return pd.DataFrame(rows)


def main(out=None):
    tables = []
    for label, filename, column in ACQUISITIONS:
        table = intervals(pd.read_csv(DATA / filename), column)
        table.insert(0, "acquisition", label)
        tables.append(table)

        print(f"\n{label}, median of {column}, "
              f"{N_DRAWS} resamplings, seed {SEED}")
        for _, r in table.iterrows():
            print(f"  {r['class']:<14} n = {r['n']:>4}   "
                  f"{r['median_GPa']:7.2f}   "
                  f"[{r['ci_low_GPa']:.2f}, {r['ci_high_GPa']:.2f}]")

    combined = pd.concat(tables, ignore_index=True)
    if out:
        combined.to_csv(out, index=False, lineterminator="\n")
        print(f"\nwritten to {out}")
    return combined


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", help="write the table here as CSV")
    main(p.parse_args().out)
