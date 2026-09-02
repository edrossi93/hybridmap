"""
Uncertainty on the laminate Voigt bound, by resampling rather than by
first-order propagation.

Reviewer 3 comment 9 objected to the original procedure: each class median was
drawn from its 95 % bootstrap confidence interval, which throws away the shape
of the bootstrap distribution and misrepresents the tails. The fix he asked for
is to draw from the bootstrap replicates themselves, which is what this does.

Two sources are propagated together:

  medians     resample indents within each class with replacement, take the
              median of each resample. This is the bootstrap distribution, used
              directly, not summarised into an interval first.
  fractions   Dirichlet draw with concentration parameters equal to the
              observed class counts, which is the posterior for multinomial
              proportions under a uniform prior.

Two further terms are deliberately outside this budget, because neither is a
random quantity:

  thresholds  a choice, reported as a sensitivity in Supplementary Table S1.
  basalt      which basalt input to use is also a choice, between the shallow
              and the depth-resolved class median. It is reported as a
              sensitivity in Supplementary Table S2 and computed by
              rule_of_mixtures.substitute(), not here.

Folding a choice into a random budget would misrepresent it as noise, and would
let the quoted interval absorb a disagreement the reader should see.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

from rule_of_mixtures import CLASSES, class_table, bounds  # noqa: E402

# Fixed so the quoted +/- is the same number every time the script is run. The
# same value is used in clustering_check.py; there is nothing special about 42
# beyond it being written down.
SEED = 42

# Ten thousand replicates. The standard deviation of the combined draw is stable
# in the third decimal of a GPa from about two thousand onwards, so this is
# comfortably past convergence and still runs in a few seconds.
N_DRAWS = 10_000


def bootstrap_medians(df, n_draws=N_DRAWS, seed=SEED):
    """One column of median replicates per class, drawn independently."""
    rng = np.random.default_rng(seed)
    out = {}
    for cls in CLASSES:
        vals = df.loc[df.constituent == cls, "E_GPa"].to_numpy(float)
        # Non-finite values are dropped rather than counted as zero. They are
        # fit failures, and a zero would pull the class median down.
        vals = vals[np.isfinite(vals)]
        # Each resample has the same size as the class, which is what makes the
        # spread reflect how well that class is actually sampled: the 380-indent
        # basalt median is less certain than the 2300-indent flax one, and the
        # budget should say so.
        idx = rng.integers(0, len(vals), size=(n_draws, len(vals)))
        out[cls] = np.median(vals[idx], axis=1)
    return out


def main():
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    table = class_table(shallow)
    counts = table.n.to_numpy(float)
    point = bounds(table.fraction, table.E_median_GPa)[0]

    reps = bootstrap_medians(shallow)
    E = np.column_stack([reps[c] for c in CLASSES])

    # A separate generator, seeded the same way. The two draws are independent
    # by construction: medians resample indents within a class, fractions
    # resample how many indents each class holds, and treating them as
    # uncorrelated is what lets the two component standard deviations below add
    # in quadrature to the combined one.
    rng = np.random.default_rng(SEED)
    f = rng.dirichlet(counts, size=N_DRAWS)

    # Three sums from the same replicates: everything varying, then each source
    # varying alone against the other held at its point value. That is how the
    # components in the budget are attributed, and it is why they are reported
    # as separate standard deviations rather than as a single number.
    voigt_all = (f * E).sum(axis=1)
    voigt_med = (table.fraction.to_numpy() * E).sum(axis=1)
    voigt_frac = (f * table.E_median_GPa.to_numpy()).sum(axis=1)

    budget = {
        "voigt_point_estimate": float(point),
        "n_draws": N_DRAWS,
        "seed": SEED,
        "method": ("Class medians drawn from their bootstrap replicates "
                   "directly, not from a summarised 95 % interval; class "
                   "fractions drawn from a Dirichlet with concentration equal "
                   "to the observed counts."),
        "components_sd_GPa": {
            "class_medians": float(voigt_med.std(ddof=1)),
            "class_fractions": float(voigt_frac.std(ddof=1)),
        },
        "combined_sd_GPa": float(voigt_all.std(ddof=1)),
        "percentile_95_interval_GPa": [float(np.percentile(voigt_all, 2.5)),
                                       float(np.percentile(voigt_all, 97.5))],
        "excluded": ("Classification thresholds, the choice of basalt input, "
                     "and cross-section representativeness. The first two are "
                     "analysis choices rather than random quantities and are "
                     "reported as sensitivities in Supplementary Tables S1 and "
                     "S2; the third cannot be estimated from a single "
                     "section."),
    }

    out = DATA / "uncertainty_budget.json"
    out.write_text(json.dumps(budget, indent=2), encoding="utf8")

    lo, hi = budget["percentile_95_interval_GPa"]
    print(f"Voigt point estimate      {point:.3f} GPa")
    print(f"class medians alone       ± {voigt_med.std(ddof=1):.3f}")
    print(f"class fractions alone     ± {voigt_frac.std(ddof=1):.3f}")
    print(f"combined                  ± {voigt_all.std(ddof=1):.3f}")
    print(f"95 % percentile interval  {lo:.2f} to {hi:.2f} GPa")
    print(f"\nskew {float(pd.Series(voigt_all).skew()):+.3f}, so the interval "
          f"is {'not ' if abs(pd.Series(voigt_all).skew()) > 0.1 else ''}"
          f"symmetric about the point estimate")
    print(f"written to {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
