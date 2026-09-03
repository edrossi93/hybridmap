"""
Uncertainty on the laminate Voigt bound, by resampling rather than by
first-order propagation.

The resampling is in `hybridmap.uncertainty`. What is here is our four classes,
our seed, and the prose that says which sources this budget covers and which it
deliberately does not.

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

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

from hybridmap import uncertainty as _unc                      # noqa: E402
from rule_of_mixtures import CLASSES, MEDIAN_COL, class_table  # noqa: E402

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
    return _unc.bootstrap_medians(df, CLASSES, "E_GPa", n_draws, seed)


def main():
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    table = class_table(shallow)

    reps = bootstrap_medians(shallow)
    result = _unc.budget(table, reps, CLASSES, seed=SEED, n_draws=N_DRAWS,
                         median_col=MEDIAN_COL)

    budget = {
        "voigt_point_estimate": result["point"],
        "n_draws": N_DRAWS,
        "seed": SEED,
        "method": ("Class medians drawn from their bootstrap replicates "
                   "directly, not from a summarised 95 % interval; class "
                   "fractions drawn from a Dirichlet with concentration equal "
                   "to the observed counts."),
        "components_sd_GPa": result["components_sd"],
        "combined_sd_GPa": result["combined_sd"],
        "percentile_95_interval_GPa": result["interval"],
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
    print(f"Voigt point estimate      {result['point']:.3f} GPa")
    print(f"class medians alone       ± {result['components_sd']['class_medians']:.3f}")
    print(f"class fractions alone     ± {result['components_sd']['class_fractions']:.3f}")
    print(f"combined                  ± {result['combined_sd']:.3f}")
    print(f"95 % percentile interval  {lo:.2f} to {hi:.2f} GPa")
    skew = float(pd.Series(result["draws"]).skew())
    print(f"\nskew {skew:+.3f}, so the interval "
          f"is {'not ' if abs(skew) > 0.1 else ''}"
          f"symmetric about the point estimate")
    print(f"written to {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
