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

Those two are Type A in the sense of the GUM: they are evaluated from the
statistical spread of the data, and their standard deviations mean what a
standard deviation usually means.

Two further terms are not random quantities but are still uncertainty, and the
first version of this budget left them out entirely. Reviewer 3's objection was
that the quoted interval covered too little, so omitting them answered the wrong
half of his comment. They are included here as Type B contributions, each
evaluated as a rectangular distribution over the range the sensitivity analysis
actually spans, so the standard uncertainty is the half-range divided by root
three:

  thresholds  the classification thresholds, swept over the same 3 x 5 grid as
              Supplementary Table S1. The bound spans 20.83 to 21.44 GPa.
  poisson     the Poisson ratio, uniform 0.30 against the phase-specific values
              of Supplementary Table S3, which moves the bound to 21.11 GPa.

Using the half-range itself, rather than the half-range over root three, is what
the superseded budget did. It overstates a bounded choice by a factor of root
three and is why that budget read 0.43 GPa with the threshold term dominant.

One term stays outside the budget on purpose. Which basalt input to use, the
shallow 86.0 GPa or the depth-resolved 49.5 GPa, is not an uncertainty at all:
it is a discriminated choice with a physical criterion, set out in §2.4 before
any comparison with the flexural measurement. Folding it in would report a
decision as noise and would let the interval swallow a disagreement the reader
should be shown. It is reported as a sensitivity in Supplementary Table S2, and
the measurement lies inside the predicted bracket under either input.

Cross-section representativeness is outside for a different reason: one section
cannot estimate it. That is said in §4.3 rather than costed here.
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


# Table S3: per-class shift in the median modulus when the uniform Poisson
# ratio of 0.30 is replaced by phase-specific values (basalt 0.24, flax 0.30,
# PA-11 0.42, mixed-volume 0.34 as a Voigt-Reuss-Hill average at f_basalt = 0.5).
POISSON_SHIFT_PCT = {"flax": 0.0, "matrix": -9.49,
                     "mixed-volume": -2.81, "basalt": +3.56}

RECTANGULAR = 3 ** 0.5      # half-range -> standard uncertainty, GUM Type B


def type_b_terms(raw_shallow, table, point):
    """The two bounded analysis choices, as rectangular standard uncertainties."""
    from rule_of_mixtures import bounds, threshold_sweep

    sweep = threshold_sweep(raw_shallow)["Voigt_GPa"]
    thresholds = (sweep.max() - sweep.min()) / 2 / RECTANGULAR

    shifted = [e * (1 + POISSON_SHIFT_PCT[c] / 100.0)
               for c, e in zip(table["class"], table[MEDIAN_COL])]
    poisson_point = bounds(table.fraction, pd.Series(shifted))[0]
    poisson = abs(poisson_point - point) / 2 / RECTANGULAR

    return ({"classification_thresholds": float(thresholds),
             "poisson_ratio": float(poisson)},
            {"threshold_sweep_GPa": [float(sweep.min()), float(sweep.max())],
             "poisson_specific_voigt_GPa": float(poisson_point)})


def main():
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    raw = pd.read_csv(DATA / "shallow_map.csv")
    table = class_table(shallow)

    reps = bootstrap_medians(shallow)
    result = _unc.budget(table, reps, CLASSES, seed=SEED, n_draws=N_DRAWS,
                         median_col=MEDIAN_COL)

    type_a = result["components_sd"]
    type_b, ranges = type_b_terms(raw, table, result["point"])
    combined = sum(v ** 2 for v in
                   list(type_a.values()) + list(type_b.values())) ** 0.5

    budget = {
        "voigt_point_estimate": result["point"],
        "n_draws": N_DRAWS,
        "seed": SEED,
        "method": ("Type A by resampling: class medians drawn from their "
                   "bootstrap replicates directly, not from a summarised 95 % "
                   "interval; class fractions drawn from a Dirichlet with "
                   "concentration equal to the observed counts. Type B by the "
                   "GUM rectangular convention: half the range of each "
                   "sensitivity analysis, divided by root three."),
        "type_a_sd_GPa": type_a,
        "type_a_combined_sd_GPa": result["combined_sd"],
        "type_b_sd_GPa": type_b,
        "type_b_ranges": ranges,
        "combined_sd_GPa": combined,
        "share_of_variance": {k: v ** 2 / combined ** 2 for k, v in
                              list(type_a.items()) + list(type_b.items())},
        "random_only_percentile_95_interval_GPa": result["interval"],
        "excluded": ("The choice of basalt input, which is a discriminated "
                     "choice on measurement-physics grounds rather than an "
                     "uncertainty and is reported in Supplementary Table S2; "
                     "and cross-section representativeness, which cannot be "
                     "estimated from a single section and is addressed in "
                     "§4.3."),
    }

    out = DATA / "uncertainty_budget.json"
    # Newlines pinned to "\n" rather than the platform default, so running
    # this on Windows does not rewrite every line of a file that has not
    # changed and leave a diff that looks like a result moved.
    out.write_text(json.dumps(budget, indent=2) + "\n",
                   encoding="utf8", newline="\n")

    print(f"Voigt point estimate            {result['point']:.3f} GPa\n")
    print("  Type A, evaluated by resampling")
    for k, v in type_a.items():
        print(f"    {k:<28} ± {v:.3f}   {100 * v ** 2 / combined ** 2:4.1f} %")
    print(f"    {'subtotal':<28} ± {result['combined_sd']:.3f}\n")
    print("  Type B, rectangular over the sensitivity range")
    for k, v in type_b.items():
        print(f"    {k:<28} ± {v:.3f}   {100 * v ** 2 / combined ** 2:4.1f} %")
    print(f"\n  combined                      ± {combined:.3f}")
    print(f"  reported as                     {result['point']:.2f} "
          f"± {combined:.2f} GPa")
    lo, hi = budget["random_only_percentile_95_interval_GPa"]
    print(f"\n  random-only 95 % interval     {lo:.2f} to {hi:.2f} GPa")
    skew = float(pd.Series(result["draws"]).skew())
    print(f"  skew {skew:+.3f}, so that interval is "
          f"{'not ' if abs(skew) > 0.1 else ''}symmetric")
    print(f"\nwritten to {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
