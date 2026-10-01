"""
Unsupervised cross-check of the rule-based classification.

The method is in `hybridmap.cluster`. What is here is our feature set, our k,
and our seed.

The point of this is not to select the number of populations. The silhouette
score peaks at k = 2, with a secondary plateau at k = 4-5, and the
Gaussian-mixture BIC falls to k = 7 and turns up at k = 8, so the unsupervised
analysis on its own does not pick four (`figs_supplementary.cluster_diagnostics`,
Supplementary Figure S4). What it does is tell us
whether a partition built without any architectural knowledge lands on
something compatible with the rule.

The feature vector is the nine slice values, hardness, modulus and phase angle
at 100, 150 and 200 nm, z-scored. They are three correlated triplets, so a
three-component PCA keeps about 94 % of the variance; the comparison is run on
both representations to show the answer does not depend on that choice.

ARI is chance-corrected, so its expected value for unrelated partitions is 0,
not 1/k. That is the calibration for reading 0.40.

ARI* collapses the two compliant classes on both sides: the two most compliant
clusters into one, and flax and matrix into one compliant rule class. k-means
splits the compliant family into two groups by stiffness, and that split is not
the matrix/flax boundary, which the mechanical signal cannot see. Merging
removes a distinction the data do not support and is the comparison reported
as ARI* in the paper.

The merge is applied on both sides, as §2.3 describes: the two compliant
clusters are collapsed, and so are the rule's flax and matrix labels. There is
no Hungarian or other matching step, because ARI is invariant to how either
partition numbers its classes.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybridmap import cluster as _hm                           # noqa: E402
from hybridmap.cluster import order_by_stiffness               # noqa: E402,F401

# Three properties at three depths. Depth is included deliberately: a single
# slice cannot separate a genuine fibre contact from one that starts stiff and
# softens as the probed volume grows, and that difference is the whole point of
# the mixed-volume class. The nine are three correlated triplets rather than
# nine independent features, which is why the PCA variant below exists.
FEATURES = ["H_100", "H_150", "H_200",
            "E_100", "E_150", "E_200",
            "PA_100", "PA_150", "PA_200"]

# The modulus the clusters are ordered by, so the numbering means the same
# thing in every run and in every method.
MODULUS = "E_150"

# Fixed so the partition is identical between runs. The value carries no
# meaning; what matters is that it is stated, and that the seed sweep in
# __main__ shows the partition does not depend on it.
SEED = 42

# k-means is restarted this many times and the best inertia kept. Twenty is
# enough that repeated runs agree to four decimals on this data; the default of
# ten occasionally landed in a worse local minimum.
N_INIT = 20

# The two most compliant classes on each side are merged before scoring ARI*:
# the two lowest-modulus clusters, and the two lowest-modulus rule classes,
# which on this map are flax and matrix.
MERGE_LOWEST = 2


def run(df, k=4, seed=SEED, use_pca=False, model="kmeans"):
    return _hm.run(df, FEATURES, MODULUS, k=k, seed=seed, n_init=N_INIT,
                   model=model, use_pca=use_pca, merge_lowest=MERGE_LOWEST)


def bootstrap_stability(df, k=4, n_rep=20, frac=0.9, seed=SEED):
    """Refit on random 90 % subsamples and compare each to the full fit."""
    return _hm.bootstrap_stability(df, FEATURES, k=k, n_rep=n_rep, frac=frac,
                                   seed=seed, n_init=N_INIT)


if __name__ == "__main__":
    data = Path(__file__).resolve().parents[1] / "data"
    df = pd.read_csv(data / "classified_depth_resolved.csv")

    print("representation                    n      ARI    ARI*   silhouette")
    for label, kw in [("k-means, 9 z-scored features", dict()),
                      ("k-means, 3 principal components", dict(use_pca=True)),
                      ("Gaussian mixture, full covariance", dict(model="gmm"))]:
        r = run(df, **kw)
        print(f"{label:<34}{r['n']:>5}{r['ARI']:>9.3f}{r['ARI_star']:>8.3f}"
              f"{r['silhouette']:>13.3f}")

    print("\nARI is chance-corrected, so the value expected for unrelated "
          "partitions is 0.000")
    print("ARI* merges the two most compliant classes on both sides, the two "
          "lowest clusters and\nflax with matrix, before scoring")

    # Plain ARI between the full fit and each refit, at k = 4 on both sides:
    # nothing is merged and the rule takes no part, so this figure is neither
    # the ARI against the rule nor ARI*.
    m, s = bootstrap_stability(df)
    print(f"cluster stability over 20 subsamples at k = 4: ARI {m:.3f} +/- {s:.3f}"
          f"  (full fit against each refit, no merge, rule not involved)")

    print("\nseed check")
    for seed in (0, 1, 42, 123, 2024):
        r = run(df, seed=seed)
        print(f"  seed {seed:>5}: ARI {r['ARI']:.4f}   ARI* {r['ARI_star']:.4f}")
