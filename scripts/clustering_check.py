"""
Unsupervised cross-check of the rule-based classification.

The point of this is not to select the number of populations. The silhouette
score peaks at k = 2 and the Gaussian-mixture BIC keeps falling, so the
unsupervised analysis on its own does not pick four. What it does is tell us
whether a partition built without any architectural knowledge lands on
something compatible with the rule.

The feature vector is the nine slice values, hardness, modulus and phase angle
at 100, 150 and 200 nm, z-scored. They are three correlated triplets, so a
three-component PCA keeps about 94 % of the variance; the comparison is run on
both representations to show the answer does not depend on that choice.

ARI is chance-corrected, so its expected value for unrelated partitions is 0,
not 1/k. That is the calibration for reading 0.40.

ARI* collapses the two compliant clusters. k-means splits the compliant family
into two groups by stiffness, and that split is not the matrix/flax boundary,
which the mechanical signal cannot see. Merging them removes a distinction the
data do not support and is the comparison reported as ARI* in the paper.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Three properties at three depths. Depth is included deliberately: a single
# slice cannot separate a genuine fibre contact from one that starts stiff and
# softens as the probed volume grows, and that difference is the whole point of
# the mixed-volume class. The nine are three correlated triplets rather than
# nine independent features, which is why the PCA variant below exists.
FEATURES = ["H_100", "H_150", "H_200",
            "E_100", "E_150", "E_200",
            "PA_100", "PA_150", "PA_200"]

# Fixed so the partition is identical between runs. The value carries no
# meaning; what matters is that it is stated, and that the seed sweep in
# __main__ shows the partition does not depend on it.
SEED = 42

# k-means is restarted this many times and the best inertia kept. Twenty is
# enough that repeated runs agree to four decimals on this data; the default of
# ten occasionally landed in a worse local minimum.
N_INIT = 20


def order_by_stiffness(labels, E, k):
    """Relabel clusters 0..k-1 in order of increasing median modulus, so the
    numbering is stable between runs and between methods."""
    order = np.argsort([np.nanmedian(E[labels == c]) for c in range(k)])
    lut = {old: new for new, old in enumerate(order)}
    return np.array([lut[l] for l in labels])


def run(df, k=4, seed=SEED, use_pca=False, model="kmeans"):
    # QC failures are dropped rather than imputed. An indent whose contact never
    # settled has no meaningful property vector, and filling one in would let
    # the clustering agree with the rule for the wrong reason.
    d = df[(df.constituent != "QC/invalid") & df[FEATURES].notna().all(axis=1)]

    # Standardised because the three properties differ by two orders of
    # magnitude: unscaled, modulus in GPa would dominate the distance and the
    # phase angle would contribute almost nothing.
    X = StandardScaler().fit_transform(d[FEATURES].to_numpy())
    if use_pca:
        # Three components, which hold about 94 % of the variance here. The
        # nine features are three correlated triplets, so this tests whether
        # the agreement with the rule survives removing that redundancy.
        X = PCA(n_components=3, random_state=seed).fit_transform(X)

    if model == "kmeans":
        lab = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X).labels_
    elif model == "gmm":
        lab = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=10, random_state=seed).fit(X).predict(X)
    else:
        raise ValueError(model)

    lab = order_by_stiffness(lab, d.E_150.to_numpy(), k)
    rule = d.constituent.to_numpy()

    ari = adjusted_rand_score(rule, lab)
    ari_star = adjusted_rand_score(rule, np.where(lab <= 1, 0, lab))
    return {"n": len(d), "ARI": ari, "ARI_star": ari_star,
            "silhouette": silhouette_score(X, lab), "labels": lab, "rule": rule}


def bootstrap_stability(df, k=4, n_rep=20, frac=0.9, seed=SEED):
    """Refit on random 90 % subsamples and compare each to the full fit."""
    rng = np.random.default_rng(seed)
    d = df[(df.constituent != "QC/invalid") & df[FEATURES].notna().all(axis=1)]
    X = StandardScaler().fit_transform(d[FEATURES].to_numpy())
    base = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X).labels_
    scores = []
    for _ in range(n_rep):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        sub = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X[idx]).labels_
        scores.append(adjusted_rand_score(base[idx], sub))
    return float(np.mean(scores)), float(np.std(scores))


if __name__ == "__main__":
    from pathlib import Path
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

    m, s = bootstrap_stability(df)
    print(f"cluster stability over 20 subsamples at k = 4: ARI {m:.3f} +/- {s:.3f}")

    print("\nseed check")
    for seed in (0, 1, 42, 123, 2024):
        r = run(df, seed=seed)
        print(f"  seed {seed:>5}: ARI {r['ARI']:.4f}   ARI* {r['ARI_star']:.4f}")
