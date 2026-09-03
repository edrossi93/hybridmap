"""
Unsupervised cross-check of a rule-based classification.

This does not choose how many populations there are, and it should not be
reported as if it did. What it answers is narrower and more useful: does a
partition built with no architectural knowledge land somewhere compatible with
the rule that used it.

ARI is chance-corrected, so the value expected for unrelated partitions is 0,
not 1/k. That is the calibration for reading whatever number comes out.

ARI* merges the `merge_lowest` most compliant clusters before comparing. A
distance-based partition tends to split a compliant family by stiffness, and
that split is often not the boundary the rule draws, because the mechanical
signal cannot see it. Merging removes a distinction the data do not support.
Set it to 0 and ARI* is just ARI again.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import io
from .classify import INVALID


def order_by_stiffness(labels, modulus, k):
    """Renumber clusters 0..k-1 by increasing median modulus.

    Cluster numbering is arbitrary, so without this the same partition prints
    differently between runs and between methods, and nothing can be compared
    by eye.
    """
    order = np.argsort([np.nanmedian(modulus[labels == c]) for c in range(k)])
    lut = {old: new for new, old in enumerate(order)}
    return np.array([lut[l] for l in labels])


def _matrix(d, features, seed, use_pca, n_components):
    # Standardised because the properties differ by orders of magnitude:
    # unscaled, a modulus in GPa dominates the distance and a phase angle in
    # degrees contributes almost nothing.
    X = StandardScaler().fit_transform(d[features].to_numpy())
    if use_pca:
        # Features that come as correlated groups, one property at several
        # depths, carry less information than their count suggests. This tests
        # whether the agreement survives removing that redundancy.
        X = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    return X


def run(df, features, modulus_col, k=4, seed=0, n_init=10, model="kmeans",
        use_pca=False, n_components=3, merge_lowest=0,
        label_col="constituent", invalid=INVALID):
    """Fit, align the numbering, and score against the rule's labels."""
    # A tuple indexes a DataFrame as one key, not as a column list.
    features = list(features)
    io.require(df, features + [modulus_col], "the clustering cross-check")

    # Quality failures are dropped rather than imputed. An indent whose contact
    # never settled has no meaningful property vector, and filling one in would
    # let the clustering agree with the rule for the wrong reason.
    d = df[(df[label_col] != invalid) & df[features].notna().all(axis=1)]
    X = _matrix(d, features, seed, use_pca, n_components)

    if model == "kmeans":
        lab = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X).labels_
    elif model == "gmm":
        lab = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=10, random_state=seed).fit(X).predict(X)
    else:
        raise ValueError(f"unknown clustering model {model!r}; "
                         f"expected 'kmeans' or 'gmm'")

    lab = order_by_stiffness(lab, d[modulus_col].to_numpy(), k)
    rule = d[label_col].to_numpy()
    merged = np.where(lab <= merge_lowest - 1, 0, lab)
    return {"n": len(d),
            "ARI": adjusted_rand_score(rule, lab),
            "ARI_star": adjusted_rand_score(rule, merged),
            "silhouette": silhouette_score(X, lab),
            "labels": lab, "rule": rule}


def bootstrap_stability(df, features, k=4, n_rep=20, frac=0.9, seed=0,
                        n_init=10, label_col="constituent", invalid=INVALID):
    """Refit on random subsamples and score each against the full fit.

    A partition that dissolves when a tenth of the data is removed is not
    evidence of anything, whatever its silhouette says.
    """
    features = list(features)
    rng = np.random.default_rng(seed)
    d = df[(df[label_col] != invalid) & df[features].notna().all(axis=1)]
    X = StandardScaler().fit_transform(d[features].to_numpy())
    base = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X).labels_
    scores = []
    for _ in range(n_rep):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        sub = KMeans(n_clusters=k, n_init=n_init,
                     random_state=seed).fit(X[idx]).labels_
        scores.append(adjusted_rand_score(base[idx], sub))
    return float(np.mean(scores)), float(np.std(scores))
