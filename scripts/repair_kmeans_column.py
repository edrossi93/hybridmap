"""`kmeans_k4` in the deposited table was not the partition the paper reports.

The column carried three labels, 0 to 2, under a name that promises four, and
its -1 did not mean what the data dictionary says it means: 157 of the 433
indents rejected by the strict phase-angle criterion were marked -1, while 276
of them carried a cluster index and nine indents that passed quality control
carried none. It was left over from an earlier pass with a different quality
mask.

No published number came from the column: the paper's ARI of 0.40 is computed by
`clustering_check.py`, which reruns the clustering. But Supplementary Figure S1
shows this partition, and a deposit that ships a mislabelled column invites a
reader to reproduce something the paper never claimed.

The column is rebuilt from the canonical run: k = 4, the nine-dimensional
feature vector, seed 42, clusters ordered by median modulus so the index means
the same thing in every run, and -1 exactly where `qc_flag_strict` is 1. The
agreement with the rule-based labels is printed, and it must come back at the
0.398 the paper rounds to 0.40.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clustering_check as C                                    # noqa: E402

CSV = HERE.parent / "data" / "classified_depth_resolved.csv"


def main():
    d = pd.read_csv(CSV)
    before = sorted(d.kmeans_k4.unique())

    result = C.run(d)
    labels = np.asarray(result["labels"])
    keep = d.qc_flag_strict == 0
    if keep.sum() != len(labels):
        sys.exit(f"  [FAIL] {keep.sum()} quality-controlled indents against "
                 f"{len(labels)} labels")

    column = np.full(len(d), -1, dtype=int)
    column[keep.to_numpy()] = labels
    d["kmeans_k4"] = column

    ari = adjusted_rand_score(d.loc[keep, "constituent"], labels)
    if abs(ari - result["ARI"]) > 1e-9:
        sys.exit(f"  [FAIL] the column does not reproduce the run's ARI: "
                 f"{ari:.4f} against {result['ARI']:.4f}")
    if abs(ari - 0.398) > 0.002:
        sys.exit(f"  [FAIL] ARI is {ari:.4f}; the paper reports 0.40")

    d.to_csv(CSV, index=False)
    print(f"  [ok ] kmeans_k4 rebuilt: {before} -> "
          f"{sorted(d.kmeans_k4.unique())}")
    print(f"  [ok ] -1 on exactly the {int((~keep).sum())} indents the strict "
          f"criterion rejects")
    print(f"  [ok ] ARI against the rule {ari:.4f}, ARI* {result['ARI_star']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
