# Nanoindentation data and analysis code

Data and code behind *Measured-input rule-of-mixtures prediction of the flexural modulus of a flax/basalt/PA-11 hybrid laminate from high-speed nanoindentation mapping*.

Everything the paper reports can be recomputed from what is in here. Start with `scripts/reproduce_paper_values.py`; it prints each published number next to the recomputed one and tells you if anything has drifted.

---

## Quick start

```
pip install -r requirements.txt
cd scripts
python reproduce_paper_values.py
```

Expected last line: `34 of 34 values reproduce`.

Tested on Python 3.10 and 3.12. Nothing here needs a GPU or more than a couple of GB of RAM.

---

## What is in `data/`

Eight CSVs and two JSON summaries. The first three are the exported per-indent arrays and are primary inputs to this deposit, the next two carry the classification used in the paper, two cover the mechanical testing, and two hold the backscattered-electron segmentation behind Figure 2.

| file | rows | what it is |
|---|---|---|
| `shallow_map.csv` | 8553 | The fixed-depth high-speed map, one row per indent. Every indent in the acquisition, including the 3000 that sit on the embedding resin. |
| `depth_resolved_curves.csv` | 101515 | The CSM map as curves. Each of the 2000 indents is resampled onto a common 5 nm depth grid from 50 to 300 nm, so it is one tidy table instead of 2000 ragged ones. |
| `depth_resolved_slices.csv` | 2000 | The same map reduced to slice values at 100, 150 and 200 nm, computed on the resampled grid. Not the numbers the paper reports; see *Slice extraction* below. |
| `classified_shallow.csv` | 5553 | The shallow map after the resin cut, carrying the zone and the constituent label. |
| `classified_depth_resolved.csv` | 2000 | The CSM map with slice values, zone, constituent label, quality flag and the k-means assignment. |
| `flexural_curves.csv` | 8371 | Three-point bending, stress against strain, **five specimens per configuration**. |
| `flexural_moduli.csv` | 10 | Per-specimen chord moduli, raw and after ASTM D790 Annex A1.1 toe compensation, plus the modulus over the 0.28-0.43 % window. |
| `bse_segmentation_tiles.csv` | 91 | One row per imaged tile: zone pixel counts and the fibre/matrix split inside each zone. |
| `bse_segmentation_summary.json` | - | The constituent volume fractions of Figure 2e, with the aggregation stated. |
| `uncertainty_budget.json` | - | The resampled uncertainty on the Voigt bound, drawn from bootstrap replicates. |

Units are in the column names. Moduli and hardness in GPa, positions in µm, depths in nm, loads in mN, phase angles in degrees.

### Which axis is through the thickness

The two acquisitions store the same physical direction under different column
names, so this is worth stating plainly before anyone plots them:

| file | through-thickness axis | in-plane axis |
|---|---|---|
| `shallow_map.csv`, `classified_shallow.csv` | `x_um` (resin cut at x < 380, zone split at 730) | `y_um`, 97.5 µm |
| `classified_depth_resolved.csv` | `y_um` (three zone bands at 0-78, 81-168, 171-297) | `x_um`, 57 µm |

In both cases the map's **long** axis runs through the laminate thickness, which
is the physically meaningful statement; only the column naming differs.
`depth_resolved_slices.csv` carries no position columns at all: join it to
`classified_depth_resolved.csv` on `indent_id` if you need coordinates. In the
backscattered-electron mosaic of Figure 2 the thickness direction is vertical,
rotated ninety degrees relative to the property maps of Figures 3 to 5.

### Class names

`flax`, `matrix`, `mixed-volume`, `basalt`, and `QC/invalid` for indents that fail the phase-angle check.

`mixed-volume` means the stressed volume under the tip spanned both a fibre and the surrounding matrix. It is a description of the contact, not of a distinct material phase, and its population fraction depends on contact radius and grid pitch. It should not be read as an interphase property.

### On the fractions

The class fractions in the paper are **indent-class fractions**: the number of indents carrying each label over the number of classified indents. They are not measured constituent volume fractions. Inside the flax-rich half of the section the rule assigns every compliant indent to flax, so polyamide sitting in that half is counted as flax. This does not move the Voigt sum, because the two compliant medians are 4.70 and 4.66 GPa, but it does mean the numbers should not be quoted as volume fractions.

### The flexural replicates

All ten curves are here: five specimens per configuration, every one of them, not a representative curve per configuration. So the statistics the paper quotes can be recomputed rather than taken on trust:

| configuration | ASTM D790, toe-compensated | mean above the 0.28 % crossover |
|---|---|---|
| basalt face in tension | 18.85 ± 0.54 GPa | 19.91 GPa |
| flax face in tension | 19.66 ± 0.48 GPa | 18.48 GPa |

The ranking reverses across the crossover, which is why both windows are reported. `reproduce_paper_values.py` checks these together with the Welch statistic and Hedges *g*.

One caveat on the flax standard deviation: recomputing it from `flexural_moduli.csv` gives 0.49, not the 0.48 above. See the note on precision below.

---

## What is in `scripts/`

Fourteen files, in three groups. They depend on numpy, pandas, scipy, scikit-learn and matplotlib, and on nothing else.

**Start here.**

| file | what it does |
|---|---|
| `reproduce_paper_values.py` | Runs the analysis and prints all 34 published values next to the recomputed ones. This is the one to run first. |

**Analysis.**

| file | what it does |
|---|---|
| `classify.py` | The rule-based classification: spatial zones plus modulus thresholds. |
| `rule_of_mixtures.py` | Voigt, Reuss and Hill bounds, threshold sweep, input substitutions. |
| `clustering_check.py` | k-means and Gaussian-mixture cross-check, ARI against the rule. |
| `uncertainty_budget.py` | The resampled uncertainty on the Voigt bound. Writes `data/uncertainty_budget.json`. |
| `make_flexural_moduli.py` | Chord moduli from the bending curves. Reproduces `data/flexural_moduli.csv` exactly. |

**Figures.** Each writes the figure it is named for, from the deposited data only.

| file | figure |
|---|---|
| `figstyle.py` | Shared style: sizes, fonts, the constituent colours, the layout check and the save helper. Imported by the rest, not run on its own. |
| `fig05_classification.py` | Figure 5, classification panels |
| `fig06_property_relationships.py` | Figure 6 |
| `fig07_curves_per_class.py` | Figure 7 |
| `fig08_rom.py` | Figure 8, the rule-of-mixtures construction |
| `fig09_flexural.py` | Figure 9, three-point bending |
| `fig_map_coverage.py`, `reply_figures.py` | Figures supporting the response to reviewers rather than the paper |

The backscattered-electron panels of Figures 2 and 5 are not regenerated here.
They are drawn from the tile images rather than from any deposited table, and
those images are the primary input described below. What
the deposit carries for them is the measurement: `bse_segmentation_tiles.csv`
and `bse_segmentation_summary.json`, with the procedure written out under
*Files provided as primary inputs*.

The analysis scripts each run on their own and print something worth reading:

```
python classify.py            # reprints the class counts, checks them against the deposit
python rule_of_mixtures.py    # bounds, zone-weighting check, sensitivity
python clustering_check.py    # ARI across three representations, stability, seed sweep
python uncertainty_budget.py  # the +/- on the bound, by component
```

### Where the per-indent tables come from

`shallow_map.csv`, `depth_resolved_curves.csv` and `depth_resolved_slices.csv` are **primary inputs to this deposit**. They are the per-indent arrays exported from the two nanoindentation acquisitions, and everything else here is computed from them by the scripts in `scripts/`.

The instrument files they were exported from are not part of this deposit and are not distributed. The acquisition and the proprietary export that produces these arrays are the instrument's, not the analysis's, and nothing in the paper depends on them: every reported number is recomputed from the tables above by `reproduce_paper_values.py`. The three files are complete, in the sense that no indent has been removed from them — `shallow_map.csv` still carries the ~3000 indents that landed on the embedding resin, and `depth_resolved_curves.csv` is unfiltered, artefacts and all.

So the reproduction boundary of this deposit is the exported per-indent table. Below that line is the instrument. Above it, everything is here.

---

## Reproducing specific things

**Class medians and counts (Table 2).** `classify.py` rebuilds the labels from `shallow_map.csv` and checks them against `classified_shallow.csv`. Agreement is 100 %.

**The bounds (Section 3.4).** `rule_of_mixtures.py`. It also verifies that zone weighting and global counting give the same fractions, which the paper asserts and which is worth checking rather than believing.

**Threshold sensitivity (Supplementary Table S1).** `threshold_sweep()` in `rule_of_mixtures.py`. Reference construction at 15 and 70 GPa gives 21.33 GPa; the fifteen threshold pairs span 20.83 to 21.44 GPa, largest deviation 0.50 GPa.

**The unsupervised cross-check (Section 3.3).** `clustering_check.py`. ARI is 0.398 against the rule and 0.678 once the two compliant clusters are merged, and both are the same to four decimals across five random seeds. The comparison is also run on three principal components and on a full-covariance Gaussian mixture, and it does not move much.

One thing to be clear about: the clustering does **not** independently select four populations. The silhouette peaks at k = 2 and the Gaussian-mixture BIC falls monotonically. Four is chosen on architectural grounds, and the clustering tells us that a partition built with no architectural knowledge lands somewhere compatible with the rule. ARI is chance-corrected, so 0.398 should be read against an expectation of 0 for unrelated partitions.

---

## Slice extraction

Slice values at 100, 150 and 200 nm are the mean of the CSM response over a 20 nm window centred on the target depth, requiring at least three valid points inside the window.

The same quantity appears twice in `data/`, computed by two passes, and the two are not the same numbers. **`E_150` in `classified_depth_resolved.csv` is taken from the trace as recorded, and it is what the paper reports.** `E_150nm_GPa` in `depth_resolved_slices.csv` is the window mean over the resampled 5 nm grid of `depth_resolved_curves.csv`, which you can recompute yourself from that file.

They correlate at 0.9997. The class medians are identical for flax and matrix and differ where E(z) is steepest:

| class | `E_150`, the paper | `E_150nm_GPa`, the slices file |
|---|---|---|
| flax | 3.85 | 3.85 |
| matrix | 5.37 | 5.37 |
| mixed-volume | 29.28 | 28.76 |
| pure-axial basalt | 49.47 | 49.13 |

So a reader recomputing from the slices file will see the two stiff classes land about 2 % low. Both routes are defensible; the difference is a real, if small, sensitivity of the slice extraction on the classes where the modulus is changing fastest with depth, and it is smaller than the class-to-class separation by more than an order of magnitude. Carried through the Voigt sum it is worth 0.02 GPa on a bound of 21.33.

---

## Citation

If you use this, please cite the paper and this deposit.

Deposit DOI: **[Zenodo DOI to be inserted on acceptance]**

---

## Licence

Data under CC BY 4.0, code under MIT.

## A note on precision

`flexural_moduli.csv` stores the per-specimen moduli rounded to two decimals,
which is the precision at which they are reported. The summary statistics in the
paper were computed from the unrounded values.

The two differ in the third decimal, and for one quantity that changes the
second. Recomputing the flax-longitudinal standard deviation from this file
gives 0.4863 GPa, which rounds to 0.49; computed at full precision it is 0.4842,
which rounds to the 0.48 the paper reports. Nothing is wrong with either number.
`reproduce_paper_values.py` allows for this, but a reader recomputing by hand
should know why the last digit moves.

## Files provided as primary inputs

Most of what is here can be rebuilt from something else in the folder. Five
files cannot, and it is better to say so than to leave a reader hunting for a
script that is not there.

### `shallow_map.csv`, `depth_resolved_curves.csv`, `depth_resolved_slices.csv`

The exported per-indent arrays, described under *Where the per-indent tables
come from* above. They are the starting point of everything else here.

One of the three is in fact derivable from another: `depth_resolved_slices.csv`
is the 20 nm window mean of `depth_resolved_curves.csv` at each of the three
target depths, requiring at least three valid points in the window, and you can
reproduce it in a few lines of pandas. It is provided ready-made because it is
what most readers will want.

### `bse_segmentation_tiles.csv` and `bse_segmentation_summary.json`

These carry the constituent volume fractions reported in Figure 2e. They are
**primary inputs**: the steps above them need the backscattered-electron
micrographs, which are several gigabytes of tile images and are not deposited.
They remain with the corresponding author and are available on request.

How the tile statistics were produced:

1. The polished cross-section was imaged as 91 overlapping backscattered-electron
   tiles and stitched into a single mosaic covering 6.65 mm².
2. **The basalt-rich and flax-rich tow zones were marked by hand on the mosaic.**
   The boundary between adjacent tows is diffuse, so an interface band was left
   unmarked rather than forced into one zone or the other; it is 4.0 % of the
   mosaic area and is excluded from the fractions.
3. Within each marked zone, pixels were classified as fibre or matrix by a
   texture classifier trained on the operator's labels, and voids and cracks
   were separated by threshold.
4. Per-tile pixel counts were written to `bse_segmentation_tiles.csv`.

The manual zone marking is the one subjective step in the measurement, and it
is the reason the fractions are reported with an interval from field-to-field
scatter rather than as point values.

`bse_segmentation_summary.json` is one aggregation step above the tile file and
**is** reproducible: it renormalises the zone areas over the marked zones only
and multiplies each by its pooled in-zone fibre fraction.

### `flexural_moduli.csv`

Derived from `flexural_curves.csv` by the ASTM D790 chord fit with Annex A1.1
toe compensation. See `make_flexural_moduli.py` for the exact convention,
including which linear region is taken as the steepest.
