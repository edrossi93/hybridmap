# Nanoindentation data and analysis code

Data and code behind *Estimating the flexural modulus of a hybrid flax/basalt/PA-11 laminate from high-speed nanoindentation mapping and a rule of mixtures*.

Everything the paper reports can be recomputed from what is in here. Start with `scripts/reproduce_paper_values.py`; it prints each published number next to the recomputed one and tells you if anything has drifted.

---

## Quick start

```
pip install -r requirements.txt
cd scripts
python reproduce_paper_values.py
```

Expected last line: `34 of 34 values reproduce`.

Needs Python 3.8 or newer, and the six packages in `requirements.txt`, which says what each one is for. Tested on Python 3.10 and 3.12. Nothing here needs a GPU or more than a couple of GB of RAM.

### Where to go next

| if you want to | go to |
|---|---|
| check that the paper's numbers are what the data says | *Reproducing specific things* |
| know what is in the deposited files | *What is in `data/`* |
| run the method on your own map | *Running this on your own map* |
| know what your CSV has to look like | *What your CSV has to look like* |
| cite this, or check the licence | *Citation*, *Licence* |

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
| `mixed_class_deconvolution.json` | - | The mixed-volume class inverted back onto its end members, both acquisitions, with the independent comparison. |
| `spatial_localisation.json` | - | The neighbourhood statistics behind the claim that mixed-volume contacts sit at the phase boundaries. |
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

Twenty files, in three groups, plus the `hybridmap/` package and the example configs described under *Running this on your own map*. They depend on numpy, pandas, scipy, scikit-learn, matplotlib and PyYAML, and on nothing else. Only the last is needed for the tool rather than for the reproduction.

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
| `phase_angle_sensitivity.py` | Sweeps the phase-angle quality-control threshold at 10, 15 and 20 degrees and reports what it moves. Writes `data/phase_angle_sensitivity.json`. |
| `spacing_compliance.py` | Contact depth per indent from load and hardness, against the Phani–Oliver spacing criterion. Writes `data/spacing_compliance.json`. |
| `make_flexural_moduli.py` | Chord moduli from the bending curves. Reproduces `data/flexural_moduli.csv` exactly. |
| `table2_intervals.py` | The per-class confidence intervals of Table 2, by the construction its caption declares. |
| `mixed_class_deconvolution.py` | Inverts every indent in the basalt-rich zone onto the basalt/matrix tie line, under parallel and series mixing, and compares the two acquisitions with each other and with the imaging. Writes `data/mixed_class_deconvolution.json`. |
| `repair_kmeans_column.py` | Rebuilds `kmeans_k4` in `classified_depth_resolved.csv` from the canonical clustering run. The submitted column held three labels under a name that promises four, and a quality mask that matched neither `qc_flag_strict` nor the paper. |

**Figures.** Each writes the figure it is named for, from the deposited data only.

| file | figure |
|---|---|
| `figstyle.py` | Shared style: sizes, fonts, the constituent colours, the layout check and the save helper. Imported by the rest, not run on its own. |
| `fig06_classification.py` | Figure 5, classification panels |
| `fig07_property_relationships.py` | Figure 6 |
| `fig05_curves_per_class.py` | Figure 7 |
| `fig08_rom.py` | Figure 8, the rule-of-mixtures construction |
| `fig09_flexural.py` | Figure 9, three-point bending |
| `figs_supplementary.py` | Supplementary Figures S1, S2 and S4, and the spatial statistics of §4.1. Writes `data/spatial_localisation.json`. |
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
python phase_angle_sensitivity.py  # what the QC threshold does and does not move
python spacing_compliance.py       # how many indents meet the spacing criterion
python mixed_class_deconvolution.py  # the class inverted onto its end members
python figs_supplementary.py         # Supplementary Figures S1, S2, S4 and their statistics
```

### Where the per-indent tables come from

`shallow_map.csv`, `depth_resolved_curves.csv` and `depth_resolved_slices.csv` are **primary inputs to this deposit**. They are the per-indent arrays exported from the two nanoindentation acquisitions, and everything else here is computed from them by the scripts in `scripts/`.

The instrument files they were exported from are not part of this deposit and are not distributed. The acquisition and the proprietary export that produces these arrays are the instrument's, not the analysis's, and nothing in the paper depends on them: every reported number is recomputed from the tables above by `reproduce_paper_values.py`. The three files are complete, in the sense that no indent has been removed from them: `shallow_map.csv` still carries the ~3000 indents that landed on the embedding resin, and `depth_resolved_curves.csv` is unfiltered, artefacts and all.

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

## Running this on your own map

Everything above reproduces one laminate. The same chain, classify, aggregate, bound, compare against a macroscopic measurement, runs on someone else's map from a configuration file. You should never have to open a `.py` to analyse your own data.

```
cd scripts
python -m hybridmap validate  --config examples/paper_shallow.yaml
python -m hybridmap histogram --config examples/paper_shallow.yaml --out figs
python -m hybridmap report    --config examples/paper_shallow.yaml
```

| command | what it does |
|---|---|
| `init` | Reads your CSV and writes a starter config with your own column names in it. Start here. |
| `validate` | Checks the config against the data and reports what it found: values usable and missing, the modulus range, whether your cuts fall inside it. |
| `histogram` | The modulus distribution with your cuts drawn on it. This is how a cut gets chosen. |
| `report` | The whole chain. `--out DIR` also writes the tables and figures. |
| `classify` | The per-indent labels, as a CSV. |
| `sweep` | The bounds across a grid of cut values. |
| `uncertainty` | The resampled spread on the bound, and where it comes from. |
| `cluster` | The unsupervised cross-check, if you have per-indent features. |
| `macro` | Reduces bending curves to per-specimen moduli. |

Any of them takes `--set` to change one value without editing the file, for example `--set cuts.lower=12`. A configuration problem exits with a message naming what to fix rather than a traceback.

Configs are YAML, and JSON with the same keys is read too. YAML is the documented form because a threshold should carry its reason on the line above it.

### Starting from your own CSV

```
cd scripts
python -m hybridmap init      --map /path/to/your_map.csv --out mine.yaml
python -m hybridmap validate  --config mine.yaml
python -m hybridmap histogram --config mine.yaml --out figs
# look at figs/modulus_histogram.png, then edit the cuts in mine.yaml
python -m hybridmap report    --config mine.yaml --out results
```

Those commands run from `scripts/` because that is where the package sits. To
use the tool on data kept somewhere else, install it once and drop the `cd`:

```
pip install -e .
hybridmap report --config /path/to/mine.yaml --out results
```

Installing is optional and changes nothing else. The reproduction path above
needs no install at all.

`init` guesses which column is which from the column names and lays out one cut per phase boundary at even quantiles of your data. Those quantile cuts are a placeholder and the file says so: they spread the indents evenly between phases and mean nothing physically. The histogram is how you replace them with a choice you can defend.

---

## What your CSV has to look like

The tool reads two kinds of table and never anything else. It does not rename your columns and it does not convert your units.

### The map, one row per indent

| role | required | what it must be |
|---|---|---|
| `modulus` | yes | A number. This is what the phases are cut on. Empty cells are allowed and become unusable indents; text is not. |
| `position` | only if `zones` cuts bands on a coordinate | A number, in whatever unit you like. |
| `id` | no | Anything, carried through so you can join results back to your own records. |
| columns named by `qc` or `clustering` | only if you use those sections | Numbers. |

Any other column in the file is carried through untouched. There is no required naming convention: `E_GPa`, `reduced_modulus_GPa` and `Er` are all fine, because you say which is which in the config.

If your export is not comma separated, or uses a decimal comma, say so and nothing else changes:

```yaml
map:
  path: your_map.csv
  read: {sep: ";", decimal: ",", encoding: latin-1, na_values: ["---"]}
```

The tool does not try to detect that for you. Guessing a decimal separator is how a factor of a thousand gets into an analysis without anyone noticing.

### The curves, one row per recorded point

Needed only if you want the tool to reduce your bending data rather than being told the answer. Two shapes are accepted.

**Already reduced**, which is what testing software usually exports:

```yaml
macro:
  from_curves:
    path: your_curves.csv
    columns: {specimen: id, group: configuration, strain: strain_pct, stress: stress_MPa}
    chord: [0.05, 0.25]       # per cent strain, ASTM D790
    toe_window: 0.15
```

**As recorded**, force against displacement, which is what the machine actually writes. Give the specimen geometry in millimetres and the tool converts with the standard formulae for a rectangular bar in three-point bending, `stress = 3PL/2bd²` and `strain = 6Dd/L²`:

```yaml
macro:
  from_curves:
    path: your_curves.csv
    columns: {specimen: id, group: face, displacement: disp_mm, force: force_N}
    geometry: {span: 40.0, width: 12.7, thickness: 3.2}
    chord: [0.05, 0.25]
    toe_window: 0.15
```

Either way the chord comes out as a slope in stress-unit per per-cent strain. `modulus_scale` converts it to the unit your map is in, and defaults to `0.1`, which is right when stress is in MPa and you want GPa. Set it to `1` if your stress is already in GPa. The printed output always names the scale it used.

With two groups, the tool also reports the mean, standard deviation and count per group, Welch's *t* with its *p*, and Hedges *g*.

### The two worked examples

| file | what it is |
|---|---|
| `examples/paper_shallow.yaml` | This paper's shallow map. Reproduces the published class counts, medians and bounds from the config alone. |
| `examples/paper_depth_resolved.yaml` | This paper's depth-resolved map. The other input shape, and the other pure-fibre cut. |

Read them side by side before writing your own. What differs between them is exactly what the tool had to stop assuming.

### What a config decides

| section | what it says | optional |
|---|---|---|
| `map` | the CSV to read, whether it is single-depth or depth-resolved, and which column holds the identifier, the modulus and the position | no |
| `zones` | bands cut on a coordinate, or zone names read from a column; a band may be marked `discard` for mounting medium or edge effects | yes |
| `cuts` | named modulus thresholds | no |
| `phases` | one per mechanical population: a modulus range between named cuts, optionally restricted to one zone, with a name and a colour | no |
| `qc` | a stored rejection flag, or a phase-angle criterion | yes |
| `sweep` | the values to try for each cut | yes |
| `macro` | a measured modulus with its uncertainty, or bending curves to reduce to one | yes |
| `clustering` | the feature columns, k, and the seed for the unsupervised cross-check | yes |
| `sensitivity` | phase inputs to substitute and report beside the answer | yes |
| `uncertainty` | how many resampling draws, and the seed | yes |

Three points about the schema, because they are the ones that surprise people.

**Cuts are named once and phases refer to them by name.** A phase says it sits below the cut called `lower`, not below 15.0. Writing the number into each phase would let the copies drift, and it would leave the sweep with nothing well defined to move.

**Phases are matched in the order you write them and the first match wins.** That is a rule a reader can follow by eye, which matters more here than flexibility. Two, three or five phases all work; ours happens to use four.

**Zones are optional.** A modulus-only classification is legitimate when there is no architecture to exploit. Ours uses two bands along the thickness direction because the laminate has two tow zones; yours might have one, three, or none.

Nothing renames your columns and nothing converts your units. The unit lives in the column name, as it does in `data/`.

### Choosing the cuts

Making a threshold configurable is nearly useless if you have no way to choose it, so the tool ships the two diagnostics that justified ours.

`python -m hybridmap histogram` draws the modulus distribution with your cuts on it and the phases they produce shaded in their own colours. A cut is defended by the shape of the distribution it sits in: if there is no thinning in the histogram where you have put a line, the line is a decision you have not yet earned.

`python -m hybridmap sweep` redoes the whole classification for every combination of cut values you list and prints the bounds for each. It reports the departure from your configured cuts, which is the number to quote. For this paper the fifteen threshold pairs span 20.83 to 21.44 GPa, a largest departure of 0.50 GPa on a bound of 21.33.

### The package

`scripts/hybridmap/` is the reusable part. The analysis scripts above are configurations of it, so there is one implementation of each step rather than one per script.

| file | what it does |
|---|---|
| `config.py` | Reads and validates a config. Most of it is error messages. |
| `io.py` | Reads the map CSV and resolves which column plays which role. |
| `classify.py` | Zones, the phase rule, and the quality check. |
| `aggregate.py` | Counts, fractions and per-phase median modulus. |
| `mixtures.py` | Voigt, Reuss and Hill, input substitution, and the cut sweep. |
| `uncertainty.py` | The resampled spread on the bound, and the interval on each phase median. |
| `macro.py` | The measured comparison, including the chord-modulus route to one. |
| `cluster.py` | k-means and Gaussian-mixture cross-check on named features. |
| `plots.py` | The histogram, the sweep and the prediction against the measurement. |
| `__main__.py` | The command line. |
| `__init__.py` | What the package exports, and a short example of using it directly. |

To use it from your own code rather than from the command line:

```python
from hybridmap import config, io, classify, aggregate, mixtures
cfg = config.load("examples/paper_shallow.yaml")
frame = io.read_map(cfg.map)
cols = io.resolve(frame, cfg.map)
labelled = classify.apply(frame, cfg, cols)
table = aggregate.class_table(labelled, cfg.phase_names, cols["modulus"])
voigt, hill, reuss = mixtures.bounds(table.fraction, table["median"])
```

### What the tool will not do

It will not read an instrument file. It starts from an exported per-indent table, which is the reproduction boundary of this deposit.

It contains no science that is not in the paper. There is no classifier, bound or statistic here beyond the ones reported, and adding one is not a configuration change.

---

## Checking the deposit

```
python tests/test_deposit.py
python tests/test_tool.py
```

| file | what it checks |
|---|---|
| `test_deposit.py` | This README and the data dictionary against the actual files: row counts, columns in both directions, the script and data inventories, and that every third-party import is in `requirements.txt`. |
| `test_tool.py` | That the tool runs end to end on a second dataset, that `examples/paper_shallow.yaml` still reproduces the published bounds, and that the config and the scripts still agree on every threshold. |
| `make_synthetic_map.py` | Writes the second dataset. It is synthetic and is not a measurement; see below. |

The second dataset is generated, not measured. `make_synthetic_map.py` draws three populations from fixed distributions with a stated seed. `test_tool.py` writes it to a temporary directory that does not outlive the test, so no synthetic file is ever left sitting in the deposit; run the generator by hand and it writes to `tests/_generated/` instead, which is not committed. It has different column names, three phases and one zone, so it fails if anything in the tool has this laminate baked into it. Because the populations are planted, `test_tool.py` can check that the tool recovers what was put in: it labels 99.8 % of the points with the phase they were drawn from, the rest being the tails that genuinely cross a cut.

The first thing this test found, on the first run, was that a single-zone configuration crashed.

---

## Citation

If you use this, please cite the paper and this deposit.

Deposit DOI: **10.5281/zenodo.22283997**. The DOI is reserved: it is permanent, and it resolves once the record is published on acceptance of the paper.

---

## Licence

The code in `scripts/` and `tests/` is under the MIT licence: `LICENSE`. That
file holds the licence text and nothing else, so GitHub reports the repository
as MIT; anything appended to it makes the detection fall back to "Other".

The data in `data/` is under CC BY 4.0: `data/LICENSE`.

Copyright (c) 2026 Edoardo Rossi, Università degli Studi Roma Tre. Use either
freely, with attribution; `CITATION.cff` has the machine-readable form.

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
