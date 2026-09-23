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

Expected last line: `276 of 276 values reproduce`.

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

Eight CSVs and thirteen JSON summaries. The first three are the exported per-indent arrays and are primary inputs to this deposit, the next two carry the classification used in the paper, two cover the mechanical testing, and three hold the backscattered-electron segmentation behind Figure 2 and its validation. The rest are the summaries the later analyses write; `data/DATA_DICTIONARY.md` documents every key of every one.

| file | rows | what it is |
|---|---|---|
| `shallow_map.csv` | 8553 | The load-controlled high-speed map, one row per indent. Every indent in the acquisition, including the 3000 that sit on the embedding resin. |
| `depth_resolved_curves.csv` | 101515 | The CSM map as curves. Each of the 2000 indents is resampled onto a common 5 nm depth grid from 50 to 300 nm, so it is one tidy table instead of 2000 ragged ones. |
| `depth_resolved_indents.csv` | 2000 | The same map reduced to slice values at 100, 150 and 200 nm, read from the instrument trace. **These are the numbers the paper reports**; see *Slice extraction* below. |
| `classified_shallow.csv` | 5553 | The shallow map after the resin cut, carrying the zone and the constituent label. |
| `classified_depth_resolved.csv` | 2000 | The CSM map with slice values, zone, constituent label, quality flag and the k-means assignment. |
| `flexural_curves.csv` | 8371 | Three-point bending, stress against strain, **five specimens per configuration**. |
| `flexural_moduli.csv` | 10 | Per-specimen chord moduli, raw and after ASTM D790 Annex A1.1 toe compensation, plus the modulus over the 0.28-0.43 % window. |
| `bse_segmentation_tiles.csv` | 91 | One row per imaged tile: zone pixel counts and the fibre/matrix split inside each zone. |
| `bse_segmentation_summary.json` | - | The constituent volume fractions of Figure 2e, with the aggregation stated. |
| `mixed_class_deconvolution.json` | - | The mixed-volume class inverted back onto its end members, both acquisitions, with the independent comparison. |
| `spatial_localisation.json` | - | The neighbourhood statistics behind the claim that mixed-volume contacts sit at the phase boundaries. |
| `uncertainty_budget.json` | - | The uncertainty on the estimate, term by term, from the spatial block bootstrap and the zone-share sweep. |
| `zone_mask_rle.json` | - | The architectural zone of every mosaic pixel, run-length encoded. |
| `zone_share_sensitivity.json` | - | How much of the section each zone occupies, and how that depends on the window you look through. |
| `basalt_input_sweep.json` | - | The basalt input under every classification cut in the swept range, both acquisitions. |
| `flexural_comparisons.json` | - | Every planned bending comparison, with the Holm correction under three family groupings. |
| `phase_angle_sensitivity.json` | - | Supplementary Table S3: what moving the phase-angle quality limit does to the rejected count, to the basalt input and to the estimate built on it. |
| `bse_validation_metrics.json` | - | Leave-one-field-out validation of the segmentation against the operator's own labels: the confusion matrix, and precision and recall per class. Behind Supplementary Table S6. |
| `spacing_compliance.json` | - | Contact depth per indent against the Phani-Oliver spacing criterion. |
| `poisson_sensitivity.json` | - | Supplementary Table S1: the per-class medians under each Poisson convention, and what they do to the estimate. |
| `PROVENANCE.json` | - | The manifest: every file declared a measurement or a product, with the producing script and a hash. |

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
`depth_resolved_indents.csv` carries `x_um` and `y_um` on the same convention
as `classified_depth_resolved.csv`, which is derived from it. In the
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
| flax face in tension | 19.66 ± 0.49 GPa | 18.48 GPa |

The ranking reverses across the crossover, which is why both windows are reported. `reproduce_paper_values.py` checks these together with the Welch statistic, Hedges *g*, the peak tangent modulus, the strength, the softening and the Holm correction under each of the three families in `flexural_comparisons.json`.

One caveat on the flax standard deviation: 0.49 is what this file gives, and 0.4842 is what the unrounded per-specimen values give. See the note on precision below.

---

## What is in `scripts/`

Twenty-nine files, in three groups, plus the `hybridmap/` package and the example configs described under *Running this on your own map*. They depend on numpy, pandas, scipy, scikit-learn, matplotlib and PyYAML, and on nothing else. Only the last is needed for the tool rather than for the reproduction.

**Start here.**

| file | what it does |
|---|---|
| `reproduce_paper_values.py` | Runs the analysis and prints all 276 published values next to the recomputed ones. This is the one to run first. |

**Analysis.**

| file | what it does |
|---|---|
| `classify.py` | The rule-based classification: spatial zones plus modulus thresholds. |
| `rule_of_mixtures.py` | Voigt, Reuss and Hill bounds, threshold sweep, input substitutions. |
| `clustering_check.py` | k-means and Gaussian-mixture cross-check, ARI against the rule. |
| `input_decomposition.py` | Separates the effect of the constituent moduli from that of the volume fractions in the comparison between the tabulated and the measured construction. |
| `uncertainty_budget.py` | The uncertainty on the laminate estimate, term by term: the spatial block bootstrap over the indents, the tile bootstrap over the zone share, and the two bounded analysis choices. Writes `data/uncertainty_budget.json`. |
| `phase_angle_sensitivity.py` | Supplementary Table S3: sweeps the phase-angle quality limit at 10, 15 and 20 degrees and reports the rejected count, the depth-resolved basalt median and the estimate built on it. Writes `data/phase_angle_sensitivity.json`. |
| `spacing_compliance.py` | Contact depth per indent from load and hardness, against the Phani–Oliver spacing criterion. Writes `data/spacing_compliance.json`. |
| `border_effect.py` | Whether the spacing shortfall leaves a trace: indents on the outer row of the short axis against the interior indents of the same class in the same zone, with a Holm correction over the five comparisons. |
| `border_rejection.py` | Whether the quality rejections of the depth-resolved map are an edge artefact: the rejected fraction on the two outer rows of the short axis against the fraction inside them, with the counts, and the same split by architectural band. |
| `make_flexural_moduli.py` | Chord moduli from the bending curves. Reproduces `data/flexural_moduli.csv` exactly. |
| `table2_intervals.py` | The per-class confidence intervals of Table 3, by the construction its caption declares. The file name is from the round in which that table was numbered 2. |
| `poisson_sensitivity.py` | Supplementary Table S1: each class median under the uniform Poisson ratio and under a phase-specific one, by inverting the Oliver-Pharr relation on the deposited reduced modulus. Writes `data/poisson_sensitivity.json`. |
| `mixed_class_deconvolution.py` | Inverts every indent in the basalt-rich zone onto the basalt/matrix tie line, under parallel and series mixing, and compares the two acquisitions with each other and with the imaging. Writes `data/mixed_class_deconvolution.json`. |
| `make_classified_depth.py` | Derives `classified_depth_resolved.csv` from `depth_resolved_indents.csv`: the zone bands, the modulus rule, the phase-angle quality flag and the cluster index. Before it existed, that file was both a record and a product, which is why nothing could regenerate it. |
| `zone_share.py` | The share of the section each zone occupies, from the mosaic, the two-level construction built on it, and the classification-threshold grid of Supplementary Table S2. Writes `data/zone_share_sensitivity.json`. |
| `basalt_input_sweep.py` | The basalt input under every classification cut in the swept range. Writes `data/basalt_input_sweep.json`. |
| `flexural_comparisons.py` | Every planned bending comparison, with the Holm correction under three family groupings. Writes `data/flexural_comparisons.json`. |
| `provenance_check.py` | Declares every file in `data/` a measurement or a product, checks the products against their scripts, and records `data/PROVENANCE.json`. |

**Figures.** Each writes the figure it is named for, from the deposited data only.

| file | figure |
|---|---|
| `figstyle.py` | Shared style: sizes, fonts, the constituent colours, the layout check and the save helper. Imported by the rest, not run on its own. |
| `fig05_curves_per_class.py` | Figure 5, per-class loading curves and depth signatures |
| `fig06_classification.py` | Figure 6, classification panels |
| `fig07_property_relationships.py` | Figure 7, relationships between the measured properties |
| `fig08_rom.py` | Figure 8, the rule-of-mixtures construction |
| `fig09_flexural.py` | Figure 9, three-point bending |
| `figs_supplementary.py` | Supplementary Figures S3, S4, S5 and S6, and the spatial statistics of §4.1. Writes `data/spatial_localisation.json`. |
| `figS7_basalt_depth.py` | Supplementary Figure S7: the stiff contacts of the basalt-centre zone followed from 50 to 300 nm, selected at 80 nm so the behaviour below is read and not chosen. |
| `fig_map_coverage.py`, `reply_figures.py` | Figures supporting the response to reviewers rather than the paper |

Every figure listed above is byte-identical to the picture in the paper. That
is checked rather than asserted, by decoding both and comparing pixels, and it
is how five figures were caught still showing the counts the analysis had
before it was redone.

**What the deposit does not draw, and why.** Seven pictures in the two documents
come from no script here.

| picture | why |
|---|---|
| Figure 1 | the production scheme: a drawing, not a plot |
| Figure 2 | the backscattered-electron mosaic: a micrograph, drawn from the tile images rather than from any deposited table |
| Figures 3 and 4 | the two property maps, laid out against the micrograph by hand. Every value they show is a column of `classified_shallow.csv` and `classified_depth_resolved.csv`, so they can be redrawn from what is here, but the layout was not scripted |
| Supplementary Figure S1 | secondary-electron micrographs of the constituents |
| Supplementary Figure S2 | where the two maps sit in the laminate: a schematic |
| the letterhead | a logo |

For the micrographs, what the deposit carries is the measurement rather than the
picture: `bse_segmentation_tiles.csv` and `bse_segmentation_summary.json`, with
the procedure written out under *Files provided as primary inputs*.

The analysis scripts each run on their own and print something worth reading:

```
python classify.py            # reprints the class counts, checks them against the deposit
python rule_of_mixtures.py    # per-class table, zone-weighting check, sensitivity
python zone_share.py          # the zone share, the estimate built on it, the sliding window
python clustering_check.py    # ARI across three representations, stability, seed sweep
python uncertainty_budget.py  # the +/- on the estimate, by component
python phase_angle_sensitivity.py  # what the quality limit does and does not move
python spacing_compliance.py       # how many indents meet the spacing criterion
python border_rejection.py         # whether the rejections crowd the edge of the map
python mixed_class_deconvolution.py  # the class inverted onto its end members
python figs_supplementary.py         # Supplementary Figures S3, S4, S5, S6 and their statistics
```

### Where the per-indent tables come from

`shallow_map.csv`, `depth_resolved_curves.csv` and `depth_resolved_indents.csv` are **primary inputs to this deposit**. They are the per-indent arrays exported from the two nanoindentation acquisitions, and everything else here is computed from them by the scripts in `scripts/`.

The instrument files they were exported from are not part of this deposit and are not distributed. The acquisition and the proprietary export that produces these arrays are the instrument's, not the analysis's, and nothing in the paper depends on them: every reported number is recomputed from the tables above by `reproduce_paper_values.py`. The three files are complete, in the sense that no indent has been removed from them: `shallow_map.csv` still carries the ~3000 indents that landed on the embedding resin, and `depth_resolved_curves.csv` is unfiltered, artefacts and all.

So the reproduction boundary of this deposit is the exported per-indent table. Below that line is the instrument. Above it, everything is here.

---

## Reproducing specific things

**Class medians and counts (Table 3).** `classify.py` rebuilds the labels from `shallow_map.csv` and checks them against `classified_shallow.csv`. Agreement is 100 %.

**The laminate estimate (Section 3.4).** `zone_share.py` builds it the way the paper does, in two levels: the class fractions and class medians inside each architectural zone, and the share the two zones occupy on the imaged section, 36.2 % basalt-rich. That gives 18.50 GPa, with a Hill average of 12.67 and a Reuss value of 6.85. Weighting the two zones by indent counts instead, which describes the window that was mapped rather than the section, gives 21.33 GPa; the paper reports that figure as the comparison it is. `rule_of_mixtures.py` holds the arithmetic and verifies that within-zone weighting and global counting give the same class fractions, which the paper asserts and which is worth checking rather than believing.

**Threshold sensitivity (Supplementary Table S2).** `threshold_grid()` in `zone_share.py`. The reference construction at 15 and 70 GPa gives 18.50 GPa; the fifteen threshold pairs span 18.00 to 18.84 GPa, largest deviation 0.50 GPa. `threshold_sweep()` in `rule_of_mixtures.py` is the same sweep without the zone weighting, which is what the command-line tool runs on a single map; it spans 20.83 to 21.44 GPa about 21.33.

**The unsupervised cross-check (Section 3.3).** `clustering_check.py`. ARI is 0.408 against the rule and 0.652 once the two compliant clusters are merged. Across the five seeds it runs, four give 0.4080 and one gives 0.4092; ARI\* is 0.652 on all five. Refitting on twenty random 90 % subsamples gives 0.979 ± 0.016 against the full fit. The comparison is also run on three principal components, 0.410 and 0.653, and on a full-covariance Gaussian mixture, 0.352 and 0.713.

One thing to be clear about: the clustering does **not** independently select four populations. The silhouette peaks at k = 2, and the Gaussian-mixture BIC falls steeply to k = 7 and turns up at k = 8, with no feature at k = 4. Four is chosen on architectural grounds, and the clustering tells us that a partition built with no architectural knowledge lands somewhere compatible with the rule. ARI is chance-corrected, so 0.408 should be read against an expectation of 0 for unrelated partitions.

---

## Slice extraction

Slice values at 100, 150 and 200 nm are the mean of the CSM response over a 20 nm window centred on the target depth, requiring at least three valid points inside the window.

There is one of them in the deposit, and there used to be two. `E_150` in `depth_resolved_indents.csv` is taken from the instrument trace as recorded; `classified_depth_resolved.csv` carries the same column, value for value, and it is what the paper reports. The second convention, a window mean recomputed over the resampled 5 nm grid, was deposited as `depth_resolved_slices.csv` and has been removed: two files holding one quantity under two conventions is how a quality mask that matched neither survived two rounds of review, because whichever number was checked the other was available to explain it away.

The resampled route is still open to anyone who wants it. `depth_resolved_curves.csv` holds the full traces, so the window mean can be recomputed from them. Expect the two compliant classes to agree and the two stiff classes to land about 2 % low, which is where E(z) is steepest; the two routes correlate at 0.9997.

The estimate the paper reports does not depend on any of this: its basalt input is the median of the load-controlled map, which carries no trace and no slices. The convention reaches only the alternative input of Supplementary Table S4, and a 2 % shift in the depth-resolved basalt median moves the estimate built on that input by less than a tenth of a GPa.

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
| `examples/paper_shallow.yaml` | This paper's shallow map. Reproduces the published class counts and class medians, and the single-map Voigt, Hill and Reuss values, from the config alone. |
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

`python -m hybridmap sweep` redoes the whole classification for every combination of cut values you list and prints the bounds for each. It reports the departure from your configured cuts, which is the number to quote. On this paper's load-controlled map the tool weights every class by indent count over the whole map, and the fifteen threshold pairs then span 20.83 to 21.44 GPa about 21.33, a largest departure of 0.50 GPa. The published Supplementary Table S2 is the same sweep carried through the two-level construction, 18.00 to 18.84 GPa about 18.50, a largest departure of the same 0.50 GPa. The zone weighting is a property of this laminate rather than of the tool, so it lives in `zone_share.py` and not in a config key.

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
| `test_tool.py` | That the tool runs end to end on a second dataset, that `examples/paper_shallow.yaml` still reproduces the single-map Voigt, Hill and Reuss values, and that the config and the scripts still agree on every threshold. |
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
which is the precision at which they are reported.

The rounded and the unrounded values differ in the third decimal, and for one
quantity that changes the second. Recomputing the flax-longitudinal standard
deviation from this file gives 0.4863 GPa, which rounds to the 0.49 the paper
reports; from the unrounded values it is 0.4842, which would round to 0.48.
Nothing is wrong with either number, and the paper prints the one this file
supports so that the deposit and the page agree. A reader recomputing by hand
should still know why the last digit can move.

## Files provided as primary inputs

Most of what is here can be rebuilt from something else in the folder. Six files
cannot, and it is better to say so than to leave a reader hunting for a script
that is not there. `data/PROVENANCE.json` says which is which for every file, and
`provenance_check.py` holds the products to their scripts.

### `shallow_map.csv`, `depth_resolved_curves.csv`, `depth_resolved_indents.csv`

The exported per-indent arrays, described under *Where the per-indent tables
come from* above. They are the starting point of everything else here.

The three are close relatives but not derivable from one another.
`depth_resolved_indents.csv` holds the 20 nm window mean at each target depth as
the instrument recorded the trace; `depth_resolved_curves.csv` holds the same
traces resampled onto a common 5 nm grid, which is what makes them one tidy
table. Recomputing the slice values from the resampled grid gives numbers that
correlate at 0.9997 with the deposited ones and land about 2 % low on the two
stiff classes. The paper uses the deposited ones.

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

### `bse_validation_metrics.json`

The leave-one-field-out validation of that segmentation against the operator's
own labels, field by field: the confusion matrix, and precision and recall per
class, per zone. It is behind Supplementary Table S6 and behind the per-class
figures quoted in Section 3.4.

It is a **primary input** for the same reason as the tile file: recomputing it
needs the tile images and the hand-painted fields, which are several gigabytes
and stay with the corresponding author.

Two denominators are in the file and both are worth reading. Over the two solid
phases, which is the denominator the reported volume fractions use, the
classifier separates them at 96.0 % in the basalt zone and 97.9 % in the flax
zone. With void included as a third class, which no reported fraction uses as a
denominator, the same fields give 82.8 % and 88.1 %. Void is where the confusion
sits, and Supplementary Table S6 gives the per-class precision and recall behind
both figures.

### `flexural_moduli.csv`

Derived from `flexural_curves.csv` by the ASTM D790 chord fit with Annex A1.1
toe compensation. See `make_flexural_moduli.py` for the exact convention,
including which linear region is taken as the steepest.
