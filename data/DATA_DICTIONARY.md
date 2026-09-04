# Data dictionary

Eleven files: eight CSVs and three JSON summaries. Every column, its unit, its
range, and what a blank cell means.

Two conventions hold throughout unless a column says otherwise:

- **Units are in the column name**: `_um`, `_nm`, `_GPa`, `_MPa`, `_mN`,
  `_pct`, `_deg`, `_N_per_m`.
- **A blank cell is an instrument or fit failure on that indent**, not a zero
  and not a censored value. Blanks are left in place rather than dropped so
  that row counts stay comparable across files; every analysis script filters
  them explicitly.

## Which axis is through the thickness

The two acquisitions name the same physical direction differently. In both, the
map's **long** axis runs through the laminate thickness; only the column letter
differs.

| file | through-thickness | in-plane |
|---|---|---|
| `shallow_map.csv`, `classified_shallow.csv` | `x_um` | `y_um` |
| `classified_depth_resolved.csv` | `y_um` | `x_um` |

In the backscattered-electron mosaic of Figure 2 the thickness direction is
**vertical**, rotated ninety degrees from the property maps of Figures 3 to 5.

---

## `shallow_map.csv` - 8553 rows

Every indent of the fixed-depth high-speed acquisition, including the ~3000 that
landed on the embedding resin.

| column | unit | range | notes |
|---|---|---|---|
| `indent_id` | - | 1–8553 | unique, and the key to `classified_shallow.csv` |
| `row`, `col` | - | 1–40, 1–214 | position in the acquisition grid |
| `x_um` | µm | 0–1086 | **through the thickness**; resin below 380 µm |
| `y_um` | µm | 0–97.5 | in plane |
| `peak_load_mN` | mN | 0.45–3.15 | nominally 0.53; the spread is the load-control tolerance |
| `contact_depth_nm` | nm | −401 to 21174 | negative and very large values are failed contacts, 1 blank |
| `max_depth_nm` | nm | −353 to 21264 | as above |
| `stiffness_N_per_m` | N m⁻¹ | 2792–53160 | 1 blank |
| `Er_GPa` | GPa | 0.03–145 | reduced modulus, Oliver–Pharr, 2 blanks |
| `E_GPa` | GPa | 0.03–152 | indentation modulus at ν = 0.30, 2 blanks |
| `H_GPa` | GPa | 0–13.3 | hardness, 2 blanks |
| `S2_over_P_GPa` | GPa | 16–5654 | S²/P, a load-independent stiffness measure |

## `classified_shallow.csv` - 5553 rows

The same map after the resin cut at x ≥ 380 µm, with labels.
Reproducible: `classify.py` regenerates the labels at **100 %**.

Columns as above, plus:

| column | values | notes |
|---|---|---|
| `zone` | `basalt`, `flax` | architectural half, from the x position |
| `constituent` | `flax`, `matrix`, `mixed-volume`, `basalt`, `QC/invalid` | three stiffness levels, with the compliant one split into flax and matrix by position |

`y_um` runs −17.5 to 80.0 here: the origin was re-zeroed on the retained
indents, so it is offset by −17.5 µm from `shallow_map.csv`. The spacing is
unchanged.

## `depth_resolved_curves.csv` - 101,515 rows

The CSM acquisition as curves: 2000 indents resampled onto a common 5 nm grid
from 50 to 300 nm, roughly 51 points each.

| column | unit | range | notes |
|---|---|---|---|
| `indent_id` | - | 1–2000 | |
| `depth_nm` | nm | 50–300 | the resampled grid |
| `load_mN` | mN | 0–14.4 | |
| `stiffness_N_per_m` | N m⁻¹ | −1198 to 137900 | 30 blanks; negatives are fit failures near the surface |
| `E_GPa`, `H_GPa` | GPa | −218 to 267, −0.01 to 28 | unfiltered, so early-depth artefacts are present and negative values occur |
| `phase_angle_deg` | ° | −90 to 88 | |
| `S2_over_P_GPa` | GPa | −1738 to 2.1×10⁶ | |

**Nothing here is quality-filtered.** Use `qc_flag_strict` in
`classified_depth_resolved.csv` to reject the 433 indents the paper excludes.

## `depth_resolved_slices.csv` - 2000 rows

The same acquisition reduced to three depths. Each value is the mean of the CSM
response over a **± 10 nm window** centred on the target depth, per Supplementary
Note S1.

There are **no position columns here**. Join on `indent_id` to
`classified_depth_resolved.csv` if you need coordinates or labels.

| column | unit | blanks |
|---|---|---|
| `indent_id` | - | key to `classified_depth_resolved.csv`, 1–2000 |
| `E_100nm_GPa`, `H_100nm_GPa`, `PA_100nm_deg` | GPa, GPa, ° | 9 |
| `E_150nm_GPa`, `H_150nm_GPa`, `PA_150nm_deg` | GPa, GPa, ° | 15 |
| `E_200nm_GPa`, `H_200nm_GPa`, `PA_200nm_deg` | GPa, GPa, ° | 11 |

**These are not the values the paper reports.** See the note below.

## `classified_depth_resolved.csv` - 2000 rows

The depth-resolved map with zones, labels and the clustering result. **This is
the file the paper's depth-resolved numbers come from.**

| column | unit | range | notes |
|---|---|---|---|
| `indent_id` | - | 1–2000 | |
| `x_um` | µm | 0–57 | in plane |
| `y_um` | µm | 0–297 | **through the thickness** |
| `E_100`, `E_150`, `E_200` | GPa | 0.02–106 | 19, 19 and 12 blanks |
| `H_100`, `H_150`, `H_200` | GPa | 0.004–5.7 | 107, 61 and 46 blanks |
| `PA_100`, `PA_150`, `PA_200` | ° | 0.01–74.6 | 17, 46 and 26 blanks |
| `SRS` | dimensionless | −0.218 to 0.430 | see below |
| `qc_flag_strict` | 0 or 1 | | 1 marks the **433** indents rejected on the strict near-surface phase-angle criterion |
| `zone` | `basalt-centre`, `flax-bottom`, `flax-top` | | the three architectural bands along y |
| `constituent` | as above | | |
| `kmeans_k4` | −1 to 2 | | unsupervised cluster index; **−1 means not assigned**, for indents excluded by QC |

### `E_150` and `E_150nm_GPa` are not the same numbers

They are the same *quantity* computed by two passes of the pipeline. `E_150` is
taken from the raw instrument trace; `E_150nm_GPa` is the ± 10 nm window mean
over the resampled 5 nm grid. They correlate at **0.9997** and differ by a
median of 0.02 GPa.

The difference is largest where E(z) is steepest:

| class | difference in the class median |
|---|---|
| flax, matrix | **0.00 GPa** |
| mixed-volume | 0.52 |
| pure-axial basalt | 0.34 |

**The paper uses `E_150`.** Carried through the Voigt sum the basalt difference
is worth 0.02 GPa on a bound of 21.33, so the choice does not affect any
reported result, but a reader recomputing from the slices file will see the
third decimal move and should know why.

### `SRS`

The **strain-rate sensitivity index**: the exponent *m* in E ∝ (strain rate)^*m*,
fitted from the CSM records. Dimensionless.

It separates the constituents by chemistry rather than by architecture:

| class | median |
|---|---|
| pure-axial basalt | 0.019 |
| mixed-volume | 0.025 |
| PA-11 matrix | 0.058 |
| flax | 0.073 |

Near zero on the mineral fibre, several times higher on the two organic
constituents, as viscoelasticity predicts. About 1 % of values are slightly
negative: noise on near-rate-independent contacts.

## `flexural_curves.csv` - 8371 rows

Three-point bending, five specimens per configuration.

| column | unit | notes |
|---|---|---|
| `specimen_id` | - | e.g. `16Hybr_Bas_Long_1` |
| `configuration` | - | `basalt_face` or `flax_face`, the yarn aligned with the load |
| `strain_pct` | % | raw, **not** toe-compensated |
| `stress_MPa` | MPa | |

## `flexural_moduli.csv` - 10 rows

Derived from the curves by `make_flexural_moduli.py`, which reproduces this file
exactly.

| column | unit | notes |
|---|---|---|
| `specimen_id` | - | matches `flexural_curves.csv`, e.g. `16Hybr_Bas_Long_1` |
| `configuration` | - | `basalt_face` or `flax_face`, as in the curves file |
| `toe_intercept_pct` | % strain | zero-stress intercept of the steepest linear region, found with a sliding window **0.15 % of strain wide** |
| `E_D790_raw_GPa` | GPa | chord over 0.05–0.25 % on the raw strain axis |
| `E_D790_toe_compensated_GPa` | GPa | the same chord after subtracting the toe intercept. **This is the modulus the paper reports** |
| `E_window_0p28_0p43_GPa` | GPa | chord over 0.28–0.43 %, **on the raw axis**, deliberately: the 0.28 % crossover was located there, so the window reporting the ranking above it sits on the same axis |

All three chords are **least-squares fits** through the points inside the
interval, not secants between the endpoints. The curve is slightly concave, so a
secant reads low by up to 0.39 GPa.

## `bse_segmentation_tiles.csv` - 91 rows

Per-tile statistics from the backscattered-electron mosaic. **A primary input**:
rebuilding it needs the tile images, which are several gigabytes and are not
deposited. See the README for how the zones were marked.

| column | unit | notes |
|---|---|---|
| `file` | - | source tile |
| `n_px` | pixels | tile size, constant at 1,005,732 |
| `n_basaltzone`, `n_flaxzone` | pixels | area of each marked tow zone in that tile. A tile lying wholly in one zone has 0 for the other |
| `basaltzone_fib`, `basaltzone_mat` | fraction | fibre and matrix share **within** the basalt zone; sum to 1. **9 blanks**, the tiles with no basalt zone |
| `flaxzone_fib`, `flaxzone_mat` | fraction | as above for the flax zone. **1 blank** |

Pixels not in either zone are the diffuse interface band, 4.0 % of the mosaic,
excluded from the fractions.

## `bse_segmentation_summary.json`

The aggregation of the above into the volume fractions of Figure 2e. Zone areas
are taken from the stitched mask and renormalised over the marked zones only,
then multiplied by the pooled in-zone fibre fraction. **Reproducible** from the
tile file.

## `uncertainty_budget.json`

The ± 0.40 GPa on the Voigt bound: class medians drawn from bootstrap
replicates, class fractions from a Dirichlet with the observed counts. Written
by `uncertainty_budget.py`.

The budget covers the two quantities that are random under resampling. It
**excludes** the classification thresholds and the basalt input, which are
analysis choices reported as sensitivities in Supplementary Tables S1 and S2,
and cross-section representativeness, which one section cannot bound.

---

## A note on precision

The flexural moduli are stored rounded to two decimals; the summary statistics
in the paper were computed at full precision. Recomputing the flax-longitudinal
standard deviation from this file gives 0.4863, which rounds to 0.49; at full
precision it is 0.4842, which rounds to the 0.48 the paper reports. Neither is
wrong.

## `phase_angle_sensitivity.json`

Written by `scripts/phase_angle_sensitivity.py`, and the source of
Supplementary Table S7.

| key | meaning |
|---|---|
| `reference_voigt_GPa` | the Voigt bound from the fixed-depth inputs, 21.33 GPa. The phase-angle criterion cannot move it: the fixed-depth map has no phase-angle channel. |
| `note` | that fact, stated in the file so it travels with the numbers |
| `rows` | one entry per threshold. `phase_angle_limit_deg` is the criterion, `rejected` and `rejected_pct` the indents it discards, `E_<class>_GPa` the resulting depth-resolved medians, `alternative_basalt_input_GPa` the basalt median, and `Voigt_with_that_input_GPa` the bound that input gives when substituted into the reference table. |

The first row is the published flag, computed on the full continuous-stiffness
trace, which rejects 433 indents. The three below recompute the same criterion
from the slice values held here, which rejects 320 at the same threshold: the
difference is indents whose phase angle crosses the limit between the slices.
The sweep is internally consistent and is not an attempt to reproduce the
published count.
