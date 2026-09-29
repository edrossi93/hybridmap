# Data dictionary

Twenty-three files: eight CSVs and fifteen JSON summaries. Every column, its unit, its
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

Every indent of the load-controlled high-speed acquisition, including the ~3000 that
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
| `S2_over_P_GPa` | GPa | 16–5654 | the exported S²/P channel, which does not pass through the area function; equal to (4/π) Er²/H of the two columns above to four decimals, and not to the square of `stiffness_N_per_m` over the load (see the note under `depth_resolved_curves.csv`) |

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
| `depth_nm` | nm | 50–300 | the resampled grid of indenter displacement z, the instrument's depth channel after frame-compliance and drift correction; the contact depth is z − 0.75 P/S |
| `load_mN` | mN | 0–14.4 | |
| `stiffness_N_per_m` | N m⁻¹ | −1198 to 137900 | the exported stiffness channel; 30 blanks; negatives are fit failures near the surface. See the note on the two stiffness records below |
| `E_GPa`, `H_GPa` | GPa | −236 to 290, −0.01 to 28 | E is the sample modulus at ν = 0.30 from the reduced modulus, with E_i = 1140 GPa and ν_i = 0.07; unfiltered, so early-depth artefacts are present and negative values occur |
| `phase_angle_deg` | ° | −90 to 88 | |
| `S2_over_P_GPa` | GPa | −1738 to 2.1×10⁶ | the exported S²/P channel, which does not pass through the area function. See the note below |

**Two stiffness records.** `stiffness_N_per_m` and `S2_over_P_GPa` are both channels of the export, and they do not agree with each other. The S²/P channel agrees with the exported modulus and hardness through the identity S²/P = (4/π) Er²/H, to four decimals on every indent of `shallow_map.csv` (there the same relation holds between the same three columns), so it carries the stiffness the instrument used for the modulus. The stiffness channel squared over the load runs above it in this file, by 13 % in the median with no trend in depth, and on the stiff contacts of §4.4 it falls from above it at 50 nm to about equal at 300 nm; in `shallow_map.csv` the same quotient, on the peak load, runs 6 % below the S²/P channel. The step the software takes between the two is not in the export and is not guessed at here. Everything the paper prints from a stiffness, the S²/P medians of §4.4 and the contact stiffness quoted beside them, is read off the S²/P channel, the stiffness being taken as the square root of S²/P times the load; `scripts/depth_softening.py` computes it that way and writes the stiffness channel's own figures beside it, unprinted, in `depth_softening.json`.

**Nothing here is quality-filtered.** Use `qc_flag_strict` in
`classified_depth_resolved.csv` to reject the 304 indents the paper excludes.

## `depth_resolved_indents.csv` - 2000 rows

**The primary record of the depth-resolved acquisition.** One row per indent,
carrying its position and the slice values at 100, 150 and 200 nm. Each value
is the mean of the CSM response over a **20 nm window** centred on the target
depth, requiring at least three valid points inside the window, per
Supplementary Note S1.

The elastic modulus is the sample modulus at ν = 0.30, computed from the
instrument's reduced modulus with the diamond indenter constants, E_i = 1140 GPa
and ν_i = 0.07, as in `shallow_map.csv`.

Nothing in the deposit produces this file; everything else about the
depth-resolved map is produced from it. `make_classified_depth.py` reads it and
writes `classified_depth_resolved.csv`, which is this file plus the zone, the
class, the quality flag and the cluster index. It is not regenerable from
`depth_resolved_curves.csv` either: the window mean recomputed over that file's
resampled 5 nm grid reproduces `E_150` to a median of 0.013 GPa and not
exactly, the largest difference being 2.5 GPa. The README says more under
*Slice extraction*.

| column | unit | range | blanks |
|---|---|---|---|
| `indent_id` | - | 1-2000 | key to `classified_depth_resolved.csv` |
| `x_um` | µm | 0-57 | in plane |
| `y_um` | µm | 0-297 | **through the thickness** |
| `E_100`, `H_100`, `PA_100` | GPa, GPa, ° | | 9 each |
| `E_150`, `H_150`, `PA_150` | GPa, GPa, ° | | 15 each |
| `E_200`, `H_200`, `PA_200` | GPa, GPa, ° | | 11 each |
| `SRS` | dimensionless | -0.218 to 0.430 | 3 |
| `qc_flag_instrument` | 0 or 1 | | 1 marks the 433 indents the instrument's own quality flag rejects, computed at acquisition on the full continuous-stiffness trace; no rule applied to the deposited phase-angle values reproduces that set. Supplementary Table S3, row 1, applies it | none |

A blank is an indent whose trace had no valid window at that depth. The three
depths fail together on an indent whose contact never established, which is why
the counts come in threes.

## `classified_depth_resolved.csv` - 2000 rows

The depth-resolved map with zones, labels and the clustering result. **This is
the file the paper's depth-resolved numbers come from.**

| column | unit | range | notes |
|---|---|---|---|
| `indent_id` | - | 1–2000 | |
| `x_um` | µm | 0–57 | in plane |
| `y_um` | µm | 0–297 | **through the thickness** |
| `E_100`, `E_150`, `E_200` | GPa | −0.14 to 187 | ν = 0.30; 9, 15 and 11 blanks |
| `H_100`, `H_150`, `H_200` | GPa | −0.001 to 25.6 | 9, 15 and 11 blanks |
| `PA_100`, `PA_150`, `PA_200` | ° | −46.6 to 76.8 | 9, 15 and 11 blanks |
| `SRS` | dimensionless | −0.218 to 0.430 | see below |
| `qc_flag_strict` | 0 or 1 | | 1 marks the **304** indents rejected by the phase-angle criterion: the corrected phase angle exceeds 15° in magnitude at one of the three depths, or that depth has no valid window |
| `zone` | `basalt-centre`, `flax-bottom`, `flax-top` | | the three architectural bands along y |
| `constituent` | as above | | |
| `kmeans_k4` | −1 to 3 | | unsupervised cluster index from the k = 4 run of `clustering_check.py`, ordered by median modulus so that 0 is the most compliant cluster; **−1 on the 304 indents the phase-angle criterion rejects**, and nowhere else. Rebuilt by `clustering_check.py` on the regenerated classification: the submitted column held three labels under a name that promises four, and a quality mask that matched neither `qc_flag_strict` nor the paper |

### `SRS`

The **strain-rate sensitivity index**: the exponent *m* in E ∝ (strain rate)^*m*,
fitted from the CSM records. Dimensionless.

It separates the constituents by chemistry rather than by architecture:

| class | median |
|---|---|
| pure-axial basalt | 0.021 |
| mixed-volume | 0.028 |
| PA-11 matrix | 0.062 |
| flax | 0.073 |

Near zero on the mineral fibre, several times higher on the two organic
constituents, as viscoelasticity predicts. About 1 % of values are slightly
negative: noise on near-rate-independent contacts.

The flax and matrix figures are the two §4.3.2 prints, and they are checked by
`reproduce_paper_values.py`; the flax one also bounds the rate departure of §2.5.

## `flexural_curves.csv` - 7621 rows

Three-point bending, five basalt-longitudinal specimens and four
flax-longitudinal.

Four and not five in the flax configuration. The source workbook carried one of
its flax-longitudinal curves twice: two of its five flax-longitudinal records
were identical point for point over all 750 points of the shorter one, which was
a single final point short of the longer, so they were one specimen under two
names. The shorter copy, `16Hybr_Flax_Long_4`, is
not in this file and takes no part in any statistic computed from it. The
specimen numbering that remains is the workbook's, so the flax specimens are 1,
2, 3 and 5. The basalt configuration carries no such repeat and is five.

| column | unit | notes |
|---|---|---|
| `specimen_id` | - | e.g. `16Hybr_Bas_Long_1` |
| `configuration` | - | `basalt_face` or `flax_face`, the yarn aligned with the load |
| `strain_pct` | % | raw, **not** toe-compensated |
| `stress_MPa` | MPa | |

## `flexural_moduli.csv` - 9 rows

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

The ± 1.4410 GPa on the laminate estimate of 18.4979 GPa, term by term. Written
by `uncertainty_budget.py`, and the source of Supplementary Table S5.

| key | meaning |
|---|---|
| `voigt_point_estimate`, `hill`, `reuss` | the estimate and the two other mixing limits, at the imaged zone share |
| `voigt_at_indent_count_zone_share` | the same sum with the zones weighted by indent counts instead, 21.3330, which describes the window that was mapped |
| `w_basalt_mosaic`, `w_basalt_indent_counts` | the two zone shares behind those two numbers, 0.3620 and 0.4970 |
| `type_a_sd_GPa` | the three sampling terms, each with the other two frozen: `fractions` 1.0800, `zone_share` 0.6114, `medians` 0.3447 |
| `type_a_combined_sd_GPa` | the same three together in one resampling, 1.4189, which is smaller than their sum because they are correlated |
| `type_b_sd_GPa` | the three bounded analysis choices as rectangular distributions: `classification_thresholds` 0.2427, `poisson_ratio` 0.0556 and `runaway_contacts` 0.0367, the last being the fifty-nine flax-zone contacts below 1 GPa that the Supplementary Figure S3 caption argues are failed contacts and the map keeps |
| `type_b_ranges` | the ranges those three are taken over: the threshold grid, 17.998 to 18.839 GPa, the phase-specific Poisson estimate, 18.3051 GPa, and the estimate with the runaway contacts in and out, 18.4979 and 18.6251 GPa |
| `combined_sd_GPa` | the Type A terms and the three Type B terms in quadrature, 1.4410; it was 1.4406 before the third Type B term was added, and prints 1.441 either way |
| `share_of_variance_pct` | each term squared over the combined value squared, which is the last column of Supplementary Table S5: 56.2, 5.7 and 18.0 for the three frozen sampling terms, 96.9 for their joint value, 2.8, 0.15 and 0.06 for the three bounded choices. One term at a time, so they do not sum to 100 |
| `percentile_95_interval_GPa` | 15.776 to 21.341, from the draws rather than from a coverage factor |
| `block_size_sensitivity_sd_GPa` | the joint sampling term alone, Type A, at 2, 4 and 8 grid columns, 1.21, 1.38 and 1.68 GPa, from 1000 draws at each width rather than the 6000 of the budget, so the block width can be seen rather than trusted. With the three Type B terms in quadrature those read 1.23, 1.41 and 1.70 GPa, as they did with two; the script prints both forms |
| `n_draws`, `seed`, `block_columns` | 6000 draws, seed 42, blocks four grid columns wide |
| `excluded` | the one term left out on purpose, and why |

The indent resampling is a **spatial block bootstrap**, not a draw over
individual indents: contacts 2.5 µm apart on fibres 10 to 20 µm across are not
independent samples, so blocks of four grid columns, about 20 µm, are resampled
with replacement inside each zone. The zone share is resampled over the 91
mosaic tiles beside them, in the same pass, so that the correlation between the
three sampling terms is carried rather than assumed away.

The classification thresholds are **in** this budget, as its largest Type B
term, and so are the fifty-nine runaway contacts, as its smallest. What stays out is the choice of basalt input, which is a
discriminated choice on contact mechanics and is reported as the sensitivity of
Supplementary Table S4 instead, and cross-section representativeness, which one
section cannot bound.

---

## The JSON summaries

The twelve files below were added when the analysis they record was added. Each
is written by one script and read by the manuscript or the supplement, except
`zone_mask_rle.json` and `bse_validation_metrics.json`, which are measurements
nothing here produces. None of the twelve is edited by hand.

## `PROVENANCE.json`

Written by `provenance_check.py --rebuild`, which re-runs every product's script
and records the hash only where the file came back byte for byte as the manifest
already holds it. Recording anything else takes `--accept-changes`, so accepting
drift is a decision someone makes rather than something a routine run does. The
manifest holds, for every file in `data/`, whether it is a **measurement**
(nothing here produces it) or a **product** (a named script does), and a SHA-256
of the bytes.

| key | meaning |
|---|---|
| `primary` | file name to SHA-256, for the ten measurements |
| `derived` | file name to SHA-256, for the thirteen products |

Which script writes which product is declared in `provenance_check.py` rather
than repeated in the manifest, so the two cannot disagree about it.

Run `provenance_check.py` to compare the manifest against what is on disk. It
fails if a file has changed without the manifest being re-recorded, which is the
point: a derived table that no longer matches its script is the failure mode
this deposit had.

## `zone_mask_rle.json`

The architectural zone of every pixel of the backscattered-electron mosaic,
run-length encoded so the mask travels as text rather than as an image.

| key | meaning |
|---|---|
| `width`, `height` | mosaic size in pixels |
| `um_per_px` | scale, µm per pixel |
| `classes` | the two zone names, in label order |
| `note` | the label convention, 1 for the flax zone and 2 for the basalt zone with 0 unmarked, and that the mask is laid out on the nominal tile grid |
| `mask_rle` | alternating run lengths, starting on class 0 |
| `provenance` | whose marking it is and when it was exported |

Decode by repeating each class index for its run length and reshaping to
(`height`, `width`). The lengths sum to `width * height`; if they do not, the
file is truncated.

## `zone_share_sensitivity.json`

How much of the section each zone occupies, and how that answer depends on where
you look. This is the term the published construction left out: it weighted the
zones by how many indents landed in each, which describes the window that was
mapped rather than the section.

| key | meaning |
|---|---|
| `w_basalt_indent_counts` | basalt-zone share implied by indent counts, 0.497 |
| `w_basalt_mosaic` | basalt-zone share of the section, 0.362, from the tile counts in `bse_segmentation_tiles.csv`. **This is the one the estimate uses.** |
| `w_basalt_downsampled_mask` | the same marking read from `zone_mask_rle.json`, 0.367. Downsampling redraws the zone boundary and moves the share by half a point; the mask is used only for the sliding window below |
| `w_basalt_depth_resolved_strip` | the basalt-centre band's share of the depth-resolved strip by indent count, 0.300, which is where that window is placed on its own distribution |
| `windows` | three windows, each slid over the mosaic at an 8 px stride with its long axis through the thickness: the load-controlled map's full footprint, the 704 µm of laminate its retained indents cover, and the depth-resolved footprint. Per window: `footprint_um_along_thickness_by_in_plane`, `n_placements`, the mean and standard deviation of the basalt share and of the Voigt sum over the placements, the 5, 25, 50, 75 and 95 % points of the Voigt sum, `mapped_window_share` (the share the real window saw), and the percentage of placements at or below and at or above it |

The gap between the first two numbers is why the estimate moved from 21.33 to
18.50 GPa. What the windows give: over the full load-controlled footprint,
29 880 placements, a share of 0.369 ± 0.070 and a sum of 18.65 ± 1.46 GPa;
over the retained 704 µm span, 52 290 placements, 0.360 ± 0.141 and
18.47 ± 2.96 GPa, with the mapped window's 0.497 at or above 85 % of them.

## `basalt_input_sweep.json`

The basalt input under every classification cut in the swept range, for both
acquisitions, so that the choice of cut can be seen rather than argued.

| key | meaning |
|---|---|
| `shallow` | one entry per pair of cuts, the lower at 10, 15 and 20 GPa and the upper from 60 to 80: the cuts, the class size, the class median, and the Voigt sum at the indent-count and at the mosaic zone share |
| `depth_resolved` | the basalt median and class size at upper cuts of 30 to 50 GPa on the depth-resolved map, on the 304-rejection mask |
| `reading_depth` | two entries: the basalt input read from the selected basalt-centre contacts of `depth_softening.json` at 80 nm, 84.67 GPa, and at 50 nm, 97.93 GPa, each substituted into the two-level construction with everything else held, as Supplementary Table S4 substitutes the depth-resolved class median: the estimate, 18.43 and 19.14 GPa, and its relative gap to each measured flexural modulus on the table's convention |
| `note`, `reading_depth_note` | what each sweep shows, in one sentence |

This is the evidence behind a withdrawal. The shallow class is the tail above
the cut, so its median follows the cut, from 76.70 GPa at 60 to 93.07 at 80,
and the published range for bare basalt fibre, 70 to 95 GPa, contains all of
them. Agreement with that range was therefore built in and is no longer offered
as evidence. The `reading_depth` block reads the input at a different depth
rather than above a different cut, on the population Supplementary Figure S7
follows, so that Table S4 can carry the reading depth as a sensitivity beside
the cut.

## `depth_softening.json`

Written by `scripts/depth_softening.py`, and the arithmetic behind every figure
§4.4 and the Supplementary Figure S7 caption quote about the depth dependence of
the stiff contacts of the basalt-centre band. The selection is the figure's:
every indent of the band whose trace reads above the load-controlled map's own
70 GPa cut at 80 nm. It is made at 80 nm and the behaviour beyond is read
afterwards.

| key | meaning |
|---|---|
| `selection` | the band, the depth and the cut the selection is made at, how many indents meet it (86), how many indents the band holds (600) and how many of them are classified (534), the class the classification gives the selected indents at 150 nm (all basalt), and their identifiers |
| `E_median_GPa_by_depth` | the median modulus of the selected contacts at 50, 60, 70, 80, 100, 150, 200 and 300 nm |
| `E300_over_E80` | the ratio E(300)/E(80) indent by indent: its median, how many are at or below 0.5, its range, the share below 1, and by how much the median modulus falls between the two depths |
| `E300_over_E150` | the same ratio between 150 and 300 nm on the selected contacts |
| `bands_by_E80_over_classified` | E(300)/E(80) over all 534 classified contacts of the band, in four bands of the modulus read at 80 nm, each `(lower, upper]` in GPa with `null` for an open end: the count, the median ratio and the share below 1 |
| `called_basalt_at_150_nm` | the 222 contacts the classification itself calls basalt, a population selected at 150 nm and not at 80: the median and softening share on E(300)/E(80) and on E(300)/E(150) |
| `distance_to_nearest_compliant_contact` | whether the ratio follows the distance to the nearest contact reading below 15 GPa at 80 nm, drawn from every indent in the band as the neighbourhoods of §4.1 are: the grid pitch, the range of distances, the Spearman correlation and its p, and the split between the indents more than two pitches from any such contact and the rest, with the count and median ratio of each. The shell at exactly two pitches counts as near |
| `eight_neighbour_basalt_fraction` | the same question through the eight-neighbour basalt fraction of §4.1: the indents with at least six of eight neighbours called basalt against the rest, with the count and median ratio of each and a two-sided Mann-Whitney p |
| `H300_over_H80` | the hardness of the selected contacts between 80 and 300 nm: the median ratio and how many read lower at 300 |
| `S2_over_P` | S²/P, which does not pass through the area function: its median at 50 and at 300 nm, the median of the per-indent ratio and the ratio of the medians, how many indents have both values and on how many it falls |
| `contact_stiffness_S` | the contact stiffness itself between 250 and 300 nm, where the load is still rising: the median at each depth in kN m⁻¹, on how many of the selected contacts it falls, the median load at each depth, and S(300)/S(80) on the selection against the same ratio on the 115 matrix contacts of the band |
| `cut_free_shares` | two shares that involve no classification cut at 150 nm: the selection as a share of the band's 534 classified contacts, and the contacts the load-controlled map's own 70 GPa cut calls basalt as a share of the 2759 classified in its basalt-rich zone |

`reproduce_paper_values.py` holds the selection to a live recomputation from
the curves and then holds every printed figure of §4.4 to this file.

## `flexural_comparisons.json`

Every planned comparison between the two bending configurations, with the
correction applied to the family it belongs to.

| key | meaning |
|---|---|
| `per_specimen` | nine rows, one per specimen: configuration, chord modulus, peak tangent modulus, strength, softening, strain at break |
| `comparisons` | per quantity: the two means, Welch t, uncorrected p, Hedges g |
| `families` | the same p values under three Holm groupings |

The three families are reported because the conclusion depends on which one is
used: the stiffness comparison corrects to 0.0675 in the family as
pre-specified and to 0.0357 in either alternative. The manuscript reports the
difference itself, about 5 %, rather than a verdict that moves with the
grouping.

## `poisson_sensitivity.json`

Supplementary Table S1, and the only route to it. The instrument measures a
reduced modulus, which does not depend on the Poisson ratio assumed for the
specimen; the specimen modulus does. Both columns of that table follow from
`Er_GPa` by inverting the Oliver-Pharr relation at each value of nu.

| key | meaning |
|---|---|
| `indenter` | the diamond constants the inversion uses, E = 1140 GPa and nu = 0.07 |
| `nu_uniform` | the convention the paper adopts, 0.30 |
| `per_class` | one entry per class: the median reduced modulus, the median specimen modulus under each convention, the phase-specific nu and the shift between them |
| `laminate_estimate_GPa` | the estimate under each convention, at the fractions Eq. (1) uses, and the shift |
| `mixed_volume_nu` | the 0.34 assumed for the mixed-volume class, stated as what it is, and the Voigt, Reuss and Hill means of the basalt and matrix ratios at basalt fractions of 0.50, 0.54 and 0.67 |

The phase-specific ratios are inputs, not measurements: basalt 0.24, flax
0.30, PA-11 0.42 and 0.34 for the mixed-volume class. The 0.34 is an assumed
intermediate value between the basalt and the matrix figures. It is not the
Voigt-Reuss-Hill average of the two at a basalt fraction of 0.50: at 0.50 the
Voigt mean is 0.330, the Reuss mean 0.305 and the Hill average 0.318, and at the
0.54 and 0.67 the deconvolution of §4.1 puts on the mixed-volume contact the
Hill average is 0.311 and 0.289. The value is kept, because the table tests a plausible
phase-specific set and 0.34 lies inside the interval the two end members
bound; the file carries the averages so that the description can be checked.

`uncertainty_budget.py` reads the four percentages from this script rather
than holding its own copy, so the two cannot disagree.

## `bse_validation_metrics.json`

Accuracy of the texture classifier inside the marked zones, scored against the
operator's hand-painted fields by leaving one field out at a time. One block per
zone. **A primary input**: rebuilding it needs the tile images and the painted
fields, which are not deposited. It is the source of Supplementary Table S6.

| key | meaning |
|---|---|
| `n_fields` | painted fields in that zone, 16 in the basalt zone and 21 in the flax zone |
| `labelled_pixels` | hand-labelled pixels in that zone, which is the denominator |
| `accuracy_pct` | accuracy over all three classes including void, 82.8 % and 88.1 % |
| `accuracy_pct_phases_only` | accuracy over the two solid phases with void set aside, 96.0 % and 97.9 %, which is the denominator the reported volume fractions use |
| `note_phases_only` | that distinction, stated in the file so it travels with the numbers |
| `classes` | per class: `labelled_pixels`, `fields`, `precision_pct` and `recall_pct`, which are the cells of Supplementary Table S6 |
| `confusion_true_by_predicted` | the full matrix, true class by predicted class, in pixels |
| `model` | the classifier and the validation scheme |
| `reference` | what it was scored against, which is an operator's reading and not an independent measurement |

Two denominators are in the file and the paper uses one of them, so both are
named here. Void is a label the classifier has to assign and no reported
fraction uses it as a denominator, which is why the phases-only figure is the
one quoted beside the volume fractions; the all-three figure is the one a reader
opening the file first sees. The confusion is between matrix and void rather
than between the two solid phases: in the flax zone, matrix is called void on
19.3 % of its labelled pixels and flax on 2.5 % of them.

## A note on precision

The flexural moduli are stored rounded to two decimals. Recomputing the
flax-longitudinal standard deviation from this file gives 0.4994 and from the
unrounded values 0.4971, and both round to the 0.50 the paper reports, so the
deposit and the page agree whichever route a reader takes. That was not true
while the duplicate flax curve was in the file, where the two routes gave 0.49
and 0.48; the agreement is a property of these nine specimens and not a
guarantee, and recomputing from the stored file remains the route the printed
figures follow.

## `phase_angle_sensitivity.json`

Written by `scripts/phase_angle_sensitivity.py`, and the source of
Supplementary Table S3.

| key | meaning |
|---|---|
| `reference_voigt_GPa` | the laminate estimate, 18.4979 GPa, built on the load-controlled medians and the imaged zone share. The phase-angle limit cannot move it: the load-controlled map has no phase-angle channel. |
| `note` | that fact, stated in the file so it travels with the numbers |
| `criterion` | the rule being swept, in the words the Methods use |
| `rows` | one entry per limit. `phase_angle_limit_deg` is the limit, `rejected` and `rejected_pct` the indents it discards, `E_<class>_GPa` the resulting depth-resolved medians, `alternative_basalt_input_GPa` the basalt median among them, `Voigt_with_that_input_GPa` the estimate that median gives when substituted for the basalt input, `n_basalt` the size of that class, and `source` what the row applies |

The criterion is the one the Methods state: reject an indent when the magnitude
of its corrected phase angle exceeds the limit at any of the three analysis
depths, or when one of those depths has no valid window. At 15° it rejects 304
indents, and the script checks that its own answer agrees with `qc_flag_strict`
indent by indent rather than assuming it.

The first row applies the instrument's own quality flag, deposited as
`qc_flag_instrument` in `depth_resolved_indents.csv`: it rejects 433 indents.
The class rule and the basalt median under it are computed on the deposited
slice values like every other row. The estimate on that basalt median differs
from the one under the stated rule at 15° by 0.14 GPa, the figure §4.3.3 gives.

## `spacing_compliance.json`

Written by `scripts/spacing_compliance.py`, and the source of the compliance
figures in §2.2.

| key | meaning |
|---|---|
| `criterion` | pitch at least ten times the maximum depth |
| `load_controlled_pitch_nm` | the two pitches of the load-controlled grid, 2477 nm on the short axis and 5096 nm on the long one |
| `classified_indents` | 5551, the classified indents of the load-controlled map |
| `rows` | per class: `n`, the median maximum depth, the pitch that median would require, and the share of indents breaching the criterion on each axis |
| `depth_resolved` | the depth-resolved grid: 3000 nm pitch, 300 nm maximum depth, and their ratio |

Compliance is counted per indent rather than summarised, because a class median
hides how many contacts breach the criterion. The load-controlled map carries
no quality filter, so failed contacts, which run several micrometres deep,
stay in the counts and breach the criterion like any other deep contact.

## `contact_geometry.json`

Written by `scripts/contact_geometry.py`, and the source of the contact sizes
§3.3, §4.1 and §4.4 give.

| key | meaning |
|---|---|
| `radius_factor` | √(24.5/π), the equivalent contact radius of a Berkovich contact per unit contact depth |
| `load_controlled` | per class: median contact depth, equivalent radius and median maximum depth at 0.53 mN |
| `depth_resolved_by_displacement` | per class at z = 80, 150 and 300 nm: median contact depth h_c = z − 0.75 P/S, the gap z − h_c, and the equivalent radius |
| `basalt_centre_reread` | the classified contacts of the basalt-centre band re-read at displacements from 50 to 300 nm with the 15 and 40 GPa cuts: the mixed-volume, matrix and basalt shares and the median contact depth of the stiff contacts |
| `load_controlled_basalt_zone` | the mixed-volume share of the load-controlled basalt-rich zone on the 40 and on the 70 GPa upper cut, and the median contact depth of its stiff contacts |
| `stiff_contacts_matched` | the contacts selected above 70 GPa at 80 nm, read at the displacement where their contact depth is closest to the load-controlled contact depth on basalt: that displacement, the contact depth, the median modulus and load there, and their equivalent radius at 300 nm |
| `mixed_volume_fall_vs_distance` | E(300)/E(100) of the band's mixed-volume contacts against the distance to the nearest contact below 15 GPa at 80 nm: the medians within and beyond one pitch, and a Spearman correlation |

## `mixed_class_deconvolution.json`

Written by `scripts/mixed_class_deconvolution.py`. The mixed-volume class does
not survive a change of contact footprint; what it is built out of does, and
this is that calculation.

| key | meaning |
|---|---|
| `acquisitions` | per acquisition: the zone, the indent count, the two end members, the deconvolved basalt fraction under each mixing rule with its 95 % interval, the class fraction for comparison, and the end-member sweep |
| `two_class_sum_GPa` | the two-class Voigt sum under each rule, both acquisitions rebuilt on one common pair of end members, with the spread between them |
| `cross_acquisition_agreement` | the same sweep read the other way: how far the absolute fraction moves with the end members, against how far the two acquisitions move apart |
| `independent_comparison` | the basalt packing inside the same zone from backscattered-electron imaging, which shares no input with the indentation |

Every indent in the zone is inverted, not only those the rule called
mixed-volume, so no classification threshold enters the result. That is what
makes it a test of the objection rather than a restatement of it.

## `spatial_localisation.json`

Written by `scripts/figs_supplementary.py`, and the source of every spatial
number in §4.1 and of Supplementary Figure S4.

| key | meaning |
|---|---|
| `neighbours`, `pool`, `test` | the definitions the statistic depends on: eight nearest neighbours, drawn from every indent in the zone including the quality-rejected ones, one-sided Mann-Whitney U |
| `median_basalt_neighbour_fraction` | per class, the median share of the eight neighbours that are pure basalt |
| `median_difference_ci95` | bootstrap interval on the median difference between the mixed-volume and matrix classes |
| `p_depth_resolved`, `p_max_over_k_3_to_20` | the test at k = 8, and the worst case over the sweep |
| `edge_band_um`, `p_excluding_edge_bands`, `n_after_edge_exclusion` | the same test with a band excluded at each end of the zone, so that no indent counted can have neighbours outside it |
| `p_load_controlled_map`, `n_load_controlled_zone` | the same test on the other acquisition, the load-controlled map |

The pool matters and is the reason this file exists: counting only the
classified indents changes p at k = 8. The published value is the one that
includes the rejected indents, which is the right choice, because a rejected
indent still occupies its place on the grid.
