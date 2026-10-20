# Proteomics Toolkit v26.9.0 Release Notes

## Overview

Feature release with a breaking change to every default moderated p-value. The default variance
prior for `statistical_test_method="moderated_linear_model"` is now chosen by level, as
Skyline-PRISM chooses it: `"limma-trend-intensity-n-peptides"` (limma-trend with each protein's
peptide count as a second covariate) for protein-level data with a peptide-count column, and
`"limma-trend"` (limma's `eBayes(trend=TRUE)`) otherwise, including at peptide level. Both fit
their prior to the design's own residuals. They replace the intensity trend, which is deprecated:
it was calibrated on average but tilted by intensity, optimistic for bright features and
conservative for dim ones, and found fewer true effects. `intensity-peptide-trend`, which adds a
peptide-count stage to the intensity trend, is deprecated too. The priors now carry Skyline-PRISM's
names (`global`, `limma-trend`, `limma-trend-intensity-n-peptides`, `peptide-count`,
`intensity-trend`), so one setting means the same prior in both tools. Moderated p-values are also
now read on limma's capped degrees of freedom, for every prior. Both priors are pinned to R limma
3.66.0, and Skyline-PRISM makes the same changes, so the two tools agree on default runs.
The toolkit also adopts the three conventions that still separated a default run from
Skyline-PRISM on real data: no log pseudocount (a value of 0 or less is missing), R's `lowess()`
for the LOWESS-based priors, and limma's floor for zero residual variances.

## New Features

- **The default prior is chosen by level.** Leave `config.moderation` unset (`None`, the new
  default) and the model uses `"limma-trend-intensity-n-peptides"` when the data carry
  `config.peptide_count_column` (default `n_peptides`, which PRISM writes to the protein parquet
  only), and `"limma-trend"` otherwise. A peptide has no peptide count, so peptide-level data
  always gets limma-trend. Skyline-PRISM chooses from the level it analyzes, so on the same PRISM
  output both tools pick the same prior. The run prints which prior it chose, and `test_method`
  records it. A count column that does not look protein-level stops the run rather than fall back.
  With too few proteins to fit both covariates, the default uses limma-trend and prints that it
  did, as Skyline-PRISM does; named outright, the new prior refuses.
- **`moderation="limma-trend-intensity-n-peptides"`, for protein-level analysis only.** limma-trend
  with the protein's log peptide count as a second covariate: the residual log-variance is
  regressed on limma-trend's spline of `AveExpr` plus a natural spline of log peptide count, both
  fitted to the design's own residuals, and the prior df is estimated around the combined curve.
  limma-trend does not use peptide counts and tilts by them on protein-level data, where variance
  falls with the count: on random-label nulls of a 3,595-protein serum cohort, 6.9% of null
  p < 0.05 for one-peptide proteins and 3.0% for those with ten or more (unpaired 5 vs 5). This
  prior gave 4.4% and 4.2%, raised the prior df by 50-60%, and found 28% of two-fold spike-ins
  against limma-trend's 23%. On a second cohort (4,657 protein groups) it removed the same tilt in
  small designs and changed nothing with 35 subjects, where the prior barely matters. Expect fewer
  single-peptide discoveries and more among well-covered proteins. It **refuses input that is not
  protein-level** - a missing, fractional or sub-1 count, counts that never vary, or rows that
  repeat a feature identifier - rather than run on it or fall back to limma-trend. Results carry
  `limma_trend_intensity_n_peptides_s0_sq` and `peptide_count_used`.
- **`moderation="limma-trend"`**, limma's `eBayes(trend=TRUE)`: the prior variance follows a
  natural-spline curve of the residual log-variance on each feature's mean log abundance
  (`AveExpr`), fitted to the design's own residuals, and the prior df is estimated around that
  curve. This is limma's `fitFDist` with a covariate, with its spline df (1 to 4, by the number of
  features). Nothing comes from QC or reference samples, so the curve follows whatever mix of
  technical and biological variance the design leaves. It works at protein and peptide level, for
  unpaired, paired, linear-trend and interaction designs, with covariates and `robust=True`.
  Results carry `limma_trend_s0_sq`, with `limma_s0_sq` as the global prior for comparison.
- **New `total_df` column** on every moderated result: the degrees of freedom the p-value was
  read on (see Bug Fixes). `posterior_df` keeps `residual_df + d0`.

## Bug Fixes

- **Moderated p-values were read on uncapped degrees of freedom, so they were smaller than
  limma's wherever the cap binds.** limma's `eBayes` caps the moderated t's df at the residual
  df pooled over every feature, `pmin(df.residual + df.prior, sum(df.residual))`. The toolkit
  used `residual_df + d0` uncapped, for every prior, so with an infinite prior df it read the t
  off a normal distribution. The cap binds whenever the prior df is infinite, and otherwise only
  when few features are tested and the prior df is large. For example, 8 features at 4 residual
  df with an infinite prior df: a t of 5.49 gave p = 4.0e-8, and gives 4.8e-6 on the capped 32
  df. On 3,000 features at 8 df the cap moves a p-value by 0.1% at t = 3 and 4.5% at t = 8.
  Skyline-PRISM made the same fix in dotnet-v26.29.0 (maccoss/skyline-prism#137).
- **A paired or within-subject feature with a subject entirely missing was dropped, or given a
  wrong result.** The per-feature fit inverted the design restricted to the observed samples,
  and a subject with no value for a feature leaves that design singular. Usually the inverse
  failed and the feature vanished from the results. When the missing subject was the one the
  intercept stands for, the inverse was numerically singular instead, and the feature got a
  meaningless answer: on a 6-subject paired test set, a protein with a real change came back
  t = 0.03, p = 0.98, on 3 residual df where 4 was right. That bad variance also entered the
  prior every other feature was shrunk toward. Each feature is now fitted as limma's `lmFit`
  fits one with missing values: on its observed samples, with the design columns they can
  estimate (left to right, as R's `lm.fit` pivots). A subject absent from a feature drops out of
  that feature's comparison. A feature missing a whole arm reports NaN for its contrast, as
  limma's is NA, but its residual variance and df now go into the prior and the pooled df, as in
  limma. So does an interaction feature with an empty cell (no Control/Post value, say), whose
  difference of differences cannot be estimated: keeping design columns left to right alone would
  have reported Treatment's Post - Pre change as the interaction, so each feature's tested column
  is also checked against the span of the others on its observed samples. `AveExpr` is reported
  for every feature with an observed value. All of this is pinned to R limma 3.66.0
  (`eBayes(legacy = TRUE)`, since missing values make the residual df unequal). With complete
  data nothing changes: the fit is bit-identical, and features are fitted one missing-value
  pattern at a time, so no more than one pattern's design is held in memory.

- **Adjusted p-values counted features that could not be tested.**
  `apply_multiple_testing_correction` filled a missing p-value with 1.0 before correcting. So a
  feature with no p-value (an arm entirely missing, or too few values for its test) still counted
  as a comparison, and every adjusted p was larger by up to n_total / n_tested. R's `p.adjust`
  leaves an `NA` out of `n`, and so do limma's `topTable` and Skyline-PRISM. The toolkit now does
  too, for every test and every correction method, and an untested feature's `adj.P.Val` is NaN
  rather than 1.0. Skyline-PRISM's end-to-end golden found it: one untested feature in 12 made the
  toolkit's adjusted p 12/11 of limma's. With complete data nothing changes.

## Breaking Changes

- **The default prior changed, so every default moderated p-value changes.** A config that does
  not set `moderation` used to run the intensity trend and now runs the prior its level calls for
  (see New Features). On random-label nulls the intensity trend gave 6-9% of p-values below 0.05
  for the brightest fifth of features and 2-5% for the dimmest, where a calibrated test gives 5% in
  both (a 327-peptide cohort, a 3,595-protein serum cohort, and simulations; unpaired, paired and
  trend designs). limma-trend gave 4.7-6.0% in both the dimmest and the brightest fifth of every
  case. The intensity trend also roughly halved the prior df and lost power: in simulation at 5%
  FDR it found 52% of true two-fold effects against limma-trend's 71% (unpaired 5 vs 5), and 89%
  against 95% (paired, 6 subjects). Set `moderation="intensity-trend"` to reproduce an earlier
  result. Its estimator is unchanged; only its p-values move, and only where the df cap binds.
- **The priors take Skyline-PRISM's names.** `moderation` accepts `"global"`, `"limma-trend"`,
  `"limma-trend-intensity-n-peptides"`, `"peptide-count"`, `"intensity-trend"` and
  `"intensity-peptide-trend"` (the list is `MODERATIONS`), the tokens `prism differential --prior`
  uses. Underscores are accepted for the hyphens, so `"intensity_trend"` still works. Two names
  changed: `"limma"` is now `"global"` and `"deqms"` is now `"peptide-count"`; both old names still
  work, with a `FutureWarning`, and will be removed in a future release. `test_method` records the
  new names (`moderated_linear_model[global]`). Result column names are unchanged.
- **`moderation="intensity-trend"` is deprecated** and emits a `FutureWarning` saying why. It is a
  `FutureWarning` rather than a `DeprecationWarning` because Python hides a `DeprecationWarning`
  raised inside a library, and notebook users are who need to see it.
- **`moderation="intensity-peptide-trend"` is deprecated too**, with its own `FutureWarning`. Its
  peptide-count stage is shaped on the same QC/reference pools or design groups as its intensity
  stage, so it inherits the intensity trend's mismatch instead of correcting it: pooled technical
  noise averages down over a protein's peptides, while the biology in a study residual does not.
  On a simulated protein-level null (peptide counts 1-64, technical variance falling with
  intensity and as 1/count, flat biology), a pool-sourced shape gave 2.8% of null p-values below
  0.05 for the dimmest fifth of proteins and 6.2% for the brightest, 3.0% for the fifth with the
  fewest peptides and 6.7% for the most, the smallest prior df of any prior tested (2.1), and
  found 55% of true two-fold effects at 5% FDR against limma-trend's 69% (unpaired 5 vs 5). A
  design-group shape was calibrated unpaired but tilted the other way paired (8.4% dimmest, 2.9%
  brightest, 6 subjects), as the intensity trend's does. Random-label nulls on the serum cohort
  showed the pool-sourced shape tilted by intensity in both unpaired nulls. Use it only to
  reproduce an earlier result. It was never in Skyline-PRISM.
- **`moderation="peptide-count"` (DEqMS) now refuses input that is not protein-level**, with the
  same checks as `limma-trend-intensity-n-peptides`: a missing, fractional or sub-1 peptide count,
  counts that never vary, or rows that repeat a feature identifier (through
  `run_comprehensive_statistical_analysis`, the `Protein` column). It used to check only that the
  count column existed, so it would run on peptide rows given their protein's count. Valid
  protein-level input gives the same results as before.
- **`variance_prior_group_column` now warns when it has no effect.** It only ever shaped the
  intensity-trend priors. Until this release that was the default, so a config that sets it was
  written for that prior; under any other moderation it is now ignored with a `UserWarning`
  rather than silently.
- **P-values change wherever the df cap binds, under every prior** (see Bug Fixes). The t
  statistics, posterior variances and prior df are unchanged.
- **No log pseudocount by default.** When the toolkit takes the log (`log_transform_before_stats`),
  a value of 0 or less now becomes missing and nothing is added to the rest, as Skyline-PRISM logs
  its linear parquet. The default used to add the smallest value / 100 to every value, and shift
  everything up if any value was negative; on a 3,595-protein cohort that moved low-abundance log
  ratios by up to 1.4e-3 against PRISM. Every test handles the missing values (the moderated model
  as limma does). Setting `config.log_pseudocount` to a number keeps the earlier behavior.
- **The peptide-count and intensity-trend priors smooth with R's `lowess()`,** at its default
  interpolation distance of 1% of the x range, as Skyline-PRISM does. They used statsmodels'
  `lowess` at `delta=0`, which on the same cohort moved the intensity trend by up to 1.4e-3
  relative against PRISM. statsmodels at 1% still differs from R where points crowd at the top of
  the range (2e-6 on 600 points), so the toolkit now carries R's algorithm (`_clowess`), which
  agrees with R to ~5e-14. The intensity trend on peptide-level data is also much faster: the fit
  is close to linear in the number of points rather than quadratic (80,000 points in about 1 s).
- **A residual variance of exactly zero is floored, not dropped,** as limma's `fitFDist` floors it:
  at 1e-5 x the median, with limma's warning ("Zero sample variances detected, have been offset
  away from zero"), and Skyline-PRISM does the same. Dropping such features from the prior fit
  changed it for every feature: on the new golden cases, a prior df of 3.3 against limma's 0.68,
  and p-values off by orders of magnitude.
- **The peptide-count prior leaves out every residual variance at or below limma's floor**, not
  just exact zeros, from its LOWESS on log peptide count. Replicates that agree exactly give a
  variance of 0 from one least-squares solve and 1e-29 from another, and the 1e-29 entered the fit as
  a log variance of -66 that bent the whole curve: on a cohort with three exactly constant proteins
  that moved p by 4e-3 between the toolkit and Skyline-PRISM, which uses the same rule.

## Testing

- `TestLimmaEbayesGolden` pins the moderated t to R limma 3.66.0's own `lmFit` + `eBayes`, called
  whole: `tests/fixtures/limma_ebayes.json`, written by `tests/fixtures/generate_limma_ebayes.R`,
  which imports nothing from the toolkit. Fourteen cases cover limma-trend at all three spline
  df (2, 3, 4), finite and infinite prior df, unpaired, paired, linear-trend and within-subject
  trend designs, and the global prior. Five of the cases have missing values (`legacy=TRUE`):
  scattered NaNs, an arm entirely missing, a paired subject missing (the reference subject and
  another), and a within-subject trend with whole subjects missing. Five make the df cap bind,
  and the rest show it changes nothing when it does not. Measured agreement is 1e-11 or better;
  the stated tolerances (1e-12 to 1e-9) are explained in the test's docstring.
- `TestMissingValuesInTheDesign` reproduces the dropped and wrong paired results directly: the
  fold change of a feature missing one subject equals the mean within-subject difference over the
  other subjects, on the right residual df; a feature missing a whole arm, and an interaction
  feature with an empty cell, report no contrast but contribute their residual df.
- `TestLimmaTrendIntensityNPeptidesGolden` pins `limma-trend-intensity-n-peptides` to R limma
  3.66.0: `tests/fixtures/limma_trend_intensity_n_peptides.json`, written by
  `tests/fixtures/generate_limma_trend_intensity_n_peptides.R`. limma's `fitFDist` takes one
  covariate, so the generator transcribes it with the peptide-count columns added to its design and
  refuses to write a case unless the transcription without them reproduces limma's own
  `eBayes(trend = TRUE)` to 1e-12. Seven cases cover unpaired, paired and linear-trend designs, both
  Amean spline sizes (3 and 4 columns), both count blocks (a spline, and a straight line - for two
  distinct counts, and for a median count on the boundary, where R's `ns()` refuses), and an
  infinite prior df where limma-trend's is finite. Measured agreement is
  about 1e-14 in the prior and posterior variances and the prior df, 3e-13 in the p-values.
- `TestModerationNamesAndDefault` pins the names to Skyline-PRISM's, the default chosen by level
  (directly and through `run_comprehensive_statistical_analysis`), the refusal of a count column
  that does not look protein-level under the default, the warnings for `"limma"` and `"deqms"`,
  and the silent acceptance of underscores.
- `TestPriorCalibrationByIntensity` reruns a seeded null with real data's variance structure
  (technical noise falling with intensity, flat biology; 6 draws of 2,000 features, unpaired 5
  vs 5) and checks calibration in each intensity fifth, which an overall 5% rate cannot see.
  limma-trend must stay within 3.5-6.5% in every fifth, and the pool-shaped intensity trend must
  still tilt by more than 3 points. Marked `slow` (about 35 s).
- `TestLimmaTrendIntensityNPeptidesCalibration` holds the protein-level default within 3.5-6.5%
  in every intensity fifth and every peptide-count fifth of the protein-level null
  (`_protein_calibration_null`), unpaired and paired, with a larger prior df than limma-trend's.
  Marked `slow` (about 6 s).
- `TestIntensityPeptideTrendCalibration` reruns the protein-level null (6 draws of 2,000 proteins
  with peptide counts) and pins the three facts the `intensity-peptide-trend` deprecation rests on:
  the pool-shaped prior tilts by intensity and by peptide count (unpaired), the design-group-shaped
  one tilts the other way (paired), and limma-trend stays within 3.5-6.5% in every intensity fifth
  while tilting by peptide count. Each tilt bound is 2 points, more than three binomial sd. Marked
  `slow` (about 90 s together).
- `TestLimmaTrendIntensityNPeptidesProteinLevelOnly` and `TestModeratedLinearModelDeqms` check
  that both peptide-count priors refuse peptide-level input: no count column, missing, fractional
  or zero counts, counts without spread, and rows that repeat a feature identifier, directly and
  through the dispatcher; and that the new prior refuses too few proteins rather than dropping the
  counts, while the default on the same few proteins uses limma-trend and says so.
- `TestTrendCalibration` checks limma-trend's overall null rate on its unpaired and paired
  simulations, beside the intensity trend's.
- Two zero-variance cases in `limma_ebayes.json` (the global prior and limma-trend, each with
  three features whose replicates agree exactly) pin limma's floor; the code before this release
  fails both. `TestLimmaVarianceFloor` checks limma's two warnings.
- `TestPriorLowessMatchesR` pins the priors' LOWESS to R's `lowess()`: `tests/fixtures/lowess.json`,
  written by `tests/fixtures/generate_lowess.R`, on crowded intensities, tied peptide counts and
  hand-placed points either side of `delta`. Agreement is ~5e-14 against a 1e-12 tolerance, and the
  peptide-count prior computed end to end from R's smoother matches to 1e-12.
- `TestMultipleTestingCorrection` pins Benjamini-Hochberg, Bonferroni, Holm and
  Benjamini-Yekutieli with a missing p-value to R 4.5.1's `p.adjust`; the code before this release
  fails all four.
- `TestLogTransformDefault` checks that values of 0 or less become missing with nothing added, and
  that an explicit pseudocount keeps the earlier behavior.
- An end-to-end comparison with Skyline-PRISM (2,500 proteins, 32 samples, scattered blanks, whole
  arms and whole subjects missing, exactly constant proteins; unpaired, paired and within-subject
  trend designs; every prior) agreed on which proteins were tested and on p to 1.6e-12 (the default,
  limma-trend and global priors), 6e-11 (peptide-count) and 3e-10 (intensity-trend), with t to 3e-11.
  It found the peptide-count rounding difference above and a Skyline-PRISM difference in the
  intensity trend's one-value groups, both fixed.
- New tests for both deprecation warnings and their reasons, the warning for an ignored
  `variance_prior_group_column`, and both residual-fitted priors with `robust=True`.

## Documentation

- `docs/06-statistical-analysis.md` gains "Same results as Skyline-PRISM": which reference each
  piece is pinned to in both tools and how closely, the three conventions the toolkit now shares
  with PRISM (no pseudocount, R's LOWESS, limma's zero-variance floor), missing values fitted as
  limma's `lmFit` fits them in both (in PRISM from maccoss/skyline-prism#148), adjusted p-values
  over the features tested, and the options only one tool has.
  "Log transformation" states the new default.
- `docs/06-statistical-analysis.md` uses Skyline-PRISM's names throughout, states the default
  chosen by level, and has a table of which priors work at which level. It explains why limma-trend
  replaced the intensity trend, with the null-calibration measurements by intensity fifth, and
  documents limma's df cap and the new `total_df` column. The one deliberate difference from limma
  is recorded: with missing values limma 3.58+ switches to `fitFDistUnequalDF1`, and the toolkit
  keeps `fitFDist`'s moments.
- A new section, "Does the peptide-count stage fix it?", has the protein-level null measured by
  intensity fifth and by peptide-count fifth for every prior, unpaired and paired, from pools and
  from design groups, with power and prior df, two sensitivity checks (biology falling with peptide
  count too; count tracking intensity), and the same comparison on random-label nulls and
  spike-ins of a 3,595-protein serum cohort. The peptide-count prior's earlier cross-validation
  figures are kept, with a note that they measure fit to technical replicates, not calibration on
  study samples.
- Another, "limma-trend with peptide count (protein level only)", documents the protein-level
  default: what it fits, the protein-level checks and the one case they cannot see, the evidence
  from both cohorts, the shift of discoveries away from single-peptide proteins, and the golden.
- The README, the overview, the QC-plot and pitfalls pages, and the tutorial notebook use the new
  names and the default chosen by level; the tutorial gains a section that runs the default on its
  protein-level data. `plot_variance_vs_intensity` is documented as a diagnostic for the intensity
  priors only, and the docs and tutorial show limma's `plotSA` drawn by hand instead.
