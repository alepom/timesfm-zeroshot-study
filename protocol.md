# Study protocol — How far does zero-shot go? (TimesFM 3)

**Status: v1 (2026-10-28). Frozen by the git tag `protocol-frozen-v1`.** No model is run on, and no data are downloaded for, the
post-release test windows (Tier B, W-post, W-new, P) before that tag exists; the code enforces the earliest download date. Every
later change is listed in §11 (Deviations) with its date and reason. v1 incorporates an independent adversarial review of the draft
(2026-10-28); the changes it caused are summarised in §10.

## 1. Questions and hypotheses

| ID | Question | Tier |
|---|---|---|
| RQ1 | Do the published per-task TimesFM-3 results, and the claim they support, reproduce with the public code and weights? | A |
| RQ2 | Does TimesFM-3's advantage hold on data that post-date its release, and is there a trace of memorisation? | B, W, P |
| RQ3 | Under which controlled conditions does zero-shot forecasting fail? | C |
| RQ4 | Are the quantile forecasts calibrated? | A, B, C, P |
| RQ5 | When do multivariate mode and covariates help, and when do they hurt? | C-mv, B |
| RQ6 | How much do inference-time options and internal components contribute? | ablations |
| RQ7 | Is zero-shot worth it against classical methods (and, descriptively, a global model), accounting for compute? | B, A |

### Pre-declared hypotheses

Each hypothesis has one primary test (§5.3); all primary tests form family H.

- **H1 (descriptive, RQ2).** The advantage of TimesFM-3 over a statistical ensemble is smaller on post-release data (Tier B) than on
  the public benchmark (Tier A). Tier A and Tier B differ in datasets, frequencies, horizons and possible exposure, so the comparison
  is reported descriptively, without a test, and no lineage-stratum contrast inside Tier B is tested.
- **H2 (RQ2, memorisation).** On Wikipedia pageviews the advantage of TimesFM-3 over Chronos-2, log(MASE TimesFM-3) −
  log(MASE Chronos-2), is more negative on W-seen (pages and period inside TimesFM's pageview corpus) than on W-post (same pages,
  post-release period). Chronos-2 is the reference: like TimesFM-3 it is affected by the difficulty and noise level of each period,
  which a classical reference would not track. Assumption, stated as such: by the published descriptions of its training data,
  Chronos-2's corpus contains Wikipedia pageviews only from periods before 2018 (Kaggle web traffic, wiki-rolling), not 2023. A
  null result does not show the absence of memorisation (minimum detectable effect in §10).
- **H3 (RQ3/RQ6).** The hard-coded linear detrending (`use_linear_detrending`, model.py:472-522) causes trend continuation on the
  saturating logistic process: disabling it lowers MASE over steps 25–48 on D7 with n = 200 and H = 48 (MASE of those steps,
  scaled as usual).
- **H4 (RQ5).** Multivariate mode (`tfm3`) lowers MASE relative to univariate mode (`tfm3-uni`) when a known cross-series
  dependence exists (H4a lead-lag 24, scored on the follower series only, the one that can gain from the leader; H4b known future
  covariate, SNR 2), and is equivalent within ±3% when none exists (H4c eight independent series; H4d spurious past covariate).
- **H5 (RQ2).** On the near-random-walk Tier B series (10 ECB exchange rates; FRED: 8 Treasury yields, WTI, Brent), TimesFM-3
  (`tfm3-uni`) does not improve on the random walk (`naive`) by more than 10% in MASE: one-sided non-superiority test, H0 ratio
  ≤ 0.90, rejected when the 95% lower bound (5th bootstrap percentile) exceeds 0.90. (An equivalence test at ±5% was considered;
  the pilot's interval, about ±11%, gives it no power.)

## 2. Models and configurations

All weights are loaded by Hugging Face revision sha (recorded in every run file), and the code bundle sha256 is recorded.

| Key | Model | Configuration |
|---|---|---|
| `tfm3` | TimesFM-3, `google/timesfm-3.0-pytorch` | `TimesFM3Forecaster.predict_batch` package defaults (no symmetric averaging, no `make_positive`, sorted quantiles); multivariate mode: all targets of a task and its covariates in one pass |
| `tfm3-uni` | TimesFM-3 | same defaults; each target alone, no covariates. **The TimesFM-3 configuration of the information-matched contrasts (family P)** |
| `tfm3-eval` | TimesFM-3 | flags of Google's benchmark runners: `use_symmetric_averaging=True`, `make_positive=True` (multivariate, as `tfm3`) |
| `tfm3-c2016` | TimesFM-3 | as `tfm3-uni` with the classical models' context (last 2,016 points); secondary |
| `tfm25` | TimesFM-2.5, `google/timesfm-2.5-200m-pytorch` | univariate, no covariates; `normalize_inputs`, continuous quantile head, flip invariance, `infer_is_positive`, `fix_quantile_crossing`; compiled per context bucket (multiples of 1,024) |
| `chronos2`, `chronos2-uni` | Chronos-2, `amazon/chronos-2` | `predict_df`; `chronos2` with all targets and covariates of a task, `chronos2-uni` each target alone without covariates |
| `tirex2`, `tirex2-uni` | TiRex-2, `NX-AI/TiRex-2`, revision `30788a31` | as Chronos-2; maximum horizon 320 (longer horizons are not run); sLSTM "vanilla" backend on the T4 (the fused FlashRNN kernels need Ampere) |
| `snaive`, `naive`, `mean` | Seasonal Naive, Naive, historic average | StatsForecast 2.1.1 |
| `ets`, `arima`, `theta` | AutoETS, AutoARIMA, AutoTheta | StatsForecast defaults, season per frequency (§4.4). On hourly data `ets` and `theta` are run and reported as the off-the-shelf configuration but are not ensemble members; `arima` is not run on hourly data |
| `mstl`, `mstl-theta`, `mstl-arima` | MSTL (seasons 24 and 168) + AutoETS(ZZN), AutoTheta or non-seasonal AutoARIMA on the deseasonalised series | season-24 data only. A season is used only if the context holds two of its cycles; with none, the remainder model is fitted to the series |
| `stat-ens` | equal-weight combination (quantiles averaged) | season-24 data: `mstl`, `mstl-theta`, `mstl-arima`; other data: `ets`, `arima`, `theta` |
| ablations | `tfm3` with one change | `tfm3-nodetrend` (`use_linear_detrending=False`), `tfm3-nocpmrevin`, `tfm3-novarattn`, `tfm3-nostitch` (module attributes set after loading, because a Hub load overrides `ModelConfig` flags), `tfm3-sym`, `tfm3-pos`, `tfm3-znorm`, `tfm3-edge` (inference flags) |

Classical quantiles come from the 20/40/60/80% prediction intervals and the point forecast. Classical contexts: at most the last
2,016 observations (hourly, synthetic), 1,095 (daily) or 780 (business daily); TSFM contexts: at most 15,360 (Chronos-2 truncates at
8,192). With main-study history from 2025-01-01, Tier B contexts hold about 14,500 hourly points, about 600 days or about 430
business days, so on daily and business-daily data all models see the same history and only hourly data differ (2,016 vs about
14,500 points; `tfm3-c2016` measures that difference). Tier W contexts are 400 days for all models. A failed classical fit falls
back to the naive forecast; fallbacks are flagged per task in the metrics (and inherited by `stat-ens`) and reported. MSTL
intervals ignore the uncertainty of the seasonal component, so `stat-ens` coverage is reported next to its scores. A trained
global model (LightGBM) is not run; fev's published LightGBM results are reported descriptively for Tier A.

Group tasks give `tfm3`, `tfm3-eval`, `chronos2` and `tirex2` all targets at once; the `-uni` configurations, `tfm25` and the
classical models forecast each target separately.

## 3. Tiers and data

### 3.1 Tier A — public benchmark (reproduction)

- **fev-bench** (v1): 30 of the 100 tasks, stratified by domain × frequency (`numpy.random.default_rng(17)`), after excluding the
  3 tasks with string covariates, which Google's runner encodes with Python `hash()` (not reproducible across processes).
  Manifest `data/manifests/tierA_fev.csv`; code `tfmstudy/tiera.py`. GIFT-Eval is not re-run.
- Metrics with fev's own functions; fev 0.9.0, dataset revision recorded per run. Conventions of Google's runner for all Tier A
  models: missing values carried forward; context cut to 15,360; `tfm3-eval` through Google's `TimesFM3Evaluator` (chunks of 32
  variates). Models: `tfm3-eval`, `tfm3`, `chronos2`, `tirex2`, `snaive`.
- **Pipeline control.** Our Seasonal Naive must match fev's published results (|MASE difference| < 5·10⁻⁵) and our Chronos-2
  fev's published Chronos-2 results (|SQL deviation| ≤ 0.5%) on every task; the RQ1 summaries are reported on the tasks passing both
  controls and, separately, on all tasks, with the failing tasks listed.
- **RQ1, reproduction of the claim (primary for RQ1, descriptive).** The geometric mean over tasks of SQL(TimesFM-3) / SQL(Chronos-2),
  ours (`tfm3-eval` / our `chronos2`) against published (Google's CSV / fev's `chronos-2.csv`).
- **RQ1, per task.** Relative deviation of our `tfm3-eval` SQL, MASE, WQL from Google's CSV; share of tasks within ±3% SQL; median,
  range and a sign test of the deviations. GPU test on two eligible, unselected tasks (2026-10-28): Chronos-2 matched fev to 4
  decimals on epf_de and within 0.5% on ETT_1D; `tfm3-eval` deviated from Google on epf_de by +1.2% SQL (Google used another GPU
  and torch 2.9.1); the TimesFM-3 / Chronos-2 SQL ratio was 0.995 (ours) vs 0.983 (published) on that task.
- **H1 (descriptive).** Per task, SQL and MASE of our `tfm3` (Google's inputs, covariates included) relative to fev's published
  `stat_ensemble`; summarised next to the Tier B advantage of `tfm3` over `stat-ens` on univariate tasks (covariates included, as in
  Tier A). Our ensemble differs from fev's in members and seasonality handling, which is stated.
- **TiRex-2 contrast dropped.** TiRex-2-f (`NX-AI/TiRex-2-fevbench`, described as trained without the fev-bench datasets) has weights
  byte-identical to `NX-AI/TiRex-2` (all 283 tensors; same source checkpoint sha256 `184b160f…`; bit-identical forecasts). The same
  checkpoint was uploaded to both repositories on 2026-07-02, replacing the earlier TiRex-2 checkpoint. The TiRex-2 paper reports one
  checkpoint per benchmark without saying which one is published as TiRex-2. Reported as a neutral provenance note; evidence in
  `data/manifests/tirex2_provenance.json` and `reports/provenance_tirex2.md`.

### 3.2 Tier B — post-release real data

- TimesFM-3 release: 2026-08-31. **Test windows: 2026-09-01 … 2026-10-20.** Data are downloaded once, on or after 2026-10-29 (after
  the freeze; enforced in `build.py`), when ERA5 (about six days of lag) covers the windows; download time and URLs are stored. The
  last days of ERA5 are preliminary (ERA5T) and may later be revised; the values used are the ones archived at download.
- Origins: every `min(horizon, one week)` steps (hourly 168, daily 7, business daily 5), so 24-hour, 168-hour, 7-day and 5-day
  windows do not overlap and the 28-day and 20-day windows overlap; the bootstrap accounts for it (§5.2).

| Domain | Source (licence) | Series | Frequency | Horizons | Covariates |
|---|---|---|---|---|---|
| Electricity load | Energy-Charts (CC BY 4.0) | 12 European countries | hourly | 24, 168 | temperature at the capital (Open-Meteo ERA5), observed values for the forecast window: an **oracle**, used only by `tfm3`, `tfm3-eval`, `chronos2`, `tirex2` |
| Electricity price | Energy-Charts (CC BY 4.0) | day-ahead price, 12 bidding zones | hourly | 24, 168 | none; one group task over the 12 zones |
| Weather | Open-Meteo, ERA5 (CC BY 4.0) | temperature and wind speed, 20 cities | hourly | 24, 168 | — |
| River flow | USGS (public domain, provisional) | 20 gauges | daily | 7, 28 | precipitation (Open-Meteo ERA5), oracle as above |
| FX | ECB (free reuse) | 10 currencies vs EUR | business daily | 5, 20 | none; one group task |
| FRED | FRED | 8 Treasury yields, WTI, Brent | business daily | 5, 20 | none; one group task |

Series lists were fixed by rules on 2025 data (`catalog.py`, `build.py select`; `data/manifests/tierB_series.csv`). Lineage notes
(`catalog.DOMAINS`) are descriptive: same-source data of earlier periods are in GiftEvalPretrain for load and price (Spanish ENTSO-E
data), weather (ERA5 1989–2018) and FRED (only monthly FRED-MD, a weak link); none is known for rivers and FX. The full TimesFM-3
corpus is not public.

### 3.3 Tier W — Wikipedia memorisation probe

English Wikipedia, daily user pageviews, horizons 7 and 28, contexts of 400 days.

- **W-seen**: 200 pages sampled (seed 17) among articles in at least 3 of the 2022 monthly top-1000 lists (April 2022 missing from the
  API) with ≤5% zero-view days in 2025-03…08; test windows **2023-09-01 … 2023-10-20**, inside the TimesFM pageview corpus (to
  November 2023).
- **W-post**: the same pages, test windows 2026-09-01 … 2026-10-20 (same calendar span).
- **W-new**: pages in the 2024-01…2025-08 top-1000 lists absent from the 2022–2023 lists, first revision 2023-12-01…2025-07-28,
  ≤5% zero-view days in 2025-03…08 and no views at all under their title in 2021-01-01…2023-11-30; matched to W-seen/W-post on
  quintiles of 2025-03…08 median views (up to 40 per quintile, seed 17); same windows as W-post. The two top quintiles are pooled
  (W-new has 8 pages in the fourth), giving four popularity groups with equal weight.
- **Redirect rule.** A page whose title is a redirect or missing at the latest status check is excluded, because its views move to
  another title. Check of 2026-10-28: 10 W-seen/W-post and 8 W-new titles excluded (`data/manifests/tierW_redirects_2026-10-28.csv`),
  leaving 190 and 128 pages; the check is repeated at download time and any newly moved page is excluded and listed.
- W-new contexts never start before the page's first revision. W-new pages (mostly 2024–25 releases and events) differ in dynamics
  from the long-lived W-seen/W-post pages; the W-new comparison is therefore secondary and adjusted for context features (§5.4).

### 3.4 Tier P — prospective

- **Two issues**, S = 2026-10-30 and S = 2026-11-02 (their scoring windows share four of seven days). On the morning of S the data published up to S − 1 are downloaded and each series
  is forecast from its last available observation (group tasks: the earliest last observation of their members).
- **Scoring window**: target timestamps in [S + 2 days, S + 9 days) UTC. Forecasts are sealed before S + 1 10:00 UTC; at that time no
  value in the scoring window is published (day-ahead prices for S + 2 appear at about 11:45 UTC on S + 1).
- **Series**: load, price, river, FX, FRED (Tier B lists) and the W-post and W-new pages (minus redirects); group tasks for price,
  FX, FRED. The §6 context rules are applied at issue time (the targets do not exist yet). Weather is
  not included (ERA5's lag would end its contexts a week before S). Horizons cover the window from the earliest plausible origin:
  240 hours, 10 days, 10 business days. No covariates (their future values are unknown). Models: as Tier B.
- **Seal** (`prospective seal`): sha256 of the contexts (`series.parquet`), tasks, `ISSUE.json`, code bundle, run records (model
  revisions, configurations) and every forecast file, and of their sorted list; the combined hash receives a public timestamp
  before the outcomes exist; files are not modified afterwards.
- **Outcomes** are downloaded once the window is published (planned 2026-11-10 and 2026-11-13) with a fresh cache; step k of a
  forecast is the k-th observation after the origin in the source's calendar; a task with fewer than 80% of its window steps
  observed is excluded and listed. Analysis in §5.5.

### 3.5 Tier C — controlled synthetic data

Seeds: replication r of cell c uses `default_rng([17, crc32(c), r])` (`synthetic._rng`), except C-replication, which uses the
reference study's own seeding (`ff_simulate`); 200 replications per cell (7,600 tasks).

- **C-replication** (arXiv:2610.00589, doi:10.5281/zenodo.22976253, CC BY 4.0): D6 local level with a break, D7 logistic growth, D8
  intermittent demand; n ∈ {48, 200}; the reference study's main horizon H = 12 and its 48-step check (same seeds), replications 0–199.
  Re-implemented from the published code with the same parameters and seeding; bitwise identity not verified.
- **C-uni**: two seasonalities (24, 168) with 1 or 4 weeks of context; season longer than the context; context ladder 64…15,360;
  horizons up to 1,024; regime change near the origin; outliers in the last 5% of the context; random walk; white noise.
- **C-mv**: VAR(1) (dimensions 2, 8, 32, 64); lead-lag pair (lag 1, 5, 24); target driven by a known future covariate (SNR 0.5, 2);
  spurious past covariate; independent series stacked as variates.
- **Property tests**: scale, shift, sign flip, variate permutation, input shape, determinism (smoke test).

### 3.6 Ablations (RQ6)

On all Tier C cells and on the univariate tasks of a fixed 20% sample of the Tier B series (21 of 104, `numpy.random.default_rng(17)`;
`forecast.ablation_series`): the eight ablation configurations of §2, each against `tfm3`. They are run only where they can act
(`forecast._ablation_applies`): CPM, iterative RevIN and stitching above 64 steps (shorter horizons are decoded from one token);
`tfm3-novarattn` on multivariate tasks (Tier C group and covariate cells; Tier B group tasks and sampled tasks with covariates);
`tfm3-edge` on tasks with future covariates. Not run on Tiers W and P.

## 4. Metrics

- **Point**: MASE of the median (primary); RMSSE scaled by the seasonal differences of the context (as MASE, not M5's lag 1); WAPE.
- **Probabilistic**: SQL over the nine levels (fev-bench definition, primary); WQL; coverage and width of the 80% and 60% intervals
  (missing truth values are skipped); PIT histogram, described.
- **Aggregation**: model / Seasonal Naive per series and window, geometric means within domain × horizon, then equal weight over
  horizons and domains. Win rate and skill score are also reported.
- **Seasons**: hourly 24, daily 7, business daily 5.

## 5. Statistics

### 5.1 Estimand
For a contrast A vs B: d = log(score A) − log(score B) on identical series and windows (pairs with a non-positive or missing score are
dropped and counted). The estimate is the mean of d within each domain × horizon cell (Tier W: horizon × popularity group; Tier C:
cell), averaged with equal weight over horizons and then over domains, reported as exp(·) (< 1: A better). Each family uses one
complete-case set: the series-windows that all models of the family have scored.

### 5.2 Uncertainty
Two-way cluster bootstrap with 10,000 replicates (seed 17): series clusters (a series; the two weather variables of a city form one
cluster) are resampled within domains, and ISO weeks of the first forecast step are resampled within their year, with weights
multiplied, so that overlapping windows and shocks common to several series are both reflected. Tier C resamples replications (a
task with all its variates) within cells; Tier P uses series clusters only. A cell left without weight in a replicate is dropped from
that replicate's average; the share of such replicates is reported. Percentile 95% intervals; two-sided bootstrap p-values (smallest
1/10,000). The one-way (series-only) interval is reported next to every two-way interval as a sensitivity analysis.

### 5.3 Families and decisions
Decisions use Holm-adjusted p-values (α = 0.05) within a family; 95% intervals are reported unadjusted.

- **Family P (primary, Tier B)**: P1 `tfm3-uni` vs `stat-ens`; P2 `tfm3-uni` vs `chronos2-uni`; P3 `tfm3-uni` vs `tfm25` — the
  information-matched contrasts (univariate tasks of the structured domains load, price, weather, river; no covariates for any model);
  P5 `tfm3-eval` vs `tfm3` on the same tasks (identical inputs for both, covariates included); P4a `tfm3` vs `tfm3-uni` on the three
  real group tasks (price, FX, FRED; strata = domain). Each on MASE and SQL: 10 tests.
- **Family H (hypotheses)**: H2 (DiD, MASE), H3, H4a, H4b (two-sided p, sign as predicted), H4c, H4d (equivalence by two one-sided
  tests on the bootstrap: p = max of the two tail shares beyond the margin, i.e. the 90% interval inside the margin), H5
  (non-superiority: p = share of bootstrap estimates at or below log 0.90). Always seven
  tests: a test that cannot be computed enters Holm with p = 1 and is reported as missing.
  A hypothesis is supported if its Holm-adjusted p < 0.05 and, for directional tests, the estimate has the predicted sign.
- **Secondary families** (Holm within each): S1 the P1–P3 contrasts per domain; S2 Wikipedia: H2 with references `stat-ens` and
  `tirex2`, H2 for `tfm25` (also exposed to pre-2024 pageviews), W-post vs W-new (advantage over Chronos-2), and the adjusted DiDs of
  §5.4; S3 covariates as an upper bound: P4b `tfm3` vs `tfm3-uni` on load and river (oracle covariates), and `chronos2` vs
  `chronos2-uni`; S4 `tfm3-c2016` vs `tfm3-uni` on hourly domains; S5 ablations (each vs `tfm3`, per tier); S6 Tier P (§5.5).
- **Exploratory** (no correction, labelled): spike anticipation on W (windows whose maximum exceeds 3× the context's 90th percentile:
  ratio of the forecast 0.9-quantile maximum to that context percentile, by stratum and model); results by lineage note; latency.
- Results that are not resolved are reported as such; equivalence is claimed only through the TOST rule above.

### 5.4 Difference in differences (Tier W)
Advantage a = log(MASE TimesFM-3) − log(MASE reference) per page-window. DiD = mean a(W-seen) − mean a(W-post), each mean taken with
equal weight over horizons and popularity groups. Pages are the clusters and appear in both strata, so they are resampled jointly
(paired); weeks of 2023 and 2026 are separate time clusters. W-post vs W-new uses independent page resampling. **Adjusted DiD**
(secondary): weighted least squares of a on stratum indicators, horizon and popularity-group indicators and context features computed
from each window's 400-day context only (log mean level, coefficient of variation, OLS slope of log(1 + views) per 100 days,
lag-7 autocorrelation of log(1 + views), share of days above 3× the context median), standardised; each stratum × horizon ×
popularity-group cell gets equal total weight, as in the unadjusted estimate; the adjusted W-post vs W-new contrast has the same sign
convention as the unadjusted one (W-post minus W-new); bootstrap as above.

### 5.5 Tier P
Both issues pooled, the issue being a fixed averaging level (issue × domain × nominal horizon cells): `tfm3-uni` vs `stat-ens`,
`chronos2-uni`, `tfm25` and `naive` on MASE and SQL, univariate tasks, structured domains (load, price, river) and Wikipedia
separately (16 tests, Holm within S6); series clusters only. With two overlapping issues, inference generalises over series, not
over time; Tier P is a check of Tier B's direction, not an estimate of its own.

### 5.6 Negative controls and calibration
White noise (best: `mean`), random walk (best: `naive`), and the FX and FRED series: a model beating the optimal baseline by a resolved
margin would point to an evaluation problem. Coverage with Wilson intervals over series.

## 6. Exclusion rules (before any model runs)

A series window is excluded if: more than 5% missing values in the TSFM context or in the (shorter) classical context; any target value
missing in the window; constant context; context shorter than two seasons (Tier B). Members of group tasks are checked on the calendar of
the first target, as the runner builds them. Interior gaps are filled by linear interpolation for all models. Tier W: the redirect rule
(§3.3). Exclusions are listed with reasons in the task tables.

## 7. Compute and determinism

TSFMs and ablations on Google Colab (Tesla T4, fp32), possibly from two Google accounts sharing the results folder; classical models
on a local CPU (StatsForecast, deterministic, parallel over series; parallel and serial runs were verified identical). Every run records
GPU, driver, package versions, model revisions and the code bundle sha256; runs are resumable, and several sessions may share a job
(duplicate forecasts are identical; one is kept). Latency per model at batch sizes 1, 32, 256 on a Tier B subset.

## 8. Outputs released

Per-series, per-window quantile forecasts of every configuration, metrics, manifests, code, analysis outputs, and the post-release data
where licences allow (otherwise the download code with timestamps). TimesFM-3 weights are not redistributed (non-commercial licence).

## 9. What is not done

No fine-tuning; no hyperparameter search on test data; no use of post-release data before the freeze; no claims about commercial use.

## 10. Smoke test, pilot and review (before the freeze)

The smoke test and the pilot used synthetic data and pre-release data only; they are not part of the confirmatory analysis.

- **Pilot B/W**: Tier B and W designs one year earlier (windows 2025-09-01 … 11-02, non-overlapping origins). **W-seen disclosure:**
  W-seen windows are 2023 windows in both pilot and main, so pilot W-seen forecasts partly coincide with confirmatory ones; they were
  produced but not scored (the scoring code holds them out for the pilot), and are re-run under the frozen code.
- **Pilot C**: a separate synthetic set (5 replications per cell, seed 99, reference-study replications 900–904).
- Results, run times, precision and minimum detectable effects: `reports/pilot_synthetic.md`, `reports/pilot_BW.md`. In short:
  **H2** MDE at 80% power about 0.05 in log (5% on the ratio of advantages) at α = 0.05 and 0.06 at the strictest Holm step, from
  the W-post standard error and the conservative bound DiD SE ≤ √2 × SE; W-post vs W-new MDE about 10%. **H4c/H4d** standard errors
  with 200 replications about 0.0004 and 0.0002 (the ±3% margin is a relevance threshold). **H5** pilot interval about ±11%, hence the
  non-superiority form. **Family P** pilot two-way 95% intervals: P1 ±7%, P2 ±2.5%, P3 ±4%, P4a ±6%, P5 ±1.5% (MASE).
- **Changes made after the pilot**: hourly classical ensemble through MSTL (AutoETS with season 24 selected non-seasonal models; pilot
  load MASE relative to Seasonal Naive 3.14 at h = 24); Open-Meteo pinned to ERA5 (the default product fills recent days with other
  models).
- **Changes made after the adversarial review**: information-matched primary contrasts (oracle covariates had entered P1–P3) and no
  double counting of series in uni and group tasks; H1 made descriptive with fev's published ensemble as the Tier A reference; H2
  re-specified with a TSFM reference, joint page resampling, pre-specified adjustment and MDE; Tier P redesigned (scoring window after
  sealing, complete seal, analysis plan); two-way bootstrap, horizon-equal weights, explicit families with Holm-based decisions and
  10,000 replicates; equivalence margins for H4c, H4d, H5; Chronos-2 pipeline control and claim reproduction for RQ1; redirect rule;
  exclusion checks on the classical context and on group calendars; coverage ignoring missing truth; replication cells with H = 12.
- **Changes for the schedule** (before any 2026 data were downloaded): test windows end on 2026-10-20 instead of 2026-11-02, origins
  every week for long horizons, two short prospective issues instead of one long one.

## 11. Deviations

None yet.
