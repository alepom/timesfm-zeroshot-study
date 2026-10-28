# Pilot B/W — 28/10/2026

Not part of the confirmatory analysis (protocol §10). Tier B and W designs replicated one year earlier: test windows
2025-09-01…2025-11-02, history from 2024-01-01, non-overlapping origins, Open-Meteo default product. **W-seen was not scored**
(held out by `evaluate.HELD_OUT`), because its 2023 windows coincide in part with the confirmatory ones.

## Runs

- TSFMs on Colab T4 (tfmstudy 0.3.0–0.4.0); classical models on the local CPU (8 cores, niced; parallel and serial fits verified
  bit-identical). Failures: 0 in every run. Naive fallbacks of classical models: 0 on B and W.
- Times on the T4 (B, 5,132 tasks): TimesFM-3 about 40 min per multivariate configuration, 20 min univariate; TiRex-2 13 min
  (vanilla sLSTM); W (5,896 tasks): TimesFM-3 1.3 min, Chronos-2 4 min, TiRex-2 22 min. Local CPU: ETS on hourly B 21 min,
  ARIMA on W 59 min.
- Colab's free GPU quota ran out once; a second Google account sharing the Drive folder continued the run (resumable parts).

## Tier B (relative MASE vs Seasonal Naive, geometric means)

| domain | tfm3-uni | tfm3 | tfm3-eval | chronos2 | tirex2 | tfm25 | stat-ens (MSTL) | ets (raw) | naive |
|---|---|---|---|---|---|---|---|---|---|
| load | 0.415 | 0.392 | 0.389 | 0.424 | 0.471 | 0.464 | 0.490 | 2.860 | 3.014 |
| price | 0.654 | 0.648 | 0.643 | 0.661 | 0.671 | 0.714 | 0.793 | 1.064 | 1.123 |
| weather | 0.706 | 0.706 | 0.703 | 0.722 | 0.722 | 0.729 | 0.844 | 1.091 | 1.240 |
| river | 0.654 | 0.680 | 0.682 | 0.741 | 0.692 | 0.710 | 0.797 | 0.794 | 0.746 |
| fx | 0.807 | 0.797 | 0.799 | 0.765 | 0.755 | 0.773 | 0.755 | 0.756 | 0.753 |
| rates | 0.699 | 0.689 | 0.686 | 0.685 | 0.708 | 0.666 | 0.669 | 0.681 | 0.676 |

(`tfm3`, `chronos2`, `tirex2` use the oracle covariates on load and river and joint forecasts on group tasks.)

**Finding used to change the design:** AutoETS with season 24 selected non-seasonal models (e.g. ETS(A,N,N), flat forecasts) on
hourly series; the hourly ensemble is now MSTL + {ETS, Theta, ARIMA} (load: 1.264 → 0.490).

Coverage of the 80% interval (univariate tasks): TimesFM-3 0.81–0.85 in every domain; Chronos-2 0.71–0.78 on load, price, weather
and river (under-coverage); `stat-ens` 0.78–0.96.

## Family P precision (two-way bootstrap, 4,000 replicates; `reports/pilot_familyP.csv`)

| test | ratio | 95% CI two-way | 95% CI one-way |
|---|---|---|---|
| P1 tfm3-uni vs stat-ens, MASE | 0.843 | 0.788–0.902 | 0.808–0.880 |
| P1, SQL | 0.649 | 0.591–0.700 | 0.604–0.698 |
| P2* tfm3-uni vs chronos2 (price, weather only: no covariates there), MASE | 0.967 | 0.943–0.989 | 0.954–0.981 |
| P3 tfm3-uni vs tfm25, MASE | 0.925 | 0.891–0.964 | 0.907–0.943 |
| P4a tfm3 vs tfm3-uni on group tasks, MASE | 0.953 | 0.907–1.025 | 0.919–0.987 |
| P5 tfm3-eval vs tfm3, MASE | 0.993 | 0.978–1.009 | 0.985–1.001 |

The two-way intervals are 1.5–2× wider than series-only ones (common weekly shocks). Holm-adjusted, P1–P3 would be resolved; P4a and
P5 would not. About 10% of the pilot's bootstrap replicates left the river h = 28 cell empty (two windows, two weeks); with weekly
origins the main study has four such windows and the share should fall to well under 1%.

## Tier W (W-post and W-new only)

| stratum | tfm3 | chronos2 | tirex2 | tfm25 | stat-ens | naive |
|---|---|---|---|---|---|---|
| W-post | 0.695 | 0.727 | 0.722 | 0.708 | 0.922 | 1.126 |
| W-new | 0.705 | 0.762 | 0.767 | 0.750 | 1.448 | 1.156 |

Classical models have heavy right tails on pageviews (90th percentile of relative MASE 2.4–2.9: they extrapolate spikes at the end of
the context); medians are close to Seasonal Naive. TSFMs under-cover slightly (80% coverage 0.73–0.76).

## Minimum detectable effects and power (main study)

- **H2** (DiD W-seen vs W-post, TimesFM-3 vs Chronos-2): SE of the W-post mean advantage 0.0124 (log); the paired DiD SE is at most
  √2 × that, 0.0176, ignoring the positive correlation between a page's two periods. MDE at 80% power: 0.049 in log (about 5% on the
  ratio of advantages) at α = 0.05, 0.062 at the strictest Holm step (α/7).
- **W-post vs W-new** (secondary): pilot DiD 1.024 (0.964–1.099); adjusted 0.973 (0.889–1.075). MDE about 10%.
- **H4c, H4d**: pilot SE with 5 replications 0.0028 and 0.0010; with 200 replications about 0.0004 and 0.0002, so the ±3% margin is
  a relevance threshold, not a power constraint. Pilot ratios 0.995 and 1.000.
- **H4a** (follower only): pilot ratio 0.540, SE 0.080 with 5 replications (about 0.013 with 200).
- **H5**: pilot `tfm3-uni` vs `naive` on FX and FRED 1.058 (95% CI 0.944–1.171): an equivalence test at ±5% would have no power;
  H5 is therefore a non-superiority test at 10% (H0: ratio ≤ 0.90).
