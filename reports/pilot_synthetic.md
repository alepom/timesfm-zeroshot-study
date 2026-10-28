# Pilot C (synthetic_pilot) — 28/10/2026

Not part of the confirmatory analysis (protocol section 10). 5 replications per cell, seed 99; reference-study replications 900–904.
Run on Colab T4 with tfmstudy 0.2.0/0.2.1. Failures: 0. MSTL fell back to the naive forecast on 5 of 60 tasks.
Run times (classical, Colab CPU): ARIMA 4.3 s/task (non-hourly only), Theta 0.5 s/task, MSTL 0.5 s/task.

## Geometric mean of MASE relative to Seasonal Naive

| cell                     |   tfm3 |   tfm3-eval |   tfm3-uni |   tfm25 |   chronos2 |   tirex2 |   stat-ens |   naive |   mean |
|:-------------------------|-------:|------------:|-----------:|--------:|-----------:|---------:|-----------:|--------:|-------:|
| mv:independent:8         |   0.72 |        0.72 |       0.73 |    0.72 |       0.73 |     0.72 |       0.87 |    1    |   0.74 |
| mv:known_cov:snr0.5      |   0.6  |        0.6  |       0.6  |    0.6  |       0.59 |     0.59 |       0.68 |    1    |   0.61 |
| mv:known_cov:snr2        |   0.37 |        0.37 |       0.76 |    0.76 |       0.35 |     0.36 |       0.76 |    1    |   0.76 |
| mv:lead_lag:1            |   0.78 |        0.78 |       0.78 |    0.75 |       0.82 |     0.81 |       0.91 |    1    |   0.74 |
| mv:lead_lag:24           |   0.69 |        0.69 |       0.93 |    0.94 |       0.6  |     0.95 |       0.95 |    1    |   0.99 |
| mv:lead_lag:5            |   0.86 |        0.83 |       0.95 |    0.95 |       0.73 |     0.95 |       0.96 |    1    |   1.13 |
| mv:spurious_past_cov     |   0.76 |        0.76 |       0.75 |    0.77 |       0.82 |     0.76 |       0.89 |    1    |   0.77 |
| mv:var1:2                |   0.65 |        0.65 |       0.67 |    0.67 |       0.66 |     0.66 |       0.76 |    1    |   0.69 |
| mv:var1:32               |   0.73 |        0.73 |       0.74 |    0.74 |       0.73 |     0.74 |       0.74 |    1    |   0.73 |
| mv:var1:64               |   0.76 |        0.76 |       0.76 |    0.76 |       0.76 |     0.76 |       0.76 |    1    |   0.76 |
| mv:var1:8                |   0.83 |        0.83 |       0.84 |    0.84 |       0.83 |     0.83 |       0.85 |    1    |   0.83 |
| rep:D6:n200              |   0.82 |        0.8  |       0.82 |    0.76 |       0.9  |     1.41 |       0.72 |    1    |   4.05 |
| rep:D6:n48               |   2.26 |        1.21 |       2.26 |    1.09 |       1.42 |     1.59 |       2.18 |    1    |   5.74 |
| rep:D7:n200              |   5.44 |        7.56 |       5.44 |    8.58 |       3.61 |     3.26 |       2.79 |    1    |  40.27 |
| rep:D7:n48               |   0.81 |        0.81 |       0.81 |    0.89 |       0.51 |     0.71 |       0.49 |    1    |   1.49 |
| rep:D8:n200              |   0.76 |        0.76 |       0.76 |    0.76 |       0.76 |     0.76 |       1.1  |    1    |   1.05 |
| rep:D8:n48               |   0.67 |        0.69 |       0.67 |    0.68 |       0.64 |     0.69 |       0.98 |    1    |   0.93 |
| uni:context_ladder:1024  |   0.34 |        0.34 |       0.34 |    0.35 |       0.35 |     0.38 |       0.76 |    1.67 |   1.54 |
| uni:context_ladder:15360 |   0.25 |        0.25 |       0.25 |    0.26 |       0.25 |     0.26 |       0.56 |    1.21 |   1.2  |
| uni:context_ladder:256   |   0.38 |        0.36 |       0.38 |    0.38 |       0.35 |     0.72 |       0.86 |    1.85 |   1.43 |
| uni:context_ladder:4096  |   0.24 |        0.24 |       0.24 |    0.25 |       0.24 |     0.26 |       0.57 |    1.17 |   1.09 |
| uni:context_ladder:64    |   0.82 |        0.84 |       0.82 |    0.75 |       0.89 |     0.77 |       1.15 |    1.15 |   1.27 |
| uni:long_horizon:1024    |   0.25 |        0.25 |       0.25 |    0.26 |       0.24 |   nan    |       0.64 |    1.21 |   1.11 |
| uni:long_horizon:256     |   0.23 |        0.23 |       0.23 |    0.23 |       0.22 |     0.24 |       0.53 |    1.04 |   1.01 |
| uni:long_horizon:512     |   0.24 |        0.24 |       0.24 |    0.24 |       0.24 |   nan    |       0.55 |    1.12 |   1.1  |
| uni:long_period:ctx200   |   0.62 |        0.62 |       0.62 |    0.61 |       0.48 |     0.87 |       0.93 |    1    |   3.78 |
| uni:outliers             |   0.27 |        0.27 |       0.27 |    0.26 |       0.37 |     0.32 |       0.83 |    1.66 |   1.4  |
| uni:random_walk          |   1.09 |        1.09 |       1.09 |    1.15 |       1.05 |     0.98 |       1.08 |    1    |   1.91 |
| uni:regime_change        |   1.07 |        1.1  |       1.07 |    0.72 |       0.38 |     0.88 |       0.88 |    2.16 |   2.81 |
| uni:two_seasons:ctx168   |   0.58 |        0.59 |       0.58 |    0.5  |       0.49 |     0.79 |       0.71 |    1.4  |   1.33 |
| uni:two_seasons:ctx672   |   0.31 |        0.31 |       0.31 |    0.32 |       0.31 |     0.35 |       0.74 |    1.44 |   1.33 |
| uni:white_noise          |   0.91 |        0.92 |       0.91 |    0.92 |       0.92 |     0.92 |       0.92 |    1    |   0.92 |

Mean 80% coverage: {'tfm3': 0.797, 'tfm3-eval': 0.799, 'tfm3-uni': 0.796, 'tfm25': 0.8, 'chronos2': 0.792, 'tirex2': 0.795, 'stat-ens': 0.833, 'naive': 0.972, 'mean': 0.763}

## Observations used to finalise the protocol
- Negative controls behave as expected: on white noise no model beats the historic mean (0.91-0.92 vs 0.92); on the random walk no TSFM beats the naive forecast by a clear margin.
- TiRex-2 has a 320-step horizon limit (long-horizon cells not applicable).
- Large failures of all TSFMs on the saturating logistic process (D7, n=200), largest with Google's benchmark flags (tfm3-eval) and TimesFM-2.5: consistent with arXiv:2610.00589 and with hypothesis H3.
- TimesFM-3 reacts poorly to a level shift 24 steps before the origin (1.07 vs 0.38 for Chronos-2).
- Multivariate mode helps when a real cross-series dependence exists (lead-lag 24: 0.69 vs 0.93 univariate; known covariate SNR 2: 0.37 vs 0.76) and is neutral on independent series.
- Precision expected with 200 replications (from the pilot variance): 95% half-width of TimesFM-3 vs stat-ens about +-2.6% (median cell), +-12% in the noisiest cell (D6, n=48).
