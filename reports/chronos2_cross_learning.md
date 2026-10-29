# Chronos-2 pipeline control: cause of the deviations from fev's published results (29/10/2026)

**Finding.** fev's published Chronos-2 results (`chronos-2.csv`, fev 0.8.0, commit f323f6c) were produced by fev's script
`examples/chronos-2/evaluate_model.py`, which calls `pipeline.predict_fev(task, batch_size=100, as_univariate=False,
cross_learning=True)`. With cross-learning, the series of a batch (up to 100) share information through Chronos-2's group
attention. Our Tier A runs use the library default (`predict_df`, no cross-learning), so each series is forecast on its own, as
TimesFM-3 does.

**Evidence** (exploratory diagnostic, Colab T4; `reports/diag_chronos2_variants.csv`, `reports/diag_chronos2_fev_script.csv`):

| task | fev script, cross_learning=True | published | fev script, cross_learning=False | our Tier A run |
|---|---|---|---|---|
| australian_tourism | 0.6769555 | 0.6769555 | 0.7396415 | 0.7396415 |
| ecdc_ili | 2.2713938 | 2.2713937 | 2.4665179 | 2.4665179 |
| hospital | 0.6860471 | 0.6860471 | 0.7013574 | 0.7013574 |
| SZ_TAXI_1H | 0.3977234 | 0.3977235 | 0.4166707 | 0.4166707 |
| rohlik_sales_1W | 1.2741360 | 1.2741411 | 1.1477089 | 1.1338174 (imputed) |

(SQL.) With cross-learning on, fev's script reproduces the published values to |deviation| ≤ 4·10⁻⁶; with it off, it reproduces our
values (rohlik: our run differs by 1.2% only through Google's last-value imputation of missing values, which the protocol applies
to all Tier A models). Input preparation does not matter: one item per call vs batches of 100, imputed vs raw values, synthetic
vs real timestamps give the same SQL to 10⁻⁸ (`diag_chronos2_variants.csv`).

**Consequences.**
1. Our pipeline is validated: the Chronos-2 control fails only because the published reference uses a different inference
   option. Single-item tasks (no other series to share with) matched to four decimals, as expected.
2. The comparison published by Google (TimesFM-3 vs fev's Chronos-2) sets a model forecasting each series alone against
   Chronos-2 with cross-learning across up to 100 series, whose result also depends on batch composition. Cross-learning helped
   Chronos-2 on four of the five tasks (2–9% lower SQL) and hurt it on rohlik (+11%).
3. Our RQ1 claim reproduction (`skill_ours`) is the like-for-like comparison: both models without cross-learning.

Reported in the paper as a neutral methodological note.
