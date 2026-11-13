# v1.0.0 — TimesFM 3 zero-shot study

First public release accompanying the preprint *How Far Does Zero-Shot Go? A Leakage-Aware Evaluation of TimesFM 3 on
Post-Release, Multivariate and Controlled Data* (Alessio Pomaro, 2026).

## Contents
- `tfmstudy/`: data download and task building for every tier, model adapters (TimesFM 3 and 2.5, Chronos-2, TiRex-2, classical
  models), forecasting runner, scoring, confirmatory and robustness analyses, paper tables and figures, number check.
- Frozen protocol (`protocol.md`, git tag `protocol-frozen-v1`) with its deviations (§11); series and page manifests.
- Metrics of every configuration and tier, run records with model revisions, confirmatory and descriptive outputs.
- Tier P seals with their RFC 3161 and OpenTimestamps receipts (`reports/seals/`).
- Colab notebook that runs the same modules on a T4.

## Not included here
Quantile forecasts (about 1.1 GB) and the downloaded data are in the Zenodo data record. TimesFM 3 weights are not redistributed.

## Reproduce the paper's tables
`python -m tfmstudy.analysis && python -m tfmstudy.analysis --s6-only && python -m tfmstudy.paper` (no GPU; uses the metrics in
`results/`).
