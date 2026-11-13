# How Far Does Zero-Shot Go?

**A leakage-aware evaluation of TimesFM 3 on post-release, multivariate and controlled data.**
Alessio Pomaro, 2026. Preprint, not peer reviewed.

> Paper: [doi:10.5281/zenodo.24194671](https://doi.org/10.5281/zenodo.24194671) ·
> Code, forecasts and data: [doi:10.5281/zenodo.24194745](https://doi.org/10.5281/zenodo.24194745) · How to cite: `CITATION.cff`
>
> This repository contains the code, the frozen protocol, the data manifests, the metrics of every configuration and the Tier P
> seals needed to check every number in the paper. Quantile forecasts and downloaded data are in the Zenodo data record.

## Question

TimesFM 3 (Google, August 2026) is reported to lead the public forecasting benchmarks, whose data are old and public. Does its
advantage hold on data published after its release, and where does zero-shot forecasting fail? The protocol was frozen by a git tag
before any post-release test data were downloaded.

## Design

| Tier | What | Data |
|---|---|---|
| A | Reproduction of Google's fev-bench results | 30 fev-bench tasks |
| B | Post-release real data, 2026-09-01 to 10-20 | load and prices (12 countries), weather (20 cities), river flow (20 gauges), FX, Treasury yields and oil |
| W | Memorisation probe | the same Wikipedia pages inside (2023) and outside (2026) the model's documented pageview corpus |
| P | Prospective check | two issues (2026-10-30, 11-02) sealed by SHA-256 and publicly timestamped before their outcomes existed |
| C | Controlled synthetic data | 7,600 tasks: seasonality, short histories, multivariate structure, covariates, level shifts, saturation |

Models: TimesFM 3 (univariate, multivariate, benchmark flags, ablations), TimesFM 2.5, Chronos-2, TiRex-2, and a classical ensemble
(MSTL family or ETS/ARIMA/Theta) with seasonal naive, naive and mean. Contrasts are paired log ratios of MASE and scaled quantile
loss (SQL), with a two-way cluster bootstrap and Holm corrections within pre-declared families.

## Main results

| Contrast (TimesFM 3 univariate vs) | MASE ratio (95% CI) | SQL ratio (95% CI) |
|---|---|---|
| Classical ensemble, Tier B structured domains (P1) | 0.836 (0.786–0.882), resolved | 0.707 (0.649–0.764), resolved |
| Chronos-2, same tasks (P2) | 0.953 (0.924–0.981), resolved | 0.956 (0.933–0.979), resolved |
| TimesFM 2.5, same tasks (P3) | 0.951 (0.917–0.988), not resolved after Holm | 0.939 (0.914–0.965), resolved |
| Random walk, FX, yields, oil (H5) | 1.110 (1.049–1.178): worse | |
| Classical ensemble, Tier P load, price, river (S6) | 1.073 (0.977–1.175), not resolved | 0.750 (0.648–0.861), resolved |
| Classical ensemble, Tier P Wikipedia (S6) | 0.743 (0.703–0.783), resolved | 0.419 (0.382–0.458), resolved |

In short: on the four structured post-release domains TimesFM 3 beats the classical ensemble clearly and Chronos-2 narrowly. In two
overlapping prospective weeks its quantile-loss advantage over the ensemble is confirmed, but on load, prices and river flow its
point-forecast advantage is not, driven mainly by electricity prices after a rise in level. On near-random-walk series it is worse
than the random walk. In synthetic data it fails on a saturating process and after recent upward level shifts. The Wikipedia probe
does not identify an effect of memorisation. Google's published fev-bench results largely reproduce on a 30-task sample.

## Main limitations

Six real domains and seven weeks of post-release data; few time clusters; Tier P covers two overlapping weeks and checks direction,
not size; the memorisation probe compares harder 2026 windows with 2023 windows; all neural models ran on one T4 GPU. See the paper's
Limitations section.

## Repository layout

| Path | Content |
|---|---|
| `tfmstudy/` | all code: downloads and tasks (`data/`), model adapters, runner, scoring, analysis, robustness, paper tables |
| `protocol.md` | frozen protocol (tag `protocol-frozen-v1`) and its deviations (§11) |
| `data/manifests/` | series and page lists, selection statistics, data hashes, Tier A task list and published results |
| `results/` | metrics of every tier and configuration, run records (model revisions, configurations), analysis outputs |
| `reports/` | confirmatory, descriptive, robustness and exploratory outputs; Tier A reproduction; mechanism and configuration notes |
| `reports/seals/` | Tier P seals and their RFC 3161 and OpenTimestamps receipts, with verification instructions |
| `paper/` | LaTeX sources and generated tables |
| `notebooks/` | Colab notebook that runs the same modules on a T4 |
| `tests/` | tests of the statistics and of Tier P scoring |

## Reproduce

Python 3.12; analysis dependencies in `requirements.lock`, GPU dependencies in `requirements-colab.txt`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.lock
.venv/bin/python -m tfmstudy.paper            # tables and macros from the stored metrics and analysis outputs
.venv/bin/python -m tfmstudy.check_numbers    # every number in the text against the data files
```

Re-running the confirmatory analysis (`python -m tfmstudy.analysis`), re-scoring forecasts or re-running models needs the forecasts
and data of the Zenodo data record, unpacked into `data/` and `results/`. Model runs use the notebook in `notebooks/` on a Colab T4.

## AI assistance

The study code was written, the experiments and analyses run and the paper drafted by an AI coding agent (Claude Code) under the
author's direction. Details in the paper.

## Citation and licences

Please cite the paper (see `CITATION.cff`). Code: MIT. Paper, documentation, metrics and the forecasts of every model except
TimesFM 3: CC BY 4.0. TimesFM 3 forecasts: CC BY-NC 4.0, and their use remains subject to the TimesFM Non-Commercial License.
Third-party data keep their licences: see `LICENSES.md` and `THIRD_PARTY_NOTICES.md`.
