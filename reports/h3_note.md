# H3 — the confirmatory cell never triggers the detrending gate (29/10/2026)

**Confirmatory result (protocol as frozen):** `tfm3-nodetrend` vs `tfm3`, MASE over steps 25–48, D7, n = 200, H = 48:
ratio 1.000, CI [1.000, 1.000], p = 1. H3 is **not supported**. The two configurations produce bit-identical forecasts on all
200 replications.

**Why.** The detrending is used only if `std(detrended) < 0.5·std(original)` (R² > 0.75 of a linear fit on the context;
model.py:510, see `reports/mechanism.md`). D7 places the logistic inflection at 0.6·(n + H). With n = 200 and H = 48 the
inflection is at t ≈ 149: the 200-point context is a long flat start, the rise, and the start of saturation, and a line fits it with
R² > 0.75 in **0 of 200** contexts. So the mechanism H3 targets is switched off in the cell chosen for the test. This is a design
error of the protocol: the gate was documented but not checked on the chosen cell before the freeze. The result is reported as
pre-registered and is not replaced.

Gate activation by cell (share of the 200 contexts with R² > 0.75): n48/H48 1.0, n48/H12 1.0, n200/H48 **0.0**, n200/H12 1.0.

**Exploratory (post hoc, labelled as such in the paper).** Same contrast on the replication cells where the gate fires
(geometric mean ratio `tfm3-nodetrend`/`tfm3`, series bootstrap, 2,000 replicates, no multiplicity adjustment):

| cell | metric | ratio | 95% CI |
|---|---|---|---|
| D7 n48 H48 | MASE steps 25–48 | 0.873 | 0.810–0.941 |
| D7 n48 H48 | MASE | 0.830 | 0.775–0.889 |
| D7 n200 H12 | MASE | 0.930 | 0.876–0.985 |
| D7 n48 H12 | MASE | 1.888 | 1.765–2.011 |

Reading: when the gate fires on a saturating process, removing the detrending helps at long horizons (the added line overshoots
the asymptote) but almost doubles the error at H = 12 with a short context, where continuing the trend is right. The stage trades
short-horizon accuracy for long-horizon overshoot. It is not a uniform defect.

Tier C ablations (secondary S5) agree: `tfm3-nodetrend` 1.007 (1.004–1.010) on the whole synthetic set, ≈ 1 on Tier B.
