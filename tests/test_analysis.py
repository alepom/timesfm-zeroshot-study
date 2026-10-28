"""Regression tests of the confirmatory estimator (protocol section 5) on data with known answers."""

import numpy as np
import pandas as pd

from tfmstudy import analysis as A


def _metrics(rng, effect: float, n_series: int = 12, models=("a", "b"), tier_c: bool = False) -> pd.DataFrame:
    """Two models on unbalanced domain x horizon cells; model a = b * exp(effect) up to noise."""
    rows = []
    for dom in ("load", "price"):
        for s in range(n_series):
            for h, n_win in ((24, 6), (168, 2)):
                for k in range(n_win):
                    base = np.exp(rng.normal(0, 0.3))
                    origin = pd.Timestamp("2026-09-01", tz="UTC") + pd.Timedelta(days=7 * k + (0 if h == 24 else 3))
                    task = f"{dom}-{s}-{h}-{k}"
                    for m in models:
                        score = base * (np.exp(effect + rng.normal(0, 0.02)) if m == "a" else 1.0)
                        r = {"task_id": task, "series_id": f"{dom}:{s}", "model": m, "domain": dom, "horizon": h,
                             "mode": "uni", "origin": origin, "mase": score, "sql": score}
                        if tier_c:
                            r.update(rep=s, cell=dom)
                        rows.append(r)
    return pd.DataFrame(rows)


def test_contrast_recovers_effect_and_covers():
    rng = np.random.default_rng(1)
    df = _metrics(rng, effect=np.log(0.9))
    r = A.contrast(df, "a", "b", "mase", df.model.notna(), ["domain", "horizon"], reps=500)
    assert abs(r["ratio"] - 0.9) < 0.01
    assert r["ci95"][0] < 0.9 < r["ci95"][1]
    assert r["resolved"] if "resolved" in r else r["p"] < 0.05


def test_equal_weights_over_horizons():
    rng = np.random.default_rng(2)
    df = _metrics(rng, effect=0.0)
    # Make model a better only at h=168 (2 windows) and equal at h=24 (6 windows).
    df.loc[(df.model == "a") & (df.horizon == 168), "mase"] *= np.exp(-0.2)
    r = A.contrast(df, "a", "b", "mase", df.model.notna(), ["domain", "horizon"], reps=200)
    assert abs(np.log(r["ratio"]) - (-0.1)) < 0.01  # mean over horizons, not over windows (-0.05)


def test_tier_c_clusters_are_replications():
    rng = np.random.default_rng(3)
    df = _metrics(rng, effect=0.0, tier_c=True)
    df["series_id"] = df.task_id + ":" + df.groupby(["task_id", "model"]).cumcount().astype(str)
    pr = A.paired(df, "a", "b", "mase", df.model.notna())
    assert (pr.cluster == pr.task_id).all()


def test_family_h_always_seven_tests():
    out = A.family_h(None, None, None, reps=10)
    assert [r["test"] for r in out] == list(A.H_TESTS)
    assert not any(r["supported"] for r in out)


def test_tost_equivalence():
    rng = np.random.default_rng(4)
    df = _metrics(rng, effect=0.0, n_series=40)
    r = A.contrast(df, "a", "b", "mase", df.model.notna(), ["domain", "horizon"], margin=1.05, reps=500)
    assert r["p_tost"] < 0.05 and 1 / 1.05 < r["ci90"][0] and r["ci90"][1] < 1.05
