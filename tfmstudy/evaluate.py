"""Scores saved forecasts and computes the paired contrasts of the protocol (sections 4-5).

    python -m tfmstudy.evaluate --results results/forecast --data data/processed --tier synthetic --period pilot

Writes <results>/<period>/<tier>/metrics.parquet with one row per (task, model, target row):
mase, rmsse, sql, wql, coverage80, coverage60, width80 and the relative scores to Seasonal Naive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .forecast import load_tasks


ENSEMBLE_MEMBERS = {24: ("mstl", "mstl-theta", "mstl-arima"), "other": ("ets", "arima", "theta")}


def read_forecasts(model_dir: Path) -> dict[str, np.ndarray]:
    out = {}
    for part in sorted(model_dir.glob("part-*.parquet")):
        for row in pd.read_parquet(part).itertuples():
            out[row.task_id] = np.frombuffer(row.quantiles, dtype=np.float32).reshape(json.loads(row.shape))
    return out


def read_fallbacks(model_dir: Path) -> dict[str, int]:
    """Per-task count of rows that fell back to the naive forecast (classical models; empty for older runs)."""
    out = {}
    for part in sorted(model_dir.glob("part-*.parquet")):
        df = pd.read_parquet(part)
        if "fallbacks" in df:
            out.update(dict(zip(df.task_id, df.fallbacks.astype(int))))
    return out


def stat_ensemble(tasks: pd.DataFrame, forecasts: dict) -> dict[str, np.ndarray]:
    """Equal-weight ensemble (protocol section 2): the three MSTL models on season-24 data, else ETS+ARIMA+Theta."""
    from .baselines import ensemble

    out = {}
    for row in tasks.itertuples():
        members = ENSEMBLE_MEMBERS[24] if row.season == 24 else ENSEMBLE_MEMBERS["other"]
        parts = [forecasts.get(m, {}).get(row.task_id) for m in members]
        if all(p is not None for p in parts):
            out[row.task_id] = ensemble(parts)
    return out


# Strata not scored in a period: the pilot's W-seen windows (2023) coincide with confirmatory ones,
# so they stay unseen until the protocol is frozen (protocol section 10).
HELD_OUT = {("W", "pilot"): ("W-seen",)}


def score(tier: str, period: str, results: Path, data: Path) -> pd.DataFrame:
    tasks, make = load_tasks(tier, period, data)
    held = HELD_OUT.get((tier, period))
    if held:
        tasks = tasks[~tasks.stratum.isin(held)]
    run_dir = results / period / tier
    models = sorted(p.name for p in run_dir.iterdir() if p.is_dir() and any(p.glob("part-*.parquet")))
    forecasts = {m: read_forecasts(run_dir / m) for m in models}
    fallbacks = {m: read_fallbacks(run_dir / m) for m in models}
    forecasts["stat-ens"] = stat_ensemble(tasks, forecasts)
    fallbacks["stat-ens"] = {}
    for row in tasks.itertuples():  # an ensemble window inherits its members' fallbacks
        members = ENSEMBLE_MEMBERS[24] if row.season == 24 else ENSEMBLE_MEMBERS["other"]
        fallbacks["stat-ens"][row.task_id] = sum(fallbacks.get(m, {}).get(row.task_id, 0) for m in members)
    models.append("stat-ens")
    meta_cols = [c for c in ("domain", "mode", "lineage", "stratum", "quintile", "cell", "family", "horizon", "origin", "rep")
                 if c in tasks.columns]
    rows = []
    for row in tasks.itertuples():
        task, truth, season = make(row)
        context = task.target
        if hasattr(row, "targets"):
            target_ids = json.loads(row.targets)
        else:
            target_ids = [f"{row.task_id}:{i}" for i in range(context.shape[0])]
        scale_mase = metrics.seasonal_naive_scale(context, season)
        diffs = context[:, season:] - context[:, :-season] if context.shape[1] > season else np.diff(context, axis=1)
        scale_rmsse = np.sqrt(np.nanmean(diffs ** 2, axis=1))
        for m in models:
            q = forecasts[m].get(row.task_id)
            if q is None:
                continue
            err = truth - q[..., 4]
            loss = metrics.quantile_loss(truth, q) * 2 / 9  # mean over levels of 2 * pinball
            for i, sid in enumerate(target_ids):
                rows.append({
                    "task_id": row.task_id, "model": m, "series_id": sid,
                    "fallback": bool(fallbacks.get(m, {}).get(row.task_id, 0)),
                    **{c: getattr(row, c) for c in meta_cols},
                    "mase": float(np.nanmean(np.abs(err[i])) / scale_mase[i]),
                    "rmsse": float(np.sqrt(np.nanmean(err[i] ** 2)) / scale_rmsse[i]),
                    "sql": float(np.nanmean(loss[i]) / scale_mase[i]),
                    "wql": float(np.nanmean(loss[i]) / np.nanmean(np.abs(truth[i]))) if np.nanmean(np.abs(truth[i])) > 0 else np.nan,
                    "coverage80": _coverage(truth[i], q[i, :, 0], q[i, :, 8]),
                    "coverage60": _coverage(truth[i], q[i, :, 1], q[i, :, 7]),
                    "width80": float(np.nanmean(q[i, :, 8] - q[i, :, 0]) / scale_mase[i]),
                    # H3: steps 25-48 (1-based), where trend continuation shows on the logistic DGP.
                    "mase_25_48": (float(np.nanmean(np.abs(err[i, 24:48])) / scale_mase[i])
                                   if err.shape[1] >= 48 else np.nan),
                })
    df = pd.DataFrame(rows)
    base = df[df.model == "snaive"].set_index(["task_id", "series_id"])[["mase", "sql"]]
    df = df.join(base, on=["task_id", "series_id"], rsuffix="_snaive")
    for col in ("mase", "sql"):
        df[f"rel_{col}"] = df[col] / df[f"{col}_snaive"]
    df.to_parquet(run_dir / "metrics.parquet", index=False)
    return df


def _coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of observed (non-missing) values inside [lo, hi]."""
    ok = ~np.isnan(y)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))) if ok.any() else np.nan


def gmean(x: pd.Series) -> float:
    x = x[(x > 0) & np.isfinite(x)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else np.nan


def paired_contrast(df: pd.DataFrame, a: str, b: str, metric: str, cluster: str, strata: str | None = None,
                    reps: int = 2000, seed: int = 17) -> dict:
    """Ratio of geometric means a/b on identical (task, series) pairs, with a cluster bootstrap CI.

    Clusters (e.g. series) are resampled with replacement within strata (e.g. domains); a ratio < 1
    means `a` has the lower error. Reported as 'not resolved' when the 95% interval contains 1.
    """
    keys = ["task_id", "series_id"]
    pa = df[df.model == a].set_index(keys)[metric]
    pb = df[df.model == b].set_index(keys)[metric]
    both = pd.concat({"a": pa, "b": pb}, axis=1).dropna()
    both = both[(both.a > 0) & (both.b > 0)]
    if both.empty:
        return {"a": a, "b": b, "metric": metric, "n": 0}
    info = df.drop_duplicates(keys).set_index(keys, drop=False).loc[both.index]
    both["d"] = np.log(both.a) - np.log(both.b)
    both["cluster"] = info[cluster].to_numpy()
    both["stratum"] = info[strata].to_numpy() if strata else "all"
    per_cluster = both.groupby(["stratum", "cluster"]).d.agg(["sum", "count"]).reset_index()
    groups = [(g["sum"].to_numpy(), g["count"].to_numpy()) for _, g in per_cluster.groupby("stratum")]

    def estimate(samples) -> float:
        # Mean log ratio within each stratum, then equal weight across strata (protocol section 4.3).
        return float(np.mean([sm.sum() / ct.sum() for sm, ct in samples]))

    est = estimate(groups)
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for k in range(reps):
        sample = []
        for sm, ct in groups:
            pick = rng.integers(0, len(sm), len(sm))
            sample.append((sm[pick], ct[pick]))
        stats[k] = estimate(sample)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = 2 * min(np.mean(stats >= 0), np.mean(stats <= 0))
    return {"a": a, "b": b, "metric": metric, "n": len(both), "clusters": len(per_cluster),
            "ratio": float(np.exp(est)), "ci_low": float(np.exp(lo)), "ci_high": float(np.exp(hi)),
            "p_boot": float(max(p, 1 / reps)), "resolved": bool(lo > 0 or hi < 0)}


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/forecast")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--tier", required=True, choices=["synthetic", "B", "W"])
    parser.add_argument("--period", default="pilot")
    args = parser.parse_args()
    df = score(args.tier, args.period, Path(args.results), Path(args.data))
    group = "cell" if args.tier == "synthetic" else "domain"
    table = df.groupby([group, "model"]).agg(rel_mase=("rel_mase", gmean), rel_sql=("rel_sql", gmean),
                                            cov80=("coverage80", "mean"), n=("mase", "size")).reset_index()
    print(table.pivot(index=group, columns="model", values="rel_mase").round(3).to_string())


if __name__ == "__main__":
    main()
