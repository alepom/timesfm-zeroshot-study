"""Counts and descriptive facts quoted in the paper's text, recomputed from the data: reports/paper_facts.json.

    python -m tfmstudy.facts --synthetic-forecasts <dir with tfm3 part files of the main synthetic run>

The TimesFM-3 synthetic forecasts live on Drive (results/forecast/main/synthetic/tfm3); copy them to a scratch
directory first. Every number here is checked against the text by `tfmstudy.check_numbers`.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "paper_facts.json"


def tier_b() -> dict:
    m = pd.read_parquet(ROOT / "results/forecast/main/B/metrics.parquet", columns=["series_id", "task_id", "mode",
                                                                                    "domain", "origin", "horizon"])
    tasks = pd.read_parquet(ROOT / "data/processed/main/tasks.parquet")
    series = pd.read_parquet(ROOT / "data/processed/main/series.parquet")
    hourly = tasks[tasks.domain.isin(["load", "price"]) & tasks["mode"].eq("uni")]
    first = series.dropna(subset=["value"]).groupby("series_id").timestamp.min()
    ctx = [((pd.Timestamp(r.origin) - first[json.loads(r.targets)[0]]) / pd.Timedelta(hours=1)) + 1
           for r in hourly.itertuples()]
    return {"series": int(m.series_id.nunique()), "domains": int(m.domain.nunique()),
            "series_windows": int(len(m[["series_id", "origin", "horizon"]].drop_duplicates())),
            "series_tasks_by_mode": m[["series_id", "task_id", "mode"]].drop_duplicates().groupby("mode").size()
            .astype(int).to_dict(),
            "hourly_context_points": {"min": int(min(ctx)), "median": int(np.median(ctx)), "max": int(max(ctx))}}


def tier_a() -> dict:
    rep = pd.read_csv(ROOT / "reports/tierA_reproduction.csv")
    rep = rep[rep.model.eq("tfm3-eval") & rep.SQL_reldiff.notna()]
    fev = pd.read_csv(ROOT / "data/manifests/tierA_fev.csv").set_index("task_name")
    cov = (fev.num_known_cov + fev.num_past_cov).reindex(rep.task_name).to_numpy() > 0
    dev = rep.SQL_reldiff.abs().to_numpy() * 100
    out = rep[~rep.reproduced.astype(bool)]
    chronos_fail = rep[~rep.control_ok.astype(bool) & rep.snaive_ok.astype(bool)]  # control rule of tiera.py
    return {"tfm3_eval_tasks": int(len(rep)), "tasks_with_covariates": int(cov.sum()),
            "median_abs_sql_dev_pct_without_cov": round(float(np.median(dev[~cov])), 4),
            "median_abs_sql_dev_pct_with_cov": round(float(np.median(dev[cov])), 2),
            "chronos2_control_failures": int(len(chronos_fail)),
            "chronos2_control_failures_higher_sql": int((chronos_fail.chronos2_SQL_vs_fev > 0).sum()),
            "snaive_control_failures": int((~rep.snaive_ok.astype(bool)).sum()),
            "not_reproduced": int(len(out)),
            "not_reproduced_with_covariates": int(((fev.num_known_cov + fev.num_past_cov).reindex(out.task_name) > 0).sum()),
            "not_reproduced_range_pct": [round(float(out.SQL_reldiff.min() * 100), 1),
                                         round(float(out.SQL_reldiff.max() * 100), 1)]}


def tier_c_d7(forecast_dir: Path) -> dict:
    from .data import synthetic

    t = pd.read_parquet(ROOT / "data/processed/synthetic/tasks.parquet")
    t = t[t.cell.isin(["rep:D7:n200", "rep:D7:n48"])]
    arr = synthetic.load_arrays(ROOT / "data/processed/synthetic/arrays.parquet")
    fc = pd.concat(pd.read_parquet(f, filters=[("task_id", "in", list(t.task_id))])
                   for f in glob.glob(str(forecast_dir / "*.parquet"))).set_index("task_id")
    out = {}
    for cell, g in t.groupby("cell"):
        below, above_truth = [], []
        for r in g.itertuples():
            a = arr[r.array_id]["target"][0]
            n, h = r.context_len, r.horizon
            q = np.frombuffer(fc.loc[r.task_id, "quantiles"], dtype=np.float32).reshape(
                json.loads(fc.loc[r.task_id, "shape"]))[0, :, 4]
            later = slice(24, h)  # steps 25-48
            below.append(q[later].mean() < a[n - 1])
            above_truth.append(q[later].mean() > a[n:n + h][later].mean())
        out[cell] = {"replications": int(len(g)), "median_below_last_obs_steps25_48": int(np.sum(below)),
                     "median_above_truth_steps25_48": int(np.sum(above_truth))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-forecasts", required=True)
    args = parser.parse_args()
    cells = json.loads((ROOT / "reports/descriptive.json").read_text())["tierC_cells"]
    facts = {"tier_b": tier_b(), "tier_a": tier_a(), "tier_c_d7_tfm3": tier_c_d7(Path(args.synthetic_forecasts)),
             "tier_c_cells": {"cells": len(cells),
                              "tfm3_beats_best_classical": sum(c["tfm3"] < c["best_classical_value"] for c in cells)}}
    OUT.write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
