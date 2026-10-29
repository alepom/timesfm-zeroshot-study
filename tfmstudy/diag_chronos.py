"""Diagnostic (exploratory, not part of the frozen protocol): why our Chronos-2 deviates from fev's published results.

The Tier A pipeline control (protocol section 3.1) failed on most univariate many-series tasks while matching to four decimals
on single-item multivariate tasks. This job runs Chronos-2 on a few failing tasks in four variants and scores each with the
Tier A scorer (fev's evaluation summary):

  single-imputed   our Tier A path: one item per predict_df call, Google's last-value imputation, synthetic hourly timestamps
  single-raw       one item per call, missing values left to Chronos-2, real timestamps
  batch-imputed    all items of a window in one predict_df call (batch_size=100, as in fev's published run), imputed
  batch-raw        as fev's published run: all items, batch_size=100, raw values, real timestamps

Output: <out>/summary.csv (one row per task and variant, with fev's published SQL).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import QUANTILES
from .tiera import MANIFEST, Inputs, _fetch, _impute, score

TASKS = ("australian_tourism", "ecdc_ili", "hospital", "SZ_TAXI_1H", "rohlik_sales_1W")
VARIANTS = ("single-imputed", "single-raw", "batch-imputed", "batch-raw")


def _long(dataset, task, columns, impute: bool, synthetic: bool) -> pd.DataFrame:
    """One fev window split (HF dataset of list columns) as a long DataFrame."""
    frames = []
    for rec in dataset:
        n = len(rec[task.timestamp_column])
        df = {"id": rec[task.id_column]}
        df["timestamp"] = (pd.date_range("2000-01-01", periods=n, freq="h") if synthetic
                           else pd.to_datetime(rec[task.timestamp_column]))
        for c in columns:
            v = rec[c]
            if impute and len(v) and not isinstance(v[0], str):
                v = _impute(np.asarray(v, dtype=np.float32))
            df[c] = v
        frames.append(pd.DataFrame(df))
    return pd.concat(frames, ignore_index=True)


def _predict(pipeline, task, window, variant: str) -> dict[str, np.ndarray]:
    impute = variant.endswith("imputed")
    synthetic = variant == "single-imputed"
    past, future = window.get_input_data()
    covs = task.known_dynamic_columns + task.past_dynamic_columns
    ctx = _long(past, task, task.target_columns + covs, impute, synthetic)
    fut = None
    if task.known_dynamic_columns:
        fut = _long(future, task, task.known_dynamic_columns, impute, False)
        if synthetic:  # continue each item's synthetic clock
            last = ctx.groupby("id").timestamp.max()
            fut["timestamp"] = [last[i] + pd.Timedelta(hours=k + 1) for i, k in
                                zip(fut.id, fut.groupby("id").cumcount())]
    ctx = ctx.groupby("id", sort=False).tail(15360)  # Tier A context cap
    target = task.target_columns if len(task.target_columns) > 1 else task.target_columns[0]
    kwargs = dict(prediction_length=task.horizon, quantile_levels=list(QUANTILES), id_column="id",
                  timestamp_column="timestamp", target=target)
    ids = list(dict.fromkeys(ctx.id))
    if variant.startswith("batch"):
        preds = [pipeline.predict_df(ctx, future_df=fut, batch_size=100, **kwargs)]
    else:
        preds = [pipeline.predict_df(ctx[ctx.id == i], future_df=None if fut is None else fut[fut.id == i], **kwargs)
                 for i in ids]
    pred = pd.concat(preds, ignore_index=True)
    names = task.target_columns
    out = {}
    for i in ids:
        p = pred[pred.id == i]
        q = [np.stack([(p[p.target_name == n] if "target_name" in p else p)[str(lv)].to_numpy() for lv in QUANTILES], -1)
             for n in names]
        out[i] = np.stack(q).astype(np.float32)
    return out


def run(out: Path, tasks: list[str], device: str) -> pd.DataFrame:
    from chronos import Chronos2Pipeline

    from .adapters import Chronos2

    pipeline = Chronos2Pipeline.from_pretrained(Chronos2.hf_repo, device_map=device)  # unpinned, as in Tier A
    table = pd.read_csv(MANIFEST)
    published = pd.read_csv(_fetch("chronos-2.csv")).set_index("task_name")
    rows_out = []
    for name in tasks:
        inputs = Inputs(next(table[table.task_name == name].itertuples()))
        task = inputs.task
        rows = inputs.rows(name)
        for variant in VARIANTS:
            t0 = time.perf_counter()
            forecasts = {}
            for w in range(task.num_windows):
                pred = _predict(pipeline, task, task.get_window(w), variant)
                g = rows[rows.window == w]
                past, _ = task.get_window(w).get_input_data()
                ids = past.data.table.column(task.id_column).to_pylist()
                for tid, item in zip(g.task_id, g["item"]):
                    forecasts[tid] = pred[ids[int(item)]]
            summary = score(task, rows, forecasts, f"chronos2:{variant}")
            rec = {"task_name": name, "variant": variant, "SQL": summary["SQL"], "MASE": summary["MASE"],
                   "SQL_published": published.loc[name, "SQL"], "seconds": round(time.perf_counter() - t0, 1)}
            rec["SQL_vs_published"] = rec["SQL"] / rec["SQL_published"] - 1
            rows_out.append(rec)
            print(json.dumps(rec), flush=True)
    df = pd.DataFrame(rows_out)
    dest = out
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "summary.csv", index=False)
    return df


def run_fev(out: Path, tasks: list[str], device: str) -> pd.DataFrame:
    """fev's own Chronos-2 script (examples/chronos-2/evaluate_model.py at fev commit f323f6c): `predict_fev` with
    batch_size=100, float32, seed 123, with cross_learning=True (the script's default) and False."""
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device, torch_dtype=torch.float32)
    table = pd.read_csv(MANIFEST)
    published = pd.read_csv(_fetch("chronos-2.csv")).set_index("task_name")
    rows_out = []
    for name in tasks:
        task = Inputs(next(table[table.task_name == name].itertuples())).task
        for cross_learning in (True, False):
            torch.manual_seed(123)
            t0 = time.perf_counter()
            preds, _ = pipeline.predict_fev(task, batch_size=100, as_univariate=False, cross_learning=cross_learning)
            summary = task.evaluation_summary(preds, model_name="chronos2-fev", trained_on_this_dataset=False)
            rec = {"task_name": name, "variant": f"fev-cross_learning={cross_learning}", "SQL": summary["SQL"],
                   "MASE": summary["MASE"], "SQL_published": published.loc[name, "SQL"],
                   "seconds": round(time.perf_counter() - t0, 1)}
            rec["SQL_vs_published"] = rec["SQL"] / rec["SQL_published"] - 1
            rows_out.append(rec)
            print(json.dumps(rec), flush=True)
    df = pd.DataFrame(rows_out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary_fev.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fev", action="store_true", help="run fev's own Chronos-2 script (predict_fev) instead")
    args = parser.parse_args()
    (run_fev if args.fev else run)(Path(args.out), args.tasks.split(","), args.device)


if __name__ == "__main__":
    main()
