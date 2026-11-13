"""Forecasting job run on Colab: one tier, one or more model configurations, resumable.

    python -m tfmstudy.forecast --out <results dir> --tier synthetic|B|W|P --period pilot|main|prospective-<day> \
        --models tfm3,chronos2 [--data <data dir>] [--limit N]

Forecasts are appended to <out>/<period>/<tier>/<model_key>/part-*.parquet (columns: task_id,
quantiles as float32 bytes, shape). Tasks already present are skipped, so an interrupted Colab
session continues where it stopped. Excluded tasks (protocol section 6) are not run.

Several sessions may run the same job side by side: part files carry a per-session tag so they never
overwrite each other, and classical models (fitted row by row, so order does not matter) take their tasks
in a random order and re-read the finished tasks at every flush, which keeps the duplicated work small.
Duplicate forecasts of a task are identical; readers keep one.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import secrets
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import adapters, baselines, provenance
from .adapters import Task

# Model configurations of the protocol (section 2): key -> (adapter name, options).
CONFIGS = {
    "tfm3": ("timesfm-3.0", {"mode": "multivariate"}),
    "tfm3-eval": ("timesfm-3.0", {"mode": "multivariate", "use_symmetric_averaging": True, "make_positive": True}),
    "tfm3-uni": ("timesfm-3.0", {"mode": "univariate"}),
    # Secondary: same as tfm3-uni with the classical models' context length (protocol section 2).
    "tfm3-c2016": ("timesfm-3.0", {"mode": "univariate", "context": 2016}),
    # Added after the freeze (not pre-registered): equal-context comparison with Chronos-2, which truncates at 8,192.
    "tfm3-c8192": ("timesfm-3.0", {"mode": "univariate", "context": 8192}),
    "chronos2-c2016": ("chronos-2", {"inputs": "uni", "context": 2016}),
    # Ablations (protocol section 3.6): one inference flag or one module attribute changed from tfm3.
    "tfm3-nodetrend": ("timesfm-3.0", {"mode": "multivariate", "model_overrides": {"use_linear_detrending": False}}),
    "tfm3-nocpmrevin": ("timesfm-3.0", {"mode": "multivariate", "model_overrides": {"use_iterative_cpm_revin": False}}),
    "tfm3-novarattn": ("timesfm-3.0", {"mode": "multivariate", "model_overrides": {"use_variate_attention": False}}),
    "tfm3-nostitch": ("timesfm-3.0", {"mode": "multivariate", "model_overrides": {"use_stitching": False}}),
    "tfm3-sym": ("timesfm-3.0", {"mode": "multivariate", "use_symmetric_averaging": True}),
    "tfm3-pos": ("timesfm-3.0", {"mode": "multivariate", "make_positive": True}),
    "tfm3-znorm": ("timesfm-3.0", {"mode": "multivariate", "use_znorm": True}),
    "tfm3-edge": ("timesfm-3.0", {"mode": "multivariate", "padding_mode": "edge"}),
    "tfm25": ("timesfm-2.5", {}),
    "chronos2": ("chronos-2", {}),
    "chronos2-uni": ("chronos-2", {"inputs": "uni"}),
    "tirex2": ("tirex-2", {}),
    "tirex2-uni": ("tirex-2", {"inputs": "uni"}),
    "tirex2-g": ("tirex-2", {"hf_repo": "NX-AI/TiRex-2-gifteval-zs"}),
    "tirex2-f": ("tirex-2", {"hf_repo": "NX-AI/TiRex-2-fevbench"}),
    "mean": ("classical", {"method": "mean"}),
    "snaive": ("classical", {"method": "snaive"}),
    "naive": ("classical", {"method": "naive"}),
    "ets": ("classical", {"method": "ets"}),
    "arima": ("classical", {"method": "arima"}),
    "theta": ("classical", {"method": "theta"}),
    "mstl": ("classical", {"method": "mstl"}),
    "mstl-theta": ("classical", {"method": "mstl-theta"}),
    "mstl-arima": ("classical", {"method": "mstl-arima"}),
}
# Options applied by the runner to the inputs, for any model, instead of being passed to the adapter:
# inputs="uni" gives each target alone and no covariates; context=N keeps the last N observations.
RUN_OPTIONS = ("inputs", "context")
# Where a configuration is not run (protocol section 2): seasonal AutoARIMA is too slow on hourly
# contexts, the MSTL models are the hourly ensemble members, TiRex-2 truncates horizons above 320.
APPLICABLE = {
    "arima": lambda row: row.season != 24,
    "mstl": lambda row: row.season == 24,
    "mstl-theta": lambda row: row.season == 24,
    "mstl-arima": lambda row: row.season == 24,
    "tirex2": lambda row: row.horizon <= 320,
    "tirex2-uni": lambda row: row.horizon <= 320,
    "tirex2-g": lambda row: row.horizon <= 320,
    "tirex2-f": lambda row: row.horizon <= 320,
}
ABLATIONS = ("tfm3-nodetrend", "tfm3-nocpmrevin", "tfm3-novarattn", "tfm3-nostitch", "tfm3-sym", "tfm3-pos",
             "tfm3-znorm", "tfm3-edge")
ABLATION_SHARE, ABLATION_SEED = 0.2, 17


def ablation_series() -> set[str]:
    """The fixed 20% sample of Tier B series used for the ablations (protocol section 3.6)."""
    from .data.build import MANIFESTS

    ids = sorted(pd.read_csv(MANIFESTS / "tierB_series.csv").series_id)
    pick = np.random.default_rng(ABLATION_SEED).choice(len(ids), size=round(ABLATION_SHARE * len(ids)), replace=False)
    return {ids[i] for i in sorted(pick)}


def _ablation_applies(key: str, row) -> bool:
    """Where an ablation can have an effect (protocol section 3.6); Tier B: sampled series and group tasks."""
    if getattr(row, "cell", None) is not None:  # Tier C
        multivariate, covariate = row.mode in ("group", "cov"), "known_cov" in row.cell
    else:
        if getattr(row, "domain", None) == "wiki":
            return False
        targets = json.loads(row.targets)
        covariate = bool(json.loads(row.future_covariates))
        multivariate = row.mode == "group" or covariate
        if row.mode == "uni" and targets[0] not in _ABLATION_SERIES():
            return False
        if row.mode == "group" and key != "tfm3-novarattn":
            return False
    if key in ("tfm3-nocpmrevin", "tfm3-nostitch"):
        return row.horizon > 64  # shorter horizons are decoded from a single token
    if key == "tfm3-novarattn":
        return multivariate
    if key == "tfm3-edge":
        return covariate
    return True


_ABLATION_CACHE: set[str] | None = None


def _ABLATION_SERIES() -> set[str]:
    global _ABLATION_CACHE
    if _ABLATION_CACHE is None:
        _ABLATION_CACHE = ablation_series()
    return _ABLATION_CACHE


for _key in ABLATIONS:
    APPLICABLE[_key] = (lambda k: lambda row: _ablation_applies(k, row))(_key)

TSFM_CONTEXT = 15360
BATCH = 64
FLUSH_EVERY = 512
SESSION = secrets.token_hex(3)


# ------------------------------------------------------------------ task loading

def load_tasks(tier: str, period: str, data_dir: Path) -> tuple[pd.DataFrame, callable]:
    """Returns the task table and a function task_row -> (Task, truth, season)."""
    if tier == "synthetic":
        from .data import synthetic

        name = {"pilot": "synthetic_pilot", "regime": "synthetic_regime"}.get(period, "synthetic")
        tasks = pd.read_parquet(data_dir / name / "tasks.parquet")
        arrays = synthetic.load_arrays(data_dir / name / "arrays.parquet")

        def make(row):
            a = arrays[row.array_id]
            n, h = row.context_len, row.horizon
            fcov = a["future_cov"]
            task = Task(a["target"][:, :n], h, past_cov=a["past_cov"], future_cov=fcov, task_id=row.task_id)
            return task, a["target"][:, n:n + h], row.season
        return tasks, make

    prefix = "wiki_" if tier == "W" else ""  # B and P share the layout
    tasks = pd.read_parquet(data_dir / period / f"{prefix}tasks.parquet")
    tasks = tasks[tasks.excluded == ""].reset_index(drop=True)
    series = pd.read_parquet(data_dir / period / f"{prefix}series.parquet")
    by_id = {sid: g.set_index("timestamp")["value"] for sid, g in series.groupby("series_id")}

    def make(row):
        targets = json.loads(row.targets)
        covs = json.loads(row.future_covariates)
        idx = by_id[targets[0]].index
        pos = idx.get_loc(row.origin)
        lo = max(0, pos + 1 - TSFM_CONTEXT)
        if getattr(row, "context_start", None) is not None and not pd.isna(row.context_start):
            lo = max(lo, int(idx.searchsorted(row.context_start)))
        h = row.horizon
        target = np.stack([_interp(by_id[s].reindex(idx).iloc[lo:pos + 1].to_numpy()) for s in targets])
        truth = np.stack([by_id[s].reindex(idx).iloc[pos + 1:pos + 1 + h].to_numpy() for s in targets])
        fcov = None
        if covs:
            fcov = np.stack([_interp(by_id[c].reindex(idx).iloc[lo:pos + 1 + h].to_numpy()) for c in covs])
        return Task(target, h, future_cov=fcov, task_id=row.task_id), truth, row.season
    return tasks, make


def _interp(x: np.ndarray) -> np.ndarray:
    """Interior gaps filled by linear interpolation for all models alike (protocol section 6)."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    if ok.all() or not ok.any():
        return x
    i = np.arange(len(x))
    return np.interp(i, i[ok], x[ok])


def _prepare(task: Task, run_opts: dict) -> list[Task]:
    """The inputs a configuration sees (RUN_OPTIONS); several tasks when targets are split."""
    target, past, future = task.target, task.past_cov, task.future_cov
    n = run_opts.get("context")
    if n and target.shape[1] > n:
        cut = target.shape[1] - n
        target = target[:, cut:]
        past = None if past is None else past[:, cut:]
        future = None if future is None else future[:, cut:]
    if run_opts.get("inputs") == "uni":
        return [Task(target[i:i + 1], task.horizon, task_id=f"{task.task_id}#{i}") for i in range(target.shape[0])]
    return [Task(target, task.horizon, past_cov=past, future_cov=future, task_id=task.task_id)]


def _predict(adapter, classical: bool, tasks: list[Task], seasons: list[int]) -> list[np.ndarray]:
    return adapter.predict(tasks, seasons=seasons) if classical else adapter.predict(tasks)


def _free_gpu() -> None:
    """Releases cached CUDA memory after a failed call (an out-of-memory batch leaves the cache fragmented, and the
    one-task retries then fail too). Does not change any forecast."""
    gc.collect()
    import sys
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ output

def _done(model_dir: Path) -> set[str]:
    done: set[str] = set()
    for part in model_dir.glob("part-*.parquet"):
        done.update(pd.read_parquet(part, columns=["task_id"]).task_id)
    return done


def _flush(model_dir: Path, buffer: list[dict]) -> None:
    if not buffer:
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(model_dir.glob("part-*.parquet")))
    pd.DataFrame(buffer).to_parquet(model_dir / f"part-{n:05d}-{SESSION}.parquet", index=False)
    buffer.clear()


def run_model(key: str, tasks: pd.DataFrame, make, out_dir: Path, device: str, limit: int | None) -> dict:
    name, all_options = CONFIGS[key]
    options = {k: v for k, v in all_options.items() if k not in RUN_OPTIONS}
    run_opts = {k: v for k, v in all_options.items() if k in RUN_OPTIONS}
    model_dir = out_dir / key
    done = _done(model_dir)
    if key in APPLICABLE:
        tasks = tasks[[APPLICABLE[key](r) for r in tasks.itertuples()]]
    todo = tasks[~tasks.task_id.isin(done)]
    if limit:
        todo = todo.head(limit)
    record = {"model_key": key, "adapter": name, "options": all_options, "tasks_total": len(tasks),
              "already_done": len(done), "to_run": len(todo)}
    if todo.empty:
        return record

    if name == "classical":
        adapter = baselines.Classical(device="cpu", **options)
    else:
        adapter = adapters.REGISTRY[name](device=device, **options)
        adapter.revision = provenance.hf_revision(adapter.hf_repo)
    adapter.load()
    record["adapter_info"] = adapter.describe()

    buffer: list[dict] = []
    failures = []
    t0 = time.perf_counter()
    rows = list(todo.itertuples())
    if name == "classical":
        random.Random(SESSION).shuffle(rows)
    pos = 0
    while pos < len(rows):
        batch = rows[pos:pos + BATCH]
        pos += BATCH
        built = [make(r) for r in batch]
        task_list = [b[0] for b in built]
        pieces = [_prepare(t, run_opts) for t in task_list]
        flat = [p for ps in pieces for p in ps]
        seasons = [b[2] for b, ps in zip(built, pieces) for _ in ps]
        classical = name == "classical"
        try:
            flat_preds = _predict(adapter, classical, flat, seasons)
            flat_fb = list(getattr(adapter, "task_fallbacks", [0] * len(flat)))
        except Exception:
            flat_fb = [0] * len(flat)
            # Fall back to one task at a time so a single bad task does not lose the batch.
            flat_preds = []
            _free_gpu()
            for task, season in zip(flat, seasons):
                try:
                    flat_preds.append(_predict(adapter, classical, [task], [season])[0])
                except Exception:
                    failures.append({"task_id": task.task_id, "error": traceback.format_exc(limit=3)[-600:]})
                    flat_preds.append(None)
                    _free_gpu()
        preds, fbs, k = [], [], 0
        for ps in pieces:
            part = flat_preds[k:k + len(ps)]
            fbs.append(int(sum(flat_fb[k:k + len(ps)])) if len(flat_fb) == len(flat) else 0)
            k += len(ps)
            preds.append(None if any(q is None for q in part) else np.concatenate(part, axis=0))
        for task, q, fb in zip(task_list, preds, fbs):
            if q is not None:
                buffer.append({"task_id": task.task_id, "quantiles": q.astype(np.float32).tobytes(),
                               "shape": json.dumps(list(q.shape)), "fallbacks": fb})
        if len(buffer) >= FLUSH_EVERY:
            _flush(model_dir, buffer)
            if name == "classical":  # skip what another session finished meanwhile
                done = _done(model_dir)
                rows = rows[:pos] + [r for r in rows[pos:] if r.task_id not in done]
    _flush(model_dir, buffer)
    if hasattr(adapter, "close"):
        adapter.close()
    record.update(seconds=round(time.perf_counter() - t0, 1), failures=len(failures),
                  failure_examples=failures[:5], fallbacks=getattr(adapter, "fallbacks", 0))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--tier", required=True, help="comma-separated: synthetic, B, W")
    parser.add_argument("--period", default="pilot", help="pilot, main or prospective-<issue day>")
    parser.add_argument("--models", required=True)
    parser.add_argument("--data", default=None, help="data/processed directory (default: <out>/../../data/processed)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_root = Path(args.out)
    data_dir = Path(args.data) if args.data else out_root.parent.parent / "data" / "processed"
    for tier in args.tier.split(","):
        out_dir = out_root / args.period / tier
        tasks, make = load_tasks(tier, args.period, data_dir)
        report = {"environment": provenance.environment(), "tier": tier, "period": args.period, "runs": []}
        for key in args.models.split(","):
            rec = run_model(key, tasks, make, out_dir, args.device, args.limit)
            report["runs"].append(rec)
            print(json.dumps({"tier": tier, **{k: v for k, v in rec.items() if k != "failure_examples"}}, default=str), flush=True)
        stamp = report["environment"]["utc"].replace(":", "").replace("-", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"run_{stamp}.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
