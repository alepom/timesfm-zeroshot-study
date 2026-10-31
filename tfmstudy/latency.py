"""Latency and throughput of the foundation models on the Colab T4 (protocol section 7, RQ7).

    python -m tfmstudy.latency --out <results dir> [--models tfm3-uni,tfm3-eval,tfm25,chronos2-uni,tirex2-uni]

256 univariate Tier B tasks (hourly load and price, horizon 24, the longest contexts of the study, about 15,000 points)
are forecast in calls of 1, 32 and 256 series. For each batch size the model is loaded (TimesFM-3 fixes its batch size at
load time) and warmed up with one call; then every batch size is timed over the same 256 series, with `torch.cuda.synchronize()` around the timed calls.
TimesFM-3 receives the call size as `per_core_batch_size`; the other adapters batch internally. Output:
<out>/latency.csv (model, batch size, seconds per call, series per second, peak GPU memory) and the environment record.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import adapters, provenance
from .forecast import CONFIGS, RUN_OPTIONS, _free_gpu, _prepare, load_tasks

SIZES = (1, 32, 256)
N_SERIES = 256


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(key: str, tasks: list, device: str) -> list[dict]:
    import torch

    name, all_options = CONFIGS[key]
    options = {k: v for k, v in all_options.items() if k not in RUN_OPTIONS}
    run_opts = {k: v for k, v in all_options.items() if k in RUN_OPTIONS}
    tasks = [p for t in tasks for p in _prepare(t, run_opts)][:N_SERIES]  # e.g. inputs="uni" drops covariates
    rows = []
    for size in SIZES:
        opts = {**options, "batch_size": size} if name == "timesfm-3.0" else options
        adapter = adapters.REGISTRY[name](device=device, **opts)
        adapter.revision = provenance.hf_revision(adapter.hf_repo)
        adapter.load()
        try:
            adapter.predict(tasks[:size])  # warm-up (kernels, compilation, allocator)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            _sync()
            t0 = time.perf_counter()
            for start in range(0, len(tasks), size):
                adapter.predict(tasks[start:start + size])
            _sync()
        except torch.cuda.OutOfMemoryError:  # reported, not hidden: the T4 has 15 GB
            rows.append({"model": key, "batch_size": size, "series": len(tasks), "error": "out of memory"})
            print(json.dumps(rows[-1]), flush=True)
            del adapter
            _free_gpu()
            continue
        seconds = time.perf_counter() - t0
        calls = len(tasks) // size
        rows.append({"model": key, "batch_size": size, "series": len(tasks), "seconds_total": round(seconds, 3),
                     "seconds_per_call": round(seconds / calls, 4), "series_per_second": round(len(tasks) / seconds, 2),
                     "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
                     "revision": adapter.revision})
        print(json.dumps(rows[-1]), flush=True)
        del adapter
        _free_gpu()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default="tfm3-uni,tfm3-eval,tfm25,chronos2-uni,tirex2-uni")
    parser.add_argument("--data", default=None, help="data/processed directory (default: <out>/../../data/processed)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.out)
    data_dir = Path(args.data) if args.data else out.parent.parent / "data" / "processed"
    table, make = load_tasks("B", "main", data_dir)
    sub = table[(table["mode"] == "uni") & table.domain.isin(["load", "price"]) & (table.horizon == 24)]
    sub = sub.sort_values("task_id").head(N_SERIES)
    tasks = [make(r)[0] for r in sub.itertuples()]
    rows = []
    for key in args.models.split(","):
        rows += measure(key, tasks, args.device)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "latency.csv", index=False)
    (out / "environment.json").write_text(json.dumps(provenance.environment(), indent=1, default=str))


if __name__ == "__main__":
    main()
