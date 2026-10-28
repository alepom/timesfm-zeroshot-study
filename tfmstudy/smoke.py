"""Smoke test run on Colab: installs work, every adapter returns well-formed quantiles, rough timings.

Uses synthetic series and pre-release real data (2025) only, so no post-release test data are
looked at before the protocol is frozen. Results are not part of the study.

    python -m tfmstudy.smoke --out /content/drive/MyDrive/timesfm-zeroshot-study/results/smoke
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from . import adapters, metrics, provenance
from .adapters import Task

RNG = np.random.default_rng(17)


def synthetic_tasks() -> dict[str, tuple[Task, np.ndarray, int]]:
    """name -> (task, truth (targets, horizon), season)."""
    out = {}
    t = np.arange(1024 + 64)
    sine = 10 + 3 * np.sin(2 * np.pi * t / 24) + RNG.normal(0, 0.5, t.size)
    out["sine24"] = (Task(sine[:1024], 64), sine[None, 1024:], 24)

    walk = np.cumsum(RNG.normal(0, 1, 512 + 32)) + 100
    out["random_walk"] = (Task(walk[:512], 32), walk[None, 512:], 1)

    # y follows x with a lag of 5 steps: a multivariate model can exploit it for the first 5 steps.
    n, h, lag = 512, 24, 5
    x = np.zeros(n + h + lag)
    for i in range(1, x.size):
        x[i] = 0.9 * x[i - 1] + RNG.normal()
    y = 0.8 * np.roll(x, lag) + RNG.normal(0, 0.3, x.size)
    xy = np.stack([x, y])[:, lag:]
    out["lead_lag"] = (Task(xy[:, :n], h), xy[:, n:], 1)

    # Target driven by a covariate whose future values are known.
    c = RNG.normal(0, 1, n + h)
    noise = np.zeros(n + h)
    for i in range(1, noise.size):
        noise[i] = 0.5 * noise[i - 1] + RNG.normal(0, 0.3)
    target = 50 + 2 * c + noise
    out["known_covariate"] = (Task(target[:n], h, future_cov=c), target[None, n:], 1)
    return out


def real_task_pre_release() -> tuple[Task, np.ndarray, int] | None:
    """German hourly load (Energy-Charts, CC BY 4.0) with Berlin temperature, Sep–Oct 2025."""
    import pandas as pd
    import requests

    power = requests.get("https://api.energy-charts.info/public_power",
                         params={"country": "de", "start": "2025-09-01", "end": "2025-10-21"}, timeout=60).json()
    load = next(p["data"] for p in power["production_types"] if p["name"] == "Load")
    s = pd.Series(load, index=pd.to_datetime(power["unix_seconds"], unit="s", utc=True), dtype=float)
    s = s.resample("1h").mean()
    weather = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": 52.52, "longitude": 13.41, "start_date": "2025-09-01", "end_date": "2025-10-21",
        "hourly": "temperature_2m", "timezone": "GMT"}, timeout=60).json()["hourly"]
    temp = pd.Series(weather["temperature_2m"], index=pd.to_datetime(weather["time"], utc=True), dtype=float)
    df = pd.concat({"load": s, "temp": temp}, axis=1).dropna()
    h = 24
    y, c = df["load"].to_numpy(), df["temp"].to_numpy()
    return Task(y[:-h], h, future_cov=c), y[None, -h:], 24


MODEL_CONFIGS = [
    ("timesfm-3.0", {"mode": "multivariate"}),
    ("timesfm-3.0", {"mode": "multivariate", "use_symmetric_averaging": True, "make_positive": True}),
    ("timesfm-3.0", {"mode": "univariate"}),
    ("timesfm-2.5", {}),
    ("chronos-2", {}),
    ("tirex-2", {}),
    # Load check of the decontaminated variant used in the Tier A contamination contrast.
    ("tirex-2", {"hf_repo": "NX-AI/TiRex-2-gifteval-zs"}),
]


def evaluate(adapter, tasks) -> dict:
    results = {}
    for name, (task, truth, season) in tasks.items():
        entry: dict = {}
        try:
            t0 = time.perf_counter()
            q = adapter.predict([task])[0]
            entry["seconds"] = round(time.perf_counter() - t0, 4)
            entry["shape"] = list(q.shape)
            entry["quantile_crossing"] = bool(np.any(np.diff(q, axis=-1) < -1e-6))
            entry["mase"] = metrics.mase(truth, q, task.target, season).tolist()
            entry["wql"] = metrics.wql(truth, q).tolist()
            entry["coverage80"] = metrics.coverage(truth, q).tolist()
        except Exception:
            entry["error"] = traceback.format_exc(limit=6)
            if hasattr(adapter, "last_raw"):
                entry["raw_type"] = repr(type(adapter.last_raw))
        results[name] = entry
    return results


def timesfm3_properties(adapter) -> dict:
    """Scale/shift/sign equivariance, 1D vs 2D input, determinism (TimesFM3 forecaster defaults)."""
    model = adapter.model
    x = (np.cumsum(RNG.normal(0, 1, 600)) + 50).astype(np.float32)

    def q(arr, **kw):
        return np.asarray(next(iter(model.predict_batch(contexts=[arr], horizon=64, return_quantiles=True, **kw))).quantiles)

    base = q(x)
    rel = lambda a, b: float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9))
    return {
        "determinism_rel_diff": rel(q(x), base),
        "scale_x1000_rel_diff": rel(q(1000 * x) / 1000, base),
        "shift_plus1e4_rel_diff": rel(q(x + 1e4) - 1e4, base),
        "sign_flip_rel_diff": rel(-q(-x)[..., ::-1], base),
        "sign_flip_rel_diff_with_symavg": rel(-q(-x, use_symmetric_averaging=True)[..., ::-1],
                                              q(x, use_symmetric_averaging=True)),
        "input_1d_vs_2d_rel_diff": rel(q(x[None, :]).reshape(base.shape), base),
    }


def ablation_switch_check(adapter) -> dict:
    """Checks that a module override changes the output and is restored afterwards (functionality only)."""
    t = np.arange(600)
    trending = Task(5 + 0.05 * t + RNG.normal(0, 0.5, t.size), 48)  # strong linear trend: detrending applies
    before = adapter.predict([trending])[0]
    ablated = adapters.TimesFM3(device=adapter.device, model_overrides={"use_linear_detrending": False})
    ablated.model = adapter.model
    changed = ablated.predict([trending])[0]
    after = adapter.predict([trending])[0]
    return {
        "override_changes_output": bool(np.max(np.abs(changed - before)) > 1e-6),
        "restored_after_override": bool(np.array_equal(before, after)),
    }


def throughput(adapter, n_series=256, context=2048, horizon=64) -> dict:
    tasks = [Task(np.cumsum(RNG.normal(0, 1, context)), horizon) for _ in range(n_series)]
    adapter.predict(tasks[:4])  # warm-up
    t0 = time.perf_counter()
    adapter.predict(tasks)
    sec = time.perf_counter() - t0
    out = {"n_series": n_series, "context": context, "horizon": horizon, "seconds": round(sec, 3),
           "series_per_second": round(n_series / sec, 2)}
    try:
        import torch

        out["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except Exception:
        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", default="all", help="comma-separated adapter names or 'all'")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"environment": provenance.environment(), "models": []}

    tasks = synthetic_tasks()
    try:
        real = real_task_pre_release()
        if real:
            tasks["de_load_2025_temp"] = real
            report["real_task_context_len"] = int(real[0].target.shape[1])
    except Exception:
        report["real_task_error"] = traceback.format_exc(limit=4)

    wanted = None if args.models == "all" else set(args.models.split(","))
    loaded: dict = {}
    for name, options in MODEL_CONFIGS:
        if wanted and name not in wanted:
            continue
        record: dict = {"model": name, "options": options}
        try:
            adapter = adapters.REGISTRY[name](device=args.device, **options)
            adapter.revision = record["hf_revision"] = provenance.hf_revision(adapter.hf_repo)
            key = (name, adapter.hf_repo)
            if key in loaded:  # reuse weights across option variants
                adapter.model, adapter.load_seconds = loaded[key].model, loaded[key].load_seconds
            else:
                adapter.load()
                loaded[key] = adapter
            record["adapter"] = adapter.describe()
            record["tasks"] = evaluate(adapter, tasks)
            if name == "timesfm-3.0" and options == {"mode": "multivariate"}:
                record["properties"] = timesfm3_properties(adapter)
                record["ablation_switch"] = ablation_switch_check(adapter)
            if not options or options == {"mode": "multivariate"}:
                record["throughput"] = throughput(adapter)
        except Exception:
            record["error"] = traceback.format_exc(limit=8)
        report["models"].append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "tasks"}, default=str)[:600], flush=True)

    stamp = report["environment"]["utc"].replace(":", "").replace("-", "")
    path = out_dir / f"smoke_{stamp}.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    print("wrote", path)


if __name__ == "__main__":
    main()
