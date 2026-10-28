"""Tier A (protocol section 3.1): reproduction of Google's published fev-bench results for TimesFM-3.

    python -m tfmstudy.tiera select                    # once, locally -> data/manifests/tierA_fev.csv
    python -m tfmstudy.tiera --out <results dir> --models tfm3-eval,tfm3,chronos2,snaive [--limit N] [--tasks a,b]
    python -m tfmstudy.tiera compare --out <results dir>   # -> reports/tierA_reproduction.csv

Selection (`select`, deterministic). Pinned sources, cached in data/raw/tierA/ and recorded with their sha256 in
data/manifests/tierA_sources.json:
  - task definitions: fev-bench `tasks.yaml` of fev v0.9.0 (the fev version in Google's CSV), commit bb3c0ff;
  - domain and frequency group of each task: `DOMAIN_GROUPS` (7 domains) and `FREQUENCY_GROUPS` (sub-hourly, hourly,
    daily, weekly, monthly+) of the official fev leaderboard (HF Space autogluon/fev-leaderboard, src/task_groups.py);
  - column dtypes: dataset card of autogluon/fev_datasets;
  - fev's published Seasonal Naive results (used by `compare` as a data check).
  1. Exclusions: a task one of whose dynamic covariates (known or past-only) has string dtype, because Google's runner
     encodes them with Python `hash()`, which is salted per process; a task whose definition differs from its row in
     Google's CSV. Static covariates are not used by Google's runner and do not matter.
  2. Strata = domain x frequency group of the eligible tasks. Allocation of n = 30: quota q_s = 30 N_s / N,
     a_s = max(1, floor(q_s)); then, one task at a time, a task is added to the stratum with the largest q_s - a_s among
     those with tasks left, or removed from the one with the smallest q_s - a_s among those with a_s > 1, until the
     total is 30 (ties: stratum name).
  3. rng = numpy.random.default_rng(17); strata in sorted order; rng.choice(sorted task names, a_s, replace=False).
  The manifest lists all 100 tasks (fev Task definition as JSON, stratum, `selected`, exclusion reason).

Job. Each selected task is iterated with fev's own windows; every (window, item) is one study `Task`, task_id
"<task_name>|<window>|<item id>", run through `forecast.run_model` (same CONFIGS, APPLICABLE, adapters and part files,
in <out>/A/<model_key>/). Inputs follow Google's runner (fev_bench_timesfm3.ipynb) for all models: missing values
filled by its LastValueImputation (forward fill, leading gaps take the first value, all-missing -> 0) instead of the
linear interpolation of protocol section 6; targets kept together; target and covariates cut to the last 15,360
observations; known covariates = past + future values; past-only covariates; static columns unused. `tfm3-eval` runs
through `TimesFM3Evaluator`, Google's class, which differs from `TimesFM3Forecaster` only in splitting inputs with more
than 32 variates into chunks of 32. Scoring uses fev's `Task.evaluation_summary` (SQL, MASE, WAPE, WQL averaged over
windows, exactly as in Google's CSV); the point forecast is the 0.5 quantile, as TimesFM-3's `forecast` output.
One row per task in <out>/A/<model_key>/metrics.csv.

Google's CSV: google-research/timesfm, commit e51928e27119cb17bebc005be2696b75e0a9e688 (repository HEAD as downloaded on
2026-10-28), timesfm3-usage/benchmarks/fev_bench/fev_bench_results.csv, sha256
0ff232e46df6a7bc504ea288053ddae09aebcfcba9b72d90768615a6ebebcc46; copied to data/manifests/google_fev_bench_results.csv.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import QUANTILES, adapters, provenance
from .adapters import Task
from .forecast import TSFM_CONTEXT, run_model

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "tierA_fev.csv"
GOOGLE_CSV = ROOT / "data" / "manifests" / "google_fev_bench_results.csv"
CACHE = ROOT / "data" / "raw" / "tierA"

FEV_VERSION = "0.9.0"
SOURCES = {
    "tasks.yaml": "https://raw.githubusercontent.com/autogluon/fev/bb3c0ff87465cad18188264458ffa5ad08b3cbf1/"
                  "benchmarks/fev_bench/tasks.yaml",
    "task_groups.py": "https://huggingface.co/spaces/autogluon/fev-leaderboard/raw/"
                      "88036ff47ab3ccc6c03e0ddd21a892a515394f6a/src/task_groups.py",
    "fev_datasets_README.md": "https://huggingface.co/datasets/autogluon/fev_datasets/raw/"
                              "d3e7e720f4158eab03ebcb021f1cce779cd92acc/README.md",
    "seasonal_naive.csv": "https://raw.githubusercontent.com/autogluon/fev/bb3c0ff87465cad18188264458ffa5ad08b3cbf1/"
                          "benchmarks/fev_bench/results/seasonal_naive.csv",
    "chronos-2.csv": "https://raw.githubusercontent.com/autogluon/fev/bb3c0ff87465cad18188264458ffa5ad08b3cbf1/"
                     "benchmarks/fev_bench/results/chronos-2.csv",
    "stat_ensemble.csv": "https://raw.githubusercontent.com/autogluon/fev/bb3c0ff87465cad18188264458ffa5ad08b3cbf1/"
                         "benchmarks/fev_bench/results/stat_ensemble.csv",
}
GOOGLE_SOURCE = {
    "repo": "https://github.com/google-research/timesfm",
    "commit": "e51928e27119cb17bebc005be2696b75e0a9e688",
    "path": "timesfm3-usage/benchmarks/fev_bench/fev_bench_results.csv",
    "local_source": "/tmp/claude-1000/-home-alepom-progetti-lab-test-AP-timesFM-google/"
                    "8793d451-47c9-4afd-a2e5-a1fb6540f9e7/scratchpad/dl_tfm/timesfm-HEAD/"
                    "timesfm3-usage/benchmarks/fev_bench/fev_bench_results.csv",
    "sha256": "0ff232e46df6a7bc504ea288053ddae09aebcfcba9b72d90768615a6ebebcc46",
}
SAMPLE_SIZE = 30
SEED = 17
METRICS = ("SQL", "MASE", "WAPE", "WQL")
# Task fields compared with Google's CSV (same names as fev.Task attributes and as the CSV columns).
DEFINITION_FIELDS = ("dataset_path", "dataset_config", "horizon", "num_windows", "initial_cutoff", "window_step_size",
                     "min_context_length", "max_context_length", "seasonality", "eval_metric", "extra_metrics",
                     "quantile_levels", "id_column", "timestamp_column", "target", "generate_univariate_targets_from",
                     "known_dynamic_columns", "past_dynamic_columns", "static_columns")


# ------------------------------------------------------------------ selection

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(name: str) -> Path:
    path = CACHE / name
    if not path.exists():
        import requests

        CACHE.mkdir(parents=True, exist_ok=True)
        response = requests.get(SOURCES[name], timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def _groups(path: Path) -> dict[str, dict[str, list[str]]]:
    """DOMAIN_GROUPS and FREQUENCY_GROUPS read as literals (the file is not executed)."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("DOMAIN_GROUPS", "FREQUENCY_GROUPS"):
                out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def _card(path: Path) -> dict[str, dict]:
    """config -> {"dtypes": {column: dtype}, "download_size": bytes} from the dataset card's YAML header."""
    import yaml

    header = yaml.safe_load(path.read_text().split("---")[1])
    out = {}
    for info in header["dataset_info"]:
        dtypes = {}
        for f in info["features"]:
            dtype = f.get("dtype") or f.get("sequence") or f.get("list")
            dtypes[f["name"]] = dtype.get("dtype") if isinstance(dtype, dict) else dtype
        out[info["config_name"]] = {"dtypes": dtypes, "download_size": info.get("download_size")}
    return out


def _as_csv(value) -> str:
    return "" if value is None else str(value)


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation with at least one task per stratum (rule in the module docstring)."""
    total = sum(sizes.values())
    quota = {s: n * k / total for s, k in sizes.items()}
    alloc = {s: min(sizes[s], max(1, math.floor(q))) for s, q in quota.items()}
    while sum(alloc.values()) < n:
        s = max(sorted(s for s in sizes if alloc[s] < sizes[s]), key=lambda s: quota[s] - alloc[s])
        alloc[s] += 1
    while sum(alloc.values()) > n:
        s = min(sorted(s for s in sizes if alloc[s] > 1), key=lambda s: quota[s] - alloc[s])
        alloc[s] -= 1
    return alloc


def select() -> pd.DataFrame:
    import fev
    import yaml

    assert fev.__version__ == FEV_VERSION, f"fev {fev.__version__} installed, the manifest is built with {FEV_VERSION}"
    paths = {name: _fetch(name) for name in SOURCES}
    defs = yaml.safe_load(paths["tasks.yaml"].read_text())["tasks"]
    groups = _groups(paths["task_groups.py"])
    domain = {t: g for g, ts in groups["DOMAIN_GROUPS"].items() for t in ts}
    frequency = {t: g for g, ts in groups["FREQUENCY_GROUPS"].items() for t in ts}
    card = _card(paths["fev_datasets_README.md"])

    GOOGLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    source = Path(GOOGLE_SOURCE["local_source"])
    if source.exists() and not GOOGLE_CSV.exists():
        shutil.copyfile(source, GOOGLE_CSV)
    assert _sha256(GOOGLE_CSV) == GOOGLE_SOURCE["sha256"], "Google CSV differs from the recorded file"
    google = pd.read_csv(GOOGLE_CSV, dtype=str, keep_default_na=False).set_index("task_name")

    rows = []
    for d in defs:
        task = fev.Task(**d)
        definition = task.to_dict()
        dtypes = card[task.dataset_config]["dtypes"]
        strings = [c for c in task.dynamic_columns if dtypes.get(c) in ("string", "large_string")]
        reasons = []
        if strings:
            reasons.append("string covariates hashed by Google's runner: " + ",".join(strings))
        if task.task_name not in google.index:
            reasons.append("not in Google's CSV")
        else:
            diff = [f for f in DEFINITION_FIELDS if _as_csv(definition[f]) != google.loc[task.task_name, f]]
            if diff:
                reasons.append("definition differs from Google's CSV: " + ",".join(diff))
        rows.append({
            "task_name": task.task_name, "dataset_config": task.dataset_config,
            "domain": domain[task.task_name], "frequency": frequency[task.task_name],
            "stratum": f"{domain[task.task_name]}|{frequency[task.task_name]}",
            "horizon": task.horizon, "season": task.seasonality, "num_windows": task.num_windows,
            "num_targets": len(task.target_columns), "num_known_cov": len(task.known_dynamic_columns),
            "num_past_cov": len(task.past_dynamic_columns),
            "download_mb": round(card[task.dataset_config]["download_size"] / 2**20, 2),
            "excluded": "; ".join(reasons), "definition": json.dumps(definition),
        })
    table = pd.DataFrame(rows)
    eligible = table[table.excluded == ""]
    sizes = eligible.groupby("stratum").size().to_dict()
    alloc = allocate(sizes, SAMPLE_SIZE)
    rng = np.random.default_rng(SEED)
    chosen = []
    for stratum in sorted(alloc):
        names = sorted(eligible.loc[eligible.stratum == stratum, "task_name"])
        chosen += list(rng.choice(names, alloc[stratum], replace=False))
    table["stratum_eligible"] = table.stratum.map(sizes).fillna(0).astype(int)
    table["stratum_selected"] = table.stratum.map(alloc).fillna(0).astype(int)
    table["selected"] = table.task_name.isin(chosen)
    order = {name: i for i, name in enumerate(chosen)}
    table["order"] = table.task_name.map(order)
    table = table.sort_values(["order", "task_name"], na_position="last").drop(columns="order").reset_index(drop=True)
    table.to_csv(MANIFEST, index=False)

    sel = table[table.selected]
    stats = {
        "fev_version": FEV_VERSION,
        "sources": {name: {"url": SOURCES[name], "sha256": _sha256(p)} for name, p in paths.items()},
        "google_csv": {**GOOGLE_SOURCE, "copied_to": str(GOOGLE_CSV.relative_to(ROOT))},
        "seed": SEED, "sample_size": SAMPLE_SIZE, "tasks": len(table), "eligible": len(eligible),
        "excluded": dict(zip(table.task_name[table.excluded != ""], table.excluded[table.excluded != ""])),
        "strata": {s: {"eligible": sizes[s], "selected": alloc[s]} for s in sorted(sizes)},
        "selected_download_mb": round(sel.drop_duplicates("dataset_config").download_mb.sum(), 1),
    }
    (MANIFEST.parent / "tierA_sources.json").write_text(json.dumps(stats, indent=1))
    return table


# ------------------------------------------------------------------ inputs

def fev_task(row):
    import fev

    return fev.Task(**json.loads(row.definition))


def _impute(x: np.ndarray) -> np.ndarray:
    """LastValueImputation of Google's runner: forward fill, leading gaps take the first value, all-missing -> 0."""
    x = np.array(x, dtype=np.float32)
    gaps = np.isnan(x)
    if not gaps.any():
        return x
    if gaps.all() or len(x) == 1:
        return np.where(gaps, np.float32(0), x)
    idx = np.where(~gaps, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    x[np.isnan(x)] = x[np.flatnonzero(~gaps)[0]]
    return x


def _columns(dataset, columns: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """column -> (flat float32 values, offsets) of the list column, without per-item Python conversion."""
    import pyarrow as pa
    import pyarrow.compute as pc

    out = {}
    table = dataset.data.table
    for c in columns:
        arr = table.column(c).combine_chunks()
        values = pc.cast(arr.values, pa.float32(), safe=False).to_numpy(zero_copy_only=False)
        out[c] = (np.asarray(values, dtype=np.float32), arr.offsets.to_numpy())
    return out


class Inputs:
    """Window inputs of one fev task at a time; the job goes through tasks and windows in order."""

    def __init__(self, row):
        self.task = fev_task(row)
        self.season = int(row.season)
        self._key = None

    def rows(self, name: str) -> pd.DataFrame:
        out = []
        for w in range(self.task.num_windows):
            past, _ = self.task.get_window(w).get_input_data()
            ids = past.data.table.column(self.task.id_column).to_pylist()
            out += [{"task_id": f"{name}|{w}|{item}", "window": w, "item": i,
                     "horizon": self.task.horizon, "season": self.season} for i, item in enumerate(ids)]
        return pd.DataFrame(out)

    def _load(self, w: int) -> None:
        if self._key == w:
            return
        t = self.task
        past, future = t.get_window(w).get_input_data()
        self.past = _columns(past, t.target_columns + t.known_dynamic_columns + t.past_dynamic_columns)
        self.future = _columns(future, t.known_dynamic_columns)
        self._key = w

    def make(self, row):
        self._load(int(row.window))
        t, i = self.task, int(row.item)

        def get(store, col):
            values, offsets = store[col]
            return _impute(values[offsets[i]:offsets[i + 1]])

        target = np.stack([get(self.past, c)[-TSFM_CONTEXT:] for c in t.target_columns])
        past_cov = future_cov = None
        if t.past_dynamic_columns:
            past_cov = np.stack([get(self.past, c)[-TSFM_CONTEXT:] for c in t.past_dynamic_columns])
        if t.known_dynamic_columns:
            future_cov = np.stack([np.concatenate([get(self.past, c)[-TSFM_CONTEXT:], get(self.future, c)])
                                   for c in t.known_dynamic_columns])
        return Task(target, t.horizon, past_cov=past_cov, future_cov=future_cov, task_id=row.task_id), None, self.season


@contextlib.contextmanager
def google_evaluator(key: str):
    """`tfm3-eval` loads TimesFM3Evaluator (Google's runner class) in place of TimesFM3Forecaster."""
    base = adapters.REGISTRY["timesfm-3.0"]
    if key != "tfm3-eval":
        yield
        return

    class Evaluator(base):
        def _load(self) -> None:
            from timesfm3 import TimesFM3Evaluator

            super()._load()
            self.model.__class__ = TimesFM3Evaluator

    adapters.REGISTRY["timesfm-3.0"] = Evaluator
    try:
        yield
    finally:
        adapters.REGISTRY["timesfm-3.0"] = base


# ------------------------------------------------------------------ scoring

def read_forecasts(model_dir: Path, prefix: str) -> dict[str, np.ndarray]:
    out = {}
    for part in sorted(model_dir.glob("part-*.parquet")):
        df = pd.read_parquet(part)
        for row in df[df.task_id.str.startswith(prefix)].itertuples():
            out[row.task_id] = np.frombuffer(row.quantiles, dtype=np.float32).reshape(json.loads(row.shape))
    return out


def score(task, rows: pd.DataFrame, forecasts: dict[str, np.ndarray], key: str) -> dict:
    """fev's evaluation summary of one model on one task (all windows)."""
    import datasets

    levels = [str(q) for q in task.quantile_levels]
    index = [int(np.argmin(np.abs(np.asarray(QUANTILES) - q))) for q in task.quantile_levels]
    median = QUANTILES.index(0.5)
    per_window = []
    for _, g in rows.groupby("window", sort=True):
        q = np.stack([forecasts[t] for t in g.task_id]).astype(np.float64)  # (items, targets, horizon, 9)
        per_window.append(datasets.DatasetDict({
            col: datasets.Dataset.from_dict({"predictions": q[:, d, :, median],
                                             **{lv: q[:, d, :, j] for lv, j in zip(levels, index)}})
            for d, col in enumerate(task.target_columns)}))
    return task.evaluation_summary(per_window, model_name=key, trained_on_this_dataset=False,
                                   extra_info={"model_key": key})


def _append(path: Path, summary: dict) -> None:
    row = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in summary.items()}])
    if path.exists():
        old = pd.read_csv(path)
        row = pd.concat([old[old.task_name != summary["task_name"]], row], ignore_index=True)
    row.to_csv(path, index=False)


# ------------------------------------------------------------------ job

def _dataset_revision() -> str:
    """fev loads autogluon/fev_datasets at its current revision (no pinning), so the revision is recorded."""
    try:
        from huggingface_hub import dataset_info

        return dataset_info("autogluon/fev_datasets").sha
    except Exception as exc:  # network errors are recorded, not fatal
        return f"error: {exc}"


def run(out: Path, models: list[str], device: str, limit: int | None, names: list[str] | None,
        manifest: Path) -> dict:
    table = pd.read_csv(manifest)
    if names:  # named tasks may be eligible but unselected ones (GPU checks before the protocol is frozen)
        table = table[table.task_name.isin(names) & table.excluded.isna()]
    else:
        table = table[table.selected]
    if limit:
        table = table.head(limit)
    out_dir = out / "A"
    import fev

    report = {"environment": provenance.environment(), "tier": "A", "fev_version": fev.__version__,
              "fev_datasets_revision": _dataset_revision(), "runs": []}
    for row in table.itertuples():
        inputs = Inputs(row)
        rows = inputs.rows(row.task_name)
        for key in models:
            with google_evaluator(key):
                rec = run_model(key, rows, inputs.make, out_dir, device, None)
            rec["task_name"] = row.task_name
            model_dir = out_dir / key
            metrics_path = model_dir / "metrics.csv"
            scored = metrics_path.exists() and row.task_name in set(pd.read_csv(metrics_path).task_name)
            if rec["tasks_total"] and not scored:
                forecasts = read_forecasts(model_dir, row.task_name + "|")
                missing = len(set(rows.task_id) - set(forecasts))
                rec["missing"] = missing
                if not missing:
                    summary = score(inputs.task, rows, forecasts, key)
                    summary["forecast_seconds"] = rec.get("seconds")
                    _append(metrics_path, summary)
                    rec.update({m: summary.get(m) for m in METRICS})
            report["runs"].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "failure_examples"}, default=str), flush=True)
    stamp = report["environment"]["utc"].replace(":", "").replace("-", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"run_{stamp}.json").write_text(json.dumps(report, indent=1, default=str))
    return report


# ------------------------------------------------------------------ comparison with Google's CSV

REPRODUCTION_TOLERANCE = 0.03  # |SQL deviation| per task, descriptive (protocol section 3.1)
CONTROL_TOLERANCE = 0.005


def compare(out: Path, model: str = "tfm3-eval", path: Path = ROOT / "reports" / "tierA_reproduction.csv") -> pd.DataFrame:
    """Per-task differences from Google's CSV (ours - Google; relative: ours / Google - 1).

    fev's dataset fingerprint hashes pickled Arrow objects and changes with library versions, so it is not a data check;
    `snaive_MASE_vs_fev` is: our Seasonal Naive point forecasts against fev's published Seasonal Naive results (same data
    and windows give the same MASE; 0 when they agree).
    """
    ours = pd.read_csv(out / "A" / model / "metrics.csv").set_index("task_name")
    google = pd.read_csv(GOOGLE_CSV).set_index("task_name")
    snaive_path = out / "A" / "snaive" / "metrics.csv"
    snaive = pd.read_csv(snaive_path).set_index("task_name") if snaive_path.exists() else None
    reference = pd.read_csv(_fetch("seasonal_naive.csv")).set_index("task_name")
    rows = []
    for name in ours.index:
        row = {"task_name": name, "model": model, "num_forecasts": ours.loc[name, "num_forecasts"],
               "num_forecasts_google": google.loc[name, "num_forecasts"],
               "snaive_MASE_vs_fev": (snaive.loc[name, "MASE"] / reference.loc[name, "MASE"] - 1
                                      if snaive is not None and name in snaive.index else np.nan),
               "snaive_ok": bool(snaive is not None and name in snaive.index
                                 and abs(snaive.loc[name, "MASE"] - reference.loc[name, "MASE"]) < 5e-5)}
        for m in METRICS:
            a, b = float(ours.loc[name, m]), float(google.loc[name, m])
            row.update({m: a, f"{m}_google": b, f"{m}_absdiff": a - b, f"{m}_reldiff": a / b - 1 if b else np.nan})
        rows.append(row)
    df = pd.DataFrame(rows)
    # Pipeline control (protocol section 3.1): our Chronos-2 against fev's published Chronos-2 results.
    chronos_path = out / "A" / "chronos2" / "metrics.csv"
    if chronos_path.exists():
        ours_c = pd.read_csv(chronos_path).set_index("task_name")
        pub_c = pd.read_csv(_fetch("chronos-2.csv")).set_index("task_name")
        df["chronos2_SQL_vs_fev"] = [ours_c.loc[n, "SQL"] / pub_c.loc[n, "SQL"] - 1
                                     if n in ours_c.index and n in pub_c.index else np.nan for n in df.task_name]
        df["control_ok"] = (df.chronos2_SQL_vs_fev.abs() <= CONTROL_TOLERANCE) & df.snaive_ok
        # Reproduction of the claim: TimesFM-3 relative to Chronos-2, ours vs published (Google / fev).
        df["skill_ours"] = [ours.loc[n, "SQL"] / ours_c.loc[n, "SQL"] if n in ours_c.index else np.nan
                            for n in df.task_name]
        df["skill_published"] = [google.loc[n, "SQL"] / pub_c.loc[n, "SQL"] if n in pub_c.index else np.nan
                                 for n in df.task_name]
    # H1 (descriptive): advantage over fev's published statistical ensemble on the same tasks.
    ens = pd.read_csv(_fetch("stat_ensemble.csv")).set_index("task_name")
    for metric in ("SQL", "MASE"):
        df[f"{metric}_vs_stat_ensemble"] = [ours.loc[n, metric] / ens.loc[n, metric] if n in ens.index else np.nan
                                            for n in df.task_name]
    df["reproduced"] = df.SQL_reldiff.abs() <= REPRODUCTION_TOLERANCE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def reproduction_summary(df: pd.DataFrame) -> dict:
    """RQ1 summary on the tasks passing the pipeline controls, and on all tasks (protocol section 3.1)."""
    out = {"controls_passed": _summary_of(df[df.control_ok]) if "control_ok" in df else None, "all_tasks": _summary_of(df)}
    out["control_failures"] = df.loc[~df.control_ok, "task_name"].tolist() if "control_ok" in df else None
    return out


def _summary_of(ok: pd.DataFrame) -> dict:
    from scipy.stats import binomtest

    dev = ok.SQL_reldiff.dropna()
    out = {"tasks": int(len(ok)), "reproduced_share": float(ok.reproduced.mean()), "median_SQL_dev": float(dev.median()),
           "range_SQL_dev": [float(dev.min()), float(dev.max())],
           "sign_test_p": float(binomtest(int((dev > 0).sum()), int((dev != 0).sum())).pvalue) if len(dev) else None}
    if "skill_ours" in ok:
        s = ok[["skill_ours", "skill_published"]].dropna()
        out.update(skill_ours=float(np.exp(np.log(s.skill_ours).mean())),
                   skill_published=float(np.exp(np.log(s.skill_published).mean())), skill_tasks=int(len(s)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", nargs="?", default="run", choices=["run", "select", "compare"])
    parser.add_argument("--out", default=str(ROOT / "results" / "tiera"))
    parser.add_argument("--models", default="tfm3-eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="first N selected fev tasks")
    parser.add_argument("--tasks", default=None, help="comma-separated fev task names (default: all selected)")
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--model", default="tfm3-eval", help="compare: model key compared with Google's CSV")
    args = parser.parse_args()

    if args.step == "select":
        table = select()
        sel = table[table.selected]
        print(sel.groupby("stratum").size().to_string())
        print(f"{len(sel)} selected of {(table.excluded == '').sum()} eligible; excluded:")
        print(table.loc[table.excluded != "", ["task_name", "excluded"]].to_string(index=False))
    elif args.step == "compare":
        df = compare(Path(args.out), args.model)
        cols = ["task_name", "snaive_MASE_vs_fev"] + [c for m in METRICS for c in (m, f"{m}_google", f"{m}_reldiff")]
        print(df[cols].round(4).to_string(index=False))
    else:
        names = args.tasks.split(",") if args.tasks else None
        run(Path(args.out), args.models.split(","), args.device, args.limit, names, Path(args.manifest))


if __name__ == "__main__":
    main()
