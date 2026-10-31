"""Exploratory analyses of protocol section 5.4 (no multiplicity correction, labelled exploratory in the paper).

    python -m tfmstudy.exploratory --w-forecasts <dir with the main W model folders>

1. Spike anticipation on W: windows whose observed maximum exceeds 3x the 90th percentile of their context; for
   each such window and model, the ratio of the forecast 0.9-quantile maximum to that percentile, by stratum.
2. Tier B by lineage note (catalog.DOMAINS): TimesFM-3 uni against the statistical ensemble and Chronos-2 uni on the
   structured domains, separately for series whose source has earlier-period data in public pre-training corpora
   (L1) and series with none known (L0). Within the structured domains L0 is the river domain only, so this is a
   domain contrast as much as a lineage one.

The W forecasts are on Drive (results/forecast/main/W); copy them to a scratch directory first. Output:
reports/exploratory.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as A
from .evaluate import read_forecasts, stat_ensemble

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "exploratory.json"
SPIKE = 3.0
W_MODELS = ("tfm3", "tfm3-eval", "chronos2", "tirex2", "tfm25", "snaive", "naive", "stat-ens")
LINEAGE_REPS = 2_000


def spikes(w_dir: Path) -> dict:
    tasks = pd.read_parquet(ROOT / "data/processed/main/wiki_tasks.parquet")
    tasks = tasks[tasks.excluded.fillna("").eq("")] if "excluded" in tasks else tasks
    series = pd.read_parquet(ROOT / "data/processed/main/wiki_series.parquet")
    by_id = {sid: g.set_index("timestamp")["value"] for sid, g in series.groupby("series_id")}
    fc = {m: read_forecasts(w_dir / m) for m in W_MODELS if (w_dir / m).is_dir()}
    fc["stat-ens"] = stat_ensemble(tasks, fc)
    rows = []
    for t in tasks.itertuples():
        s = by_id[json.loads(t.targets)[0]]
        ctx = s[(s.index >= t.context_start) & (s.index <= t.origin)].to_numpy(dtype=float)
        future = s[s.index > t.origin].to_numpy(dtype=float)[:t.horizon]
        if len(future) < t.horizon or not np.isfinite(ctx).any():
            continue
        p90 = np.nanpercentile(ctx, 90)
        if not p90 > 0 or np.nanmax(future) <= SPIKE * p90:
            continue
        for m, f in fc.items():
            q = f.get(t.task_id)
            if q is not None:
                rows.append({"task_id": t.task_id, "stratum": t.stratum, "model": m,
                             "onset": "ongoing" if ctx[-1] > SPIKE * p90 else "new",  # already in the last context value?
                             "actual_ratio": float(np.nanmax(future) / p90), "forecast_ratio": float(q[0, :, 8].max() / p90)})
    df = pd.DataFrame(rows)
    out = {"definition": f"windows with max(actual) > {SPIKE} x context p90; forecast ratio = max q0.9 / context p90",
           "windows_total": int(len(tasks)), "spike_windows": int(df.task_id.nunique()),
           "spike_windows_by_stratum": df.drop_duplicates("task_id").groupby("stratum").size().astype(int).to_dict(),
           "by_stratum_model": []}
    out["spike_windows_by_onset"] = df.drop_duplicates("task_id").groupby("onset").size().astype(int).to_dict()
    out["by_onset_model"] = []
    for (onset, model), g in df.groupby(["onset", "model"]):
        out["by_onset_model"].append({"onset": onset, "model": model, "windows": int(len(g)),
                                      "median_actual_ratio": round(float(g.actual_ratio.median()), 2),
                                      "median_forecast_ratio": round(float(g.forecast_ratio.median()), 3),
                                      "share_forecast_ratio_above_spike": round(float((g.forecast_ratio > SPIKE).mean()), 3)})
    for (stratum, model), g in df.groupby(["stratum", "model"]):
        out["by_stratum_model"].append({"stratum": stratum, "model": model, "windows": int(len(g)),
                                        "median_actual_ratio": round(float(g.actual_ratio.median()), 2),
                                        "median_forecast_ratio": round(float(g.forecast_ratio.median()), 3),
                                        "share_forecast_ratio_above_spike": round(float((g.forecast_ratio > SPIKE).mean()), 3)})
    return out


def lineage(reps: int = LINEAGE_REPS) -> list[dict]:
    b = pd.read_parquet(ROOT / "results/forecast/main/B/metrics.parquet")
    uni = b["mode"].eq("uni") & b.domain.isin(A.STRUCTURED)
    rows = A.complete_cases(b, ["tfm3-uni", "stat-ens", "chronos2-uni"], ["mase", "sql"], uni)
    out = []
    for lin in sorted(b.loc[rows, "lineage"].unique()):
        sub = rows & b.lineage.eq(lin)
        for ref in ("stat-ens", "chronos2-uni"):
            for metric in ("mase", "sql"):
                r = A.contrast(b, "tfm3-uni", ref, metric, sub, ["domain", "horizon"], reps=reps)
                out.append({"lineage": lin, "domains": sorted(b.loc[sub, "domain"].unique()), "ref": ref,
                            "metric": metric, "ratio": round(r["ratio"], 3), "ci95": [round(x, 3) for x in r["ci95"]],
                            "n": r["n"], "clusters": r["clusters"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w-forecasts", required=True)
    parser.add_argument("--reps", type=int, default=LINEAGE_REPS)
    args = parser.parse_args()
    report = {"label": "exploratory, no multiplicity correction (protocol section 5.4)",
              "spike_anticipation_W": spikes(Path(args.w_forecasts)), "lineage_B": lineage(args.reps)}
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
