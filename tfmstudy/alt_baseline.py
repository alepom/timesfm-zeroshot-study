"""Sensitivity analysis requested in review: a re-specified classical ensemble for river, FX and rates (not pre-registered).

    python -m tfmstudy.alt_baseline --tree <scratch dir>   (the dir must hold main/B/{tfm3-uni,chronos2-uni,naive,snaive})

The pre-registered ensemble (AutoETS + AutoARIMA + AutoTheta) uses the domain's weekly season (7 for daily river flow,
5 for business-daily FX and rates). None of these series has a weekly cycle, and river flows are skewed. The
re-specified ensemble uses the same three models without a season, and for river flow fits them to log(1 + y) and
maps the quantiles back (quantiles are preserved by a monotone transform). Forecasts are written to the scratch tree as
ets/arima/theta, scored by evaluate.score, and compared with the main results. Output: reports/alt_baseline.json.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as A
from .baselines import _fit_row
from .evaluate import score
from .forecast import load_tasks

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("river", "fx", "rates")


def _fit(job):
    method, y, horizon, log = job
    y = np.asarray(y, float)
    if log:
        q, fb = _fit_row(method, np.log1p(np.clip(y, 0, None)), horizon, 1)
        return np.expm1(q), fb
    return _fit_row(method, y, horizon, 1)


def forecasts(tree: Path) -> None:
    tasks, make = load_tasks("B", "main", ROOT / "data" / "processed")
    tasks = tasks[(tasks["mode"] == "uni") & tasks.domain.isin(DOMAINS)]
    built = [(r, make(r)[0]) for r in tasks.itertuples()]
    with ProcessPoolExecutor(8) as pool:
        for method in ("ets", "arima", "theta"):
            jobs = [(method, t.target[0], t.horizon, r.domain == "river") for r, t in built]
            res = list(pool.map(_fit, jobs, chunksize=8))
            rows = [{"task_id": r.task_id, "quantiles": q[None].astype(np.float32).tobytes(),
                     "shape": json.dumps([1, *q.shape]), "fallbacks": int(fb)} for (r, _), (q, fb) in zip(built, res)]
            out = tree / "main" / "B" / method
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(out / "part-00000-alt.parquet", index=False)


def compare(tree: Path, reps: int = 2_000) -> dict:
    alt = score("B", "main", tree, ROOT / "data" / "processed")
    alt = alt[alt.domain.isin(DOMAINS) & alt["mode"].eq("uni")]
    main = pd.read_parquet(ROOT / "results/forecast/main/B/metrics.parquet")
    main = main[main.domain.isin(DOMAINS) & main["mode"].eq("uni")]
    both = pd.concat([main, alt[alt.model.eq("stat-ens")].assign(model="stat-ens-alt")], ignore_index=True)
    out = {"label": "sensitivity analysis requested in review; not pre-registered", "contrasts": []}
    for doms, a, b in ((("river",), "tfm3-uni", "stat-ens"), (("river",), "tfm3-uni", "stat-ens-alt"),
                       (("river",), "stat-ens-alt", "stat-ens"), (("river",), "stat-ens-alt", "naive"),
                       (("fx", "rates"), "stat-ens-alt", "stat-ens"), (("fx", "rates"), "stat-ens-alt", "naive"),
                       (("fx", "rates"), "tfm3-uni", "stat-ens-alt")):
        rows = A.complete_cases(both, [a, b], ["mase", "sql"], both.domain.isin(doms))
        for metric in ("mase", "sql"):
            r = A.contrast(both, a, b, metric, rows, ["domain", "horizon"], reps=reps)
            out["contrasts"].append({"domains": list(doms), "a": a, "b": b, "metric": metric, "ratio": round(r["ratio"], 4),
                                     "ci95": [round(x, 4) for x in r["ci95"]], "n": r["n"]})
    cov = both[both.model.isin(["stat-ens", "stat-ens-alt", "tfm3-uni"])].groupby(["domain", "model"]).agg(
        coverage80=("coverage80", "mean"), width80=("width80", "median")).round(3).reset_index()
    out["calibration"] = cov.to_dict("records")
    # P1 with the re-specified ensemble on river (the other structured domains unchanged)
    p = pd.read_parquet(ROOT / "results/forecast/main/B/metrics.parquet")
    alt_river = alt[alt.model.eq("stat-ens") & alt.domain.eq("river")]
    p = pd.concat([p[~(p.model.eq("stat-ens") & p.domain.eq("river"))], alt_river], ignore_index=True)
    uni = (p["mode"] == "uni") & p.domain.isin(A.STRUCTURED)
    rows = A.complete_cases(p, ["tfm3-uni", "stat-ens"], ["mase", "sql"], uni)
    for metric in ("mase", "sql"):
        r = A.contrast(p, "tfm3-uni", "stat-ens", metric, rows, ["domain", "horizon"], reps=reps)
        out[f"P1_with_alt_river_{metric}"] = {"ratio": round(r["ratio"], 4), "ci95": [round(x, 4) for x in r["ci95"]]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--reps", type=int, default=2_000)
    args = parser.parse_args()
    tree = Path(args.tree)
    forecasts(tree)
    report = compare(tree, args.reps)
    (ROOT / "reports" / "alt_baseline.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
