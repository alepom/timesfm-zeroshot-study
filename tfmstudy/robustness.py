"""Robustness and sensitivity analyses requested in review (not pre-registered; reported as such).

    python -m tfmstudy.robustness            -> reports/robustness.json

R1  aggregation: family P contrasts under other averaging schemes (row-pooled, series-weighted, ratio of mean errors,
    median and trimmed mean of the log ratio, leave one domain out).
R2  small-cluster inference: family P and H5 with (a) a delete-one-cluster jackknife over series and over weeks with
    t(G-1) critical values and (b) the two-way bootstrap with two-week time blocks; Holm decisions under each.
R3  context: TimesFM-3 restricted to 2,016 points against Chronos-2 and the ensemble on hourly data.
R4  ablations on Tier B by domain and only on windows where the ablation changes the forecast.
R5  Wikipedia difference in differences by page difficulty.
R6  intermittent demand (D8) on RMSSE and SQL as well as MASE.
R7  Tier A: task-bootstrap intervals for the TimesFM-3 / Chronos-2 ratios.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import analysis as A

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "robustness.json"
MAIN = ROOT / "results" / "forecast" / "main"
REPS = 10_000
ACTIVE = 1e-3  # |log MASE ratio| above which an ablation counts as having changed the forecast
P_TESTS = [("P1", "tfm3-uni", "stat-ens"), ("P2", "tfm3-uni", "chronos2-uni"), ("P3", "tfm3-uni", "tfm25"),
           ("P4a", "tfm3", "tfm3-uni"), ("P5", "tfm3-eval", "tfm3")]


def _rows(b: pd.DataFrame) -> dict:
    uni = (b["mode"] == "uni") & b.domain.isin(A.STRUCTURED)
    cc = A.complete_cases(b, ["tfm3-uni", "stat-ens", "chronos2-uni", "tfm25", "tfm3-eval", "tfm3"], ["mase", "sql"], uni)
    grp = A.complete_cases(b, ["tfm3", "tfm3-uni"], ["mase", "sql"], b["mode"] == "group")
    nrw = A.complete_cases(b, ["tfm3-uni", "naive"], ["mase"], (b["mode"] == "uni") & b.domain.isin(A.NEAR_RANDOM_WALK))
    return {"P1": cc, "P2": cc, "P3": cc, "P4a": grp, "P5": cc, "H5": nrw}


def _nested(pr: pd.DataFrame) -> float:
    return float(A.Design(pr, ["domain", "horizon"], "domain").estimate(pr.d.to_numpy())[0, 0])


def aggregation(b: pd.DataFrame) -> list[dict]:
    rows, out = _rows(b), []
    for name, a, m in P_TESTS:
        for metric in ("mase", "sql"):
            pr = A.paired(b, a, m, metric, rows[name])
            sa = b[rows[name] & b.model.eq(a)].set_index(["task_id", "series_id"])[metric]
            sb = b[rows[name] & b.model.eq(m)].set_index(["task_id", "series_id"])[metric]
            sa, sb = sa.loc[sa.index.intersection(sb.index)], sb.loc[sa.index.intersection(sb.index)]
            lo, hi = np.quantile(pr.d, [0.01, 0.99])
            r = {"test": f"{name}:{metric}", "nested_equal_weights": _nested(pr), "row_pooled": pr.d.mean(),
                 "series_weighted": pr.groupby("series_id").d.mean().mean(),
                 "ratio_of_means": float(np.log(sa.mean() / sb.mean())), "median": pr.d.median(),
                 "trimmed_1pct": pr.d[(pr.d >= lo) & (pr.d <= hi)].mean()}
            for dom in sorted(pr.domain.unique()):
                rest = pr[pr.domain != dom]
                if rest.domain.nunique() >= 1:
                    r[f"without_{dom}"] = _nested(rest)
            out.append({k: (round(float(np.exp(v)), 4) if k != "test" else v) for k, v in r.items()})
    return out


def _blocks(week: pd.Series, size: int = 2) -> pd.Series:
    order = {w: i for i, w in enumerate(sorted(week.unique()))}
    return week.map(lambda w: f"block{order[w] // size}")


def small_g(b: pd.DataFrame, reps: int = REPS) -> list[dict]:
    rows, out = _rows(b), []
    tests = [(n, a, m, met) for n, a, m in P_TESTS for met in ("mase", "sql")] + [("H5", "tfm3-uni", "naive", "mase")]
    for name, a, m, metric in tests:
        pr = A.paired(b, a, m, metric, rows[name])
        est = _nested(pr)
        var, gs = 0.0, []
        for col in ("cluster", "week"):  # delete-one-group jackknife in each dimension, variances added
            groups = pr[col].unique()
            th = np.array([_nested(pr[pr[col] != g]) for g in groups])
            var += (len(groups) - 1) / len(groups) * ((th - th.mean()) ** 2).sum()
            gs.append(len(groups))
        df = min(gs) - 1
        se = np.sqrt(var)
        t = stats.t.ppf(0.975, df)
        r = {"test": f"{name}:{metric}", "ratio": round(float(np.exp(est)), 4),
             "jackknife_ci95": [round(float(np.exp(est - t * se)), 4), round(float(np.exp(est + t * se)), 4)],
             "jackknife_df": int(df), "series_clusters": gs[0], "week_clusters": gs[1]}
        if name == "H5":  # non-superiority at 0.90: one-sided p of estimate <= log 0.90
            r["jackknife_p"] = float(stats.t.sf((est - np.log(0.90)) / se, df))
        else:
            r["jackknife_p"] = float(2 * stats.t.sf(abs(est) / se, df))
        blk = pr.assign(week=_blocks(pr.week))
        design = A.Design(blk, ["domain", "horizon"], "domain")
        boot = design.bootstrap(blk.d.to_numpy(), reps)[:, 0]
        s = A._summary(est, boot, floor=0.90 if name == "H5" else None)
        r["block2_ci95"] = [round(x, 4) for x in s["ci95"]]
        r["block2_p"] = s["p_nonsup"] if name == "H5" else s["p"]
        out.append(r)
    fam = [r for r in out if not r["test"].startswith("H5")]
    for key in ("jackknife_p", "block2_p"):
        for r, ph in zip(fam, A.holm([r[key] for r in fam])):
            r[key.replace("_p", "_p_holm")] = round(ph, 4)
            r[key.replace("_p", "_resolved")] = bool(ph < 0.05)
    return out


def context(b: pd.DataFrame, reps: int = 2_000) -> list[dict]:
    hourly = (b["mode"] == "uni") & b.domain.isin(["load", "price", "weather"])
    rows = A.complete_cases(b, ["tfm3-c2016", "tfm3-uni", "chronos2-uni", "stat-ens"], ["mase", "sql"], hourly)
    out = []
    for a, m in (("tfm3-uni", "chronos2-uni"), ("tfm3-c2016", "chronos2-uni"), ("tfm3-c2016", "stat-ens"),
                 ("tfm3-uni", "stat-ens")):
        for metric in ("mase", "sql"):
            r = A.contrast(b, a, m, metric, rows, ["domain", "horizon"], reps=reps)
            out.append({"a": a, "b": m, "metric": metric, "ratio": round(r["ratio"], 4),
                        "ci95": [round(x, 4) for x in r["ci95"]], "n": r["n"]})
    return out


def ablations(b: pd.DataFrame) -> list[dict]:
    out = []
    keys = ["task_id", "series_id"]
    base = b[b.model.eq("tfm3")].set_index(keys)
    for abl in sorted(m for m in b.model.unique() if m.startswith("tfm3-") and m not in
                      ("tfm3-uni", "tfm3-eval", "tfm3-c2016", "tfm3-novarattn")):
        x = b[b.model.eq(abl)].set_index(keys)
        idx = x.index.intersection(base.index)
        d = pd.DataFrame({"domain": x.loc[idx, "domain"], "a": x.loc[idx, "mase"], "b": base.loc[idx, "mase"]})
        d = d[np.isfinite(d.a) & np.isfinite(d.b) & (d.a > 0) & (d.b > 0)]
        d["d"] = np.log(d.a) - np.log(d.b)
        d["active"] = d.d.abs() > ACTIVE  # GPU run-to-run noise alone moves MASE by less than this
        for dom, g in d.groupby("domain"):
            act = g[g.active]
            out.append({"ablation": abl, "domain": dom, "windows": int(len(g)), "active_share": round(float(g.active.mean()), 3),
                        "ratio_all": round(float(np.exp(g.d.mean())), 4),
                        "ratio_active": round(float(np.exp(act.d.mean())), 4) if len(act) else None,
                        "active_windows": int(len(act))})
    return out


def did_by_difficulty(w: pd.DataFrame) -> list[dict]:
    rows = A.complete_cases(w, ["tfm3", "chronos2"], ["mase"], (w["mode"] == "uni") & w.stratum.isin(["W-seen", "W-post"]))
    pr = A.paired(w, "tfm3", "chronos2", "mase", rows)
    ref = w[rows & w.model.eq("chronos2")].groupby("series_id").mase.apply(lambda s: np.exp(np.log(s).mean()))
    q = pd.qcut(ref, 4, labels=["Q1 (easiest)", "Q2", "Q3", "Q4 (hardest)"])
    pr["difficulty"] = pr.series_id.map(q)
    out = []
    for lab, g in pr.groupby("difficulty", observed=True):
        seen, post = g[g.stratum.eq("W-seen")].d.mean(), g[g.stratum.eq("W-post")].d.mean()
        out.append({"page_difficulty": str(lab), "pages": int(g.series_id.nunique()),
                    "advantage_seen": round(float(np.exp(seen)), 4), "advantage_post": round(float(np.exp(post)), 4),
                    "did": round(float(np.exp(seen - post)), 4)})
    gm = lambda s: float(np.exp(np.log(s).mean()))
    for st in ("W-seen", "W-post", "W-new"):
        sub = w[w.stratum.eq(st) & w["mode"].eq("uni")]
        out.append({"stratum": st, "gmean_mase_tfm3": round(gm(sub[sub.model.eq("tfm3")].mase), 4),
                    "gmean_mase_chronos2": round(gm(sub[sub.model.eq("chronos2")].mase), 4)})
    return out


def intermittent(c: pd.DataFrame) -> list[dict]:
    out = []
    gm = lambda s: float(np.exp(np.log(s[(s > 0) & np.isfinite(s)]).mean()))
    for cell in sorted(x for x in c.cell.unique() if x.startswith("rep:D8")):
        g = c[c.cell.eq(cell)]
        r = {"cell": cell}
        for model in ("tfm3", "chronos2", "mean", "ets", "theta", "arima", "naive", "snaive"):
            s = g[g.model.eq(model)]
            if len(s):
                r[model] = {k: round(gm(s[k]), 4) for k in ("mase", "rmsse", "sql")}
        out.append(r)
    return out


def near_random_walk(b: pd.DataFrame) -> list[dict]:
    """Every model against the random walk on FX and rates (descriptive, point estimates)."""
    ms = ["tfm3-uni", "chronos2-uni", "tfm25", "tirex2-uni", "stat-ens", "naive"]
    rows = A.complete_cases(b, ms, ["mase"], (b["mode"] == "uni") & b.domain.isin(A.NEAR_RANDOM_WALK))
    out = []
    for m in ms[:-1]:
        pr = A.paired(b, m, "naive", "mase", rows)
        out.append({"model": m, "ratio_vs_naive": round(float(np.exp(_nested(pr))), 4),
                    **{f"ratio_{d}": round(float(np.exp(g.d.mean())), 4) for d, g in pr.groupby("domain")}})
    return out


def tier_a(reps: int = REPS) -> dict:
    t = pd.read_csv(ROOT / "reports" / "tierA_reproduction.csv")
    t = t[t.model.eq("tfm3-eval")][["task_name", "skill_ours", "skill_published"]].dropna()
    lo, lp = np.log(t.skill_ours.to_numpy()), np.log(t.skill_published.to_numpy())
    rng = np.random.default_rng(A.SEED)
    idx = rng.integers(0, len(t), size=(reps, len(t)))
    ci = lambda x: [round(float(np.exp(np.quantile(x, 0.025))), 4), round(float(np.exp(np.quantile(x, 0.975))), 4)]
    return {"tasks": int(len(t)), "ours": round(float(np.exp(lo.mean())), 4), "ours_ci95": ci(lo[idx].mean(1)),
            "published": round(float(np.exp(lp.mean())), 4), "published_ci95": ci(lp[idx].mean(1)),
            "ours_over_published": round(float(np.exp((lo - lp).mean())), 4), "ours_over_published_ci95": ci((lo - lp)[idx].mean(1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=REPS)
    args = parser.parse_args()
    b, w, c = (pd.read_parquet(MAIN / t / "metrics.parquet") for t in ("B", "W", "synthetic"))
    report = {"label": "robustness and sensitivity analyses requested in review; not pre-registered",
              "R1_aggregation": aggregation(b), "R2_small_clusters": small_g(b, args.reps), "R3_context": context(b),
              "R4_ablations_by_domain": ablations(b), "R5_did_by_difficulty": did_by_difficulty(w),
              "R6_intermittent": intermittent(c), "R7_tier_a": tier_a(args.reps),
              "R8_near_random_walk": near_random_walk(b)}
    OUT.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
