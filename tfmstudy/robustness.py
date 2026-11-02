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


# ------------------------------------------------------------------ second round (GPU runs of 2026-11-02, bundle 0.5.6)

def regime_grid(reps: int = 2_000) -> list[dict]:
    """R9: level-shift grid (period 'regime'): TimesFM-3 against the best classical model and against Chronos-2, per cell."""
    from .evaluate import score
    c = score("synthetic", "regime", ROOT / "results" / "forecast", ROOT / "data" / "processed")
    classical = ["ets", "theta", "mstl", "mstl-theta", "mstl-arima", "naive", "snaive", "mean", "stat-ens"]
    gm = lambda s: float(np.exp(np.log(s[(s > 0) & np.isfinite(s)]).mean()))
    out = []
    for cell, g in c.groupby("cell"):
        level = {m: gm(g[g.model.eq(m)].mase) for m in g.model.unique()}
        best = min((m for m in classical if m in level), key=level.get)
        r = {"cell": cell, "best_classical": best, **{m: round(v, 4) for m, v in level.items()}}
        for ref in (best, "chronos2"):
            rows = A.complete_cases(c, ["tfm3", ref], ["mase"], c.cell.eq(cell))
            x = A.contrast(c, "tfm3", ref, "mase", rows, ["cell"], "cell", two_way=False, reps=reps)
            r[f"tfm3_vs_{'best' if ref == best else ref}"] = round(x["ratio"], 4)
            r[f"tfm3_vs_{'best' if ref == best else ref}_ci95"] = [round(v, 4) for v in x["ci95"]]
        out.append(r)
    return out


def equal_context(tree: Path, reps: int = 2_000) -> list[dict]:
    """R10: TimesFM-3 and Chronos-2 at equal context (2,016 and 8,192 points) on the hourly Tier B domains."""
    from .evaluate import score
    b = score("B", "main", tree, ROOT / "data" / "processed")
    hourly = b["mode"].eq("uni") & b.domain.isin(["load", "price", "weather"])
    models = ["tfm3-uni", "tfm3-c8192", "tfm3-c2016", "chronos2-uni", "chronos2-c2016"]
    rows = A.complete_cases(b, models, ["mase", "sql"], hourly)
    out = []
    for a, m, what in (("tfm3-c2016", "chronos2-c2016", "both at 2,016"), ("tfm3-c8192", "chronos2-uni", "both at 8,192"),
                       ("tfm3-uni", "chronos2-uni", "native (about 15,000 vs 8,192)"),
                       ("tfm3-uni", "tfm3-c8192", "TimesFM-3: full vs 8,192"), ("tfm3-c8192", "tfm3-c2016", "TimesFM-3: 8,192 vs 2,016"),
                       ("chronos2-uni", "chronos2-c2016", "Chronos-2: 8,192 vs 2,016")):
        for metric in ("mase", "sql"):
            x = A.contrast(b, a, m, metric, rows, ["domain", "horizon"], reps=reps)
            out.append({"a": a, "b": m, "comparison": what, "metric": metric, "ratio": round(x["ratio"], 4),
                        "ci95": [round(v, 4) for v in x["ci95"]], "n": x["n"]})
    return out


def croston(c: pd.DataFrame) -> list[dict]:
    """R11: Croston-type methods on the intermittent-demand cells (point forecasts: MASE and RMSSE)."""
    from statsforecast.models import ADIDA, IMAPA, TSB, CrostonOptimized, CrostonSBA
    from .forecast import load_tasks
    tasks, make = load_tasks("synthetic", "main", ROOT / "data" / "processed")
    tasks = tasks[tasks.cell.str.startswith("rep:D8")]
    methods = {"croston-opt": lambda: CrostonOptimized(), "croston-sba": lambda: CrostonSBA(), "adida": lambda: ADIDA(),
               "imapa": lambda: IMAPA(), "tsb": lambda: TSB(alpha_d=0.2, alpha_p=0.2)}
    rows = []
    for r in tasks.itertuples():
        task, truth, season = make(r)
        y, t = task.target[0].astype(float), truth[0].astype(float)
        diffs = np.diff(y)
        scale_mase, scale_rmsse = np.mean(np.abs(diffs)), np.sqrt(np.mean(diffs ** 2))
        for name, mk in methods.items():
            f = mk().forecast(y=y, h=task.horizon)["mean"]
            rows.append({"cell": r.cell, "model": name, "mase": np.mean(np.abs(t - f)) / scale_mase,
                         "rmsse": np.sqrt(np.mean((t - f) ** 2)) / scale_rmsse})
    d = pd.DataFrame(rows)
    gm = lambda s: float(np.exp(np.log(s[(s > 0) & np.isfinite(s)]).mean()))
    out = []
    for cell, g in d.groupby("cell"):
        ref = c[c.cell.eq(cell)]
        r = {"cell": cell}
        for model in list(methods) + ["tfm3", "chronos2", "mean", "ets"]:
            s = g[g.model.eq(model)] if model in methods else ref[ref.model.eq(model)]
            r[model] = {k: round(gm(s[k]), 4) for k in ("mase", "rmsse")}
        out.append(r)
    return out


FAMILY = {"boomlet": "boomlet"}


def _family(task: str) -> str:
    import re
    base = re.sub(r"_(\d+)?[A-Z]+$", "", task.split("/")[0])
    return "boomlet" if base.startswith("boomlet") else base


def tier_a_families(reps: int = REPS) -> dict:
    """R12: Tier A ratios with tasks resampled by source family (tasks of the same source move together)."""
    t = pd.read_csv(ROOT / "reports" / "tierA_reproduction.csv")
    t = t[t.model.eq("tfm3-eval")][["task_name", "skill_ours", "skill_published"]].dropna()
    t["family"] = t.task_name.map(_family)
    t["lo"], t["lp"] = np.log(t.skill_ours), np.log(t.skill_published)
    fams = t.family.unique()
    rng = np.random.default_rng(A.SEED)
    draws = rng.integers(0, len(fams), size=(reps, len(fams)))
    by = {f: g for f, g in t.groupby("family")}
    stats_ = {k: [] for k in ("lo", "lp", "diff")}
    for row in draws:
        s = pd.concat([by[fams[i]] for i in row])
        stats_["lo"].append(s.lo.mean()); stats_["lp"].append(s.lp.mean()); stats_["diff"].append((s.lo - s.lp).mean())
    ci = lambda x: [round(float(np.exp(np.quantile(x, 0.025))), 4), round(float(np.exp(np.quantile(x, 0.975))), 4)]
    return {"tasks": int(len(t)), "families": int(len(fams)), "family_of_task": dict(zip(t.task_name, t.family)),
            "ours_ci95": ci(stats_["lo"]), "published_ci95": ci(stats_["lp"]), "ours_over_published_ci95": ci(stats_["diff"])}


def main_extra(tree: Path) -> None:
    c = pd.read_parquet(MAIN / "synthetic" / "metrics.parquet")
    report = {"label": "second-round review-requested analyses (GPU runs of 2026-11-02, bundle 0.5.6); not pre-registered",
              "R9_regime_grid": regime_grid(), "R10_equal_context": equal_context(tree), "R11_croston": croston(c),
              "R12_tier_a_families": tier_a_families()}
    (ROOT / "reports" / "robustness_extra.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float)[:8000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--extra", help="scratch tree with the Tier B forecasts for the equal-context comparison")
    args = parser.parse_args()
    if args.extra:
        return main_extra(Path(args.extra))
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
