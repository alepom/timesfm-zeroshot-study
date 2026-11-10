"""Robustness and sensitivity analyses added after the freeze (not pre-registered; reported as such).

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
R8  every model against the random walk on FX and rates.

    python -m tfmstudy.robustness --extra <tree> [--parts R10 R13 ...]   -> reports/robustness_extra.json

R9  level-shift grid; R10 equal context with the model x context interaction; R11 Croston-type methods on D8;
R12 Tier A by source family; R13 family P and H5 on non-overlapping windows and by horizon; R14 serial dependence of the
loss differentials; R15 the windows behind the SQL tails of P1; R16 excluded scores and fallbacks.
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
METHODS = {
    "estimator": "nested equal-weight mean of the per-window log ratio (cells domain x horizon, then horizons, then domains)",
    "bootstrap": ("two-way multiplicative-weight (pigeonhole) cluster bootstrap: series clusters drawn with replacement "
                  "within domain strata, times ISO weeks of the first forecast step drawn with replacement "
                  "(Owen 2007, Ann. Appl. Stat. 1:386-411); one-way over replications on synthetic data"),
    "interval": "percentile, 2.5 and 97.5 per cent of the bootstrap distribution of the log estimate, exponentiated",
    "p_value": "two-sided, 2 min(share of replicates >= 0, share <= 0), floored at 1/reps",
    "seed": 17,
    "reps": {"R2, R7, R12, R13": 10_000, "R3, R9, R10": 2_000},
    "status": "post hoc, not pre-registered; intervals and p-values are unadjusted unless a Holm column is given",
}
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
    stored = ROOT / "results" / "forecast" / "regime" / "synthetic" / "metrics.parquet"
    if stored.exists():
        c = pd.read_parquet(stored)
    else:
        from .evaluate import score
        c = score("synthetic", "regime", ROOT / "results" / "forecast", ROOT / "data" / "processed")
    classical = ["ets", "theta", "mstl", "mstl-theta", "mstl-arima", "naive", "snaive", "mean", "stat-ens"]
    gm = lambda s: float(np.exp(np.log(s[(s > 0) & np.isfinite(s)]).mean()))
    out = []
    for cell, g in c.groupby("cell"):
        level = {m: gm(g[g.model.eq(m)].mase) for m in g.model.unique()}
        best = min((m for m in classical if m in level), key=level.get)
        r = {"cell": cell, "best_classical": best, **{m: round(v, 4) for m, v in level.items()}}
        for key, ref in (("best", best), ("chronos2", "chronos2"), ("stat-ens", "stat-ens")):  # best: chosen post hoc
            rows = A.complete_cases(c, ["tfm3", ref], ["mase"], c.cell.eq(cell))
            x = A.contrast(c, "tfm3", ref, "mase", rows, ["cell"], "cell", two_way=False, reps=reps)
            r[f"tfm3_vs_{key}"] = round(x["ratio"], 4)
            r[f"tfm3_vs_{key}_ci95"] = [round(v, 4) for v in x["ci95"]]
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
    for metric in ("mase", "sql"):  # interaction: TimesFM-3's gain from 2,016 to 8,192 over Chronos-2's, jointly resampled
        keys = ["task_id", "series_id"]
        t = A.paired(b, "tfm3-c8192", "tfm3-c2016", metric, rows).set_index(keys)
        c = A.paired(b, "chronos2-uni", "chronos2-c2016", metric, rows).set_index(keys)
        pr = t.loc[t.index.intersection(c.index)].copy()
        pr["d"] = pr.d - c.loc[pr.index, "d"]
        pr = pr.reset_index()
        design = A.Design(pr, ["domain", "horizon"], "domain")
        est = design.estimate(pr.d.to_numpy())[0, 0]
        s = A._summary(est, design.bootstrap(pr.d.to_numpy(), reps)[:, 0])
        out.append({"a": "tfm3-c8192 / tfm3-c2016", "b": "chronos2-uni / chronos2-c2016",
                    "comparison": "interaction: gain from 2,016 to 8,192, TimesFM-3 over Chronos-2", "metric": metric,
                    "ratio": round(s["ratio"], 4), "ci95": [round(v, 4) for v in s["ci95"]], "p": round(s["p"], 4),
                    "n": int(len(pr))})
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


# ------------------------------------------------------------------ third round (no new forecasts)

def _end(r) -> pd.Timestamp:
    """Last forecast step of a window: hourly load, price and weather, daily river, business-daily FX and rates."""
    if r.domain in ("fx", "rates"):
        return r.origin + pd.offsets.BDay(int(r.horizon))
    return r.origin + (pd.Timedelta(days=int(r.horizon)) if r.domain == "river" else pd.Timedelta(hours=int(r.horizon)))


def non_overlapping(b: pd.DataFrame) -> pd.Series:
    """Mask keeping, per series and horizon, the earliest window and then each window that starts after the last kept
    one ends. Only the 28-day river and 20-business-day FX and rates windows (weekly origins) overlap; the others are
    all kept."""
    w = b[["series_id", "domain", "horizon", "origin"]].drop_duplicates()
    keep = set()
    for (_, _), g in w.groupby(["series_id", "horizon"]):
        last = None
        for r in g.sort_values("origin").itertuples():
            if last is None or r.origin >= last:
                keep.add((r.series_id, r.horizon, r.origin))
                last = _end(r)
    return pd.Series([k in keep for k in zip(b.series_id, b.horizon, b.origin)], index=b.index)


def _horizon_class(pr: pd.DataFrame) -> pd.Series:
    short = pr.groupby("domain").horizon.transform("min")
    return np.where(pr.horizon.eq(short), "short", "long")


def non_overlap(b: pd.DataFrame, reps: int = REPS) -> dict:
    """R13: family P and H5 on non-overlapping windows (two-way bootstrap with Holm over family P), and the same contrasts
    separately for each domain's short and long horizon (all windows and non-overlapping windows)."""
    rows, keep = _rows(b), non_overlapping(b)
    tests = [(n, a, m, met) for n, a, m in P_TESTS for met in ("mase", "sql")] + [("H5", "tfm3-uni", "naive", "mase")]
    out, by_h = [], []
    for name, a, m, metric in tests:
        n_all = 0
        for label, mask in (("all", rows[name]), ("non_overlapping", rows[name] & keep)):
            pr = A.paired(b, a, m, metric, mask)
            n_all = n_all or len(pr)
            design = A.Design(pr, ["domain", "horizon"], "domain")
            est = design.estimate(pr.d.to_numpy())[0, 0]
            s = A._summary(est, design.bootstrap(pr.d.to_numpy(), reps)[:, 0], floor=0.90 if name == "H5" else None)
            if label == "non_overlapping":
                out.append({"test": f"{name}:{metric}", "windows": int(len(pr)), "dropped": int(n_all - len(pr)), "weeks": int(pr.week.nunique()),
                            "ratio": round(s["ratio"], 4), "ci95": [round(v, 4) for v in s["ci95"]],
                            "p": s["p_nonsup"] if name == "H5" else s["p"]})
            pr["hclass"] = _horizon_class(pr)
            for hc, g in pr.groupby("hclass"):
                dg = A.Design(g, ["domain", "horizon"], "domain")
                e = dg.estimate(g.d.to_numpy())[0, 0]
                sg = A._summary(e, dg.bootstrap(g.d.to_numpy(), 2_000)[:, 0])
                by_h.append({"test": f"{name}:{metric}", "windows_used": label, "horizons": hc, "windows": int(len(g)),
                             "ratio": round(sg["ratio"], 4), "ci95": [round(v, 4) for v in sg["ci95"]]})
    fam = [r for r in out if not r["test"].startswith("H5")]
    for r, ph in zip(fam, A.holm([r["p"] for r in fam])):
        r["p_holm"], r["resolved"] = round(ph, 4), bool(ph < 0.05)
    return {"note": "post hoc; 28-day river and 20-business-day FX and rates windows thinned to non-overlapping origins",
            "family_p_and_h5": out, "by_horizon": by_h}


def loss_dependence(b: pd.DataFrame) -> list[dict]:
    """R14: lag-one autocorrelation of the loss differential between consecutive origins of the same series (series means
    removed), and the share of its variance that is common to all series of a domain in the same week."""
    rows, out = _rows(b), []
    for name, a, m in P_TESTS[:3]:
        for metric in ("mase", "sql"):
            pr = A.paired(b, a, m, metric, rows[name]).sort_values(["series_id", "horizon", "origin"])
            for (dom, h), g in pr.groupby(["domain", "horizon"]):
                x = g.d - g.groupby("series_id").d.transform("mean")
                lag = x.groupby(g.series_id).shift(1)
                ok = lag.notna()
                wk = g.d - g.groupby("week").d.transform("mean")
                out.append({"test": f"{name}:{metric}", "domain": dom, "horizon": int(h), "windows": int(len(g)),
                            "origins": int(g.origin.nunique()),
                            "lag1_autocorrelation": round(float(np.corrcoef(x[ok], lag[ok])[0, 1]), 3) if ok.sum() > 2 else None,
                            "week_share_of_variance": round(float(1 - wk.var() / g.d.var()), 3)})
    return out


def sql_tails(b: pd.DataFrame) -> dict:
    """R15: which windows the 1% trimming of P1 on SQL removes (it moves the ratio from 0.707 to about 0.80)."""
    pr = A.paired(b, "tfm3-uni", "stat-ens", "sql", _rows(b)["P1"])
    lo, hi = np.quantile(pr.d, [0.01, 0.99])
    tails = pr[(pr.d < lo) | (pr.d > hi)]
    return {"windows": int(len(pr)), "trimmed": int(len(tails)),
            "lower_tail_by_domain": tails[tails.d < lo].domain.value_counts().to_dict(),
            "upper_tail_by_domain": tails[tails.d > hi].domain.value_counts().to_dict(),
            "lower_tail_median_ratio": round(float(np.exp(tails[tails.d < lo].d.median())), 4),
            "upper_tail_median_ratio": round(float(np.exp(tails[tails.d > hi].d.median())), 4),
            "share_of_windows_by_domain": pr.domain.value_counts().to_dict(),
            "nested_without_trimmed_river": round(float(np.exp(_nested(pr[~(pr.index.isin(tails.index) & pr.domain.eq("river"))]))), 4)}


def exclusions() -> dict:
    """R16: scores excluded from the paired contrasts (non-finite or zero) and fallback forecasts, per tier."""
    out = {}
    for tier in ("B", "W", "synthetic"):
        d = pd.read_parquet(MAIN / tier / "metrics.parquet")
        bad = ~(np.isfinite(d.mase) & (d.mase > 0) & np.isfinite(d.sql) & (d.sql > 0))
        out[tier] = {"scores": int(len(d)), "excluded": int(bad.sum()), "excluded_by_model": d[bad].model.value_counts().to_dict(),
                     "fallbacks": int(d.fallback.sum())}
    t = pd.read_parquet(ROOT / "data" / "processed" / "main" / "tasks.parquet")
    out["B_windows_excluded_before_forecasting"] = {"windows": int(len(t)), "excluded": int(t.excluded.ne("").sum()),
                                                    "reasons": t[t.excluded.ne("")].excluded.str.split(": ").str[1]
                                                    .value_counts().to_dict()}
    b = pd.read_parquet(MAIN / "B" / "metrics.parquet")
    out["family_P_complete_cases"] = {"univariate_structured": int(b[(b["mode"] == "uni") & b.domain.isin(A.STRUCTURED)]
                                                                   .groupby(["task_id", "series_id"]).ngroups),
                                      "used": int(len(A.paired(b, "tfm3-uni", "stat-ens", "mase", _rows(b)["P1"])))}
    return out


EXTRA_METHODS = {
    **METHODS,
    "reps": {"R12, R13": 10_000, "R9, R10, R13 by_horizon": 2_000},
    "R9": ("DGP: 10 + 3 sin(2 pi t/24) + 2 sin(2 pi t/168) + AR(1) noise (phi 0.5, innovation sd 0.5), 1,024 points, H 48; "
           "upward level shift of size x the marginal noise sd (0.577) starting `lag` steps before the origin; 100 "
           "replications per cell, seed 2026, independent draws per cell (no common random numbers); intervals over "
           "replications; the best classical model is selected per cell after seeing the results, stat-ens is fixed"),
    "R10": "hourly Tier B windows (load, price, weather) with all five configurations; interaction resampled jointly",
    "R17": ("per Tier B domain, univariate configurations, complete cases over tfm3-uni, stat-ens, chronos2-uni and naive; "
            "two-way bootstrap within the domain, 2,000 draws; descriptive, no multiplicity adjustment"),
    "R12": ("tasks resampled by source family (22 families in order of first appearance), simple mean of log skill "
            "ratios, 10,000 draws, seed 17"),
}


def by_domain(b: pd.DataFrame, reps: int = 2_000) -> list[dict]:
    """R17: TimesFM-3 (univariate) against the classical ensemble, Chronos-2 and the random walk in each Tier B domain
    (descriptive; feeds the overview figure). Complete cases over the four models, horizons averaged with equal weight."""
    ms = ["tfm3-uni", "stat-ens", "chronos2-uni", "naive"]
    out = []
    for dom in sorted(b.domain.unique()):
        rows = A.complete_cases(b, ms, ["mase", "sql"], (b["mode"] == "uni") & b.domain.eq(dom))
        for ref in ms[1:]:
            for metric in ("mase", "sql"):
                r = A.contrast(b, "tfm3-uni", ref, metric, rows, ["horizon"], reps=reps)
                out.append({"domain": dom, "b": ref, "metric": metric, "ratio": round(r["ratio"], 4),
                            "ci95": [round(x, 4) for x in r["ci95"]], "n": r["n"], "series": r["clusters"],
                            "weeks": r["weeks"]})
    return out


def main_extra(tree: Path, parts: list[str] | None = None) -> None:
    path = ROOT / "reports" / "robustness_extra.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["label"] = "review-requested analyses of the second and third rounds; not pre-registered"
    report["methods"] = EXTRA_METHODS
    b = pd.read_parquet(MAIN / "B" / "metrics.parquet")
    jobs = {"R9_regime_grid": lambda: regime_grid(), "R10_equal_context": lambda: equal_context(tree),
            "R11_croston": lambda: croston(pd.read_parquet(MAIN / "synthetic" / "metrics.parquet")),
            "R12_tier_a_families": lambda: tier_a_families(), "R13_non_overlap": lambda: non_overlap(b),
            "R14_loss_dependence": lambda: loss_dependence(b), "R15_sql_tails": lambda: sql_tails(b),
            "R16_exclusions": exclusions, "R17_by_domain": lambda: by_domain(b)}
    for key, job in jobs.items():
        if not parts or key.split("_")[0] in parts:
            report[key] = job()
    path.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k not in ("methods", "R12_tier_a_families")}, indent=1, default=float)[:12000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--extra", help="scratch tree with the Tier B forecasts for the equal-context comparison")
    parser.add_argument("--parts", nargs="*", help="with --extra: recompute only these (e.g. R10 R13); keep the others")
    args = parser.parse_args()
    if args.extra:
        return main_extra(Path(args.extra), args.parts)
    b, w, c = (pd.read_parquet(MAIN / t / "metrics.parquet") for t in ("B", "W", "synthetic"))
    report = {"label": "robustness and sensitivity analyses added after the freeze; not pre-registered", "methods": METHODS,
              "R1_aggregation": aggregation(b), "R2_small_clusters": small_g(b, args.reps), "R3_context": context(b),
              "R4_ablations_by_domain": ablations(b), "R5_did_by_difficulty": did_by_difficulty(w),
              "R6_intermittent": intermittent(c), "R7_tier_a": tier_a(args.reps),
              "R8_near_random_walk": near_random_walk(b)}
    OUT.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
