"""Confirmatory analysis (protocol section 5).

    python -m tfmstudy.analysis --period main            # families P and H, secondary families, tables

Inputs are the per-(task, model, series) metrics written by `evaluate.score`. Every contrast is a paired
log ratio d = log(error of A) - log(error of B) on identical series and windows. Estimates average d with
equal weight over nested levels (e.g. windows within a horizon, horizons within a domain, domains), and
uncertainty comes from a two-way cluster ("pigeonhole") bootstrap that resamples series clusters within
their stratum and calendar weeks of the forecast origin jointly, so that both serial overlap of windows and
common shocks across series are reflected (Cameron, Gelbach and Miller 2011; Owen 2007).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPS = 10_000
SEED = 17
CHUNK = 250
STRUCTURED = ("load", "price", "weather", "river")
NEAR_RANDOM_WALK = ("fx", "rates")


# ------------------------------------------------------------------ clusters and samples

def series_cluster(series_id: str) -> str:
    """Resampling unit: a series, except that the variables of one weather city form one cluster."""
    parts = series_id.split(":")
    return ":".join(parts[:2]) if parts[0] == "weather" else series_id


def first_step(origin: pd.Series, domain: pd.Series) -> pd.Series:
    """Timestamp of the first forecast step (hourly: +1 h; daily: +1 day; business daily: next business day)."""
    from .data import catalog

    origin = pd.to_datetime(origin, utc=True)
    freq = domain.map(lambda d: catalog.DOMAINS[d]["freq"] if d in catalog.DOMAINS else "h")
    step = pd.to_timedelta(np.where(freq == "h", 3600, 86400), unit="s")
    out = origin + step
    business = (freq == "B").to_numpy()
    out[business] = origin[business] + pd.offsets.BDay(1)
    return out


def time_cluster(origin: pd.Series, domain: pd.Series | None = None) -> pd.Series:
    """ISO week of the first forecast step (of the origin itself if the domain is unknown)."""
    stamp = first_step(origin, domain) if domain is not None else pd.to_datetime(origin, utc=True)
    iso = pd.DatetimeIndex(stamp).isocalendar()
    return pd.Series((iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)).to_numpy(), index=origin.index)


def quintile_group(q: pd.Series) -> pd.Series:
    """Popularity quintiles 0-4 with the two top quintiles pooled (W-new has 8 pages in quintile 3)."""
    return q.clip(upper=3)


def paired(df: pd.DataFrame, a: str, b: str, metric: str, rows: pd.Series | None = None) -> pd.DataFrame:
    """One row per (task, series) with both models' finite positive scores and d = log(a) - log(b)."""
    keys = ["task_id", "series_id"]
    sub = df if rows is None else df[rows]
    pa = sub[sub.model == a].set_index(keys)[metric]
    pb = sub[sub.model == b].set_index(keys)[metric]
    both = pd.concat({"a": pa, "b": pb}, axis=1, join="inner")
    both = both[np.isfinite(both.a) & np.isfinite(both.b) & (both.a > 0) & (both.b > 0)]
    info = sub.drop_duplicates(keys).set_index(keys).loc[both.index]
    out = info.reset_index()
    out["d"] = (np.log(both.a) - np.log(both.b)).to_numpy()
    # Tier C: the replication (one task, all its variates) is the unit; otherwise series clusters.
    out["cluster"] = out.task_id if "rep" in out else out.series_id.map(series_cluster)
    out["week"] = (time_cluster(out.origin, out.domain if "domain" in out else None).to_numpy()
                   if "origin" in out else "none")
    if "quintile" in out:
        out["qgroup"] = quintile_group(out.quintile)
    return out


def complete_cases(df: pd.DataFrame, models: list[str], metrics: list[str], rows: pd.Series) -> pd.Series:
    """Mask of the rows of `df` whose (task, series) has finite positive scores for all models and metrics."""
    keys = ["task_id", "series_id"]
    sub = df[rows & df.model.isin(models)]
    ok = np.ones(len(sub), bool)
    for m in metrics:
        ok &= np.isfinite(sub[m].to_numpy()) & (sub[m].to_numpy() > 0)
    good = sub[ok].groupby(keys).model.nunique()
    good = set(good[good == len(models)].index)
    return rows & pd.Series([k in good for k in zip(df.task_id, df.series_id)], index=df.index)


# ------------------------------------------------------------------ estimator and bootstrap

@dataclass
class Design:
    """Nested equal-weight mean of d with two-way cluster bootstrap weights.

    levels: columns from the outermost to the innermost averaging level, e.g. ["domain", "horizon"]
    (mean within domain x horizon cells, then over horizons, then over domains). groups: optional
    outermost column whose levels are kept separate (for differences such as W-seen minus W-post).
    """

    data: pd.DataFrame
    levels: list[str]
    resample_strata: str | None = None
    groups: str | None = None
    two_way: bool = True
    _cache: dict = field(default_factory=dict)

    def __post_init__(self):
        d = self.data
        cols = ([self.groups] if self.groups else []) + self.levels
        frame = d[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)
        self.cells = d.groupby(cols, sort=True).ngroup().to_numpy()
        # Membership matrices from the cells up to the outermost column (kept: groups, or averaged at the end).
        self.mats = []
        for depth in range(len(cols) - 1, 0, -1):
            parent = cols[:depth]
            codes = frame.groupby(parent, sort=True).ngroup().to_numpy()
            m = np.zeros((len(frame), codes.max() + 1))
            m[np.arange(len(frame)), codes] = 1
            self.mats.append(m)
            frame = frame[parent].drop_duplicates().sort_values(parent).reset_index(drop=True)
        self.group_names = list(frame[self.groups]) if self.groups else ["all"]
        self.onehot = np.zeros((len(d), self.cells.max() + 1))
        self.onehot[np.arange(len(d)), self.cells] = 1
        strata = d[self.resample_strata].astype(str).to_numpy() if self.resample_strata else np.full(len(d), "")
        pairs = pd.Series([f"{s}|{c}" for s, c in zip(strata, d.cluster)])
        self.cl_codes, cl_index = pd.factorize(pairs)
        self.cl_stratum = np.array([k.split("|", 1)[0] for k in cl_index])
        self.wk_codes, self.weeks = pd.factorize(d.week)

    def _reduce(self, cell_means: np.ndarray) -> np.ndarray:
        """(R, cells) -> (R, groups), equal weights, ignoring empty cells."""
        v = cell_means
        for m in self.mats:
            ok = np.isfinite(v)
            num = np.where(ok, v, 0) @ m
            den = ok.astype(float) @ m
            v = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        if not self.groups:
            ok = np.isfinite(v)
            v = (np.where(ok, v, 0).sum(1) / ok.sum(1))[:, None]
        return v

    def estimate(self, values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """values: (N,), weights: (R, N) or None -> (R or 1, groups)."""
        w = np.ones((1, len(values))) if weights is None else weights
        num = (w * values) @ self.onehot
        den = w @ self.onehot
        return self._reduce(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan))

    def weights(self, rng: np.random.Generator, reps: int) -> np.ndarray:
        n_cl = len(self.cl_stratum)
        wc = np.zeros((reps, n_cl))
        for s in np.unique(self.cl_stratum):
            idx = np.flatnonzero(self.cl_stratum == s)
            wc[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=reps)
        w = wc[:, self.cl_codes]
        if self.two_way:  # weeks resampled within their year (2023 and 2026 windows stay separate)
            wt = np.zeros((reps, len(self.weeks)))
            years = np.array([str(x)[:4] for x in self.weeks])
            for y in np.unique(years):
                idx = np.flatnonzero(years == y)
                wt[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=reps)
            w = w * wt[:, self.wk_codes]
        return w

    def bootstrap(self, values: np.ndarray, reps: int = REPS, seed: int = SEED) -> np.ndarray:
        """(reps, groups) bootstrap distribution of the estimate; `empty_share` records how often a cell
        received no weight (it is then left out of that replicate's average)."""
        rng = np.random.default_rng(seed)
        out, empty = [], 0
        for start in range(0, reps, CHUNK):
            w = self.weights(rng, min(CHUNK, reps - start))
            empty += int(((w @ self.onehot) == 0).any(axis=1).sum())
            out.append(self.estimate(values, w))
        self.empty_share = empty / reps
        return np.vstack(out)


def _summary(est: float, boot: np.ndarray, margin: float | None = None, floor: float | None = None) -> dict:
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = max(2 * min(np.mean(boot >= 0), np.mean(boot <= 0)), 1 / len(boot))
    out = {"estimate": float(est), "ratio": float(np.exp(est)), "ci95": [float(np.exp(lo)), float(np.exp(hi))],
           "p": float(min(p, 1.0)), "reps": int(len(boot))}
    if margin is not None:  # TOST on the log scale: equivalence if the 90% interval is inside (-delta, delta)
        delta = np.log(margin)
        p_tost = max(np.mean(boot <= -delta), np.mean(boot >= delta), 1 / len(boot))
        lo90, hi90 = np.percentile(boot, [5, 95])
        out.update(margin=margin, ci90=[float(np.exp(lo90)), float(np.exp(hi90))], p_tost=float(p_tost))
    if floor is not None:  # non-superiority of A: H0 ratio <= floor; rejected if the 95% lower bound exceeds it
        out.update(floor=floor, p_nonsup=float(max(np.mean(boot <= np.log(floor)), 1 / len(boot))),
                   lower95=float(np.exp(np.percentile(boot, 5))))
    return out


def contrast(df: pd.DataFrame, a: str, b: str, metric: str, rows: pd.Series, levels: list[str],
             resample_strata: str | None = "domain", two_way: bool = True, margin: float | None = None,
             reps: int = REPS, floor: float | None = None) -> dict:
    pr = paired(df, a, b, metric, rows)
    if pr.empty:  # a pre-registered test that cannot be computed counts as not supported (p = 1)
        return {"a": a, "b": b, "metric": metric, "n": 0, "estimate": np.nan, "ratio": np.nan, "p": 1.0,
                "p_tost": 1.0, "p_nonsup": 1.0, "missing": True}
    design = Design(pr, levels, resample_strata, two_way=two_way)
    est = design.estimate(pr.d.to_numpy())[0, 0]
    boot = design.bootstrap(pr.d.to_numpy(), reps)[:, 0]
    out = {"a": a, "b": b, "metric": metric, "n": len(pr), "clusters": int(pr.cluster.nunique()),
           "weeks": int(pr.week.nunique()), "empty_cell_share": design.empty_share, **_summary(est, boot, margin, floor)}
    if two_way:  # sensitivity: series clusters only (protocol section 5.2)
        one = Design(pr, levels, resample_strata, two_way=False).bootstrap(pr.d.to_numpy(), reps)[:, 0]
        out["ci95_oneway"] = _summary(est, one)["ci95"]
    return out


def did(df: pd.DataFrame, a: str, ref: str, metric: str, rows: pd.Series, first: str, second: str,
        levels: list[str], reps: int = REPS) -> dict:
    """Difference in differences: mean advantage log(a/ref) in stratum `first` minus that in `second`.

    Pages are the clusters: a page present in both strata (W-seen and W-post) is resampled jointly;
    weeks of 2023 and 2026 are distinct time clusters.
    """
    pr = paired(df, a, ref, metric, rows & df.stratum.isin([first, second]))
    if not {first, second} <= set(pr.stratum):  # a stratum without data: the test is missing (p = 1)
        return {"a": a, "ref": ref, "metric": metric, "first": first, "second": second, "n": len(pr),
                "estimate": np.nan, "ratio": np.nan, "p": 1.0, "missing": True}
    pr["resample"] = np.where(pr.stratum == "W-new", "new", "seen/post")
    design = Design(pr, levels, "resample", groups="stratum")
    g = design.group_names
    i, j = g.index(first), g.index(second)
    est = design.estimate(pr.d.to_numpy())[0]
    boot = design.bootstrap(pr.d.to_numpy(), reps)
    res = _summary(est[i] - est[j], boot[:, i] - boot[:, j])
    res.update(a=a, ref=ref, metric=metric, first=first, second=second, n=len(pr), empty_cell_share=design.empty_share,
               advantage={first: float(np.exp(est[i])), second: float(np.exp(est[j]))})
    return res


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def decide(results: list[dict], key: str = "p") -> list[dict]:
    """Holm-adjusted p within a family; a test is resolved when the adjusted p is below 0.05."""
    adj = holm([r[key] for r in results])
    for r, p in zip(results, adj):
        r[f"{key}_holm"] = p
        r["resolved"] = bool(p < 0.05)
    return results


# ------------------------------------------------------------------ families (protocol section 5)

def family_p(b: pd.DataFrame, reps: int = REPS) -> list[dict]:
    """Primary family: Tier B, information-matched, structured domains; MASE and SQL; Holm over 10 tests."""
    uni = (b["mode"] == "uni") & b.domain.isin(STRUCTURED)
    grp = b["mode"] == "group"
    models_uni = ["tfm3-uni", "stat-ens", "chronos2-uni", "tfm25", "tfm3-eval", "tfm3"]
    cc_uni = complete_cases(b, models_uni, ["mase", "sql"], uni)
    cc_grp = complete_cases(b, ["tfm3", "tfm3-uni"], ["mase", "sql"], grp)
    tests = [("P1", "tfm3-uni", "stat-ens", cc_uni), ("P2", "tfm3-uni", "chronos2-uni", cc_uni),
             ("P3", "tfm3-uni", "tfm25", cc_uni), ("P4a", "tfm3", "tfm3-uni", cc_grp),
             ("P5", "tfm3-eval", "tfm3", cc_uni)]
    out = []
    for name, a, m, rows in tests:
        for metric in ("mase", "sql"):
            out.append({"test": f"{name}:{metric}", **contrast(b, a, m, metric, rows, ["domain", "horizon"], reps=reps)})
    return decide(out)


H_TESTS = ("H2", "H3", "H4a", "H4b", "H4c", "H4d", "H5")


def family_h(b: pd.DataFrame | None, w: pd.DataFrame | None, c: pd.DataFrame | None, reps: int = REPS) -> list[dict]:
    """Hypothesis family: one primary test per hypothesis, always seven tests (a test that cannot be computed
    enters Holm with p = 1 and is reported as missing)."""
    res: dict[str, dict] = {}
    if w is not None:  # H2: advantage of TimesFM-3 over Chronos-2 larger on W-seen than on W-post
        rows = complete_cases(w, ["tfm3", "chronos2"], ["mase"], w["mode"] == "uni")
        res["H2"] = {"direction": "negative", **did(w, "tfm3", "chronos2", "mase", rows, "W-seen", "W-post",
                                                     ["horizon", "qgroup"], reps)}
    if c is not None:
        if "tfm3-nodetrend" in set(c.model):  # H3 (ablation): MASE over steps 25-48 of D7, n = 200, H = 48
            rows = c.cell.eq("rep:D7:n200")
            res["H3"] = {"direction": "negative", **contrast(c, "tfm3-nodetrend", "tfm3", "mase_25_48", rows, ["cell"],
                                                             "cell", two_way=False, reps=reps)}
        for name, cell, margin in [("H4a", "mv:lead_lag:24", None), ("H4b", "mv:known_cov:snr2", None),
                                   ("H4c", "mv:independent:8", 1.03), ("H4d", "mv:spurious_past_cov", 1.03)]:
            rows = c.cell.eq(cell)
            if name == "H4a":  # only the follower (variate 1) can gain from the leader's history
                rows &= c.series_id.str.endswith(":1")
            r = contrast(c, "tfm3", "tfm3-uni", "mase", rows, ["cell"], "cell", two_way=False, margin=margin, reps=reps)
            res[name] = {"direction": "negative" if margin is None else "equivalence", **r}
    if b is not None:  # H5: TimesFM-3 does not beat the random walk by more than 10% on FX and FRED series
        rows = (b["mode"] == "uni") & b.domain.isin(NEAR_RANDOM_WALK)
        rows = complete_cases(b, ["tfm3-uni", "naive"], ["mase"], rows)
        res["H5"] = {"direction": "nonsuperiority", **contrast(b, "tfm3-uni", "naive", "mase", rows, ["domain", "horizon"],
                                                              floor=0.90, reps=reps)}
    out = []
    for name in H_TESTS:
        r = res.get(name, {"missing": True, "p": 1.0, "p_tost": 1.0, "estimate": np.nan})
        default = {"H4c": "equivalence", "H4d": "equivalence", "H5": "nonsuperiority"}.get(name, "negative")
        r["direction"] = r.get("direction", default)
        r["p_family"] = {"equivalence": r.get("p_tost", 1.0), "nonsuperiority": r.get("p_nonsup", 1.0)}.get(
            r["direction"], r.get("p", 1.0))
        r["sign_ok"] = r["direction"] != "negative" or (np.isfinite(r["estimate"]) and r["estimate"] < 0)
        out.append({"test": name, **r})
    out = decide(out, "p_family")
    for r in out:
        r["supported"] = bool(r["resolved"] and r["sign_ok"] and not r.get("missing"))
    return out


def context_features(task, target: np.ndarray) -> dict:
    """Features of one window's context (protocol section 5.4), from data up to the origin only."""
    y = np.asarray(target, dtype=float)
    y = y[np.isfinite(y)]
    ly = np.log1p(np.clip(y, 0, None))
    t = np.arange(len(ly)) / 100.0
    slope = np.polyfit(t, ly, 1)[0] if len(ly) > 2 else 0.0
    ac7 = np.corrcoef(ly[7:], ly[:-7])[0, 1] if len(ly) > 14 and ly.std() > 0 else 0.0
    med = np.median(y) if len(y) else 0.0
    return {"f_level": float(np.log1p(y.mean())), "f_cv": float(y.std() / y.mean()) if y.mean() > 0 else 0.0,
            "f_slope": float(slope), "f_ac7": float(ac7), "f_spike": float(np.mean(y > 3 * med)) if med > 0 else 0.0}


FEATURES = ["f_level", "f_cv", "f_slope", "f_ac7", "f_spike"]


def wiki_features(period: str, data: Path, held_out: tuple[str, ...] = ()) -> pd.DataFrame:
    """Context features of every Tier W task (one row per task_id)."""
    from .forecast import load_tasks

    tasks, make = load_tasks("W", period, data)
    tasks = tasks[~tasks.stratum.isin(held_out)]
    return pd.DataFrame([{"task_id": r.task_id, **context_features(r, make(r)[0].target[0])} for r in tasks.itertuples()])


def adjusted_did(df: pd.DataFrame, a: str, ref: str, metric: str, rows: pd.Series, first: str, second: str,
                 features: pd.DataFrame, reps: int = REPS) -> dict:
    """WLS of the advantage on stratum, horizon and popularity-group indicators and context features;
    the coefficient of `first` (with `second` as the base) is the adjusted DiD (protocol section 5.4)."""
    pr = paired(df, a, ref, metric, rows & df.stratum.isin([first, second]))
    if not {first, second} <= set(pr.stratum):
        return {"a": a, "ref": ref, "metric": metric, "first": first, "second": second, "n": len(pr),
                "estimate": np.nan, "ratio": np.nan, "p": 1.0, "missing": True, "adjusted": True}
    pr = pr.merge(features, on="task_id", how="left", validate="many_to_one")
    if pr[FEATURES].isna().any().any():
        raise ValueError("context features missing for some tasks")
    # Equal total weight for every stratum x horizon x popularity-group cell, as in did().
    cell_w = 1.0 / pr.groupby(["stratum", "horizon", "qgroup"]).d.transform("size").to_numpy()
    z = pr[FEATURES].astype(float)
    z = (z - z.mean()) / z.std().replace(0, 1)
    X = np.column_stack([np.ones(len(pr)), (pr.stratum == first).astype(float),
                         pd.get_dummies(pr.horizon, drop_first=True, dtype=float).to_numpy(),
                         pd.get_dummies(pr.qgroup, drop_first=True, dtype=float).to_numpy(), z.fillna(0).to_numpy()])
    y = pr.d.to_numpy()
    pr["resample"] = np.where(pr.stratum == "W-new", "new", "seen/post")
    design = Design(pr, ["horizon"], "resample")
    rng = np.random.default_rng(SEED)

    def coef(w):
        xtw = X.T * (w * cell_w)
        return np.linalg.lstsq(xtw @ X, xtw @ y, rcond=None)[0][1]

    est = coef(np.ones(len(y)))
    boot = []
    for start in range(0, reps, CHUNK):
        for w in design.weights(rng, min(CHUNK, reps - start)):
            boot.append(coef(w))
    res = _summary(est, np.array(boot))
    res.update(a=a, ref=ref, metric=metric, first=first, second=second, n=len(pr), adjusted=True)
    return res


def secondary(b, w, c, features: pd.DataFrame | None = None, reps: int = REPS) -> dict[str, list[dict]]:
    """Secondary families S1-S5 (protocol section 5.3); Holm within each."""
    fams: dict[str, list[dict]] = {}
    if b is not None:
        uni = (b["mode"] == "uni") & b.domain.isin(STRUCTURED)
        cc = complete_cases(b, ["tfm3-uni", "stat-ens", "chronos2-uni", "tfm25"], ["mase"], uni)
        s1 = []
        for dom in STRUCTURED:
            for name, m in (("P1", "stat-ens"), ("P2", "chronos2-uni"), ("P3", "tfm25")):
                s1.append({"test": f"{name}:{dom}", **contrast(b, "tfm3-uni", m, "mase", cc & b.domain.eq(dom),
                                                                ["domain", "horizon"], reps=reps)})
        fams["S1"] = decide(s1)
        cov = (b["mode"] == "uni") & b.domain.isin(["load", "river"])
        s3 = [{"test": "P4b", **contrast(b, "tfm3", "tfm3-uni", "mase", complete_cases(b, ["tfm3", "tfm3-uni"], ["mase"], cov),
                                         ["domain", "horizon"], reps=reps)},
              {"test": "chronos2-cov", **contrast(b, "chronos2", "chronos2-uni", "mase",
                                                  complete_cases(b, ["chronos2", "chronos2-uni"], ["mase"], cov),
                                                  ["domain", "horizon"], reps=reps)}]
        fams["S3"] = decide(s3)
        hourly = uni & b.domain.isin(["load", "price", "weather"])
        s4 = [{"test": "c2016", **contrast(b, "tfm3-c2016", "tfm3-uni", "mase",
                                           complete_cases(b, ["tfm3-c2016", "tfm3-uni"], ["mase"], hourly),
                                           ["domain", "horizon"], reps=reps)}]
        fams["S4"] = decide(s4)
    if w is not None:
        s2 = []
        uni = w["mode"] == "uni"
        for a, ref in (("tfm3", "stat-ens"), ("tfm3", "tirex2"), ("tfm25", "chronos2")):
            rows = complete_cases(w, [a, ref], ["mase"], uni)
            s2.append({"test": f"DiD:{a}-vs-{ref}", **did(w, a, ref, "mase", rows, "W-seen", "W-post", ["horizon", "qgroup"], reps)})
        rows = complete_cases(w, ["tfm3", "chronos2"], ["mase"], uni)
        s2.append({"test": "post-vs-new", **did(w, "tfm3", "chronos2", "mase", rows, "W-post", "W-new", ["horizon", "qgroup"], reps)})
        if features is not None:
            for first, second in (("W-seen", "W-post"), ("W-post", "W-new")):
                s2.append({"test": f"adjusted:{first}-vs-{second}",
                           **adjusted_did(w, "tfm3", "chronos2", "mase", rows, first, second, features, reps)})
        fams["S2"] = decide(s2)
    s5 = []
    for tier, df in (("B", b), ("C", c)):
        if df is None:
            continue
        for abl in sorted(m for m in set(df.model) if m.startswith("tfm3-no") or m in
                          ("tfm3-sym", "tfm3-pos", "tfm3-znorm", "tfm3-edge")):
            rows = complete_cases(df, [abl, "tfm3"], ["mase"], df.model.notna())
            if abl in ("tfm3-nocpmrevin", "tfm3-nostitch"):
                rows &= df.horizon > 64
            levels, strata, tw = (["domain", "horizon"], "domain", True) if tier == "B" else (["cell"], "cell", False)
            s5.append({"test": f"{tier}:{abl}", **contrast(df, abl, "tfm3", "mase", rows, levels, strata, tw, reps=reps)})
    if s5:
        fams["S5"] = decide(s5)
    return fams


def family_s6(p: pd.DataFrame, reps: int = REPS) -> list[dict]:
    """Tier P (protocol section 5.5): both issues pooled, issue as a fixed averaging level, series clusters only."""
    out = []
    groups = {"structured": p.domain.isin(["load", "price", "river"]), "wiki": p.domain.eq("wiki")}
    for gname, g in groups.items():
        rows = complete_cases(p, ["tfm3-uni", "stat-ens", "chronos2-uni", "tfm25", "naive"], ["mase", "sql"],
                              g & p["mode"].eq("uni"))
        for ref in ("stat-ens", "chronos2-uni", "tfm25", "naive"):
            for metric in ("mase", "sql"):
                out.append({"test": f"P:{gname}:{ref}:{metric}",
                            **contrast(p, "tfm3-uni", ref, metric, rows, ["issue", "domain", "horizon"], "domain",
                                       two_way=False, reps=reps)})
    return decide(out)


# ------------------------------------------------------------------ entry point

def load(results: Path, period: str, tier: str) -> pd.DataFrame | None:
    path = results / period / tier / "metrics.parquet"
    return pd.read_parquet(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/forecast")
    parser.add_argument("--period", default="main")
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--out", default="results/analysis")
    parser.add_argument("--data", default="data/processed")
    args = parser.parse_args()
    res = Path(args.results)
    b, w, c = (load(res, args.period, t) for t in ("B", "W", "synthetic"))
    report = {"period": args.period, "reps": args.reps}
    prospective = [pd.read_parquet(f) for f in sorted(res.glob("prospective-*/P/metrics.parquet"))]
    if prospective:
        report["family_S6"] = family_s6(pd.concat(prospective, ignore_index=True), args.reps)
    if b is not None:
        report["family_P"] = family_p(b, args.reps)
    report["family_H"] = family_h(b, w, c, args.reps)
    feats = wiki_features(args.period, Path(args.data)) if w is not None else None
    report.update({f"family_{k}": v for k, v in secondary(b, w, c, feats, args.reps).items()})
    out = Path(args.out) / args.period
    out.mkdir(parents=True, exist_ok=True)
    (out / "confirmatory.json").write_text(json.dumps(report, indent=1, default=str))
    for fam in [k for k in report if k.startswith("family_")]:
        if report[fam]:
            print(pd.DataFrame(report[fam])[[c for c in ("test", "ratio", "ci95", "p", "p_tost", "resolved", "supported")
                                             if c in pd.DataFrame(report[fam])]].to_string(index=False))


if __name__ == "__main__":
    main()
