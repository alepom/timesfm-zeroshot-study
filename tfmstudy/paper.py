"""Tables, numbers and figure data of the paper, generated from the analysis outputs.

    python -m tfmstudy.paper              # paper/generated/*.tex
    python -m tfmstudy.paper --refresh    # also recompute the descriptive analyses (reports/descriptive.json)

Every number quoted in the text comes from paper/generated/numbers.tex (LaTeX macros) or from a generated table.
Inputs: results/analysis/main/confirmatory.json (confirmatory families, protocol section 5), reports/tierA_*.{csv,json}
(Tier A), the per-series metrics in results/forecast/main/*/metrics.parquet and the run records next to them.
Descriptive analyses that are not part of the confirmatory families (H1, calibration, Tier C cells, compute) are
computed here and cached in reports/descriptive.json; they are labelled as descriptive in the paper.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "paper" / "generated"
CONFIRMATORY = ROOT / "results" / "analysis" / "main" / "confirmatory.json"
DESCRIPTIVE = ROOT / "reports" / "descriptive.json"
S6 = ROOT / "results" / "analysis" / "prospective" / "confirmatory_s6.json"
METRICS = ROOT / "results" / "forecast" / "main"
DESCRIPTIVE_REPS = 2_000

NAMES = {"tfm3": "TimesFM-3", "tfm3-uni": "TimesFM-3 uni", "tfm3-eval": "TimesFM-3 eval", "tfm3-c2016": "TimesFM-3 c2016",
         "tfm25": "TimesFM-2.5", "chronos2": "Chronos-2", "chronos2-uni": "Chronos-2 uni", "tirex2": "TiRex-2",
         "tirex2-uni": "TiRex-2 uni", "stat-ens": "Stat.\\ ensemble", "naive": "Naive", "snaive": "Seasonal Naive",
         "mean": "Mean", "ets": "AutoETS", "arima": "AutoARIMA", "theta": "AutoTheta", "mstl": "MSTL-ETS",
         "mstl-theta": "MSTL-Theta", "mstl-arima": "MSTL-ARIMA"}
CLASSICAL = ("naive", "snaive", "mean", "ets", "arima", "theta", "mstl", "mstl-theta", "mstl-arima", "stat-ens")


def name(key: str) -> str:
    if key.startswith("tfm3-") and key not in NAMES:
        return f"\\texttt{{{key}}}"
    return NAMES.get(key, key)


def r3(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:,.0f}".replace(",", "{,}") if abs(x) >= 100 else f"{x:.3f}"  # broken ablations reach 10^4


def r4(x: float) -> str:
    """Four decimals for ratios near 1, where three would print 1.000."""
    if x is None or not np.isfinite(x):
        return "--"
    return r3(x) if abs(x) >= 100 else f"{x:.4f}"


def ci(pair, fmt=r3) -> str:
    return "--" if not pair else f"[{fmt(pair[0])}, {fmt(pair[1])}]"


def pval(p: float) -> str:
    return "--" if p is None or not np.isfinite(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def tabular(spec: str, header: list[str], rows: list[list[str] | str]) -> str:
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        out.append(r if isinstance(r, str) else " & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def write(fname: str, text: str) -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / fname).write_text(text)


# ------------------------------------------------------------------ descriptive analyses

def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def _calibration(df: pd.DataFrame, models: list[str], unit: str) -> list[dict]:
    """Mean empirical coverage of the 80% and 60% central intervals; the series (`unit`) is the unit of the
    Wilson interval, so the interval reflects the number of independent series, not of forecast steps."""
    out = []
    for m in models:
        sub = df[df.model == m]
        if sub.empty:
            continue
        rec = {"model": m, "series": int(sub[unit].nunique())}
        for lvl in ("80", "60"):
            per_series = sub.groupby(unit)[f"coverage{lvl}"].mean().dropna()
            p = float(per_series.mean())
            rec[f"coverage{lvl}"] = p
            rec[f"wilson{lvl}"] = list(wilson(p, len(per_series)))
        out.append(rec)
    return out


def _cell_table(c: pd.DataFrame) -> list[dict]:
    """Tier C: geometric mean over replications of MASE relative to Seasonal Naive, per cell and model."""
    models = ["tfm3", "tfm3-uni", "tfm25", "chronos2", "tirex2", "stat-ens", "naive", "mean", "ets", "theta", "arima",
              "mstl", "mstl-theta", "mstl-arima"]
    rows = []
    for (family, cell), g in c[c.model.isin(models)].groupby(["family", "cell"]):
        rec = {"family": family, "cell": cell, "horizon": int(g.horizon.iloc[0])}
        for m, gm in g.groupby("model"):
            v = gm.rel_mase.to_numpy()
            v = v[np.isfinite(v) & (v > 0)]
            rec[m] = float(np.exp(np.log(v).mean())) if len(v) else None
        classical = {m: rec[m] for m in CLASSICAL if m in rec and m != "stat-ens" and rec[m] is not None}
        best = min(classical, key=classical.get)
        rec["best_classical"], rec["best_classical_value"] = best, classical[best]
        tsfm = {m: rec[m] for m in ("tfm3", "tfm25", "chronos2", "tirex2") if rec.get(m) is not None}
        rec["best_tsfm"] = min(tsfm, key=tsfm.get)
        rows.append(rec)
    return rows


def _compute_seconds() -> list[dict]:
    """Wall-clock seconds per model and tier from the run records (GPU: Tesla T4; classical: 8 CPU cores), including model
    loading; records of resumed runs with no task left to run are skipped."""
    out = {}
    for tier in ("B", "W", "synthetic"):
        for f in sorted(glob.glob(str(METRICS / tier / "run_*.json"))):
            rec = json.loads(Path(f).read_text())
            for run in rec.get("runs", []):
                if not run.get("to_run"):  # a resumed run whose tasks were all cached: no work, not a zero cost
                    continue
                key = (tier, run["model_key"])
                o = out.setdefault(key, {"tier": tier, "model": run["model_key"], "tasks": 0, "seconds": 0.0,
                                         "device": "GPU" if rec.get("environment", {}).get("gpu", {}).get("cuda_available") else "CPU"})
                o["tasks"] += run.get("to_run", 0)
                o["seconds"] += run.get("seconds", 0.0)
    return list(out.values())


def descriptive(reps: int = DESCRIPTIVE_REPS) -> dict:
    b = analysis.load(ROOT / "results" / "forecast", "main", "B")
    w = analysis.load(ROOT / "results" / "forecast", "main", "W")
    c = pd.read_parquet(METRICS / "synthetic" / "metrics.parquet")
    out = {"reps": reps}
    # H1 (descriptive): tfm3 with Google's inputs (covariates included, as in Tier A) vs stat-ens, univariate structured tasks.
    uni = (b["mode"] == "uni") & b.domain.isin(analysis.STRUCTURED)
    for metric in ("mase", "sql"):
        rows = analysis.complete_cases(b, ["tfm3", "stat-ens"], [metric], uni)
        out[f"H1_tierB_{metric}"] = analysis.contrast(b, "tfm3", "stat-ens", metric, rows, ["domain", "horizon"], reps=reps)
    # Information-matched contrasts on the near-random-walk domains, for context next to H5.
    nrw = (b["mode"] == "uni") & b.domain.isin(analysis.NEAR_RANDOM_WALK)
    for ref in ("stat-ens", "chronos2-uni"):
        rows = analysis.complete_cases(b, ["tfm3-uni", ref], ["mase"], nrw)
        out[f"nrw_tfm3uni_vs_{ref}"] = analysis.contrast(b, "tfm3-uni", ref, "mase", rows, ["domain", "horizon"], reps=reps)
    # Calibration (RQ4).
    bu = b[(b["mode"] == "uni") & b.domain.isin(analysis.STRUCTURED)]
    out["calibration"] = {
        "B": _calibration(bu, ["tfm3-uni", "tfm3", "tfm3-eval", "tfm25", "chronos2-uni", "tirex2-uni", "stat-ens"], "series_id"),
        "W-post": _calibration(w[w.stratum == "W-post"], ["tfm3", "tfm25", "chronos2", "tirex2", "stat-ens"], "series_id"),
        "C": _calibration(c.assign(unit=c.task_id), ["tfm3", "tfm25", "chronos2", "tirex2", "stat-ens"], "unit"),
    }
    out["tierC_cells"] = _cell_table(c)
    out["compute"] = _compute_seconds()
    DESCRIPTIVE.write_text(json.dumps(out, indent=1, default=float))
    return out


# ------------------------------------------------------------------ tables

P_LABELS = {"P1": ("tfm3-uni", "stat-ens"), "P2": ("tfm3-uni", "chronos2-uni"), "P3": ("tfm3-uni", "tfm25"),
            "P4a": ("tfm3", "tfm3-uni"), "P5": ("tfm3-eval", "tfm3")}


def table_family_p(conf: dict) -> None:
    rows = []
    for r in conf["family_P"]:
        test, metric = r["test"].split(":")
        rows.append([test, f"{name(r['a'])} vs {name(r['b'])}", metric.upper(), r3(r["ratio"]), ci(r["ci95"]),
                     ci(r.get("ci95_oneway")), pval(r["p_holm"]), "resolved" if r["resolved"] else "not resolved"])
    write("table_family_p.tex", tabular("lllrllll", ["Test", "Contrast (A vs B)", "Metric", "Ratio", "95\\% CI (two-way)",
                                                       "95\\% CI (series)", "$p_{\\mathrm{Holm}}$", "Decision"], rows))


H_TEXT = {"H2": "W-seen vs W-post DiD, TimesFM-3 vs Chronos-2", "H3": "no detrending vs default, D7, steps 25--48",
          "H4a": "multivariate vs univariate, lead-lag 24", "H4b": "multivariate vs univariate, known covariate",
          "H4c": "multivariate vs univariate, independent series", "H4d": "multivariate vs univariate, spurious covariate",
          "H5": "TimesFM-3 uni vs Naive, FX and FRED"}
H_FORM = {"negative": "ratio $<$ 1", "equivalence": "within $\\pm$3\\%", "nonsuperiority": "not $<$ 0.90"}


def table_family_h(conf: dict) -> None:
    rows = []
    for r in conf["family_H"]:
        band = ci(r["ci90"], r4) + "$^{90}$" if r["direction"] == "equivalence" else ci(r.get("ci95"), r4)
        rows.append([r["test"], H_TEXT[r["test"]], H_FORM[r["direction"]], r4(r["ratio"]), band,
                     pval(r["p_family_holm"]), "supported" if r["supported"] else "not supported"])
    write("table_family_h.tex", tabular("lllrlll", ["", "Contrast", "Prediction", "Ratio", "95\\% CI",
                                                      "$p_{\\mathrm{Holm}}$", "Decision"], rows))


S6_REF = {"stat-ens": "Stat.\\ ensemble", "chronos2-uni": "Chronos-2 uni", "tfm25": "TimesFM-2.5", "naive": "Naive"}


def table_family_s6(s6: dict | None) -> None:
    """Tier P (protocol section 5.5): TimesFM-3 uni against four references, two domain groups, two metrics."""
    if s6 is None:
        write("table_family_s6.tex", "\\textit{[Pooled S6 decision: after issue 2 is scored on 2026-11-13.]}")
        return
    rows = []
    for group, label in (("structured", "Load, price, river"), ("wiki", "Wikipedia")):
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for r in s6["family_S6"]:
            _, g, ref, metric = r["test"].split(":")
            if g == group:
                rows.append([f"vs {S6_REF[ref]}", metric.upper(), r3(r["ratio"]), ci(r["ci95"]), pval(r["p_holm"]),
                             "resolved" if r["resolved"] else "not resolved"])
    write("table_family_s6.tex", tabular("llrlll", ["TimesFM-3 uni", "Metric", "Ratio", "95\\% CI (series)",
                                                      "$p_{\\mathrm{Holm}}$", "Decision"], rows))


TIER_P = ROOT / "reports" / "tier_p_issues.json"


def tier_p_issues(reps: int = DESCRIPTIVE_REPS) -> dict | None:
    """Tier P issue by issue (descriptive; the confirmatory S6 pools both issues): the S6 contrasts and, per domain,
    TimesFM-3 uni against the statistical ensemble and Chronos-2; series clusters only, as in S6."""
    paths = sorted(METRICS.parent.glob("prospective-*/P/metrics.parquet"))
    if not paths:
        return None
    out = {"label": "Tier P per issue, descriptive (the pre-registered S6 pools both issues)", "reps": reps, "issues": {}}
    for path in paths:
        p = pd.read_parquet(path)
        day = str(p.issue.iloc[0])
        excluded = pd.read_csv(path.parent / "excluded_at_scoring.csv") if (path.parent / "excluded_at_scoring.csv").stat().st_size > 1 else pd.DataFrame()
        rec = {"tasks_scored": int(p.task_id.nunique()), "excluded_at_scoring": int(len(excluded)), "groups": [], "domains": []}
        groups = {"structured": p.domain.isin(["load", "price", "river"]), "wiki": p.domain.eq("wiki")}
        for gname, g in groups.items():
            rows = analysis.complete_cases(p, ["tfm3-uni", "stat-ens", "chronos2-uni", "tfm25", "naive"], ["mase", "sql"],
                                           g & p["mode"].eq("uni"))
            for ref in ("stat-ens", "chronos2-uni", "tfm25", "naive"):
                for metric in ("mase", "sql"):
                    r = analysis.contrast(p, "tfm3-uni", ref, metric, rows, ["domain", "horizon"], "domain", two_way=False, reps=reps)
                    rec["groups"].append({"group": gname, "b": ref, "metric": metric, "ratio": round(r["ratio"], 4),
                                          "ci95": [round(x, 4) for x in r["ci95"]], "n": r["n"]})
        for dom in sorted(p.domain.unique()):
            if dom == "wiki":
                continue
            rows = analysis.complete_cases(p, ["tfm3-uni", "stat-ens", "chronos2-uni"], ["mase", "sql"],
                                           p.domain.eq(dom) & p["mode"].eq("uni"))
            for ref in ("stat-ens", "chronos2-uni"):
                for metric in ("mase", "sql"):
                    r = analysis.contrast(p, "tfm3-uni", ref, metric, rows, ["horizon"], None, two_way=False, reps=reps)
                    rec["domains"].append({"domain": dom, "b": ref, "metric": metric, "ratio": round(r["ratio"], 4),
                                           "ci95": [round(x, 4) for x in r["ci95"]], "n": r["n"]})
        out["issues"][day] = rec
    TIER_P.write_text(json.dumps(out, indent=1))
    return out


def table_tier_p_issues(tp: dict | None) -> None:
    """Descriptive table of Tier P by issue: ratio and series-cluster interval for each S6 contrast."""
    if not tp:
        write("table_tier_p_issues.tex", "\\textit{[Tier P outcomes not yet scored.]}")
        return
    days = sorted(tp["issues"])
    rows = []
    for group, label in (("structured", "Load, price, river"), ("wiki", "Wikipedia")):
        rows.append(f"\\multicolumn{{{2 + 2 * len(days)}}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for ref in ("stat-ens", "chronos2-uni", "tfm25", "naive"):
            cells = []
            for day in days:
                for metric in ("mase", "sql"):
                    r = next(x for x in tp["issues"][day]["groups"] if x["group"] == group and x["b"] == ref and x["metric"] == metric)
                    cells.append(f"{r3(r['ratio'])} {ci(r['ci95'])}")
            n = next(x for x in tp["issues"][days[0]]["groups"] if x["group"] == group and x["b"] == ref)["n"]
            rows.append([f"vs {S6_REF[ref]}", str(n)] + cells)
    head = ["TimesFM-3 uni", "$n$"] + [f"{d[5:]} {m}" for d in days for m in ("MASE", "SQL")]
    write("table_tier_p_issues.tex", tabular("ll" + "l" * 2 * len(days), head, rows))


def table_secondary(conf: dict) -> None:
    rows = ["\\multicolumn{6}{l}{\\textit{S1: family P contrasts by domain (MASE)}} \\\\"]
    for r in conf["family_S1"]:
        test, dom = r["test"].split(":")
        rows.append([f"{test}, {dom}", f"{name(r['a'])} vs {name(r['b'])}", r3(r["ratio"]), ci(r["ci95"]),
                     pval(r["p_holm"]), "yes" if r["resolved"] else "no"])
    rows.append("\\midrule")
    rows.append("\\multicolumn{6}{l}{\\textit{S2: Wikipedia (MASE; DiD: W-seen minus W-post advantage)}} \\\\")
    labels = {"DiD:tfm3-vs-stat-ens": "DiD, reference stat.\\ ensemble", "DiD:tfm3-vs-tirex2": "DiD, reference TiRex-2",
              "DiD:tfm25-vs-chronos2": "DiD for TimesFM-2.5", "post-vs-new": "W-post vs W-new",
              "adjusted:W-seen-vs-W-post": "adjusted DiD", "adjusted:W-post-vs-W-new": "adjusted W-post vs W-new"}
    for r in conf["family_S2"]:
        rows.append([labels[r["test"]], f"{name(r['a'])} vs {name(r['ref'])}", r3(r["ratio"]), ci(r["ci95"]),
                     pval(r["p_holm"]), "yes" if r["resolved"] else "no"])
    rows.append("\\midrule")
    rows.append("\\multicolumn{6}{l}{\\textit{S3: oracle covariates, load and river (upper bound); S4: classical context}} \\\\")
    for fam, label in (("family_S3", None), ("family_S4", None)):
        for r in conf[fam]:
            lab = {"P4b": "P4b", "chronos2-cov": "Chronos-2 covariates", "c2016": "S4, hourly domains"}[r["test"]]
            rows.append([lab, f"{name(r['a'])} vs {name(r['b'])}", r3(r["ratio"]), ci(r["ci95"]), pval(r["p_holm"]),
                         "yes" if r["resolved"] else "no"])
    write("table_secondary.tex", tabular("llrlll", ["Test", "Contrast (A vs B)", "Ratio", "95\\% CI",
                                                      "$p_{\\mathrm{Holm}}$", "Resolved"], rows))


ABLATION_TEXT = {"tfm3-nodetrend": "linear detrending off", "tfm3-nocpmrevin": "CPM and iterative RevIN off",
                 "tfm3-novarattn": "variate attention off", "tfm3-nostitch": "stitching off",
                 "tfm3-sym": "symmetric averaging on", "tfm3-pos": "make\\_positive on",
                 "tfm3-znorm": "z-normalised inputs", "tfm3-edge": "covariate edge handling"}


def table_ablations(conf: dict) -> None:
    by = {}
    for r in conf["family_S5"]:
        tier, key = r["test"].split(":")
        by.setdefault(key, {})[tier] = r
    rows = []
    for key, txt in ABLATION_TEXT.items():
        cells = [f"\\texttt{{{key}}}", txt]
        for tier in ("B", "C"):
            r = by.get(key, {}).get(tier)
            cells += ["--", "--", "--"] if r is None else [r4(r["ratio"]), ci(r["ci95"], r4), pval(r["p_holm"])]
        rows.append(cells)
    head = ["Configuration", "Change", "Ratio B", "95\\% CI", "$p_{\\mathrm{Holm}}$", "Ratio C", "95\\% CI",
            "$p_{\\mathrm{Holm}}$"]
    write("table_ablations.tex", tabular("llrllrll", head, rows))


def table_robustness() -> None:
    """Review-requested sensitivity of family P: other aggregations and small-cluster inference (not pre-registered)."""
    path = ROOT / "reports" / "robustness.json"
    if not path.exists():
        return
    rob = json.loads(path.read_text())
    small = {r["test"]: r for r in rob["R2_small_clusters"]}
    rows = []
    for r in rob["R1_aggregation"]:
        s = small[r["test"]]
        dec = lambda k: "yes" if s[k] else "no"
        rows.append([r["test"].replace(":", " ").upper().replace("P4A", "P4a"), r3(r["nested_equal_weights"]), r3(r["row_pooled"]),
                     r3(r["series_weighted"]), r3(r["ratio_of_means"]), r3(r["median"]),
                     r3(r.get("without_river", float("nan"))), pval(s["jackknife_p_holm"]), dec("jackknife_resolved"),
                     pval(s["block2_p_holm"]), dec("block2_resolved")])
    head = ["Test", "Nested", "Pooled", "Series", "Ratio of means", "Median", "w/o river", "$p$ jack.", "Res.",
            "$p$ 2-wk", "Res."]
    write("table_robustness.tex", tabular("lrrrrrrrlrl", head, rows))


def table_regime() -> None:
    """Review-requested level-shift grid: TimesFM-3 MASE relative to the best classical model and to Chronos-2."""
    path = ROOT / "reports" / "robustness_extra.json"
    if not path.exists():
        return
    grid = {r["cell"]: r for r in json.loads(path.read_text())["R9_regime_grid"]}
    lags = (6, 24, 96, 336)
    rows = []
    for size in (1, 2, 3, 5):
        cells = [grid[f"regime:shift{size}:lag{lag}"] for lag in lags]
        rows.append([f"{size} sd"] + [r3(c["tfm3_vs_best"]) for c in cells] + [r3(c["tfm3_vs_chronos2"]) for c in cells])
    ctrl = grid["regime:shift0"]
    rows.append(["none (control)", f"\\multicolumn{{4}}{{c}}{{{r3(ctrl['tfm3_vs_best'])}}}",
                 f"\\multicolumn{{4}}{{c}}{{{r3(ctrl['tfm3_vs_chronos2'])}}}"])
    head = ["Shift"] + [f"{lag}" for lag in lags] + [f"{lag}" for lag in lags]
    body = tabular("lrrrrrrrr", head, rows)
    body = body.replace("\\toprule\n", "\\toprule\n & \\multicolumn{4}{c}{vs best classical, steps since the shift} & "
                        "\\multicolumn{4}{c}{vs Chronos-2, steps since the shift} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n", 1)
    write("table_regime.tex", body)


def table_latency() -> None:
    """Protocol section 7: seconds per call and series per second on the T4, by model and call size."""
    lat = pd.read_csv(ROOT / "reports" / "latency.csv")
    rows = []
    for key, g in lat.groupby("model", sort=False):
        cells = [name(key)]
        for size in (1, 32, 256):
            r = g[g.batch_size.eq(size)].iloc[0]
            cells += ["\\multicolumn{2}{c}{out of memory}"] if isinstance(r.get("error"), str) else \
                [f"{r.seconds_per_call:.2f}", f"{r.series_per_second:.1f}"]
        cells.append(f"{g.peak_gpu_gb.max():.1f}")
        rows.append(cells)
    head = ["Model", "s/call (1)", "series/s (1)", "s/call (32)", "series/s (32)", "s/call (256)", "series/s (256)",
            "Peak GB"]
    write("table_latency.tex", tabular("lrrrrrrr", head, rows))


def table_tier_a() -> None:
    s = json.loads((ROOT / "reports" / "tierA_summary.json").read_text())
    rows = []
    for model in ("tfm3-eval", "tfm3"):
        for subset, label in (("all_tasks", "all tasks"), ("controls_passed", "tasks passing both controls")):
            x = s[model][subset]
            rows.append([name(model), label, str(x["tasks"]), f"{100 * x['reproduced_share']:.0f}\\%",
                         f"{abs(100 * x['median_SQL_dev']):.2f}\\%" if abs(x['median_SQL_dev']) < 5e-5 else f"{100 * x['median_SQL_dev']:+.2f}\\%",
                         f"[{100 * x['range_SQL_dev'][0]:+.1f}\\%, {100 * x['range_SQL_dev'][1]:+.1f}\\%]",
                         r3(x["skill_ours"]), r3(x["skill_published"])])
    write("table_tier_a.tex", tabular("llrrrlrr", ["Model", "Tasks", "$n$", "$\\pm$3\\%", "Median",
                                                     "Range", "Ratio ours", "Ratio publ."], rows))
    # Appendix: per-task reproduction of tfm3-eval.
    t = pd.read_csv(ROOT / "reports" / "tierA_reproduction.csv")
    t = t[t.model == "tfm3-eval"].sort_values("task_name")
    rows = [[r.task_name.replace("_", "\\_"), r3(r.SQL), r3(r.SQL_google), f"{100 * r.SQL_reldiff:+.2f}\\%",
             f"{100 * r.MASE_reldiff:+.2f}\\%", "yes" if r.control_ok else "no", r3(r.SQL_vs_stat_ensemble)]
            for r in t.itertuples()]
    write("table_tier_a_tasks.tex", tabular("lrrrrlr", ["Task", "SQL ours", "SQL Google", "$\\Delta$SQL", "$\\Delta$MASE",
                                                          "Controls", "SQL / stat.\\ ens."], rows))


def table_calibration(desc: dict) -> None:
    rows = []
    for tier, label in (("B", "Tier B (structured, univariate tasks)"), ("W-post", "Tier W, W-post"),
                        ("C", "Tier C (all cells)")):
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for r in desc["calibration"][tier]:
            rows.append([name(r["model"]), str(r["series"]), f"{r['coverage80']:.3f}", f"{r['coverage60']:.3f}"])
    # Tier P (both issues, univariate tasks of load, price, river and Wikipedia): the scoring records the 80% interval only
    paths = sorted((ROOT / "results" / "forecast").glob("prospective-*/P/metrics.parquet"))
    if paths:
        p = pd.concat([pd.read_parquet(f) for f in paths], ignore_index=True)
        p = p[p["mode"].eq("uni")]
        for doms, label in ((["load", "price", "river"], "Tier P, both issues: load, price, river"),
                            (["wiki"], "Tier P, both issues: Wikipedia")):
            rows.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{label}}}}} \\\\")
            for m in ["tfm3-uni", "tfm25", "chronos2-uni", "tirex2-uni", "stat-ens"]:
                sub = p[(p.model == m) & p.domain.isin(doms)]
                if not sub.empty:
                    cov = sub.groupby("series_id").coverage80.mean().dropna()
                    rows.append([name(m), str(int(cov.size)), f"{cov.mean():.3f}", "---"])
    write("table_calibration.tex", tabular("lrrr", ["Model", "Series", "Coverage 80\\%", "Coverage 60\\%"], rows))


CELL_TEXT = {"rep:D6": "D6 level shift", "rep:D7": "D7 logistic", "rep:D8": "D8 intermittent"}


def _cell_label(cell: str) -> str:
    return cell.replace("_", "\\_").replace(":", " ")


def table_tier_c(desc: dict) -> None:
    rows = []
    family = None
    for r in desc["tierC_cells"]:
        if r["family"] != family:
            if family is not None:
                rows.append("\\midrule")
            family = r["family"]
            rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{family}}}}} \\\\")
        ratio = r["tfm3"] / r["best_classical_value"] if r.get("tfm3") else None
        rows.append([_cell_label(r["cell"]), str(r["horizon"]), r3(r.get("tfm3")), r3(r.get("chronos2")),
                     r3(r.get("tirex2")), r3(r.get("tfm25")),
                     f"{r3(r['best_classical_value'])} ({name(r['best_classical'])})",
                     ("\\textbf{" + r3(ratio) + "}") if ratio and ratio > 1 else r3(ratio)])
    write("table_tier_c.tex", tabular("lrrrrrlr", ["Cell", "$H$", "TimesFM-3", "Chronos-2", "TiRex-2", "TimesFM-2.5",
                                                     "Best classical", "TFM-3 / best"], rows))


# ------------------------------------------------------------------ numbers

def _wins(path: Path, model: str) -> tuple[float, int]:
    """Share of Tier A tasks where SQL is below fev's statistical ensemble, over tasks where fev reports the ensemble."""
    t = pd.read_csv(path)
    s = t.loc[t.model.eq(model), "SQL_vs_stat_ensemble"].dropna()
    return float((s < 1).mean()), int(len(s))


def _macro(name_: str, value: str) -> str:
    return f"\\newcommand{{\\{name_}}}{{{value}}}"


def numbers(conf: dict, desc: dict) -> None:
    m = []
    get = {r["test"]: r for fam in conf.values() if isinstance(fam, list) for r in fam}
    roman = {"P1": "Pone", "P2": "Ptwo", "P3": "Pthree", "P4a": "Pfoura", "P5": "Pfive"}
    for test, key in roman.items():
        for metric in ("mase", "sql"):
            r = get[f"{test}:{metric}"]
            tag = key + metric.capitalize()
            m += [_macro(tag, r3(r["ratio"])), _macro(tag + "Lo", r3(r["ci95"][0])), _macro(tag + "Hi", r3(r["ci95"][1]))]
    for test in ("H2", "H3", "H4a", "H4b", "H4c", "H4d", "H5"):
        r = get[test]
        tag = "H" + {"H2": "two", "H3": "three", "H4a": "foura", "H4b": "fourb", "H4c": "fourc", "H4d": "fourd",
                     "H5": "five"}[test]
        m += [_macro(tag, r3(r["ratio"])), _macro(tag + "Lo", r3(r["ci95"][0])), _macro(tag + "Hi", r3(r["ci95"][1]))]
    for k, tag in (("H1_tierB_mase", "HoneBMase"), ("H1_tierB_sql", "HoneBSql"),
                   ("nrw_tfm3uni_vs_stat-ens", "NrwStatEns"), ("nrw_tfm3uni_vs_chronos2-uni", "NrwChronos")):
        r = desc[k]
        m += [_macro(tag, r3(r["ratio"])), _macro(tag + "Lo", r3(r["ci95"][0])), _macro(tag + "Hi", r3(r["ci95"][1]))]
    a = json.loads((ROOT / "reports" / "tierA_summary.json").read_text())["tfm3-eval"]
    m += [_macro("AReproShare", f"{100 * a['all_tasks']['reproduced_share']:.0f}"),
          _macro("ATasks", str(a["all_tasks"]["tasks"])),
          _macro("AControlTasks", str(a["controls_passed"]["tasks"])),
          _macro("ASignP", f"{a['all_tasks']['sign_test_p']:.2f}"),
          _macro("ASignPCtl", f"{a['controls_passed']['sign_test_p']:.2f}"),
          _macro("ASkillOurs", r3(a["all_tasks"]["skill_ours"])), _macro("ASkillPubl", r3(a["all_tasks"]["skill_published"])),
          _macro("ASkillOursCtl", r3(a["controls_passed"]["skill_ours"])),
          _macro("ASkillPublCtl", r3(a["controls_passed"]["skill_published"])),
          _macro("AStatEnsSql", r3(a["vs_stat_ensemble_geo"]["SQL"])), _macro("AStatEnsMase", r3(a["vs_stat_ensemble_geo"]["MASE"])),
          _macro("AStatEnsWins", f"{100 * _wins(ROOT / 'reports' / 'tierA_reproduction.csv', 'tfm3-eval')[0]:.0f}")]
    # tasks failing the controls (mostly cross-learning on fev's Chronos-2 side): geometric means over those tasks
    rep = pd.read_csv(ROOT / "reports" / "tierA_reproduction.csv")
    rep = rep[rep.model.eq("tfm3-eval")]
    fail = rep[~rep.control_ok.astype(bool)][["skill_ours", "skill_published"]].dropna()
    m += [_macro("AFailTasks", str(len(fail))), _macro("ASkillOursFail", r3(float(np.exp(np.log(fail.skill_ours).mean())))),
          _macro("ASkillPublFail", r3(float(np.exp(np.log(fail.skill_published).mean()))))]
    # H1 (protocol section 3.1) uses tfm3 (package defaults, Google's inputs) against fev's published stat_ensemble.
    a3 = json.loads((ROOT / "reports" / "tierA_summary.json").read_text())["tfm3"]
    m += [_macro("HoneASql", r3(a3["vs_stat_ensemble_geo"]["SQL"])), _macro("HoneAMase", r3(a3["vs_stat_ensemble_geo"]["MASE"])),
          _macro("HoneAWins", f"{100 * _wins(ROOT / 'reports' / 'tierA_tfm3.csv', 'tfm3')[0]:.0f}"),
          _macro("HoneATasks", str(_wins(ROOT / "reports" / "tierA_tfm3.csv", "tfm3")[1]))]
    h2 = get["H2"]  # minimum detectable effect at 80% power from the realised standard error (two-sided 5%)
    se = (math.log(h2["ci95"][1]) - math.log(h2["ci95"][0])) / (2 * 1.96)
    m.append(_macro("HtwoMde", f"{100 * (math.exp((1.96 + 0.8416) * se) - 1):.0f}"))
    cal = {r["model"]: r for r in desc["calibration"]["B"]}
    m += [_macro("CovBtfm", f"{cal['tfm3-uni']['coverage80']:.3f}"), _macro("CovBstat", f"{cal['stat-ens']['coverage80']:.3f}")]
    write("numbers.tex", "% generated by tfmstudy.paper; do not edit\n" + "\n".join(m) + "\n")


# ------------------------------------------------------------------ figure data (pgfplots)

def forest(conf: dict) -> None:
    """Coordinates for the forest plot of the primary and hypothesis contrasts (log scale in pgfplots)."""
    items = []
    for r in conf["family_P"]:
        test, metric = r["test"].split(":")
        if metric == "mase":
            items.append((f"{test}: {name(r['a'])} vs {name(r['b'])}", r))
    for r in conf["family_H"]:
        if r["test"] != "H3":
            items.append((f"{r['test']}: {H_TEXT[r['test']]}", r))
    lines = []
    for i, (label, r) in enumerate(reversed(items)):
        lo, hi = r["ci95"]
        fam = 0 if r["test"].startswith("P") else 1  # filled marker: resolved (P) or supported (H) after Holm
        ok = r["resolved"] if fam == 0 else r["supported"]
        lines.append(f"{r['ratio']:.4f} {i} {r['ratio'] - lo:.4f} {hi - r['ratio']:.4f} {fam} {int(bool(ok))}")
    ticks = ",".join(str(i) for i in range(len(items)))
    labels = ",".join("{" + label + "}" for label, _ in reversed(items))
    write("forest_data.tex", "\\pgfplotstableread{\nx y em ep fam res\n" + "\n".join(lines) + "\n}\\forestdata\n"
          f"\\pgfplotsset{{forest ticks/.style={{ytick={{{ticks}}}, yticklabels={{{labels}}}, "
          f"ymin=-0.7, ymax={len(items) - 0.3}}}}}\n")


DOMAIN_ORDER = ("load", "price", "weather", "river", "fx", "rates")
DOMAIN_NAMES = {"load": "Electricity load", "price": "Electricity price", "weather": "Weather", "river": "River flow",
                "fx": "Exchange rates", "rates": "Yields and oil"}


def fig_domains() -> None:
    """Overview figure: TimesFM-3 against three references in each Tier B domain (R17, descriptive)."""
    path = ROOT / "reports" / "robustness_extra.json"
    rows = json.loads(path.read_text()).get("R17_by_domain") if path.exists() else None
    if not rows:
        return
    out = []
    for ref, key in (("stat-ens", "ens"), ("chronos2-uni", "chr"), ("naive", "rw")):
        lines = []
        for i, dom in enumerate(reversed(DOMAIN_ORDER)):
            for metric, off in (("mase", 0.14), ("sql", -0.14)):
                r = next(x for x in rows if x["domain"] == dom and x["b"] == ref and x["metric"] == metric)
                lo, hi = r["ci95"]
                lines.append(f"{metric} {r['ratio']:.4f} {i + off:.2f} {r['ratio'] - lo:.4f} {hi - r['ratio']:.4f}")
        mase = "\n".join(l.split(" ", 1)[1] for l in lines if l.startswith("mase"))
        sql = "\n".join(l.split(" ", 1)[1] for l in lines if l.startswith("sql"))
        out.append(f"\\pgfplotstableread{{\nx y em ep\n{mase}\n}}\\dom{key}mase\n"
                   f"\\pgfplotstableread{{\nx y em ep\n{sql}\n}}\\dom{key}sql\n")
    labels = ",".join("{" + DOMAIN_NAMES[d] + "}" for d in reversed(DOMAIN_ORDER))
    out.append(f"\\pgfplotsset{{domain ticks/.style={{ytick={{{','.join(str(i) for i in range(len(DOMAIN_ORDER)))}}}, "
               f"yticklabels={{{labels}}}, ymin=-0.6, ymax={len(DOMAIN_ORDER) - 0.4}}}}}\n")
    write("fig_domains_data.tex", "".join(out))


def fig_regime() -> None:
    """Heat maps of the level-shift grid (R9): log of TimesFM-3 MASE over the best classical model and over Chronos-2."""
    path = ROOT / "reports" / "robustness_extra.json"
    if not path.exists():
        return
    grid = {r["cell"]: r for r in json.loads(path.read_text())["R9_regime_grid"]}
    out = []
    for key, name_ in (("tfm3_vs_best", "regimebest"), ("tfm3_vs_chronos2", "regimechr")):
        lines, labels = [], []
        for j, size in enumerate((1, 2, 3, 5)):
            for i, lag in enumerate((6, 24, 96, 336)):
                r = grid[f"regime:shift{size}:lag{lag}"][key]
                lines.append(f"{i} {j} {math.log(r):.4f}")
                ink = "white" if abs(math.log(r)) > 0.55 else "vzink"  # white on the saturated cells
                labels.append(f"\\node[font=\\footnotesize, text={ink}] at (axis cs:{i},{j}) {{{r:.2f}}};")
            lines.append("")
        out.append(f"\\pgfplotstableread{{\nx y c\n" + "\n".join(lines).strip() + f"\n}}\\{name_}\n"
                   + _macro(f"{name_}labels", "".join(labels)) + "\n")
    ctrl = grid["regime:shift0"]
    out.append(_macro("RegimeControlBest", f"{ctrl['tfm3_vs_best']:.2f}") + "\n"
               + _macro("RegimeControlChr", f"{ctrl['tfm3_vs_chronos2']:.2f}") + "\n")
    write("fig_regime_data.tex", "".join(out))

VERSION = "1.0.0"
DOI = None  # assigned by Zenodo at publication


def meta() -> None:
    doi = f"doi:{DOI}" if DOI else "DOI to be assigned"
    write("meta.tex", "\n".join([
        _macro("PaperAuthors", "Alessio Pomaro\\\\[2pt] {\\normalsize Independent Researcher}\\\\ "
               "{\\normalsize\\texttt{info@alessiopomaro.it}}\\\\ {\\normalsize ORCID: "
               "\\href{https://orcid.org/0009-0008-6975-996X}{0009-0008-6975-996X}}"),
        _macro("PaperVersionLine", f"Preprint, version {VERSION} --- {doi} --- \\today"),
        _macro("PaperStatus", "Preprint. Not peer reviewed." + ("" if S6.exists() else
              " Draft: the second prospective issue and the pooled Tier P test (S6) are not yet scored.")),
    ]) + "\n")
    write("availability.tex",
          "Code, the frozen protocol, data manifests, the post-release data where licences allow (otherwise the download code "
          "with request timestamps and hashes), per-series and per-window quantile forecasts of every configuration, metrics and "
          "analysis outputs are released under the licences listed in the repository (MIT for code; CC BY 4.0 for the paper, "
          "documentation and the forecasts of the other models). Forecasts produced by TimesFM~3 are Outputs under the TimesFM "
          "Non-Commercial License v1.0, which does not permit their use for commercial or production purposes; they are released "
          "under CC BY-NC 4.0, and their use remains subject to Google's licence. Repository: \\url{https://github.com/alepom/timesfm-zeroshot-study} "
          f"(tag \\texttt{{v{VERSION}}}). The protocol was frozen by the git tag \\texttt{{protocol-frozen-v1}} "
          "(commit \\texttt{9f15b38}) before any post-release data were downloaded; the two prospective issues were sealed "
          "by SHA-256 hashes before their outcomes existed (Section~\\ref{sec:prospective}). TimesFM~3 weights are not "
          "redistributed (non-commercial licence); every run records the Hugging Face revision of each model.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="recompute reports/descriptive.json")
    args = parser.parse_args()
    conf = json.loads(CONFIRMATORY.read_text())
    desc = descriptive() if args.refresh or not DESCRIPTIVE.exists() else json.loads(DESCRIPTIVE.read_text())
    table_family_p(conf)
    table_family_h(conf)
    table_secondary(conf)
    table_ablations(conf)
    table_tier_a()
    table_latency()
    table_robustness()
    table_regime()
    table_calibration(desc)
    table_tier_c(desc)
    table_family_s6(json.loads(S6.read_text()) if S6.exists() else None)
    table_tier_p_issues(tier_p_issues())
    numbers(conf, desc)
    forest(conf)
    fig_domains()
    fig_regime()
    meta()
    print("written:", ", ".join(sorted(p.name for p in GEN.glob("*.tex"))))


if __name__ == "__main__":
    main()
