"""Tier P: prospective forecasts issued before the outcomes exist (protocol section 3.4).

    python -m tfmstudy.data.prospective issue --date 2026-10-30   # morning of the issue day: data, tasks
    (Colab)  forecast --tier P --period prospective-<date> --models ...
    python -m tfmstudy.data.prospective seal --date 2026-10-30    # hash contexts, tasks, code, forecasts
    python -m tfmstudy.data.prospective outcomes --date 2026-10-30   # once the scoring window is published
    python -m tfmstudy.data.prospective score --date 2026-10-30

Each series is forecast from its last observation available on the issue day S (publication lags differ by
source). Only target timestamps in the scoring window [S + 2 days, S + 9 days) are scored: on S and S + 1
some sources already publish values for those days (day-ahead prices for S + 1 at about 11:45 UTC on S;
load in near real time), so the forecasts are sealed before S + 1 10:00 UTC and nothing scored is public
at sealing time. No covariates (their future values are unknown). Weather is not part of Tier P: ERA5
arrives about six days late, so its contexts would end a week before the issue day.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalog, sources, wiki
from .build import MANIFESTS, PROCESSED, RAW, ROOT

ISSUES = ("2026-10-30", "2026-11-02")
HISTORY_START = "2025-01-01"
# Horizons cover the scoring window even from the earliest plausible origin (S - 1, or S - 3 for FRED weekends).
HORIZONS = {"h": 240, "D": 10, "B": 10}
DOMAINS = ("load", "price", "river", "fx", "rates")
SCORE_FROM_DAYS, SCORE_TO_DAYS = 2, 9
SEAL_DEADLINE = dt.timedelta(days=1, hours=10)
MIN_OBSERVED = 0.8  # share of scored steps with an outcome, else the task is excluded from scoring


def period_name(issue_date: str) -> str:
    return f"prospective-{issue_date}"


def _download(start: str, end: str, raw_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFESTS / "tierB_series.csv")
    manifest = manifest[manifest.domain.isin(DOMAINS)]
    out = []

    def add(sid, s):
        out.append(pd.DataFrame({"series_id": sid, "timestamp": s.index, "value": s.to_numpy(dtype=float)}))

    for row in manifest.itertuples():
        if row.domain == "load":
            add(row.series_id, sources.energy_load(row.key, start, end, raw_dir))
        elif row.domain == "price":
            add(row.series_id, sources.energy_price(row.key, start, end, raw_dir))
        elif row.domain == "river":
            add(row.series_id, sources.usgs_daily(row.key, start, end, raw_dir)[0])
        elif row.domain == "fx":
            add(row.series_id, sources.ecb_fx(row.key, start, end, raw_dir))
        elif row.domain == "rates":
            add(row.series_id, sources.fred(row.key, start, end, raw_dir))
    for row in _pages().itertuples():
        add(row.series_id, sources.wiki_pageviews(row.title, start, end, raw_dir))
    series = pd.concat(out, ignore_index=True)
    return series.drop_duplicates(["series_id", "timestamp"]).sort_values(["series_id", "timestamp"])


def _pages() -> pd.DataFrame:
    """W-post and W-new pages, minus those excluded by the latest redirect check."""
    pages = pd.read_csv(MANIFESTS / "tierW_pages.csv")
    moved, _ = wiki._moved_titles()
    return pages[~pages.title.isin(moved)]


def _last_valid(s: pd.Series) -> pd.Timestamp:
    return s.dropna().index.max()


def _context_exclusion(by_id: dict, targets: list[str], origin: pd.Timestamp, season: int, days: int | None) -> str:
    """Protocol section 6 rules that can be checked at issue time (the targets do not exist yet)."""
    from ..baselines import CLASSICAL_CONTEXT
    from .build import CONTEXT_CAP

    index = by_id[targets[0]].index
    for sid in targets:
        s = by_id[sid].reindex(index)
        s = s[s.index <= origin]
        context = s[s.index > origin - pd.Timedelta(days=days)] if days else s.iloc[-CONTEXT_CAP:]
        recent = context.iloc[-CLASSICAL_CONTEXT.get(season, 2016):]
        if context.isna().mean() > 0.05 or recent.isna().mean() > 0.05:
            return f"{sid}: more than 5% missing in context"
        if np.nanstd(context.to_numpy()) == 0:
            return f"{sid}: constant context"
        if len(context.dropna()) < 2 * season:
            return f"{sid}: context shorter than two seasons"
    return ""


def issue(issue_date: str) -> pd.DataFrame:
    """Downloads data published up to the issue day and builds the prospective tasks."""
    day = pd.Timestamp(issue_date, tz="UTC")
    raw_dir = RAW / f"prospective_issue_{issue_date}"
    series = _download(HISTORY_START, issue_date, raw_dir)
    series = series[series.timestamp < day]  # nothing stamped on the issue day itself
    by_id = {sid: g.set_index("timestamp")["value"] for sid, g in series.groupby("series_id")}
    manifest = pd.read_csv(MANIFESTS / "tierB_series.csv")
    manifest = manifest[manifest.domain.isin(DOMAINS)]
    rows = []

    def add(domain, mode, targets, stratum=""):
        meta = catalog.DOMAINS[domain]
        origin = min(_last_valid(by_id[t]) for t in targets)
        rows.append({"domain": domain, "mode": mode, "targets": json.dumps(targets), "future_covariates": "[]",
                     "origin": origin, "horizon": HORIZONS[meta["freq"]], "season": meta["season"],
                     "lineage": meta["lineage"], "stratum": stratum,
                     "excluded": _context_exclusion(by_id, targets, origin, meta["season"],
                                                    wiki.HISTORY_DAYS if domain == "wiki" else None)})

    for row in manifest.itertuples():
        add(row.domain, "uni", [row.series_id])
    for domain in ("price", "fx", "rates"):
        add(domain, "group", manifest.loc[manifest.domain == domain, "series_id"].tolist())
    for row in _pages().itertuples():
        add("wiki", "uni", [row.series_id], "W-new" if row.stratum == "W-new" else "W-post")

    tasks = pd.DataFrame(rows)
    # Same context rules as Tiers B and W: up to 15,360 points; Wikipedia 400 days.
    tasks["context_start"] = [o - pd.Timedelta(days=wiki.HISTORY_DAYS - 1) if d == "wiki" else pd.NaT
                              for o, d in zip(tasks.origin, tasks.domain)]
    tasks["score_from"] = day + pd.Timedelta(days=SCORE_FROM_DAYS)
    tasks["score_to"] = day + pd.Timedelta(days=SCORE_TO_DAYS)
    tasks.insert(0, "task_id", [f"P{issue_date.replace('-', '')}-{i:05d}" for i in range(len(tasks))])
    dest = PROCESSED / period_name(issue_date)
    dest.mkdir(parents=True, exist_ok=True)
    series.to_parquet(dest / "series.parquet", index=False)
    tasks.to_parquet(dest / "tasks.parquet", index=False)
    (dest / "ISSUE.json").write_text(json.dumps({
        "issued_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "issue_day": issue_date,
        "seal_deadline_utc": (day + SEAL_DEADLINE).isoformat(), "tasks": len(tasks),
        "scoring_window": [str(day + pd.Timedelta(days=SCORE_FROM_DAYS)), str(day + pd.Timedelta(days=SCORE_TO_DAYS))],
        "latest_origin_by_domain": tasks.groupby("domain").origin.max().astype(str).to_dict()}, indent=1))
    return tasks


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(issue_date: str, results: Path, bundle: Path) -> dict:
    """sha256 of the contexts, tasks, code bundle, run records and every forecast file, and of their list."""
    period = period_name(issue_date)
    deadline = pd.Timestamp(issue_date, tz="UTC") + SEAL_DEADLINE
    now = pd.Timestamp.now(tz="UTC")
    if now > deadline:
        raise SystemExit(f"too late to seal: deadline {deadline}")
    run_dir = results / period / "P"
    data_dir = PROCESSED / period
    files = {f"data/{p.name}": p for p in sorted(data_dir.glob("*")) if p.suffix in (".parquet", ".json")}
    files.update({f"forecasts/{p.relative_to(run_dir)}": p for p in sorted(run_dir.rglob("*"))
                  if p.is_file() and p.name != "SEAL.json"})
    files[f"code/{bundle.name}"] = bundle
    digests = {k: _sha(p) for k, p in sorted(files.items())}
    combined = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in digests.items()).encode()).hexdigest()
    record = {"sealed_utc": now.isoformat(), "seal_deadline_utc": deadline.isoformat(), "issue_day": issue_date,
              "files": digests, "combined_sha256": combined}
    (run_dir / "SEAL.json").write_text(json.dumps(record, indent=1))
    return record


def outcomes(issue_date: str) -> pd.DataFrame:
    """Realised values of the scoring window, downloaded with a fresh cache (later revisions are seen)."""
    day = pd.Timestamp(issue_date, tz="UTC")
    end = (day + pd.Timedelta(days=SCORE_TO_DAYS + 1)).strftime("%Y-%m-%d")
    start = (day - pd.Timedelta(days=21)).strftime("%Y-%m-%d")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M")
    series = _download(start, end, RAW / f"prospective_outcomes_{issue_date}_{stamp}")
    series.to_parquet(PROCESSED / period_name(issue_date) / "outcomes.parquet", index=False)
    return series


def truth_for(task, context: pd.Series | None, outcome: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Outcome aligned to the forecast steps, and the mask of steps inside the scoring window.

    Step k is the k-th observation after the origin in the source's own calendar (as in the contexts:
    business days for ECB, FRED's calendar for FRED, hourly/daily otherwise).
    """
    after = outcome[outcome.index > task.origin]
    freq = {"h": "h", "D": "D"}.get(catalog.DOMAINS[task.domain]["freq"])
    if freq:  # regular calendars: steps are fixed timestamps, missing outcomes stay missing
        stamps = pd.date_range(task.origin, periods=task.horizon + 1, freq=freq)[1:]
        values = after.reindex(stamps).to_numpy(dtype=float)
    else:
        stamps = after.index[:task.horizon]
        values = np.full(task.horizon, np.nan)
        values[:len(stamps)] = after.to_numpy(dtype=float)[:task.horizon]
        stamps = stamps.append(pd.DatetimeIndex([pd.NaT] * (task.horizon - len(stamps)), tz="UTC"))
    in_window = np.asarray((stamps >= task.score_from) & (stamps < task.score_to))
    return values, in_window


def score(issue_date: str, results: Path) -> pd.DataFrame:
    """Per-(task, model, series) metrics on the scoring window, like `evaluate.score` (protocol section 3.4).

    Only steps inside the window with a published outcome are scored; a task with fewer than MIN_OBSERVED
    of its window steps observed is excluded and listed. MASE/SQL scales come from the context as in Tier B.
    """
    from .. import metrics
    from ..evaluate import read_forecasts, stat_ensemble
    from ..forecast import load_tasks

    period = period_name(issue_date)
    tasks, make = load_tasks("P", period, PROCESSED)
    outcome = pd.read_parquet(PROCESSED / period / "outcomes.parquet")
    by_id = {sid: g.set_index("timestamp")["value"] for sid, g in outcome.groupby("series_id")}
    run_dir = results / period / "P"
    models = sorted(p.name for p in run_dir.iterdir() if p.is_dir() and any(p.glob("part-*.parquet")))
    forecasts = {m: read_forecasts(run_dir / m) for m in models}
    forecasts["stat-ens"] = stat_ensemble(tasks, forecasts)
    rows, dropped = [], []
    for row in tasks.itertuples():
        task, _, season = make(row)
        targets = json.loads(row.targets)
        aligned = [truth_for(row, None, by_id.get(t, pd.Series(dtype=float))) for t in targets]
        truth = np.stack([np.where(win, val, np.nan) for val, win in aligned])
        window = aligned[0][1]
        observed = np.isfinite(truth[:, window]).mean() if window.any() else 0.0
        if observed < MIN_OBSERVED:
            dropped.append({"task_id": row.task_id, "observed_share": float(observed)})
            continue
        scale = metrics.seasonal_naive_scale(task.target, season)
        for m, fc in forecasts.items():
            q = fc.get(row.task_id)
            if q is None:
                continue
            err = np.abs(truth - q[..., 4])
            loss = metrics.quantile_loss(truth, q) * 2 / 9
            for i, sid in enumerate(targets):
                ok = np.isfinite(truth[i])
                rows.append({"task_id": row.task_id, "model": m, "series_id": sid, "domain": row.domain,
                             "mode": row.mode, "stratum": row.stratum, "issue": issue_date, "horizon": row.horizon,
                             "n_scored": int(ok.sum()), "origin": row.origin,
                             "mase": float(np.nanmean(err[i]) / scale[i]), "sql": float(np.nanmean(loss[i]) / scale[i]),
                             "coverage80": float(np.mean((truth[i, ok] >= q[i, ok, 0]) & (truth[i, ok] <= q[i, ok, 8])))})
    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "metrics.parquet", index=False)
    pd.DataFrame(dropped).to_csv(run_dir / "excluded_at_scoring.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["issue", "seal", "outcomes", "score"])
    parser.add_argument("--date", required=True, choices=ISSUES, help="issue day S")
    parser.add_argument("--results", default=str(ROOT / "results" / "forecast"))
    parser.add_argument("--bundle", help="seal: the code bundle used for the forecasts")
    args = parser.parse_args()
    if args.step == "issue":
        t = issue(args.date)
        print(t.groupby(["domain", "mode"]).agg(n=("task_id", "size"), last_origin=("origin", "max")))
    elif args.step == "seal":
        print(json.dumps(seal(args.date, Path(args.results), Path(args.bundle)), indent=1)[:3000])
    elif args.step == "outcomes":
        print(outcomes(args.date).groupby("series_id").size().describe())
    else:
        df = score(args.date, Path(args.results))
        print(df.groupby(["domain", "model"]).mase.median().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
