"""Builds the Tier B / P datasets and their forecasting tasks.

    python -m tfmstudy.data.build select              # pre-release selection rules -> data/manifests/tierB_series.csv
    python -m tfmstudy.data.build series --period pilot|main
    python -m tfmstudy.data.build tasks  --period pilot|main

Outputs (per period) in data/processed/<period>/:
    series.parquet   long table: series_id, timestamp, value
    tasks.parquet    one row per forecasting task (see `make_tasks`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalog, sources

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "data" / "manifests"
PROCESSED = ROOT / "data" / "processed"

CONTEXT_CAP = 15360  # protocol section 2


# ------------------------------------------------------------------ selection (pre-release data only)

def select() -> pd.DataFrame:
    year = catalog.SELECTION_YEAR
    start, end = f"{year}-01-01", f"{year}-12-31"
    rows = []

    loads = {}
    for code in catalog.COUNTRIES:
        try:
            s = sources.energy_load(code, start, end, RAW)
        except sources.SourceError as exc:
            print("load", code, exc)
            continue
        loads[code] = {"missing": float(s.isna().mean()), "total": float(s.sum())}
    complete = [c for c, v in loads.items() if v["missing"] <= 0.01]
    chosen = sorted(complete, key=lambda c: -loads[c]["total"])[: catalog.N_COUNTRIES]
    for code in chosen:
        bzn, city, lat, lon = catalog.COUNTRIES[code]
        rows.append({"series_id": f"load:{code}", "domain": "load", "source": "energy-charts", "key": code,
                     "covariate": f"temp:{city.lower()}", "lat": lat, "lon": lon,
                     "rule": f"2025 missing {loads[code]['missing']:.4f}, total load rank {chosen.index(code) + 1}"})
        try:
            p = sources.energy_price(bzn, start, end, RAW)
            if p.isna().mean() <= 0.01:
                rows.append({"series_id": f"price:{bzn}", "domain": "price", "source": "energy-charts", "key": bzn,
                             "rule": f"zone of selected country {code}; 2025 missing {p.isna().mean():.4f}"})
        except sources.SourceError as exc:
            print("price", bzn, exc)

    for city, (lat, lon) in catalog.WEATHER_CITIES.items():
        for var in catalog.WEATHER_VARIABLES:
            rows.append({"series_id": f"weather:{city}:{var}", "domain": "weather", "source": "open-meteo",
                         "key": var, "lat": lat, "lon": lon, "rule": "fixed list"})

    kept = 0
    for site in catalog.USGS_SITES:
        if kept >= catalog.N_USGS:
            break
        try:
            s, info = sources.usgs_daily(site, start, end, RAW)
        except sources.SourceError as exc:
            print("usgs", site, exc)
            continue
        if s.isna().sum() == 0:
            rows.append({"series_id": f"river:{site}", "domain": "river", "source": "usgs", "key": site,
                         "name": info["name"], "lat": info["lat"], "lon": info["lon"],
                         "covariate": f"precip:{site}", "rule": "complete 2025 daily record, list order"})
            kept += 1

    for cur in catalog.ECB_CURRENCIES:
        rows.append({"series_id": f"fx:{cur}", "domain": "fx", "source": "ecb", "key": cur, "rule": "fixed list"})
    for sid in catalog.FRED_SERIES:
        rows.append({"series_id": f"rates:{sid}", "domain": "rates", "source": "fred", "key": sid, "rule": "fixed list"})

    df = pd.DataFrame(rows)
    df["lineage"] = df["domain"].map(lambda d: catalog.DOMAINS[d]["lineage"])
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFESTS / "tierB_series.csv", index=False)
    (MANIFESTS / "tierB_selection_stats.json").write_text(json.dumps({"loads_2025": loads}, indent=1))
    return df


# ------------------------------------------------------------------ download

def download_series(period: str) -> pd.DataFrame:
    cfg = catalog.PERIODS[period]
    start, end = cfg["history_start"], cfg["test_end"]
    not_before = cfg.get("download_not_before")
    if not_before and pd.Timestamp.now(tz="UTC") < pd.Timestamp(not_before, tz="UTC"):
        raise SystemExit(f"{period} data may not be downloaded before {not_before} (protocol section 3.2)")
    weather = cfg.get("weather_model", "era5")
    manifest = pd.read_csv(MANIFESTS / "tierB_series.csv")
    out = []

    def add(series_id: str, s: pd.Series) -> None:
        out.append(pd.DataFrame({"series_id": series_id, "timestamp": s.index, "value": s.to_numpy(dtype=float)}))

    # Energy-Charts omits the last hour(s) of the final requested day, so two extra days are fetched;
    # tasks never use data after test_end.
    energy_end = (pd.Timestamp(end) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    for row in manifest.itertuples():
        if row.domain == "load":
            add(row.series_id, sources.energy_load(row.key, start, energy_end, RAW))
            add(row.covariate, sources.open_meteo(row.lat, row.lon, "temperature_2m", start, end, RAW, model=weather))
        elif row.domain == "price":
            add(row.series_id, sources.energy_price(row.key, start, energy_end, RAW))
        elif row.domain == "weather":
            add(row.series_id, sources.open_meteo(row.lat, row.lon, row.key, start, end, RAW, model=weather))
        elif row.domain == "river":
            s, _ = sources.usgs_daily(row.key, start, end, RAW)
            add(row.series_id, s)
            add(row.covariate, sources.open_meteo(row.lat, row.lon, "precipitation_sum", start, end, RAW, daily=True,
                                                    model=weather))
        elif row.domain == "fx":
            add(row.series_id, sources.ecb_fx(row.key, start, end, RAW))
        elif row.domain == "rates":
            add(row.series_id, sources.fred(row.key, start, end, RAW))
    df = pd.concat(out, ignore_index=True)
    df = df.drop_duplicates(["series_id", "timestamp"]).sort_values(["series_id", "timestamp"])
    dest = PROCESSED / period
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "series.parquet", index=False)
    return df


# ------------------------------------------------------------------ tasks

# Origin step of the main study: one horizon, at most one week (hourly 168, daily 7, business daily 5 steps),
# so the long horizons get several, overlapping windows (protocol section 3.2). The pilot used one horizon.
WEEK_STEPS = {"h": 168, "D": 7, "B": 5}


def origin_step(period: str, freq: str, horizon: int) -> int:
    rule = catalog.PERIODS[period].get("origin_step", "horizon")
    return horizon if rule == "horizon" else min(horizon, WEEK_STEPS[freq])


def _origins(index: pd.DatetimeIndex, test_start: pd.Timestamp, test_end: pd.Timestamp, horizon: int,
             step: int | None = None) -> list[int]:
    """Positions of forecast origins: windows of `horizon` steps inside the test period, `step` apart."""
    step = step or horizon
    first = int(np.searchsorted(index, test_start))  # first test step
    last = int(np.searchsorted(index, test_end, side="right")) - 1
    origins = []
    pos = first - 1
    while pos + horizon <= last:
        origins.append(pos)
        pos += step
    return origins


def make_tasks(period: str) -> pd.DataFrame:
    """One row per (target group, origin, horizon).

    Columns: task_id, domain, mode ('uni' or 'group'), targets (JSON list of series ids),
    future_covariates (JSON list), origin (timestamp of the last context step), horizon, season,
    lineage, excluded (reason or empty).
    """
    cfg = catalog.PERIODS[period]
    series = pd.read_parquet(PROCESSED / period / "series.parquet")
    manifest = pd.read_csv(MANIFESTS / "tierB_series.csv")
    by_id = {sid: g.set_index("timestamp")["value"] for sid, g in series.groupby("series_id")}
    test_start = pd.Timestamp(cfg["test_start"], tz="UTC")
    test_end = pd.Timestamp(cfg["test_end"], tz="UTC") + pd.Timedelta(hours=23)  # last hour of the last test day
    rows = []

    def add_tasks(domain: str, mode: str, targets: list[str], covs: list[str]) -> None:
        meta = catalog.DOMAINS[domain]
        idx = by_id[targets[0]].index
        for h in meta["horizons"]:
            for pos in _origins(idx, test_start, test_end, h, origin_step(period, meta["freq"], h)):
                origin = idx[pos]
                reason = _exclusion(by_id, targets, covs, origin, h, meta["season"])
                rows.append({"domain": domain, "mode": mode, "targets": json.dumps(targets),
                             "future_covariates": json.dumps(covs), "origin": origin, "horizon": h,
                             "season": meta["season"], "lineage": meta["lineage"], "excluded": reason})

    for row in manifest.itertuples():
        covs = [row.covariate] if isinstance(row.covariate, str) else []
        add_tasks(row.domain, "uni", [row.series_id], covs)
    for domain in ("price", "fx", "rates"):  # real multivariate tasks for contrast P4
        members = manifest.loc[manifest.domain == domain, "series_id"].tolist()
        if len(members) > 1:
            add_tasks(domain, "group", members, [])

    tasks = pd.DataFrame(rows)
    tasks.insert(0, "task_id", [f"{period}-{i:06d}" for i in range(len(tasks))])
    tasks.to_parquet(PROCESSED / period / "tasks.parquet", index=False)
    return tasks


def _exclusion(by_id: dict, targets: list[str], covs: list[str], origin: pd.Timestamp, h: int, season: int) -> str:
    """Protocol section 6, applied identically for all models.

    Targets are checked on the calendar of the first target, as the runner builds them; missing values
    are checked both in the TSFM context and in the shorter classical-model context.
    """
    from ..baselines import CLASSICAL_CONTEXT

    target_index = by_id[targets[0]].index
    pos = target_index.get_loc(origin)
    for sid in targets:
        s = by_id[sid].reindex(target_index)
        context = s.iloc[max(0, pos + 1 - CONTEXT_CAP): pos + 1]
        recent = context.iloc[-CLASSICAL_CONTEXT.get(season, 2016):]
        future = s.iloc[pos + 1: pos + 1 + h]
        if len(future) < h or future.isna().any():
            return f"{sid}: missing target value in test window"
        if context.isna().mean() > 0.05 or recent.isna().mean() > 0.05:
            return f"{sid}: more than 5% missing in context"
        if np.nanstd(context.to_numpy()) == 0:
            return f"{sid}: constant context"
        if len(context.dropna()) < 2 * season:
            return f"{sid}: context shorter than two seasons"
    for cid in covs:
        c = by_id.get(cid)
        if c is None:
            return f"{cid}: covariate missing"
        window = c.reindex(target_index).iloc[max(0, pos + 1 - CONTEXT_CAP): pos + 1 + h]
        if window.isna().mean() > 0.05 or window.iloc[-h:].isna().any():
            return f"{cid}: covariate missing"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["select", "series", "tasks"])
    parser.add_argument("--period", choices=list(catalog.PERIODS), default="pilot")
    args = parser.parse_args()
    if args.step == "select":
        print(select().groupby("domain").size())
    elif args.step == "series":
        df = download_series(args.period)
        print(df.groupby("series_id").size().describe())
    else:
        t = make_tasks(args.period)
        print(t.groupby(["domain", "mode", "horizon"]).agg(n=("task_id", "size"), excluded=("excluded", lambda x: (x != "").sum())))


if __name__ == "__main__":
    main()
