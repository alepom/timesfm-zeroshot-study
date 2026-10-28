"""Tier W: the Wikipedia contamination probe (protocol section 3.3).

    python -m tfmstudy.data.wiki select
    python -m tfmstudy.data.wiki series --period pilot|main
    python -m tfmstudy.data.wiki tasks  --period pilot|main
    python -m tfmstudy.data.wiki redirects             # title status today -> data/manifests/tierW_redirects_<date>.csv

Strata (same calendar months, so seasonality is comparable):
  W-seen  pages popular in 2022, the period's test span moved to 2023 (inside TimesFM-3's pageview corpus, cut off 2023-11)
  W-post  the same pages, test windows in the study period (after the release for `main`)
  W-new   pages created 2023-12-01..2025-07-28 (never in any corpus), same test windows as W-post,
          matched to W-post on pre-release popularity quintiles
All selection uses data up to 2025-08-31 only. A page whose title is a redirect (moved or merged) or missing
on the latest status check is excluded: its views move to another title and the series breaks (the check is
repeated when the main data are downloaded).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re

import numpy as np
import pandas as pd

from . import catalog, sources
from .build import MANIFESTS, PROCESSED, RAW, _exclusion, _origins, origin_step

N_PAGES = 200
SEED = 17
NEW_CREATED_FROM, NEW_CREATED_TO = "2023-12-01", "2025-07-28"
POPULARITY_WINDOW = ("2025-03-01", "2025-08-31")  # pre-release, used for matching and zero-view screening
SEEN_YEAR = 2023  # W-seen windows: the period's test span moved to 2023 (inside the pageview corpus)
HISTORY_DAYS = 400  # equal context length in all strata; W-new pages have >= 400 days of history by construction
MISSING_TOP_MONTHS: set[str] = set()

_NON_ARTICLE = re.compile(r"^(Main_Page|Special:|Wikipedia:|Portal:|File:|Help:|Category:|Template:|Talk:|User:|Draft:|Module:|-$|Undefined$)")


def _top_union(months: list[tuple[int, int]]) -> pd.Series:
    """Number of monthly top-1000 lists each article appears in."""
    counts: dict[str, int] = {}
    for y, m in months:
        try:
            articles = sources.wiki_top_month(y, m, RAW)
        except sources.SourceError as exc:  # a few months are missing from the top endpoint
            if "404" not in str(exc):
                raise
            MISSING_TOP_MONTHS.add(f"{y}-{m:02d}")
            continue
        for a in articles:
            title = a["article"]
            if not _NON_ARTICLE.match(title):
                counts[title] = counts.get(title, 0) + 1
    return pd.Series(counts, dtype=int)


def _popularity(titles: list[str]) -> pd.DataFrame:
    a, b = POPULARITY_WINDOW
    rows = []
    for t in titles:
        s = sources.wiki_pageviews(t, a, b, RAW)
        rows.append({"title": t, "median_views": float(s.median()), "zero_share": float((s == 0).mean())})
    return pd.DataFrame(rows)


def select() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    seen_pool = _top_union([(2022, m) for m in range(1, 13)])
    seen_pool = sorted(seen_pool[seen_pool >= 3].index)
    seen_pop = _popularity(seen_pool)
    seen_pop = seen_pop[seen_pop.zero_share <= 0.05]
    seen = seen_pop.iloc[rng.permutation(len(seen_pop))[:N_PAGES]].copy()

    new_months = [(2024, m) for m in range(1, 13)] + [(2025, m) for m in range(1, 9)]
    # A page in a 2022-2023 top list existed before December 2023, so it cannot be W-new;
    # dropping those first keeps the number of creation-date lookups small.
    old_titles = set(_top_union([(y, m) for y in (2022, 2023) for m in range(1, 13)]).index)
    new_pool = sorted(set(_top_union(new_months).index) - old_titles)
    created = sources.wiki_creation_dates(new_pool, RAW)
    new_pool = [t for t, c in created.items() if c and NEW_CREATED_FROM <= c[:10] <= NEW_CREATED_TO]
    new_pop = _popularity(new_pool)
    new_pop = new_pop[new_pop.zero_share <= 0.05]
    # A recent first revision is not enough: renamed, split or re-created pages keep a title that was
    # already viewed (e.g. as a redirect). W-new pages must have no views at all under their title
    # from 2021-01-01 to 2023-11-30.
    before = {t: float(sources.wiki_pageviews(t, "2021-01-01", "2023-11-30", RAW).sum()) for t in new_pop.title}
    n_before_screen = len(new_pop)
    new_pop = new_pop[new_pop.title.map(before) == 0]

    # Match W-new to the W-seen/W-post pages on quintiles of pre-release median views.
    edges = np.quantile(seen.median_views, [0.2, 0.4, 0.6, 0.8])
    seen["quintile"] = np.searchsorted(edges, seen.median_views, side="right")
    new_pop["quintile"] = np.searchsorted(edges, new_pop.median_views, side="right")
    picks = []
    for q, n in seen.quintile.value_counts().sort_index().items():
        cand = new_pop[new_pop.quintile == q]
        picks.append(cand.iloc[rng.permutation(len(cand))[: min(n, len(cand))]])
    new = pd.concat(picks)
    new["created"] = new.title.map(created)

    out = pd.concat([seen.assign(stratum="W-seen/W-post"), new.assign(stratum="W-new")], ignore_index=True)
    out["series_id"] = "wiki:" + out.title
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(MANIFESTS / "tierW_pages.csv", index=False)
    (MANIFESTS / "tierW_selection_stats.json").write_text(json.dumps({
        "seen_pool": len(seen_pool), "seen_after_zero_screen": int(len(seen_pop)),
        "new_pool_created_in_range": len(new_pool), "new_after_zero_screen": n_before_screen,
        "new_after_no_prior_views_screen": int(len(new_pop)),
        "new_selected": int(len(new)), "quintile_edges": edges.tolist(),
        "missing_top_months": sorted(MISSING_TOP_MONTHS)}, indent=1))
    return out


def download_series(period: str) -> pd.DataFrame:
    cfg = catalog.PERIODS[period]
    pages = pd.read_csv(MANIFESTS / "tierW_pages.csv")
    not_before = cfg.get("download_not_before")
    if not_before and pd.Timestamp.now(tz="UTC") < pd.Timestamp(not_before, tz="UTC"):
        raise SystemExit(f"{period} data may not be downloaded before {not_before} (protocol section 3.3)")
    start = (pd.Timestamp(_in_seen_year(cfg["test_start"])) - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    out = []
    for row in pages.itertuples():
        s = sources.wiki_pageviews(row.title, start, cfg["test_end"], RAW)
        out.append(pd.DataFrame({"series_id": row.series_id, "timestamp": s.index, "value": s.to_numpy()}))
    df = pd.concat(out, ignore_index=True)
    dest = PROCESSED / period
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / "wiki_series.parquet", index=False)
    return df


def _in_seen_year(day: str) -> str:
    return f"{SEEN_YEAR}{day[4:]}"


def redirect_check() -> pd.DataFrame:
    """Status of every selected title today; written to tierW_redirects_<date>.csv."""
    day = dt.datetime.now(dt.timezone.utc).date().isoformat()
    pages = pd.read_csv(MANIFESTS / "tierW_pages.csv")
    moved = sources.wiki_redirects(pages.title.tolist(), RAW / f"wiki_status_{day}")
    out = pages[["title", "stratum", "quintile"]].assign(
        checked=day, redirect_to=pages.title.map(moved), excluded=pages.title.isin(list(moved)))
    out.to_csv(MANIFESTS / f"tierW_redirects_{day}.csv", index=False)
    return out


def _moved_titles() -> tuple[set[str], str]:
    files = sorted(MANIFESTS.glob("tierW_redirects_*.csv"))
    if not files:
        return set(), ""
    latest = pd.read_csv(files[-1])
    return set(latest.loc[latest.excluded, "title"]), str(latest.checked.iloc[0])


def make_tasks(period: str) -> pd.DataFrame:
    cfg = catalog.PERIODS[period]
    pages = pd.read_csv(MANIFESTS / "tierW_pages.csv")
    moved, checked = _moved_titles()
    series = pd.read_parquet(PROCESSED / period / "wiki_series.parquet")
    by_id = {sid: g.set_index("timestamp")["value"] for sid, g in series.groupby("series_id")}
    meta = catalog.DOMAINS["wiki"]
    windows = {
        "W-seen": (_in_seen_year(cfg["test_start"]), _in_seen_year(cfg["test_end"]), "L2"),
        "W-post": (cfg["test_start"], cfg["test_end"], "L1"),
        "W-new": (cfg["test_start"], cfg["test_end"], "L1"),
    }
    rows = []
    for row in pages.itertuples():
        strata = ["W-seen", "W-post"] if row.stratum == "W-seen/W-post" else ["W-new"]
        s = by_id[row.series_id]
        for stratum in strata:
            a, b, lineage = windows[stratum]
            for h in meta["horizons"]:
                for pos in _origins(s.index, pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"), h,
                                    origin_step(period, meta["freq"], h)):
                    origin = s.index[pos]
                    # Context limited to HISTORY_DAYS so the strata see the same history length; for W-new
                    # it never starts before the page existed (the API reports no views as zeros).
                    context_start = origin - pd.Timedelta(days=HISTORY_DAYS - 1)
                    if stratum == "W-new":
                        context_start = max(context_start, pd.Timestamp(row.created[:10], tz="UTC"))
                    sub = {row.series_id: s.loc[context_start:]}
                    rows.append({"domain": "wiki", "mode": "uni", "targets": json.dumps([row.series_id]),
                                 "future_covariates": "[]", "origin": origin, "horizon": h,
                                 "season": meta["season"], "lineage": lineage, "stratum": stratum,
                                 "quintile": int(row.quintile), "context_start": context_start,
                                 "excluded": (f"title is a redirect or missing on {checked}" if row.title in moved
                                              else _exclusion(sub, [row.series_id], [], origin, h, meta["season"]))})
    tasks = pd.DataFrame(rows)
    tasks.insert(0, "task_id", [f"{period}-W{i:06d}" for i in range(len(tasks))])
    tasks.to_parquet(PROCESSED / period / "wiki_tasks.parquet", index=False)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["select", "series", "tasks", "redirects"])
    parser.add_argument("--period", choices=list(catalog.PERIODS), default="pilot")
    args = parser.parse_args()
    if args.step == "select":
        print(select().groupby("stratum").median_views.describe())
    elif args.step == "redirects":
        r = redirect_check()
        print(r[r.excluded].to_string(index=False))
    elif args.step == "series":
        if args.period == "main":
            redirect_check()  # status at download time (protocol section 3.3)
        print(download_series(args.period).groupby("series_id").size().describe())
    else:
        t = make_tasks(args.period)
        print(t.groupby(["stratum", "horizon"]).agg(n=("task_id", "size"), excluded=("excluded", lambda x: (x != "").sum())))


if __name__ == "__main__":
    main()
