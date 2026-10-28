"""Downloaders for the public data sources of Tiers B, W and P.

Every response is cached under `raw_dir/<source>/` with the request URL and the UTC download time,
so a series can always be traced to the exact request that produced it and re-runs do not re-download.
Each function returns a pandas Series indexed by UTC timestamps (float values, NaN for gaps).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

USER_AGENT = "tfmstudy-research/0.1 (zero-shot forecasting study; info@alessiopomaro.it)"

# Minimum seconds between requests to the same host (Energy-Charts answers 429 to bursts).
_MIN_INTERVAL = {"api.energy-charts.info": 8.0, "wikimedia.org": 0.1, "en.wikipedia.org": 0.1}
_last_call: dict[str, float] = {}


class SourceError(RuntimeError):
    pass


def _get(url: str, params: dict, raw_dir: Path, source: str, retries: int = 6) -> bytes:
    """GET with a per-host rate limit, exponential backoff and an on-disk cache."""
    key = hashlib.sha256((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:20]
    cache = raw_dir / source / f"{key}.bin"
    meta = cache.with_suffix(".json")
    if cache.exists() and meta.exists():
        return cache.read_bytes()

    host = url.split("/")[2]
    delay = 2.0
    for attempt in range(retries):
        wait = _MIN_INTERVAL.get(host, 0.5) - (time.monotonic() - _last_call.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()
        try:
            resp = requests.get(url, params=params, timeout=120, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            err = str(exc)
        else:
            if resp.status_code == 200:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(resp.content)
                meta.write_text(json.dumps({
                    "url": resp.url, "params": params, "status": 200,
                    "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "sha256": hashlib.sha256(resp.content).hexdigest(),
                }, indent=1))
                return resp.content
            if resp.status_code == 404:
                raise SourceError(f"404 for {resp.url}")
            err = f"HTTP {resp.status_code}"
        time.sleep(delay)
        delay *= 2
    raise SourceError(f"{url} {params}: {err} after {retries} attempts")


def _months(start: str, end: str) -> list[tuple[str, str]]:
    """Inclusive [start, end] date range split into calendar-month chunks (ISO dates)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    cur = s
    while cur <= e:
        nxt = min(cur + pd.offsets.MonthBegin(1), e + pd.Timedelta(days=1))
        out.append((cur.strftime("%Y-%m-%d"), (nxt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
        cur = nxt
    return out


# ---------------------------------------------------------------- Energy-Charts (CC BY 4.0)

def energy_load(country: str, start: str, end: str, raw_dir: Path) -> pd.Series:
    """Total load (MW) of a country, hourly mean of the native 15-min/hourly values."""
    parts = []
    for a, b in _months(start, end):
        body = json.loads(_get("https://api.energy-charts.info/public_power",
                               {"country": country, "start": a, "end": b}, raw_dir, "energy_charts"))
        load = next((p["data"] for p in body["production_types"] if p["name"] == "Load"), None)
        if load is None:
            raise SourceError(f"no Load series for {country}")
        idx = pd.to_datetime(body["unix_seconds"], unit="s", utc=True)
        parts.append(pd.Series(np.array(load, dtype=float), index=idx))
    return _hourly(parts, start, end)


def energy_price(bzn: str, start: str, end: str, raw_dir: Path) -> pd.Series:
    """Day-ahead price (EUR/MWh) of a bidding zone, hourly mean."""
    parts = []
    for a, b in _months(start, end):
        body = json.loads(_get("https://api.energy-charts.info/price",
                               {"bzn": bzn, "start": a, "end": b}, raw_dir, "energy_charts"))
        idx = pd.to_datetime(body["unix_seconds"], unit="s", utc=True)
        parts.append(pd.Series(np.array(body["price"], dtype=float), index=idx))
    return _hourly(parts, start, end)


def _hourly(parts: list[pd.Series], start: str, end: str) -> pd.Series:
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s = s.resample("1h").mean()
    full = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23), freq="1h")
    return s.reindex(full)


# ---------------------------------------------------------------- Open-Meteo archive (CC BY 4.0)

def open_meteo(lat: float, lon: float, variable: str, start: str, end: str, raw_dir: Path,
               daily: bool = False, model: str = "era5") -> pd.Series:
    """Hourly (or daily) weather variable from the Open-Meteo archive, UTC.

    `model` pins the product: "era5" (main study and Tier P) is ERA5 only, available with a lag of about
    six days; "best_match" (the API default, used for the pilot data) fills the most recent days with
    other models, so the product would change inside a series.
    """
    key = "daily" if daily else "hourly"
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              key: variable, "timezone": "GMT"}
    if model != "best_match":
        params["models"] = model
    body = json.loads(_get("https://archive-api.open-meteo.com/v1/archive", params, raw_dir, "open_meteo"))
    idx = pd.to_datetime(body[key]["time"], utc=True)
    return pd.Series(np.array(body[key][variable], dtype=float), index=idx)


# ---------------------------------------------------------------- USGS daily values (public domain)

def usgs_daily(site: str, start: str, end: str, raw_dir: Path) -> tuple[pd.Series, dict]:
    """Daily mean discharge (cfs). Returns the series and site info (name, lat, lon, qualifiers)."""
    body = json.loads(_get("https://waterservices.usgs.gov/nwis/dv/", {
        "sites": site, "parameterCd": "00060", "statCd": "00003",
        "startDT": start, "endDT": end, "format": "json"}, raw_dir, "usgs"))
    series = body["value"]["timeSeries"]
    if not series:
        raise SourceError(f"no daily discharge for USGS site {site}")
    ts = series[0]
    values = ts["values"][0]["value"]
    idx = pd.to_datetime([v["dateTime"][:10] for v in values], utc=True)
    s = pd.Series([float(v["value"]) if float(v["value"]) >= 0 else np.nan for v in values], index=idx)
    geo = ts["sourceInfo"]["geoLocation"]["geogLocation"]
    info = {"name": ts["sourceInfo"]["siteName"], "lat": geo["latitude"], "lon": geo["longitude"],
            "provisional_share": float(np.mean(["P" in v.get("qualifiers", []) for v in values])) if values else None}
    full = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="D")
    return s[~s.index.duplicated()].reindex(full), info


# ---------------------------------------------------------------- ECB exchange rates

def ecb_fx(currency: str, start: str, end: str, raw_dir: Path) -> pd.Series:
    """ECB reference rate, units of `currency` per euro, business days only."""
    text = _get(f"https://data-api.ecb.europa.eu/service/data/EXR/D.{currency}.EUR.SP00.A",
                {"format": "csvdata", "startPeriod": start, "endPeriod": end}, raw_dir, "ecb").decode()
    df = pd.read_csv(io.StringIO(text))
    return pd.Series(df["OBS_VALUE"].astype(float).to_numpy(), index=pd.to_datetime(df["TIME_PERIOD"], utc=True))


# ---------------------------------------------------------------- FRED (public-domain series only)

def fred(series_id: str, start: str, end: str, raw_dir: Path) -> pd.Series:
    """A FRED series through the keyless graph CSV endpoint; '.' entries become NaN and are dropped."""
    text = _get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                {"id": series_id, "cosd": start, "coed": end}, raw_dir, "fred").decode()
    df = pd.read_csv(io.StringIO(text))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(df.iloc[:, 0], utc=True))
    return s.dropna()


# ---------------------------------------------------------------- Wikimedia (CC0)

def wiki_pageviews(article: str, start: str, end: str, raw_dir: Path) -> pd.Series:
    """Daily user pageviews of an English Wikipedia article; missing days are zero views."""
    a, b = start.replace("-", ""), end.replace("-", "")
    title = article.replace(" ", "_")
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
           f"{requests.utils.quote(title, safe='')}/daily/{a}00/{b}00")
    try:
        items = json.loads(_get(url, {}, raw_dir, "wikimedia"))["items"]
    except SourceError as exc:
        if "404" in str(exc):  # no views at all in the range
            items = []
        else:
            raise
    full = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="D")
    if not items:
        return pd.Series(0.0, index=full)
    s = pd.Series([float(i["views"]) for i in items],
                  index=pd.to_datetime([i["timestamp"][:8] for i in items], format="%Y%m%d", utc=True))
    return s.reindex(full, fill_value=0.0)


def wiki_top_month(year: int, month: int, raw_dir: Path) -> list[dict]:
    """Top-1000 English Wikipedia articles of a month (all access), as returned by the API."""
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{year}/{month:02d}/all-days"
    return json.loads(_get(url, {}, raw_dir, "wikimedia"))["items"][0]["articles"]


def wiki_creation_dates(titles: list[str], raw_dir: Path) -> dict[str, str | None]:
    """Timestamp of the first revision of each title (None if missing); one request per title."""
    out = {}
    for t in titles:
        body = json.loads(_get("https://en.wikipedia.org/w/api.php", {
            "action": "query", "prop": "revisions", "rvlimit": 1, "rvdir": "newer", "rvprop": "timestamp",
            "titles": t.replace("_", " "), "format": "json", "formatversion": 2}, raw_dir, "wikipedia"))
        page = body["query"]["pages"][0]
        out[t] = page["revisions"][0]["timestamp"] if "revisions" in page else None
    return out


def wiki_redirects(titles: list[str], raw_dir: Path) -> dict[str, str]:
    """{title: target} for titles that are currently redirects (or {title: ""} if the page is missing).

    The answer depends on the day of the request, so callers pass a dated `raw_dir`.
    """
    out: dict[str, str] = {}
    for i in range(0, len(titles), 50):
        chunk = [t.replace("_", " ") for t in titles[i:i + 50]]
        body = json.loads(_get("https://en.wikipedia.org/w/api.php", {
            "action": "query", "titles": "|".join(chunk), "redirects": 1, "format": "json",
            "formatversion": 2}, raw_dir, "wikipedia"))
        query = body["query"]
        normalized = {n["to"]: n["from"] for n in query.get("normalized", [])}
        for r in query.get("redirects", []):
            out[normalized.get(r["from"], r["from"]).replace(" ", "_")] = r["to"]
        for page in query["pages"]:
            if page.get("missing"):
                out[normalized.get(page["title"], page["title"]).replace(" ", "_")] = ""
    return out
