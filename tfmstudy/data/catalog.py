"""Candidate series for Tier B/P and the rules that select them.

Candidates are fixed lists; the selection rules use only data from before the TimesFM-3 release
(2026-08-31), so the series list is determined without looking at any test window.

Lineage strata (protocol section 3.2), relative to the known pretraining corpora
(GiftEvalPretrain as used by TimesFM-3 and Chronos-2, plus TimesFM-3's Wikipedia/Trends data):
  L0  no dataset from the same source family is known to be in the corpora
  L1  the same source family is in the corpora, for earlier periods (or other entities)
  L2  the same series and period are in the corpora (Tier W-seen only)
"""

from __future__ import annotations

# Energy-Charts country codes, main bidding zone, capital (lat, lon).
COUNTRIES = {
    "de": ("DE-LU", "Berlin", 52.52, 13.41),
    "fr": ("FR", "Paris", 48.86, 2.35),
    "it": ("IT-North", "Rome", 41.90, 12.50),
    "es": ("ES", "Madrid", 40.42, -3.70),
    "pl": ("PL", "Warsaw", 52.23, 21.01),
    "nl": ("NL", "Amsterdam", 52.37, 4.90),
    "be": ("BE", "Brussels", 50.85, 4.35),
    "at": ("AT", "Vienna", 48.21, 16.37),
    "se": ("SE3", "Stockholm", 59.33, 18.07),
    "ch": ("CH", "Bern", 46.95, 7.45),
    "no": ("NO1", "Oslo", 59.91, 10.75),
    "cz": ("CZ", "Prague", 50.08, 14.44),
    "pt": ("PT", "Lisbon", 38.72, -9.14),
    "ro": ("RO", "Bucharest", 44.43, 26.10),
    "hu": ("HU", "Budapest", 47.50, 19.04),
    "dk": ("DK1", "Copenhagen", 55.68, 12.57),
    "fi": ("FI", "Helsinki", 60.17, 24.94),
    "gr": ("GR", "Athens", 37.98, 23.73),
    "sk": ("SK", "Bratislava", 48.15, 17.11),
    "bg": ("BG", "Sofia", 42.70, 23.32),
}
N_COUNTRIES = 12  # largest by 2025 total load among candidates with complete 2025 data

# Weather: 20 cities, two variables each.
WEATHER_CITIES = {
    "berlin": (52.52, 13.41), "paris": (48.86, 2.35), "rome": (41.90, 12.50), "madrid": (40.42, -3.70),
    "warsaw": (52.23, 21.01), "amsterdam": (52.37, 4.90), "vienna": (48.21, 16.37), "stockholm": (59.33, 18.07),
    "lisbon": (38.72, -9.14), "athens": (37.98, 23.73), "london": (51.51, -0.13), "dublin": (53.35, -6.26),
    "new_york": (40.71, -74.01), "chicago": (41.88, -87.63), "los_angeles": (34.05, -118.24),
    "tokyo": (35.68, 139.69), "sydney": (-33.87, 151.21), "sao_paulo": (-23.55, -46.63),
    "mumbai": (19.08, 72.88), "cairo": (30.04, 31.24),
}
WEATHER_VARIABLES = ("temperature_2m", "wind_speed_10m")

# USGS mainstem gauges of large US rivers; kept if the 2025 daily record is complete.
USGS_SITES = [
    "07010000",  # Mississippi at St. Louis
    "07289000",  # Mississippi at Vicksburg
    "05587450",  # Mississippi at Grafton
    "06934500",  # Missouri at Hermann
    "06610000",  # Missouri at Omaha
    "03611500",  # Ohio at Metropolis
    "03294500",  # Ohio at Louisville
    "14105700",  # Columbia at The Dalles
    "12472800",  # Columbia below Priest Rapids Dam
    "09380000",  # Colorado at Lees Ferry
    "08330000",  # Rio Grande at Albuquerque
    "01646500",  # Potomac near Washington
    "01570500",  # Susquehanna at Harrisburg
    "01463500",  # Delaware at Trenton
    "01358000",  # Hudson at Green Island
    "02489500",  # Pearl near Bogalusa
    "07374000",  # Mississippi at Baton Rouge
    "03377500",  # Wabash at Mt. Carmel
    "03085000",  # Monongahela at Braddock
    "05331000",  # Mississippi at St. Paul
    "11377100",  # Sacramento above Bend Bridge
    "13317000",  # Salmon at White Bird
    "06191500",  # Yellowstone at Corwin Springs
    "02226000",  # Altamaha at Doctortown
    "08313000",  # Rio Grande at Otowi Bridge
]
N_USGS = 20

ECB_CURRENCIES = ["USD", "JPY", "GBP", "CHF", "CNY", "SEK", "NOK", "PLN", "CZK", "AUD"]
FRED_SERIES = ["DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS20", "DGS30", "DCOILWTICO", "DCOILBRENTEU"]

# Domain metadata: frequency, MASE season, horizons (protocol 3.2, 4.4) and lineage stratum.
DOMAINS = {
    "load":   {"freq": "h", "season": 24, "horizons": (24, 168), "lineage": "L1",
               "lineage_note": "ENTSO-E-derived Spanish load ('spain') and other national loads in GiftEvalPretrain"},
    "price":  {"freq": "h", "season": 24, "horizons": (24, 168), "lineage": "L1",
               "lineage_note": "Spanish day-ahead prices ('spain') in GiftEvalPretrain"},
    "weather": {"freq": "h", "season": 24, "horizons": (24, 168), "lineage": "L1",
                "lineage_note": "ERA5 1989-2018 in GiftEvalPretrain; Open-Meteo archive is ERA5-based"},
    "river":  {"freq": "D", "season": 7, "horizons": (7, 28), "lineage": "L0",
               "lineage_note": "no streamflow dataset in the known corpora"},
    "fx":     {"freq": "B", "season": 5, "horizons": (5, 20), "lineage": "L0",
               "lineage_note": "no exchange-rate dataset in the known corpora"},
    "rates":  {"freq": "B", "season": 5, "horizons": (5, 20), "lineage": "L1",
               "lineage_note": "FRED-MD (monthly macro incl. rates) in GiftEvalPretrain"},
    "wiki":   {"freq": "D", "season": 7, "horizons": (7, 28), "lineage": "L1",
               "lineage_note": "Wikipedia pageviews to 2023-11 in TimesFM-3 corpus; Kaggle web traffic in GiftEvalPretrain"},
}

# Study periods. The pilot replicates the design one year earlier, entirely before the release.
# weather_model: Open-Meteo product (sources.open_meteo). download_not_before: the main data are fetched
# after the protocol freeze, once ERA5 covers the test windows (about six days of lag). origin_step:
# "horizon" (non-overlapping windows, pilot) or "week" (build.origin_step).
PERIODS = {
    "pilot": {"history_start": "2024-01-01", "test_start": "2025-09-01", "test_end": "2025-11-02",
              "weather_model": "best_match"},
    "main":  {"history_start": "2025-01-01", "test_start": "2026-09-01", "test_end": "2026-10-20",
              "weather_model": "era5", "download_not_before": "2026-10-29", "origin_step": "week"},
}
SELECTION_YEAR = "2025"  # pre-release year whose data drive the selection rules
