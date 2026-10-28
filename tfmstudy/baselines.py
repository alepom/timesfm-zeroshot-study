"""Classical baselines with the same interface as the TSFM adapters (protocol section 2).

Quantiles come from the models' prediction intervals: the 20/40/60/80% intervals give the levels
0.4/0.6, 0.3/0.7, 0.2/0.8, 0.1/0.9, and the point forecast is used as the median. Each target row is
modelled on its own; covariates are not used. Contexts are capped (CLASSICAL_CONTEXT) for tractability.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from . import QUANTILES
from .adapters import Adapter, Task, _check_quantiles

LEVELS = (20, 40, 60, 80)
# Most recent observations given to the classical models: 12 weeks of hourly data, 3 years of daily data.
CLASSICAL_CONTEXT = {24: 2016, 7: 1095, 5: 780, 1: 2016}


def _intervals_to_quantiles(point: np.ndarray, lo: dict, hi: dict) -> np.ndarray:
    """(h,) point and {level: (h,)} bounds -> (h, 9) quantiles for QUANTILES."""
    cols = [lo[80], lo[60], lo[40], lo[20], point, hi[20], hi[40], hi[60], hi[80]]
    return np.sort(np.stack(cols, axis=-1), axis=-1)


MSTL_SEASONS = (24, 168)
# Model fitted to the deseasonalised series by each MSTL configuration (protocol section 2).
MSTL_REMAINDER = {"mstl": "ets", "mstl-theta": "theta", "mstl-arima": "arima"}


def _model(m: str, season: int, n: int | None = None):
    """Model for method `m`; `n` (context length) decides which MSTL seasons are kept."""
    from statsforecast.models import MSTL, AutoARIMA, AutoETS, AutoTheta, HistoricAverage, Naive, SeasonalNaive

    if m == "mean":  # historic average: the optimal forecast for white noise (negative control)
        return HistoricAverage()
    if m == "snaive":
        return SeasonalNaive(season_length=season)
    if m == "naive":
        return Naive()
    if m == "ets":
        return AutoETS(season_length=season)
    if m == "arima":
        return AutoARIMA(season_length=season)
    if m == "theta":
        return AutoTheta(season_length=season)
    if m in MSTL_REMAINDER:
        # Hourly data: daily and weekly seasonality removed by MSTL, a non-seasonal model on the rest.
        # A season is kept only if the context holds two of its cycles; with none, the rest model alone.
        rest = {"ets": lambda: AutoETS(model="ZZN"), "theta": lambda: AutoTheta(),
                "arima": lambda: AutoARIMA()}[MSTL_REMAINDER[m]]()
        seasons = [s for s in MSTL_SEASONS if n is None or n >= 2 * s]
        return MSTL(season_length=seasons, trend_forecaster=rest) if seasons else rest
    raise ValueError(m)


def _fit_row(method: str, y: np.ndarray, horizon: int, season: int) -> tuple[np.ndarray, int]:
    """One series -> ((h, 9) quantiles, 1 if the naive fallback was used else 0)."""
    from statsforecast.models import Naive

    y = np.asarray(y, dtype=np.float64)
    y = y[-CLASSICAL_CONTEXT.get(season, 2016):]
    y = _fill(y)
    if method in ("ets", "theta", "arima", "snaive") and len(y) < 2 * season:
        season = 1
    fell_back = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fc = _model(method, season, len(y)).forecast(y=y, h=horizon, level=list(LEVELS))
        except Exception:
            # Model fitting failed (e.g. degenerate context): fall back to the naive forecast,
            # recorded as such by the runner through `fallbacks`.
            fell_back = 1
            fc = Naive().forecast(y=y, h=horizon, level=list(LEVELS))
    point = fc["mean"]
    lo = {lv: fc[f"lo-{lv}"] for lv in LEVELS}
    hi = {lv: fc[f"hi-{lv}"] for lv in LEVELS}
    return _intervals_to_quantiles(point, lo, hi), fell_back


def _fit_row_args(job: tuple) -> tuple[np.ndarray, int]:
    return _fit_row(*job)


def _cpu_count() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


class Classical(Adapter):
    """One StatsForecast model per instance.

    options: method in {mean, snaive, naive, ets, arima, theta, mstl, mstl-theta, mstl-arima}.

    Rows are fitted in parallel worker processes, one per CPU core; the results do not depend on it.
    """

    name = "classical"
    fallbacks = 0
    _executor = None

    def _load(self) -> None:
        import statsforecast  # noqa: F401 (fail early if missing)

        self.method = self.options["method"]
        self.name = self.method
        self.workers = int(self.options.get("workers", _cpu_count()))

    def describe(self) -> dict:
        return {**super().describe(), "workers": self.workers}

    def predict_one(self, y: np.ndarray, horizon: int, season: int) -> np.ndarray:
        q, fell_back = _fit_row(self.method, y, horizon, season)
        self.fallbacks += fell_back
        return q

    def predict(self, tasks: list[Task], seasons: list[int] | None = None) -> list[np.ndarray]:
        seasons = seasons or [1] * len(tasks)
        jobs = [(self.method, row, t.horizon, season) for t, season in zip(tasks, seasons) for row in t.target]
        if self.method in ("mean", "snaive", "naive") or self.workers < 2 or len(jobs) < 2:
            fitted = [_fit_row(*j) for j in jobs]
        else:
            chunk = max(1, len(jobs) // (4 * self.workers))
            fitted = list(self._pool().map(_fit_row_args, jobs, chunksize=chunk))
        self.fallbacks += sum(f for _, f in fitted)
        out, k = [], 0
        self.task_fallbacks = []  # per task: number of rows that fell back to the naive forecast
        for t in tasks:
            v = t.target.shape[0]
            q = np.stack([fitted[k + i][0] for i in range(v)])
            self.task_fallbacks.append(int(sum(fitted[k + i][1] for i in range(v))))
            k += v
            out.append(_check_quantiles(q, v, t.horizon))
        return out

    def _pool(self) -> ProcessPoolExecutor:
        # "spawn" rather than fork: the parent may hold CUDA state and thread pools from the TSFM runs.
        if self._executor is None:
            self._executor = ProcessPoolExecutor(self.workers, mp_context=get_context("spawn"))
        return self._executor

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None


def _fill(y: np.ndarray) -> np.ndarray:
    """Linear interpolation of interior gaps; leading/trailing gaps take the nearest value."""
    if not np.isnan(y).any():
        return y
    idx = np.arange(len(y))
    ok = ~np.isnan(y)
    return np.interp(idx, idx[ok], y[ok])


def ensemble(parts: list[np.ndarray]) -> np.ndarray:
    """Equal-weight combination: mean of the members' quantiles (vincentization)."""
    return np.sort(np.mean(np.stack(parts), axis=0), axis=-1)


assert len(QUANTILES) == 9
