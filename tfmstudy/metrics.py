"""Forecast accuracy metrics on (targets, horizon) truth and (targets, horizon, 9) quantiles."""

from __future__ import annotations

import numpy as np

from . import MEDIAN_INDEX, QUANTILES


def seasonal_naive_scale(history: np.ndarray, season: int) -> np.ndarray:
    """Mean absolute seasonal difference of each history row (MASE denominator)."""
    history = np.atleast_2d(history)
    m = season if history.shape[1] > season else 1
    diffs = np.abs(history[:, m:] - history[:, :-m])
    scale = np.nanmean(diffs, axis=1)
    return np.where(scale > 0, scale, np.nan)


def mase(truth: np.ndarray, quantiles: np.ndarray, history: np.ndarray, season: int) -> np.ndarray:
    """Per-target MASE of the median forecast."""
    err = np.abs(np.atleast_2d(truth) - quantiles[..., MEDIAN_INDEX])
    return np.nanmean(err, axis=1) / seasonal_naive_scale(history, season)


def quantile_loss(truth: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Pinball loss summed over levels, shape (targets, horizon)."""
    y = np.atleast_2d(truth)[..., None]
    levels = np.asarray(QUANTILES)
    diff = y - quantiles
    return np.sum(np.maximum(levels * diff, (levels - 1) * diff), axis=-1)


def wql(truth: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Per-target weighted quantile loss: 2 * mean pinball / mean |y| (GluonTS convention)."""
    loss = quantile_loss(truth, quantiles) / len(QUANTILES)
    denom = np.nanmean(np.abs(np.atleast_2d(truth)), axis=1)
    return 2 * np.nanmean(loss, axis=1) / np.where(denom > 0, denom, np.nan)


def sql(truth: np.ndarray, quantiles: np.ndarray, history: np.ndarray, season: int) -> np.ndarray:
    """Per-target scaled quantile loss (fev-bench SQL): mean pinball scaled like MASE."""
    loss = 2 * quantile_loss(truth, quantiles) / len(QUANTILES)
    return np.nanmean(loss, axis=1) / seasonal_naive_scale(history, season)


def coverage(truth: np.ndarray, quantiles: np.ndarray, lower: int = 0, upper: int = 8) -> np.ndarray:
    """Per-target share of truth values inside [q_lower, q_upper] (default 0.1–0.9, nominal 80%)."""
    y = np.atleast_2d(truth)
    inside = (y >= quantiles[..., lower]) & (y <= quantiles[..., upper])
    return inside.mean(axis=1)
