"""Tier C: controlled synthetic data (protocol section 3.5).

    python -m tfmstudy.data.synthetic            # writes data/processed/synthetic/{series,tasks}.parquet

Each task stores its own context and truth, so the runner needs no calendar. Seeds: replication r of
cell c uses default_rng([17, crc32(cell), r]); the C-replication cells use the seeding contract of the
reference study instead, so that their series are identical to the original ones.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd

from .build import PROCESSED

REPS = 200
SEED = 17


def _rng(cell: str, rep: int, seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(cell.encode()), rep])


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator, burn: int = 200) -> np.ndarray:
    e = rng.normal(0.0, sigma, n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


# ---------------------------------------------------------------- C-replication (arXiv:2610.00589)
# Re-implementation of DGPs D6-D8 of Khaled Ebrahim et al. (arXiv:2610.00589; code doi:10.5281/zenodo.22976253,
# CC BY 4.0, file code/01_dgp.py v1.2.2), with their parameters, seeding contract and order of random draws.
_FF_DGP_IDS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
_FF_LENGTHS = [24, 48, 96, 200]


def ff_simulate(dgp: str, n: int, rep: int, horizon: int) -> np.ndarray:
    rng = np.random.default_rng(1_000_000 * _FF_DGP_IDS.index(dgp) + 1_000 * _FF_LENGTHS.index(n) + rep)
    total = n + horizon
    if dgp == "D6":  # local level with a jump of 10 sigma_obs at 0.7 n
        level = 100.0 + np.cumsum(rng.normal(0.0, 0.5, total))
        y = level + rng.normal(0.0, 2.0, total)
        y[int(0.7 * n):] += 20.0
        return y
    if dgp == "D7":  # logistic growth, inflection at 0.6 * total, lower asymptote 20
        t = np.arange(total, dtype=float)
        return 20.0 + 180.0 / (1.0 + np.exp(-0.08 * (t - 0.6 * total))) + rng.normal(0.0, 3.0, total)
    if dgp == "D8":  # intermittent demand
        occurs = rng.random(total) < 0.3
        sizes = 1.0 + rng.poisson(4.0, total)
        return np.where(occurs, sizes, 0.0).astype(float)
    raise KeyError(dgp)


# ---------------------------------------------------------------- generators of the other cells
# Each returns (target (v, n + h), future_cov (c, n + h) or None, past_cov (c, n) or None).

def _two_seasons(n: int, h: int, rng) -> np.ndarray:
    t = np.arange(n + h)
    return (10 + 3 * np.sin(2 * np.pi * t / 24) + 2 * np.sin(2 * np.pi * t / 168)
            + _ar1(n + h, 0.5, 0.5, rng))[None]


def _long_period(n: int, h: int, rng) -> np.ndarray:
    t = np.arange(n + h)
    return (20 + 8 * np.sin(2 * np.pi * t / 365) + _ar1(n + h, 0.6, 1.0, rng))[None]


def _regime_change(n: int, h: int, rng) -> np.ndarray:
    y = _two_seasons(n, h, rng)[0]
    y[n - 24:] += 5 * 0.5 / np.sqrt(1 - 0.25)  # 5 innovation-scale sd of the AR(1) noise, 24 steps before the origin
    return y[None]


def _outliers(n: int, h: int, rng) -> np.ndarray:
    y = _two_seasons(n, h, rng)[0]
    pos = rng.choice(np.arange(int(0.95 * n), n), size=3, replace=False)
    y[pos] += rng.choice([-1, 1], size=3) * 8 * 0.5 / np.sqrt(1 - 0.25)
    return y[None]


def _random_walk(n: int, h: int, rng) -> np.ndarray:
    return (100 + np.cumsum(rng.normal(0, 1, n + h)))[None]


def _white_noise(n: int, h: int, rng) -> np.ndarray:
    return (50 + rng.normal(0, 1, n + h))[None]


def _var1(dim: int):
    def gen(n: int, h: int, rng):
        a = rng.normal(0, 1, (dim, dim))
        a = 0.8 * a / np.max(np.abs(np.linalg.eigvals(a)))  # spectral radius 0.8
        x = np.zeros((dim, n + h + 200))
        for t in range(1, x.shape[1]):
            x[:, t] = a @ x[:, t - 1] + rng.normal(0, 1, dim)
        return 10 + x[:, 200:]
    return gen


def _lead_lag(lag: int):
    def gen(n: int, h: int, rng):
        x = _ar1(n + h + lag, 0.9, 1.0, rng)
        y = 0.8 * x[:-lag] + rng.normal(0, 0.3, n + h)
        return 10 + np.stack([x[lag:], y])
    return gen


def _known_covariate(snr: float):
    def gen(n: int, h: int, rng):
        c = rng.normal(0, 1, n + h)
        noise = _ar1(n + h, 0.5, 1.0, rng)
        beta = snr * noise.std()
        return (50 + beta * c + noise)[None], c[None], None
    return gen


def _spurious_past_covariate(n: int, h: int, rng):
    y = 50 + _ar1(n + h, 0.7, 1.0, rng)
    # Covariate tracks the target closely in the context, so a model that leans on it gains nothing
    # (its future is unknown to the model and unrelated to the target).
    past = y[:n] + rng.normal(0, 0.2, n)
    return y[None], None, past[None]


def _independent(dim: int):
    def gen(n: int, h: int, rng):
        return 10 + np.stack([_ar1(n + h, 0.7, 1.0, rng) for _ in range(dim)])
    return gen


# cell -> (family, generator, context length, horizon, season, mode)
CELLS: dict[str, tuple] = {
    "uni:two_seasons:ctx168": ("C-uni", _two_seasons, 168, 168, 24, "uni"),
    "uni:two_seasons:ctx672": ("C-uni", _two_seasons, 672, 168, 24, "uni"),
    "uni:long_period:ctx200": ("C-uni", _long_period, 200, 60, 1, "uni"),
    "uni:regime_change": ("C-uni", _regime_change, 1024, 48, 24, "uni"),
    "uni:outliers": ("C-uni", _outliers, 1024, 48, 24, "uni"),
    "uni:random_walk": ("C-uni", _random_walk, 512, 64, 1, "uni"),
    "uni:white_noise": ("C-uni", _white_noise, 512, 64, 1, "uni"),
    **{f"uni:context_ladder:{c}": ("C-ladder", _two_seasons, c, 168, 24, "uni") for c in (64, 256, 1024, 4096, 15360)},
    **{f"uni:long_horizon:{h}": ("C-horizon", _two_seasons, 4096, h, 24, "uni") for h in (256, 512, 1024)},
    **{f"mv:var1:{d}": ("C-mv", _var1(d), 512, 24, 1, "group") for d in (2, 8, 32, 64)},
    **{f"mv:lead_lag:{k}": ("C-mv", _lead_lag(k), 512, 24, 1, "group") for k in (1, 5, 24)},
    **{f"mv:known_cov:snr{s}": ("C-mv", _known_covariate(s), 512, 24, 1, "cov") for s in (0.5, 2)},
    "mv:spurious_past_cov": ("C-mv", _spurious_past_covariate, 512, 24, 1, "cov"),
    "mv:independent:8": ("C-mv", _independent(8), 512, 24, 1, "group"),
}


# The pilot uses its own seeds (and reference-study replications 900+, unused by that study's 200),
# so the main Tier C series stay unseen until the protocol is frozen.
VARIANTS = {"synthetic": {"reps": REPS, "seed": SEED, "rep_offset": 0},
            "synthetic_pilot": {"reps": 5, "seed": 99, "rep_offset": 900}}


def build(name: str = "synthetic") -> pd.DataFrame:
    v = VARIANTS[name]
    reps, seed, offset = v["reps"], v["seed"], v["rep_offset"]
    rows, arrays = [], []
    # C-replication: the reference study's main horizon (H = 12, cells "...:h12") and its 48-step check
    # (same seeds; cells without suffix, also used for H3).
    for dgp in ("D6", "D7", "D8"):
        for n in (48, 200):
            for h, suffix in ((48, ""), (12, ":h12")):
                for r in range(offset, offset + reps):
                    y = ff_simulate(dgp, n, r, h)
                    rows.append(_row(f"rep:{dgp}:n{n}{suffix}", "C-replication", r, n, h, 1, "uni", len(arrays)))
                    arrays.append({"target": y[None], "future_cov": None, "past_cov": None})
    for cell, (family, gen, n, h, season, mode) in CELLS.items():
        for r in range(offset, offset + reps):
            out = gen(n, h, _rng(cell, r, seed))
            target, fcov, pcov = out if isinstance(out, tuple) else (out, None, None)
            rows.append(_row(cell, family, r, n, h, season, mode, len(arrays)))
            arrays.append({"target": np.asarray(target), "future_cov": fcov, "past_cov": pcov})

    tasks = pd.DataFrame(rows)
    data = pd.DataFrame({
        "array_id": range(len(arrays)),
        "target": [a["target"].astype(np.float32).tobytes() for a in arrays],
        "target_shape": [json.dumps(list(a["target"].shape)) for a in arrays],
        "future_cov": [None if a["future_cov"] is None else a["future_cov"].astype(np.float32).tobytes() for a in arrays],
        "past_cov": [None if a["past_cov"] is None else a["past_cov"].astype(np.float32).tobytes() for a in arrays],
    })
    dest = PROCESSED / name
    dest.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(dest / "tasks.parquet", index=False)
    data.to_parquet(dest / "arrays.parquet", index=False)
    return tasks


def _row(cell, family, rep, n, h, season, mode, array_id) -> dict:
    return {"task_id": f"C-{cell}-{rep:03d}", "cell": cell, "family": family, "rep": rep,
            "context_len": n, "horizon": h, "season": season, "mode": mode, "array_id": array_id}


def load_arrays(path) -> dict[int, dict]:
    """Inverse of the storage in `build`: array_id -> dict of numpy arrays (context + truth together)."""
    df = pd.read_parquet(path)
    out = {}
    for row in df.itertuples():
        shape = json.loads(row.target_shape)
        target = np.frombuffer(row.target, dtype=np.float32).reshape(shape)
        n_total = shape[1]
        fcov = None if row.future_cov is None else np.frombuffer(row.future_cov, dtype=np.float32).reshape(-1, n_total)
        pcov = None if row.past_cov is None else np.frombuffer(row.past_cov, dtype=np.float32).reshape(1, -1)
        out[row.array_id] = {"target": target, "future_cov": fcov, "past_cov": pcov}
    return out


if __name__ == "__main__":
    import sys

    t = build(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
    print(t.groupby(["family"]).size(), len(t))
