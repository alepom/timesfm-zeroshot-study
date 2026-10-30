"""Uniform interface over the forecasting models used in the study.

Every adapter takes a list of `Task` items and returns, for each item, an array of quantile
forecasts with shape (num_targets, horizon, 9) for the levels in `tfmstudy.QUANTILES`.
Model weights are downloaded from Hugging Face at a pinned revision when one is given.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from typing import Any

import numpy as np

from . import QUANTILES


@dataclasses.dataclass
class Task:
    """One forecasting problem.

    target: (num_targets, context_len) array.
    past_cov: (num_past_cov, context_len) array or None.
    future_cov: (num_future_cov, context_len + horizon) array or None.
    """

    target: np.ndarray
    horizon: int
    past_cov: np.ndarray | None = None
    future_cov: np.ndarray | None = None
    task_id: str = ""

    def __post_init__(self):
        self.target = np.atleast_2d(np.asarray(self.target, dtype=np.float32))
        if self.past_cov is not None:
            self.past_cov = np.atleast_2d(np.asarray(self.past_cov, dtype=np.float32))
        if self.future_cov is not None:
            self.future_cov = np.atleast_2d(np.asarray(self.future_cov, dtype=np.float32))


class Adapter:
    name: str = "base"
    hf_repo: str | None = None
    supports_covariates: bool = False
    supports_multivariate: bool = False

    def __init__(self, device: str = "cuda", revision: str | None = None, **options: Any):
        self.device = device
        self.revision = revision
        self.options = options
        self.load_seconds: float | None = None

    def load(self) -> None:
        t0 = time.perf_counter()
        self._load()
        self.load_seconds = time.perf_counter() - t0

    def _load(self) -> None:
        raise NotImplementedError

    def predict(self, tasks: list[Task]) -> list[np.ndarray]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "name": self.name,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "options": self.options,
            "load_seconds": self.load_seconds,
        }


def _check_quantiles(q: np.ndarray, num_targets: int, horizon: int) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    if q.ndim == 2:
        q = q[None]
    expected = (num_targets, horizon, len(QUANTILES))
    if q.shape != expected:
        raise ValueError(f"quantile array has shape {q.shape}, expected {expected}")
    return q


class TimesFM3(Adapter):
    """TimesFM 3.0 (google/timesfm-3.0-pytorch).

    options:
      mode: "multivariate" (all targets and covariates in one forward pass) or "univariate"
        (each target alone, covariates ignored).
      use_symmetric_averaging, make_positive, sort_quantiles, use_znorm, padding_mode:
        passed to `predict_batch`. Defaults are the package defaults of TimesFM3Forecaster,
        which differ from TimesFM3Evaluator (the benchmark runner).
      model_overrides: dict of `ModelConfig` fields (e.g. use_variate_attention=False) for ablations.
    """

    name = "timesfm-3.0"
    hf_repo = "google/timesfm-3.0-pytorch"
    supports_covariates = True
    supports_multivariate = True

    PREDICT_DEFAULTS = dict(
        use_symmetric_averaging=False,
        make_positive=False,
        sort_quantiles=True,
        use_znorm=False,
        padding_mode="none",
    )

    def _load(self) -> None:
        from timesfm3 import ModelConfig, TimesFM3Forecaster

        config = ModelConfig(
            checkpoint_path=self.hf_repo,
            device=self.device,
            revision=self.revision,
            per_core_batch_size=int(self.options.get("batch_size", 32)),
        )
        self.model = TimesFM3Forecaster(config)

    @contextlib.contextmanager
    def _overrides(self):
        """Temporarily sets attributes of the loaded TimesFM3Torch module (ablations).

        A Hub load replaces ModelConfig flags with the checkpoint's config.json
        (timesfm3_forecaster.py:398), so ablations cannot go through ModelConfig. They are set only
        for the duration of a call because the loaded module is shared across configurations.

        A flag is set on every submodule that has it: `use_variate_attention` is read in the forward
        pass of each MixingTransformer layer (transformer.py:516), not on the top module, where it is only
        stored (model.py:132). Setting it on the top module alone, as versions before 0.5.5 did, left
        variate attention on.
        """
        overrides = self.options.get("model_overrides", {})
        saved = []
        for field in overrides:
            holders = [m for m in self.model.model.modules() if hasattr(m, field)]
            if not holders:
                raise AttributeError(f"no submodule of TimesFM3Torch has attribute {field!r}")
            saved += [(m, field, getattr(m, field)) for m in holders]
        try:
            for m, field, _ in saved:
                setattr(m, field, overrides[field])
            yield
        finally:
            for m, field, value in saved:
                setattr(m, field, value)

    def predict(self, tasks: list[Task]) -> list[np.ndarray]:
        with self._overrides():
            return self._predict(tasks)

    def _predict(self, tasks: list[Task]) -> list[np.ndarray]:
        flags = {k: self.options.get(k, v) for k, v in self.PREDICT_DEFAULTS.items()}
        mode = self.options.get("mode", "multivariate")
        out: list[np.ndarray] = []
        # Only tasks with the same horizon and variate layout share a batch.
        def key(t: Task) -> tuple:
            shape = lambda a: None if a is None else a.shape[0]
            return (t.horizon, t.target.shape[0], shape(t.past_cov), shape(t.future_cov))

        for group_key in sorted({key(t) for t in tasks}, key=str):
            group = [t for t in tasks if key(t) == group_key]
            horizon = group_key[0]
            if mode == "univariate":
                contexts = [row for t in group for row in t.target]
                results = list(self.model.predict_batch(
                    contexts=contexts, horizon=horizon, return_quantiles=True, **flags))
                quantiles = iter(np.asarray(r.quantiles) for r in results)
                for t in group:
                    q = np.stack([next(quantiles) for _ in range(t.target.shape[0])])
                    out.append((t, _check_quantiles(q, t.target.shape[0], horizon)))
            else:
                results = list(self.model.predict_batch(
                    contexts=[t.target for t in group],
                    horizon=horizon,
                    past_only_covariates=[t.past_cov for t in group],
                    past_future_covariates=[t.future_cov for t in group],
                    return_quantiles=True,
                    **flags,
                ))
                for t, r in zip(group, results):
                    out.append((t, _check_quantiles(r.quantiles, t.target.shape[0], horizon)))
        order = {id(t): i for i, t in enumerate(tasks)}
        out.sort(key=lambda pair: order[id(pair[0])])
        return [q for _, q in out]


class TimesFM25(Adapter):
    """TimesFM 2.5 (google/timesfm-2.5-200m-pytorch), univariate only.

    The quantile output of 2.5 has 10 channels: the mean followed by the deciles 0.1..0.9.
    """

    name = "timesfm-2.5"
    hf_repo = "google/timesfm-2.5-200m-pytorch"

    MAX_CONTEXT = 15360  # same cap as TimesFM-3 (protocol section 2)

    def _load(self) -> None:
        import timesfm

        self._timesfm = timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.hf_repo, revision=self.revision)
        self._compiled_for: tuple[int, int] | None = None

    def _compile(self, max_context: int, max_horizon: int) -> None:
        # compile() pads every input to max_context, so it is compiled per length bucket rather than
        # once at the maximum (a 15,360 cap made each forecast take ~6 s on a T4 in the smoke test).
        if self._compiled_for == (max_context, max_horizon):
            return
        self.model.compile(self._timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            normalize_inputs=True,
            per_core_batch_size=int(self.options.get("batch_size", 32)),
            use_continuous_quantile_head=True,
            force_flip_invariance=bool(self.options.get("force_flip_invariance", True)),
            infer_is_positive=bool(self.options.get("infer_is_positive", True)),
            fix_quantile_crossing=True,
        ))
        self._compiled_for = (max_context, max_horizon)

    def _bucket(self, task: Task) -> tuple[int, int, int]:
        context = min(self.MAX_CONTEXT, max(1024, -(-task.target.shape[1] // 1024) * 1024))
        max_horizon = min(1024, max(128, -(-task.horizon // 128) * 128))
        return context, max_horizon, task.horizon

    def predict(self, tasks: list[Task]) -> list[np.ndarray]:
        out: list[np.ndarray | None] = [None] * len(tasks)
        groups: dict[tuple[int, int, int], list[int]] = {}
        for i, t in enumerate(tasks):
            groups.setdefault(self._bucket(t), []).append(i)
        for (context, max_horizon, horizon), indices in sorted(groups.items()):
            self._compile(context, max_horizon)
            inputs = [row.astype(np.float64) for i in indices for row in tasks[i].target]
            _, qf = self.model.forecast(horizon=horizon, inputs=inputs)
            qf = np.asarray(qf)[:, :, 1:10]
            start = 0
            for i in indices:
                n = tasks[i].target.shape[0]
                out[i] = _check_quantiles(qf[start:start + n], n, horizon)
                start += n
        return out


class Chronos2(Adapter):
    """Chronos-2 (amazon/chronos-2) through its pandas API."""

    name = "chronos-2"
    hf_repo = "amazon/chronos-2"
    supports_covariates = True
    supports_multivariate = True

    def _load(self) -> None:
        from chronos import Chronos2Pipeline

        self.model = Chronos2Pipeline.from_pretrained(self.hf_repo, device_map=self.device, revision=self.revision)

    def predict(self, tasks: list[Task]) -> list[np.ndarray]:
        import pandas as pd

        out = []
        for t in tasks:
            n_ctx = t.target.shape[1]
            ts = pd.date_range("2000-01-01", periods=n_ctx + t.horizon, freq="h")
            targets = [f"y{i}" for i in range(t.target.shape[0])]
            ctx = {"id": "s", "timestamp": ts[:n_ctx]}
            ctx.update({name: t.target[i] for i, name in enumerate(targets)})
            future = None
            if t.past_cov is not None:
                ctx.update({f"p{i}": row for i, row in enumerate(t.past_cov)})
            if t.future_cov is not None:
                ctx.update({f"f{i}": row[:n_ctx] for i, row in enumerate(t.future_cov)})
                future = pd.DataFrame({"id": "s", "timestamp": ts[n_ctx:],
                                       **{f"f{i}": row[n_ctx:] for i, row in enumerate(t.future_cov)}})
            pred = self.model.predict_df(
                pd.DataFrame(ctx), future_df=future, prediction_length=t.horizon,
                quantile_levels=list(QUANTILES), id_column="id", timestamp_column="timestamp",
                target=targets if len(targets) > 1 else targets[0],
            )
            q = np.stack([
                np.stack([pred.loc[pred["target_name"] == name, str(level)].to_numpy() for level in QUANTILES], axis=-1)
                if "target_name" in pred.columns
                else np.stack([pred[str(level)].to_numpy() for level in QUANTILES], axis=-1)
                for name in targets
            ])
            out.append(_check_quantiles(q, t.target.shape[0], t.horizon))
        return out


class TiRex2(Adapter):
    """TiRex-2 (NX-AI/TiRex-2) and its decontaminated variants (set hf_repo via options)."""

    name = "tirex-2"
    hf_repo = "NX-AI/TiRex-2"
    MAX_HORIZON = 320  # tirex2 truncates longer prediction lengths (warning in tirex2/model/tirex2.py)
    supports_covariates = True
    supports_multivariate = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "hf_repo" in self.options:
            self.hf_repo = self.options["hf_repo"]
            self.name = "tirex-2:" + self.hf_repo.split("/")[-1]

    def _load(self) -> None:
        import torch
        import tirex2.model.component.flashrnn_slstm as slstm
        from tirex2 import load_model

        # FlashRNN's fused sLSTM CUDA kernels fail on pre-Ampere GPUs such as the T4 ("no kernel
        # image is available"); the package's pure-PyTorch "vanilla" sLSTM backend runs on any device.
        backend = self.options.get("slstm_backend", "auto")
        if self.device.startswith("cuda") and (
            backend == "vanilla" or (backend == "auto" and torch.cuda.get_device_capability()[0] < 8)
        ):
            slstm._flashrnn_backend = lambda device: "vanilla"
            self.slstm_backend = "vanilla"
        else:
            self.slstm_backend = "cuda" if self.device.startswith("cuda") else "vanilla"
        hf_kwargs = {"revision": self.revision} if self.revision else None
        self.model = load_model(self.hf_repo, device=self.device, hf_kwargs=hf_kwargs)

    def describe(self) -> dict:
        return {**super().describe(), "slstm_backend": getattr(self, "slstm_backend", None)}

    def predict(self, tasks: list[Task]) -> list[np.ndarray]:
        import torch
        from tirex2 import TimeseriesType

        out = []
        for t in tasks:
            if t.horizon > self.MAX_HORIZON:
                raise ValueError(f"TiRex-2 supports horizons up to {self.MAX_HORIZON}, got {t.horizon}")
            tensor = lambda a: None if a is None else torch.from_numpy(np.array(a, copy=True))
            ts = TimeseriesType(target=tensor(t.target), past_covariates=tensor(t.past_cov),
                                future_covariates=tensor(t.future_cov))
            raw = self.model.forecast([ts], prediction_length=t.horizon, output_type="numpy")
            self.last_raw = raw
            q = _tirex_quantiles(raw)
            out.append(_check_quantiles(q, t.target.shape[0], t.horizon))
        return out


def _tirex_quantiles(raw: Any) -> np.ndarray:
    """Extracts (targets, horizon, 9) quantiles from TiRex-2 output; layout checked in the smoke test."""
    quantiles = raw[0] if isinstance(raw, (tuple, list)) else raw
    quantiles = quantiles[0] if isinstance(quantiles, (list, tuple)) else quantiles
    q = np.asarray(quantiles)
    if q.ndim == 3 and q.shape[1] == len(QUANTILES) and q.shape[2] != len(QUANTILES):
        q = np.transpose(q, (0, 2, 1))
    return q


REGISTRY = {cls.name: cls for cls in (TimesFM3, TimesFM25, Chronos2, TiRex2)}


def make(name: str, **kwargs: Any) -> Adapter:
    return REGISTRY[name](**kwargs)
