"""Shared server-side motion postprocessing for Piper action chunks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _load_overlay_module(module_name: str, filename: str) -> Any:
    overlay = Path(__file__).resolve().parents[1] / "vendor" / "cogwam_overlay" / "flexpi_smoothing"
    path = overlay / filename
    if not path.is_file():
        raise FileNotFoundError(f"motion postprocessor module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load motion postprocessor module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MotionPostprocessor:
    """Apply optional SpeedAdapter and TemporalSmoother to one action chunk."""

    def __init__(self, config: Mapping[str, Any], *, default_horizon: int) -> None:
        self.speed_adapter = None
        self.temporal_smoother = None
        self.config = dict(config)
        overlay_enabled = bool(
            self.config.get("speed_adapter_enabled", False)
            or self.config.get("temporal_smoother_enabled", False)
        )
        if not overlay_enabled:
            return

        if bool(self.config.get("speed_adapter_enabled", False)):
            module = _load_overlay_module("piper_speed_adapter", "speed_adapter.py")
            self.speed_adapter = module.SpeedAdapter(module.SpeedAdapterConfig(
                mode=str(self.config.get("speed_adapter_mode", "heuristic")),
                factor_lo=float(self.config.get("speed_factor_lo", 0.75)),
                factor_hi=float(self.config.get("speed_factor_hi", 1.25)),
                heuristic_alpha=float(self.config.get("speed_heuristic_alpha", 1.0)),
                heuristic_v_ref=float(self.config.get("speed_heuristic_v_ref", 0.05)),
                smooth_window=int(self.config.get("speed_smooth_window", 5)),
            ))

        if bool(self.config.get("temporal_smoother_enabled", False)):
            module = _load_overlay_module("piper_temporal_smoother", "temporal_smoother.py")
            self.temporal_smoother = module.TemporalSmoother(module.SmootherConfig(
                dt_ref=float(self.config.get("temporal_smoother_dt_ref", self.config.get("action_dt", 1 / 30))),
                dt_min=float(self.config.get("temporal_smoother_dt_min", 1 / 60)),
                dt_max=float(self.config.get("temporal_smoother_dt_max", 1 / 15)),
                lambda_acc=float(self.config.get("temporal_smoother_lambda_acc", 10.0)),
                lambda_time=float(self.config.get("temporal_smoother_lambda_time", 1.0)),
                horizon=int(self.config.get("temporal_smoother_horizon", default_horizon)),
                stride=int(self.config.get("temporal_smoother_stride", max(1, default_horizon // 2))),
                optim_dims=tuple(self.config.get(
                    "temporal_smoother_optim_dims",
                    (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12),
                )),
            ))

    @property
    def enabled(self) -> bool:
        return self.speed_adapter is not None or self.temporal_smoother is not None

    def process(self, actions: Any, mode: str = "smooth") -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] < 14 or len(values) == 0:
            raise ValueError(f"action chunk must have shape [T, >=14], got {values.shape}")
        values = np.ascontiguousarray(values[:, :14])
        if mode == "interp":
            return values
        if mode != "smooth":
            raise ValueError(f"unsupported motion_mode: {mode}")
        factors = self.speed_adapter.factors(values) if self.speed_adapter is not None else None
        if self.temporal_smoother is not None:
            values = self.temporal_smoother.smooth(values, factors)
        return np.asarray(values, dtype=np.float32)
