"""Reaction wheel set modeled along body axes."""

from __future__ import annotations

import numpy as np

from .config import WheelConfig


class ReactionWheelSet:
    """Per-axis torque and momentum saturation.

    Torque sign convention: `apply` returns the torque exerted ON the body;
    the wheel momentum state evolves as ḣ_w = −T_body (handled in dynamics).
    """

    def __init__(self, cfg: WheelConfig):
        self.cfg = cfg

    def apply(self, torque_cmd: np.ndarray, h_wheel: np.ndarray) -> np.ndarray:
        if self.cfg.ideal:
            return np.asarray(torque_cmd, dtype=float)
        t = np.clip(torque_cmd, -self.cfg.max_torque, self.cfg.max_torque)
        # Momentum saturation: a wheel at its momentum limit cannot keep
        # accelerating. Wheel momentum rate is −T, so torque that would push
        # |h_w| further past the limit is zeroed on that axis.
        h_dot = -t
        saturated = (np.abs(h_wheel) >= self.cfg.max_momentum) & (
            np.sign(h_dot) == np.sign(h_wheel)
        )
        t[saturated] = 0.0
        return t
