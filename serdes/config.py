"""Configuration dataclasses for the SerDes simulation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np


@dataclass(slots=True)
class SignalConfig:
    """Configuration for the transmitted symbol stream."""

    modulation: Literal["nrz", "pam4"] = "pam4"
    symbol_rate: float = 56e9
    samples_per_symbol: int = 32
    amplitude: float = 1.0
    pattern: Literal["prbs7", "prbs9", "prbs11", "prbs13", "prbs15"] = "prbs13"
    seed: int | None = None

    @property
    def sample_rate(self) -> float:
        return self.symbol_rate * self.samples_per_symbol


@dataclass(slots=True)
class TxEqualizerConfig:
    """Feed-forward equalizer configuration."""

    taps: Sequence[float] = (0.1, 1.0, -0.1)

    def impulse_response(self) -> np.ndarray:
        taps = np.asarray(self.taps, dtype=float)
        taps /= np.sum(np.abs(taps)) or 1.0
        return taps


@dataclass(slots=True)
class ChannelConfig:
    """Channel model configuration."""

    impulse_response: np.ndarray | None = None
    length: int = 512
    ui: float = 1.0
    alpha: float = 0.12
    post_cursor: int = 12

    def resolve_impulse_response(self, samples_per_symbol: int) -> np.ndarray:
        if self.impulse_response is not None:
            return np.asarray(self.impulse_response, dtype=float)

        n = self.length
        taps = np.zeros(n, dtype=float)
        taps[0] = 1.0
        decay = np.exp(-self.alpha * np.arange(1, n))
        taps[1:] = 0.5 * decay
        post = min(self.post_cursor * samples_per_symbol, n - 1)
        taps[: post + 1] *= np.hanning(post + 1)
        return taps


@dataclass(slots=True)
class RxEqualizerConfig:
    """Receiver-side equalization."""

    ctle_zero_hz: float = 8e9
    ctle_pole_hz: float = 40e9
    ctle_dc_gain: float = 1.0
    dfe_taps: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0))


@dataclass(slots=True)
class JitterConfig:
    """Clock jitter configuration."""

    rj_sigma_ui: float = 0.01
    sj_amplitude_ui: float = 0.02
    sj_frequency: float = 5e6
    enable: bool = True


@dataclass(slots=True)
class CdrConfig:
    """Clock-data recovery configuration."""

    loop_bandwidth_hz: float = 20e6
    damping: float = 0.707
    gain: float = 0.05


@dataclass(slots=True)
class EyeConfig:
    """Visualization and measurement configuration for eye analysis."""

    ui_samples: int | None = None
    bins: int = 256
    noise_sigma: float = 0.005
