"""Equalization models for the SerDes pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import RxEqualizerConfig, SignalConfig, TxEqualizerConfig
from .signal import Waveform


@dataclass(slots=True)
class TxFeedForwardEqualizer:
    config: TxEqualizerConfig

    def apply(self, waveform: Waveform) -> Waveform:
        taps = self.config.impulse_response()
        filtered = np.convolve(waveform.samples, taps, mode="same")
        return Waveform(
            time=waveform.time,
            samples=filtered,
            sample_rate=waveform.sample_rate,
            samples_per_symbol=waveform.samples_per_symbol,
        )


def _ctle_impulse(config: RxEqualizerConfig, signal: SignalConfig) -> np.ndarray:
    zero = 2 * np.pi * config.ctle_zero_hz
    pole = 2 * np.pi * config.ctle_pole_hz
    dt = 1.0 / signal.sample_rate
    alpha = np.exp(-pole * dt)
    beta = np.exp(-zero * dt)
    gain = config.ctle_dc_gain
    impulse = np.zeros(signal.samples_per_symbol * 8, dtype=float)
    impulse[0] = gain * (1 - alpha)
    for n in range(1, len(impulse)):
        impulse[n] = beta ** n - alpha ** n
    impulse *= gain
    return impulse


@dataclass(slots=True)
class RxContinuousTimeLinearEqualizer:
    config: RxEqualizerConfig
    signal_config: SignalConfig

    def apply(self, waveform: Waveform) -> Waveform:
        impulse = _ctle_impulse(self.config, self.signal_config)
        filtered = np.convolve(waveform.samples, impulse, mode="same")
        return Waveform(
            time=waveform.time,
            samples=filtered,
            sample_rate=waveform.sample_rate,
            samples_per_symbol=waveform.samples_per_symbol,
        )


@dataclass(slots=True)
class DecisionFeedbackEqualizer:
    taps: Sequence[float]

    def apply(self, symbols: np.ndarray) -> np.ndarray:
        taps = np.asarray(self.taps, dtype=float)
        corrected = np.copy(symbols)
        for idx in range(len(symbols)):
            isi = 0.0
            for tap_idx, tap in enumerate(taps, start=1):
                if idx - tap_idx >= 0:
                    isi += tap * corrected[idx - tap_idx]
            corrected[idx] -= isi
        return corrected
