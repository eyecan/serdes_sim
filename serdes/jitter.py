"""Jitter injection utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import JitterConfig
from .signal import Waveform


@dataclass(slots=True)
class JitterInjector:
    config: JitterConfig

    def _random_jitter(self, num_samples: int, ui: float) -> np.ndarray:
        sigma = self.config.rj_sigma_ui * ui
        return np.random.normal(0.0, sigma, size=num_samples)

    def _sinusoidal_jitter(self, t: np.ndarray, ui: float) -> np.ndarray:
        amp = self.config.sj_amplitude_ui * ui
        return amp * np.sin(2 * np.pi * self.config.sj_frequency * t)

    def apply(self, waveform: Waveform) -> Waveform:
        if not self.config.enable:
            return waveform

        ui = waveform.samples_per_symbol / waveform.sample_rate
        t = waveform.time
        jitter = self._random_jitter(len(t), ui) + self._sinusoidal_jitter(t, ui)
        resampled = np.interp(t + jitter, t, waveform.samples, left=0.0, right=0.0)
        return Waveform(
            time=t,
            samples=resampled,
            sample_rate=waveform.sample_rate,
            samples_per_symbol=waveform.samples_per_symbol,
        )
