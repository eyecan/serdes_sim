"""Channel modelling utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ChannelConfig
from .signal import Waveform


@dataclass(slots=True)
class LinearChannel:
    config: ChannelConfig
    samples_per_symbol: int

    def impulse_response(self) -> np.ndarray:
        return self.config.resolve_impulse_response(self.samples_per_symbol)

    def apply(self, waveform: Waveform) -> Waveform:
        h = self.impulse_response()
        convolved = np.convolve(waveform.samples, h, mode="same")
        return Waveform(
            time=waveform.time,
            samples=convolved,
            sample_rate=waveform.sample_rate,
            samples_per_symbol=waveform.samples_per_symbol,
        )

    def step_response(self) -> np.ndarray:
        h = self.impulse_response()
        return np.cumsum(h)
