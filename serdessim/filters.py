"""Behavioral equalizer models used by the SerDes simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import signal


@dataclass
class FeedForwardEqualizer:
    """Finite impulse response filter representing an FFE."""

    taps: Iterable[float]
    gain: float = 1.0

    def impulse_response(self) -> np.ndarray:
        taps = np.asarray(list(self.taps), dtype=float)
        return self.gain * taps

    def filter(self, samples: np.ndarray) -> np.ndarray:
        """Apply the equalizer to a waveform."""

        taps = self.impulse_response()
        return np.convolve(samples, taps, mode="same")


@dataclass
class ContinuousTimeLinearEqualizer:
    """Analog equalizer modeled with zero/pole pairs."""

    zeros: Iterable[complex]
    poles: Iterable[complex]
    dc_gain: float = 1.0

    def frequency_response(self, freqs: np.ndarray) -> np.ndarray:
        zeros = np.asarray(list(self.zeros), dtype=complex)
        poles = np.asarray(list(self.poles), dtype=complex)
        system = signal.ZerosPolesGain(zeros, poles, self.dc_gain)
        w = 2 * np.pi * freqs
        _, h = signal.freqresp(system, w)
        return h

    def impulse_response(self, sample_rate: float, num_points: int = 1024) -> np.ndarray:
        system = signal.ZerosPolesGain(list(self.zeros), list(self.poles), self.dc_gain)
        sys = signal.Zpk2Tf(system.zeros, system.poles, system.gain)
        t, y = signal.impulse(sys, T=np.arange(num_points) / sample_rate)
        return y

    def filter(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        response = self.impulse_response(sample_rate, num_points=len(samples))
        return np.convolve(samples, response, mode="same")
