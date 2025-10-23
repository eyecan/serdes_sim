"""Behavioral equalizer models used by the SerDes simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:  # pragma: no cover - SciPy is optional at runtime
    from scipy import signal as _scipy_signal  # type: ignore
except Exception:  # pragma: no cover - handled by fallbacks
    _scipy_signal = None


def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast convolution using the FFT with NumPy only."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size + b.size - 1
    n_fft = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(a, n_fft) * np.fft.rfft(b, n_fft)
    result = np.fft.irfft(spec, n_fft)[:n]
    return result


@dataclass
class FeedForwardEqualizer:
    """Finite impulse response filter representing an FFE."""

    taps: Iterable[float]
    gain: float = 1.0

    def impulse_response(self) -> np.ndarray:
        taps = np.asarray(list(self.taps), dtype=float)
        if taps.size == 0:
            raise ValueError("FFE requires at least one tap")
        return self.gain * taps

    def filter(self, samples: np.ndarray) -> np.ndarray:
        """Apply the equalizer to a waveform."""

        taps = self.impulse_response()
        return np.convolve(samples, taps, mode="same")


def _prod_terms(values: np.ndarray, jw: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.ones_like(jw)
    return np.prod(jw[:, None] - values[None, :], axis=1)


@dataclass
class ContinuousTimeLinearEqualizer:
    """Analog equalizer modeled with zero/pole pairs."""

    zeros: Iterable[complex]
    poles: Iterable[complex]
    dc_gain: float = 1.0

    def frequency_response(self, freqs: np.ndarray) -> np.ndarray:
        zeros = np.asarray(list(self.zeros), dtype=complex)
        poles = np.asarray(list(self.poles), dtype=complex)
        jw = 1j * 2 * np.pi * np.asarray(freqs, dtype=float)
        numerator = _prod_terms(zeros, jw)
        denominator = _prod_terms(poles, jw)
        response = self.dc_gain * numerator / denominator
        return response

    def _impulse_via_fft(self, sample_rate: float, num_points: int) -> np.ndarray:
        freqs = np.fft.rfftfreq(num_points, d=1.0 / sample_rate)
        spectrum_half = self.frequency_response(freqs)
        spectrum = np.empty(num_points, dtype=complex)
        spectrum[: spectrum_half.size] = spectrum_half
        if num_points % 2 == 0:
            mirror = np.conj(spectrum_half[1:-1][::-1])
        else:
            mirror = np.conj(spectrum_half[1:][::-1])
        spectrum[spectrum_half.size :] = mirror
        impulse = np.fft.ifft(spectrum)
        return impulse.real

    def impulse_response(self, sample_rate: float, num_points: int = 1024) -> np.ndarray:
        if num_points <= 0:
            raise ValueError("num_points must be positive")
        if _scipy_signal is not None:
            system = _scipy_signal.ZerosPolesGain(
                list(self.zeros), list(self.poles), self.dc_gain
            )
            num, den = _scipy_signal.zpk2tf(
                system.zeros, system.poles, system.gain
            )
            t = np.arange(num_points) / sample_rate
            _, y = _scipy_signal.impulse((num, den), T=t)
            return np.asarray(y, dtype=float)
        return self._impulse_via_fft(sample_rate, num_points)

    def filter(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        response = self.impulse_response(sample_rate, num_points=len(samples))
        conv = _fft_convolve(samples, response)
        start = (conv.size - samples.size) // 2
        end = start + samples.size
        return conv[start:end]
