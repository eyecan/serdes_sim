"""Signal generation utilities for SerDes simulations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple

import numpy as np


class SignalType(str, Enum):
    """Supported line encoding formats."""

    NRZ = "nrz"
    PAM4 = "pam4"

    def levels(self) -> np.ndarray:
        """Return the normalized symbol levels for the signal type."""

        if self is SignalType.NRZ:
            return np.array([-1.0, 1.0])
        return np.array([-3.0, -1.0, 1.0, 3.0]) / 3.0


@dataclass
class SignalSpec:
    """Specification for a waveform to be synthesized."""

    type: SignalType
    symbol_rate: float
    samples_per_symbol: int
    amplitude: float = 1.0
    pattern: Iterable[int] | None = None

    def normalized_pattern(self, length: int) -> np.ndarray:
        """Return a normalized symbol pattern of a given length."""

        levels = self.type.levels()
        if self.pattern is None:
            rng = np.random.default_rng()
            symbols = rng.choice(len(levels), size=length)
        else:
            pattern = np.asarray(list(self.pattern), dtype=int)
            if pattern.size == 0:
                raise ValueError("Custom pattern must not be empty")
            symbols = np.resize(pattern, length)
        return levels[symbols] * self.amplitude


def generate_waveform(spec: SignalSpec, num_symbols: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a waveform for the provided :class:`SignalSpec`.

    Returns a tuple ``(time, samples)`` where ``time`` is a numpy array in
    seconds and ``samples`` contains the waveform values.
    """

    if num_symbols <= 0:
        raise ValueError("num_symbols must be positive")

    dt = 1.0 / (spec.symbol_rate * spec.samples_per_symbol)
    total_samples = num_symbols * spec.samples_per_symbol
    time = np.arange(total_samples) * dt
    symbols = spec.normalized_pattern(num_symbols)
    samples = np.repeat(symbols, spec.samples_per_symbol)
    return time, samples


def prbs7(length: int, seed: int = 0x7F) -> np.ndarray:
    """Generate a PRBS7 sequence of the requested length.

    Parameters
    ----------
    length:
        Number of bits to emit from the pseudo-random binary sequence.
    seed:
        Seven-bit non-zero seed for the LFSR. Defaults to ``0x7F``.

    Returns
    -------
    numpy.ndarray
        Array of ``0``/``1`` integers representing the PRBS7 pattern.
    """

    if length <= 0:
        raise ValueError("PRBS length must be positive")

    state = seed & 0x7F
    if state == 0:
        raise ValueError("PRBS seed must be non-zero")

    sequence = np.empty(length, dtype=int)
    for idx in range(length):
        output_bit = (state >> 6) & 0x1
        sequence[idx] = output_bit
        feedback = ((state >> 6) ^ (state >> 5)) & 0x1
        state = ((state << 1) & 0x7F) | feedback

    return sequence
