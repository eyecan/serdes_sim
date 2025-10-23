"""Signal synthesis helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import SignalConfig


_PRBS_TAPS = {
    "prbs7": (7, (7, 6)),
    "prbs9": (9, (9, 5)),
    "prbs11": (11, (11, 9)),
    "prbs13": (13, (13, 12, 11, 8)),
    "prbs15": (15, (15, 14)),
}


@dataclass(slots=True)
class Waveform:
    time: np.ndarray
    samples: np.ndarray
    sample_rate: float
    samples_per_symbol: int

    @property
    def ui(self) -> float:
        return self.samples_per_symbol / self.sample_rate


def _prbs_sequence(order: int, taps: Iterable[int], length: int, seed: int | None) -> np.ndarray:
    state = (1 << order) - 1 if seed is None else seed & ((1 << order) - 1)
    if state == 0:
        state = 1
    seq = np.empty(length, dtype=np.uint8)
    mask = (1 << order) - 1
    for i in range(length):
        feedback = 0
        for tap in taps:
            feedback ^= (state >> (order - tap)) & 1
        seq[i] = state & 1
        state = ((state >> 1) | (feedback << (order - 1))) & mask
    return seq


def generate_symbols(config: SignalConfig, num_symbols: int) -> np.ndarray:
    order, taps = _PRBS_TAPS[config.pattern]
    required_bits = num_symbols if config.modulation == "nrz" else num_symbols * 2
    pattern = _prbs_sequence(order, taps, required_bits, config.seed)
    if config.modulation == "nrz":
        levels = 2 * pattern - 1
        return config.amplitude * levels

    dibits = pattern.reshape(-1, 2)
    gray = (dibits[:, 0] << 1) | dibits[:, 1]
    mapping = np.array([-3, -1, +3, +1], dtype=float)
    levels = mapping[gray]
    levels *= config.amplitude / np.max(np.abs(mapping))
    return levels


def pulse_shaping(symbols: np.ndarray, config: SignalConfig) -> Waveform:
    upsample = np.repeat(symbols, config.samples_per_symbol)
    sr = config.sample_rate
    t = np.arange(len(upsample)) / sr
    return Waveform(
        time=t,
        samples=upsample.astype(float),
        sample_rate=sr,
        samples_per_symbol=config.samples_per_symbol,
    )
