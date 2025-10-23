"""Eye and margin analysis."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import EyeConfig, SignalConfig


@dataclass(slots=True)
class EyeDiagram:
    time_grid: np.ndarray
    voltage_grid: np.ndarray
    density: np.ndarray


@dataclass(slots=True)
class BathtubCurve:
    offsets_ui: np.ndarray
    ber: np.ndarray


def compute_eye(samples: np.ndarray, config: SignalConfig, eye_cfg: EyeConfig) -> EyeDiagram:
    sps = config.samples_per_symbol
    ui = 1.0 / config.symbol_rate
    eye_samples = samples.reshape(-1, sps)
    if eye_cfg.ui_samples is None:
        ui_samples = sps
    else:
        ui_samples = eye_cfg.ui_samples
    time_grid = np.linspace(-0.5 * ui, 0.5 * ui, ui_samples)
    vmin, vmax = samples.min(), samples.max()
    voltage_grid = np.linspace(vmin, vmax, eye_cfg.bins)
    density = np.zeros((eye_cfg.bins, ui_samples), dtype=float)

    noise = np.random.normal(0.0, eye_cfg.noise_sigma, size=eye_samples.shape)
    eye_samples = np.clip(eye_samples + noise, vmin, vmax)

    for symbol_eye in eye_samples:
        resampled = np.interp(
            time_grid,
            np.linspace(-0.5 * ui, 0.5 * ui, sps),
            symbol_eye,
        )
        bin_indices = np.digitize(resampled, voltage_grid) - 1
        bin_indices = np.clip(bin_indices, 0, eye_cfg.bins - 1)
        for idx, bin_idx in enumerate(bin_indices):
            density[bin_idx, idx] += 1

    density /= density.max() or 1.0
    return EyeDiagram(time_grid=time_grid, voltage_grid=voltage_grid, density=density)


def compute_bathtub(samples: np.ndarray, config: SignalConfig) -> BathtubCurve:
    sps = config.samples_per_symbol
    ui = 1.0 / config.symbol_rate
    eye_samples = samples.reshape(-1, sps)
    crossing_points = np.linspace(-0.5 * ui, 0.5 * ui, 101)
    offsets_ui = crossing_points / ui
    ber = np.empty_like(offsets_ui)
    for idx, offset in enumerate(offsets_ui):
        sample_index = int(round((offset + 0.5) * (sps - 1)))
        sample_index = np.clip(sample_index, 0, sps - 1)
        column = eye_samples[:, sample_index]
        mean = float(np.mean(column))
        sigma = float(np.std(column)) + 1e-12
        decision = np.sign(mean) or 1.0
        argument = decision * mean / (math.sqrt(2.0) * sigma)
        penalties = 0.5 * math.erfc(argument)
        ber[idx] = penalties
    return BathtubCurve(offsets_ui=offsets_ui, ber=ber)
