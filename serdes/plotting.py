"""Visualization helpers for the SerDes toolkit."""
from __future__ import annotations

import matplotlib.pyplot as plt

from .analysis import BathtubCurve, EyeDiagram


def plot_waveform(time: np.ndarray, samples: np.ndarray, *, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    ax.plot(time * 1e9, samples, linewidth=0.8)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title("Waveform")
    return ax


def plot_eye(eye: EyeDiagram, *, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    extent = [
        eye.time_grid[0] * 1e12,
        eye.time_grid[-1] * 1e12,
        eye.voltage_grid[0],
        eye.voltage_grid[-1],
    ]
    ax.imshow(
        eye.density,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_xlabel("Time offset (ps)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title("Eye Diagram")
    return ax


def plot_bathtub(bathtub: BathtubCurve, *, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    ax.semilogy(bathtub.offsets_ui, bathtub.ber)
    ax.set_xlabel("Timing offset (UI)")
    ax.set_ylabel("BER estimate")
    ax.grid(True, which="both", linestyle=":")
    ax.set_title("Bathtub Curve")
    return ax
