"""Plotting helpers for visualizing simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:  # pragma: no cover - Matplotlib is optional
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - keep importable without Matplotlib
    plt = None  # type: ignore

from .channel import SimulationArtifacts, SimulationResult


def _require_matplotlib() -> None:
    if plt is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Matplotlib is required for plotting but is not installed."
        )


@dataclass
class PlotConfig:
    """Configuration values for plot generation."""

    dpi: int = 120
    figsize: tuple[float, float] = (6.0, 4.0)
    output_dir: Optional[Path] = None


class Plotter:
    """Create the requested plots from simulation output."""

    def __init__(self, config: PlotConfig | None = None) -> None:
        self.config = config or PlotConfig()

    def _prepare_figure(self, title: str):
        _require_matplotlib()
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        fig.suptitle(title)
        return fig

    def eye_density(self, artifacts: SimulationArtifacts):
        fig = self._prepare_figure("Eye Density")
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(
            artifacts.eye_density,
            aspect="auto",
            origin="lower",
            extent=[
                artifacts.eye_time[0],
                artifacts.eye_time[-1],
                0,
                len(artifacts.amplitude_pdf),
            ],
            cmap="viridis",
        )
        fig.colorbar(im, ax=ax, label="Density")
        ax.set_xlabel("UI")
        ax.set_ylabel("Amplitude Bins")
        self._maybe_save(fig, "eye_density.png")
        return fig

    def ber_plots(self, result: SimulationResult):
        fig = self._prepare_figure("BER Plots")
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(result.ber_curve[:, 0], result.ber_curve[:, 1], label="BER Curve")
        ax.plot(result.timing_bathtub[:, 0], result.timing_bathtub[:, 1], label="Timing Bathtub")
        ax.set_yscale("log")
        ax.set_xlabel("UI")
        ax.set_ylabel("BER")
        ax.legend()
        self._maybe_save(fig, "ber_plots.png")
        return fig

    def pdf_plots(self, artifacts: SimulationArtifacts):
        fig = self._prepare_figure("PDF Plots")
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(artifacts.eye_time, artifacts.pdf)
        ax1.set_title("Timing PDF")
        ax1.set_xlabel("UI")
        ax1.set_ylabel("Probability")
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(np.linspace(-1, 1, len(artifacts.amplitude_pdf)), artifacts.amplitude_pdf)
        ax2.set_title("Amplitude PDF")
        ax2.set_xlabel("Amplitude")
        ax2.set_ylabel("Probability")
        fig.tight_layout()
        self._maybe_save(fig, "pdf_plots.png")
        return fig

    def _maybe_save(self, figure, name: str) -> None:
        if self.config.output_dir is None:
            return
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.config.output_dir / name)
