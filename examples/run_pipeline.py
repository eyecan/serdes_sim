"""Minimal command-line entry point for the SerDes pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def _ensure_repo_on_path() -> None:
    """Allow running the example directly via ``python examples/run_pipeline.py``.

    When executed as a script, Python only places the ``examples`` directory on the
    module search path. Insert the repository root so the local ``serdes`` package
    can be imported without requiring a prior ``pip install -e .``.
    """

    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


_ensure_repo_on_path()

from serdes import (  # noqa: E402  (import after path fix)
    CdrConfig,
    ChannelConfig,
    EyeConfig,
    JitterConfig,
    RxEqualizerConfig,
    SerDesPipeline,
    SignalConfig,
    TxEqualizerConfig,
)
from serdes.plotting import plot_bathtub, plot_eye, plot_waveform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=2048, help="Number of data symbols to simulate")
    parser.add_argument("--modulation", choices=["nrz", "pam4"], default="pam4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = SerDesPipeline(
        signal=SignalConfig(symbol_rate=56e9, samples_per_symbol=32, modulation=args.modulation),
        tx=TxEqualizerConfig(taps=(0.12, 1.0, -0.14)),
        channel=ChannelConfig(alpha=0.18, post_cursor=10),
        rx=RxEqualizerConfig(ctle_zero_hz=10e9, ctle_pole_hz=38e9, dfe_taps=(0.05, -0.02, 0.01)),
        jitter=JitterConfig(rj_sigma_ui=0.01, sj_amplitude_ui=0.02, sj_frequency=6e6),
        cdr=CdrConfig(loop_bandwidth_hz=20e6, damping=0.8, gain=0.05),
        eye=EyeConfig(bins=160),
    )

    result = pipeline.run(num_symbols=args.symbols)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_waveform(result.waveform_rx.time, result.waveform_rx.samples, ax=axes[0])
    plot_eye(result.eye, ax=axes[1])
    plot_bathtub(result.bathtub, ax=axes[2])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
