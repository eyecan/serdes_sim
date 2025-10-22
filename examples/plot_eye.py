"""Generate eye, BER, and PDF plots from a statistical simulation."""

from __future__ import annotations

from pathlib import Path

from serdessim import (
    PlotConfig,
    Plotter,
    SignalSpec,
    SignalType,
    SimulationConfiguration,
    Simulator,
)

from basic_simulation import build_channel


def main() -> None:
    signal = SignalSpec(
        type=SignalType.NRZ,
        symbol_rate=28e9,
        samples_per_symbol=32,
        amplitude=0.5,
    )

    config = SimulationConfiguration(
        signal=signal,
        channel=build_channel(num_points=512, sample_rate=signal.symbol_rate * signal.samples_per_symbol),
        statistical=True,
        bit_by_bit=False,
    )

    results = Simulator(config).run()
    if not results.statistical:
        raise RuntimeError("Statistical simulation did not run")

    output_dir = Path("example_outputs")
    plotter = Plotter(PlotConfig(output_dir=output_dir))
    plotter.eye_density(results.statistical.artifacts)
    plotter.ber_plots(results.statistical)
    plotter.pdf_plots(results.statistical.artifacts)
    print(f"Saved plots to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
