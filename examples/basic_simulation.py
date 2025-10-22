"""Basic SerDes link simulation example."""

from __future__ import annotations

import numpy as np

from serdessim import (
    ChannelChain,
    ContinuousTimeLinearEqualizer,
    FeedForwardEqualizer,
    SignalSpec,
    SignalType,
    SParameterBlock,
    SimulationConfiguration,
    Simulator,
)


def build_channel(num_points: int, sample_rate: float) -> ChannelChain:
    """Create a simple low-pass style channel chain."""

    frequency = np.linspace(1e6, sample_rate / 2, num_points)
    flat = np.ones_like(frequency)
    roll_off = np.exp(-frequency / frequency.max())

    tx = SParameterBlock(frequency=frequency, sdd21=flat)
    tx_pkg = SParameterBlock(frequency=frequency, sdd21=roll_off)
    chan = SParameterBlock(frequency=frequency, sdd21=roll_off)
    rx_pkg = SParameterBlock(frequency=frequency, sdd21=roll_off)
    rx = SParameterBlock(frequency=frequency, sdd21=flat)
    return ChannelChain(tx, tx_pkg, chan, rx_pkg, rx)


def run_simulation() -> None:
    """Configure and execute a SerDes simulation."""

    signal = SignalSpec(
        type=SignalType.PAM4,
        symbol_rate=56e9,
        samples_per_symbol=16,
        amplitude=0.6,
    )

    sample_rate = signal.symbol_rate * signal.samples_per_symbol
    channel = build_channel(num_points=512, sample_rate=sample_rate)

    tx_ffe = FeedForwardEqualizer(taps=[0.15, 1.0, -0.12])
    rx_ctle = ContinuousTimeLinearEqualizer(
        zeros=[-2e9],
        poles=[-35e9, -1.5e9],
        dc_gain=1.0,
    )

    config = SimulationConfiguration(
        signal=signal,
        tx_ffe=tx_ffe,
        rx_ctle=rx_ctle,
        channel=channel,
        num_symbols=4096,
        ncores=2,
    )

    results = Simulator(config).run()

    if results.statistical:
        print(f"Statistical BER estimate: {results.statistical.ber:.3e}")
        print(f"Eye height: {results.statistical.eye_metrics['eye_height']:.3f}")

    if results.bit_by_bit:
        time, response = results.bit_by_bit
        print(f"Captured {len(response)} waveform samples over {time[-1]:.3e} seconds")


if __name__ == "__main__":
    run_simulation()
