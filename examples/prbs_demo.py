from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np

from serdessim import SignalSpec, SignalType, SimulationConfiguration, Simulator, prbs7


def decode_rx_waveform(response: np.ndarray, samples_per_symbol: int, num_symbols: int) -> np.ndarray:
    """Return detected bits from the received waveform."""

    center = samples_per_symbol // 2
    sampled = response[center::samples_per_symbol][:num_symbols]
    return (sampled >= 0).astype(int)


def main() -> None:
    """Generate a PRBS7 pattern and observe it at the receiver."""

    num_symbols = 127
    pattern = prbs7(num_symbols)

    signal = SignalSpec(
        type=SignalType.NRZ,
        symbol_rate=25e9,
        samples_per_symbol=32,
        amplitude=1.0,
        pattern=pattern,
    )

    config = SimulationConfiguration(
        signal=signal,
        num_symbols=num_symbols,
        statistical=False,
        bit_by_bit=True,
    )

    results = Simulator(config).run()
    if not results.bit_by_bit:
        raise RuntimeError("Bit-by-bit simulation was not executed")

    time, response = results.bit_by_bit
    rx_bits = decode_rx_waveform(response, signal.samples_per_symbol, num_symbols)

    print("Transmitted PRBS7 bits (first 32):")
    print(" ".join(str(bit) for bit in pattern[:32]))

    print("Received PRBS7 bits (first 32):")
    print(" ".join(str(bit) for bit in rx_bits[:32]))

    tx_levels = signal.normalized_pattern(num_symbols)
    rx_levels = response[signal.samples_per_symbol // 2 :: signal.samples_per_symbol][:num_symbols]

    print("\nTX NRZ levels (first 16):")
    print(np.array2string(tx_levels[:16], precision=2))

    print("RX NRZ levels sampled at eye center (first 16):")
    print(np.array2string(rx_levels[:16], precision=2))

    print(f"\nCaptured {len(response)} RX samples over {time[-1]:.3e} seconds")


if __name__ == "__main__":
    main()
