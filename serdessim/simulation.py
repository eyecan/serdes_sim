"""High-level simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .channel import BitByBitSimulator, ChannelChain, SimulationResult, StatisticalSimulator
from .filters import ContinuousTimeLinearEqualizer, FeedForwardEqualizer
from .ibis import IbisAmiModel, TxRxModelFactory
from .signal import SignalSpec, SignalType, generate_waveform


@dataclass
class SimulationConfiguration:
    """Aggregated configuration for a SerDes link simulation."""

    signal: SignalSpec
    tx_ffe: FeedForwardEqualizer | None = None
    tx_ctle: ContinuousTimeLinearEqualizer | None = None
    rx_ffe: FeedForwardEqualizer | None = None
    rx_ctle: ContinuousTimeLinearEqualizer | None = None
    channel: ChannelChain | None = None
    jitter_sigma: float = 0.01
    num_symbols: int = 4096
    statistical: bool = True
    bit_by_bit: bool = True
    ncores: int = 1


@dataclass
class SimulationResultContainer:
    """Bundle the outcomes of the various simulation flows."""

    statistical: SimulationResult | None
    bit_by_bit: Tuple[np.ndarray, np.ndarray] | None
    tx_model: IbisAmiModel
    rx_model: IbisAmiModel


class Simulator:
    """Coordinates statistical and time-domain analyses."""

    def __init__(self, config: SimulationConfiguration) -> None:
        self.config = config

    def _tx_model(self) -> IbisAmiModel:
        return TxRxModelFactory.from_equalizers(
            name="TxModel",
            ffe=self.config.tx_ffe,
            ctle=self.config.tx_ctle,
        )

    def _rx_model(self) -> IbisAmiModel:
        return TxRxModelFactory.from_equalizers(
            name="RxModel",
            ffe=self.config.rx_ffe,
            ctle=self.config.rx_ctle,
        )

    def _channel_impulse(self, sample_rate: float, num_points: int) -> np.ndarray:
        if self.config.channel is None:
            return np.concatenate(([1.0], np.zeros(num_points - 1)))
        return self.config.channel.combined_impulse(sample_rate, num_points)

    def _prepare_waveform(self) -> Tuple[np.ndarray, np.ndarray]:
        return generate_waveform(self.config.signal, self.config.num_symbols)

    def run(self) -> SimulationResultContainer:
        time, waveform = self._prepare_waveform()
        sample_rate = self.config.signal.symbol_rate * self.config.signal.samples_per_symbol
        impulse = self._channel_impulse(sample_rate, len(time))

        statistical_result: SimulationResult | None = None
        if self.config.statistical:
            stats = StatisticalSimulator(impulse, self.config.signal.symbol_rate)
            statistical_result = stats.run(noise_sigma=self.config.jitter_sigma)

        bit_by_bit_result: Tuple[np.ndarray, np.ndarray] | None = None
        if self.config.bit_by_bit:
            symbols = waveform[:: self.config.signal.samples_per_symbol]
            bits_per_symbol = 1 if self.config.signal.type == SignalType.NRZ else 2
            bbb = BitByBitSimulator(
                impulse=impulse,
                symbol_rate=self.config.signal.symbol_rate,
                samples_per_symbol=self.config.signal.samples_per_symbol,
                bits_per_symbol=bits_per_symbol,
                ncores=self.config.ncores,
            )
            bit_by_bit_result = bbb.run(symbols)

        return SimulationResultContainer(
            statistical=statistical_result,
            bit_by_bit=bit_by_bit_result,
            tx_model=self._tx_model(),
            rx_model=self._rx_model(),
        )
