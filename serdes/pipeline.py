"""High-level simulation pipeline."""
from __future__ import annotations

from dataclasses import dataclass

from .analysis import BathtubCurve, EyeDiagram, compute_bathtub, compute_eye
from .cdr import ClockDataRecovery
from .channel import LinearChannel
from .config import (
    CdrConfig,
    ChannelConfig,
    EyeConfig,
    JitterConfig,
    RxEqualizerConfig,
    SignalConfig,
    TxEqualizerConfig,
)
from .filters import DecisionFeedbackEqualizer, RxContinuousTimeLinearEqualizer, TxFeedForwardEqualizer
from .jitter import JitterInjector
from .signal import Waveform, generate_symbols, pulse_shaping


@dataclass(slots=True)
class PipelineResult:
    waveform_tx: Waveform
    waveform_channel: Waveform
    waveform_rx: Waveform
    cdr_samples: list[float]
    cdr_times: list[float]
    equalized_symbols: list[float]
    eye: EyeDiagram
    bathtub: BathtubCurve


@dataclass(slots=True)
class SerDesPipeline:
    signal: SignalConfig
    tx: TxEqualizerConfig
    channel: ChannelConfig
    rx: RxEqualizerConfig
    jitter: JitterConfig
    cdr: CdrConfig
    eye: EyeConfig

    def run(self, num_symbols: int = 4096, warmup: int = 128) -> PipelineResult:
        symbols = generate_symbols(self.signal, num_symbols + warmup)
        waveform = pulse_shaping(symbols, self.signal)

        tx_filter = TxFeedForwardEqualizer(self.tx)
        waveform_tx = tx_filter.apply(waveform)

        channel = LinearChannel(self.channel, self.signal.samples_per_symbol)
        waveform_channel = channel.apply(waveform_tx)

        ctle = RxContinuousTimeLinearEqualizer(self.rx, self.signal)
        waveform_rx_ctle = ctle.apply(waveform_channel)

        jitter_injector = JitterInjector(self.jitter)
        waveform_jitter = jitter_injector.apply(waveform_rx_ctle)

        cdr = ClockDataRecovery(self.cdr)
        sample_times, recovered = cdr.recover(waveform_jitter)

        trimmed = recovered[warmup:]
        dfe = DecisionFeedbackEqualizer(self.rx.dfe_taps)
        equalized = dfe.apply(trimmed)

        sps = self.signal.samples_per_symbol
        eye_samples = waveform_jitter.samples[warmup * sps : (warmup + num_symbols) * sps]
        eye = compute_eye(eye_samples, self.signal, self.eye)
        bathtub = compute_bathtub(eye_samples, self.signal)

        return PipelineResult(
            waveform_tx=waveform_tx,
            waveform_channel=waveform_channel,
            waveform_rx=waveform_jitter,
            cdr_samples=list(recovered),
            cdr_times=list(sample_times),
            equalized_symbols=list(equalized),
            eye=eye,
            bathtub=bathtub,
        )
