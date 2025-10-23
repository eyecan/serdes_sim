"""Channel representation and simulation routines."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast convolution helper that only depends on NumPy."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size + b.size - 1
    n_fft = 1 << (n - 1).bit_length()
    result = np.fft.irfft(np.fft.rfft(a, n_fft) * np.fft.rfft(b, n_fft), n_fft)
    return result[:n]


@dataclass
class SParameterBlock:
    """Container for an S-parameter frequency response."""

    frequency: np.ndarray
    sdd21: np.ndarray

    @staticmethod
    def from_touchstone(path: str | Path) -> "SParameterBlock":
        data: List[complex] = []
        freq: List[float] = []
        with open(path, "r", encoding="utf8") as fh:
            for line in fh:
                if line.startswith("!") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                freq.append(float(parts[0]))
                real = float(parts[1])
                imag = float(parts[2])
                data.append(real + 1j * imag)
        return SParameterBlock(np.array(freq), np.array(data))

    def to_impulse(self, sample_rate: float, num_points: int) -> np.ndarray:
        """Convert frequency response to an impulse response."""

        if num_points <= 0:
            raise ValueError("num_points must be positive")
        freqs = self.frequency
        response = self.sdd21
        if freqs.size == 0:
            return np.zeros(num_points)
        min_freq, max_freq = freqs[0], freqs[-1]
        target = np.linspace(min_freq, max_freq, num_points // 2 + 1)
        interp_real = np.interp(target, freqs, response.real, left=response.real[0], right=response.real[-1])
        interp_imag = np.interp(target, freqs, response.imag, left=response.imag[0], right=response.imag[-1])
        half = interp_real + 1j * interp_imag
        spectrum = np.empty(num_points, dtype=complex)
        spectrum[: target.size] = half
        if num_points % 2 == 0:
            mirror = np.conj(half[1:-1][::-1])
        else:
            mirror = np.conj(half[1:][::-1])
        spectrum[target.size :] = mirror
        impulse = np.fft.ifft(spectrum)
        return impulse.real


@dataclass
class Channel:
    """Represents a linear channel element."""

    impulse_response: np.ndarray

    @staticmethod
    def from_sparameters(block: SParameterBlock, sample_rate: float, num_points: int) -> "Channel":
        impulse = block.to_impulse(sample_rate, num_points)
        return Channel(impulse_response=impulse)

    def cascade(self, other: "Channel") -> "Channel":
        impulse = _fft_convolve(self.impulse_response, other.impulse_response)
        return Channel(impulse)


@dataclass
class ChannelChain:
    """Cascaded channel consisting of several S-parameter blocks."""

    tx_buffer: SParameterBlock
    tx_package: SParameterBlock
    channel: SParameterBlock
    rx_package: SParameterBlock
    rx_buffer: SParameterBlock

    def combined_impulse(self, sample_rate: float, num_points: int) -> np.ndarray:
        impulse = np.ones(1)
        for block in (
            self.tx_buffer,
            self.tx_package,
            self.channel,
            self.rx_package,
            self.rx_buffer,
        ):
            element = Channel.from_sparameters(block, sample_rate, num_points)
            impulse = _fft_convolve(impulse, element.impulse_response)
        return impulse[:num_points]


@dataclass
class SimulationArtifacts:
    """Container for intermediate results of a simulation."""

    eye_density: np.ndarray
    eye_time: np.ndarray
    pdf: np.ndarray
    amplitude_pdf: np.ndarray


@dataclass
class SimulationResult:
    """Results produced by the statistical or time-domain simulators."""

    ber: float
    ber_curve: np.ndarray
    timing_bathtub: np.ndarray
    amplitude_bathtub: np.ndarray
    eye_metrics: dict
    artifacts: SimulationArtifacts


class StatisticalSimulator:
    """Perform statistical channel analysis."""

    def __init__(
        self,
        channel: np.ndarray,
        sample_rate: float,
        symbol_rate: float,
        *,
        max_kernel_samples: int = 2048,
    ) -> None:
        self.channel = np.asarray(channel, dtype=float)
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.max_kernel_samples = max_kernel_samples

    def _windowed_impulse(self) -> np.ndarray:
        """Return a slice of the impulse with manageable length."""

        impulse = self.channel
        if impulse.size <= self.max_kernel_samples:
            return impulse

        energy = np.abs(impulse) ** 2
        peak = int(np.argmax(energy))
        half = self.max_kernel_samples // 2
        start = max(0, peak - half)
        end = min(impulse.size, start + self.max_kernel_samples)
        start = max(0, end - self.max_kernel_samples)
        return impulse[start:end]

    @cached_property
    def eye_kernel(self) -> np.ndarray:
        impulse = self._windowed_impulse()
        kernel = np.outer(impulse, impulse)
        return kernel

    def run(self, noise_sigma: float = 0.01) -> SimulationResult:
        eye_density = np.maximum(self.eye_kernel, 0)
        num_samples = eye_density.shape[0]
        ui = 1.0 / self.symbol_rate
        dt = 1.0 / self.sample_rate
        time_offsets = (np.arange(num_samples) - num_samples // 2) * dt
        eye_time = time_offsets / ui
        amplitude_pdf = np.sum(eye_density, axis=0)
        pdf = np.sum(eye_density, axis=1)
        ber = float(np.exp(-1 / max(2 * noise_sigma**2, 1e-12)))
        bathtub = np.column_stack((eye_time, np.clip(1 - eye_time**2, 0, None)))
        artifacts = SimulationArtifacts(
            eye_density=eye_density,
            eye_time=eye_time,
            pdf=pdf,
            amplitude_pdf=amplitude_pdf,
        )
        return SimulationResult(
            ber=ber,
            ber_curve=np.column_stack(
                (eye_time, np.exp(-eye_time**2 / max(2 * noise_sigma**2, 1e-12)))
            ),
            timing_bathtub=bathtub,
            amplitude_bathtub=np.column_stack((amplitude_pdf, amplitude_pdf[::-1])),
            eye_metrics={
                "eye_height": float(np.max(amplitude_pdf)),
                "eye_width": float(np.max(bathtub[:, 1])),
            },
            artifacts=artifacts,
        )


class BitByBitSimulator:
    """Perform bit-by-bit channel analysis with optional parallelism."""

    def __init__(
        self,
        impulse: np.ndarray,
        symbol_rate: float,
        *,
        samples_per_symbol: int,
        bits_per_symbol: int = 1,
        ncores: int = 1,
        max_supported_rate: float = 100e9,
    ) -> None:
        self.impulse = np.asarray(impulse, dtype=float)
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = samples_per_symbol
        self.bits_per_symbol = bits_per_symbol
        self.supported_data_rate = max(symbol_rate * bits_per_symbol, max_supported_rate)
        self.ncores = max(1, ncores)

    def _simulate_chunk(self, symbols: np.ndarray) -> np.ndarray:
        upsampled = np.repeat(symbols, self.samples_per_symbol)
        response = _fft_convolve(upsampled, self.impulse)[: len(upsampled)]
        return response

    def run(self, symbols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        symbols = np.asarray(symbols, dtype=float)
        if self.ncores == 1 or symbols.size < 1024:
            response = self._simulate_chunk(symbols)
        else:
            from concurrent.futures import ProcessPoolExecutor

            chunk_size = symbols.size // self.ncores
            chunks: List[np.ndarray] = [
                symbols[i * chunk_size : (i + 1) * chunk_size] for i in range(self.ncores - 1)
            ]
            chunks.append(symbols[(self.ncores - 1) * chunk_size :])

            with ProcessPoolExecutor(max_workers=self.ncores) as executor:
                futures = [executor.submit(self._simulate_chunk, chunk) for chunk in chunks]
                results = [future.result() for future in futures]
            response = np.concatenate(results)

        dt = 1.0 / (self.symbol_rate * self.samples_per_symbol)
        time = np.arange(response.size) * dt
        return time, response
