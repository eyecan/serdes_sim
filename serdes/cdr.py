"""Clock data recovery model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CdrConfig
from .signal import Waveform


@dataclass(slots=True)
class ClockDataRecovery:
    config: CdrConfig

    def recover(self, waveform: Waveform) -> tuple[np.ndarray, np.ndarray]:
        samples_per_symbol = waveform.samples_per_symbol
        dt = 1.0 / waveform.sample_rate
        ui = samples_per_symbol * dt
        phase = 0.0
        loop_gain = self.config.gain
        loop_bw = self.config.loop_bandwidth_hz
        damping = self.config.damping
        k0 = 4 * damping * loop_bw
        k1 = 4 * loop_bw ** 2

        time = waveform.time
        data = waveform.samples
        sample_points = []
        recovered = []
        freq_error = 0.0

        for symbol_idx in range(len(data) // samples_per_symbol):
            center_time = symbol_idx * ui + phase
            early_time = center_time - 0.25 * ui
            late_time = center_time + 0.25 * ui
            center = np.interp(center_time, time, data)
            early = np.interp(early_time, time, data)
            late = np.interp(late_time, time, data)
            error = np.sign(center) * (late - early)

            freq_error += k1 * error * loop_gain
            phase += (k0 * error + freq_error) * loop_gain / waveform.sample_rate

            sample_points.append(center_time)
            recovered.append(center)

        return np.asarray(sample_points), np.asarray(recovered)
