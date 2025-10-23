# Modular SerDes Simulation Framework

This repository provides a lightweight, notebook-friendly SerDes (serializer/
deserializer) simulation environment that targets interactive research in
JupyterLab, GitHub Codespaces, or any Python runtime.  The focus is on quick
experimentation with high-speed serial links such as PCIe 5.0, xGMI, or 200G+
Ethernet PAM4 implementations.

## Highlights

- **End-to-end signal path** – Model TX FFE → Channel → RX CTLE/DFE → CDR →
  Eye/Bathtub analysis entirely in Python.
- **Interactive notebooks** – Ready-to-run Jupyter notebooks under
  [`notebooks/`](notebooks/) walk through exploratory studies and compliance
  sweeps.
- **Research-grade modularity** – The `serdes/` package exposes signal
  synthesis, equalization, jitter injection, CDR, and post-processing blocks for
  algorithm prototyping.
- **Compliance-style metrics** – Estimate BER, timing margin, and visualize eye
  openings with bathtub curves inspired by COM-style link analysis.

## Getting started

Create a Python environment and install the minimal dependencies:

```bash
pip install -r requirements.txt
```

Launch JupyterLab from the repository root:

```bash
jupyter lab
```

Open `notebooks/01_quickstart.ipynb` for an interactive tour of the pipeline.
The notebook demonstrates how to configure the transmitter, channel, receiver,
CDR, and visualization stages for sweep-style analysis.

## Minimal example (script)

```python
from serdes import (
    CdrConfig,
    ChannelConfig,
    EyeConfig,
    JitterConfig,
    RxEqualizerConfig,
    SerDesPipeline,
    SignalConfig,
    TxEqualizerConfig,
)

pipeline = SerDesPipeline(
    signal=SignalConfig(symbol_rate=56e9, samples_per_symbol=32, modulation="pam4"),
    tx=TxEqualizerConfig(taps=(0.1, 1.0, -0.12)),
    channel=ChannelConfig(alpha=0.18, post_cursor=8),
    rx=RxEqualizerConfig(ctle_zero_hz=12e9, ctle_pole_hz=38e9, dfe_taps=(0.05, -0.03, 0.01)),
    jitter=JitterConfig(rj_sigma_ui=0.008, sj_amplitude_ui=0.015, sj_frequency=8e6),
    cdr=CdrConfig(loop_bandwidth_hz=18e6, damping=0.8, gain=0.05),
    eye=EyeConfig(bins=200),
)

result = pipeline.run(num_symbols=2048)
print("Equalized symbols:", result.equalized_symbols[:10])
print("BER @ 0 UI:", result.bathtub.ber[result.bathtub.offsets_ui == 0.0])
```

Pair this script with the plotting helpers in `serdes.plotting` to visualize the
TX waveform, channel response, recovered data, and margin plots.

## Repository layout

```
serdes/            # Python package with modular SerDes blocks
notebooks/         # Guided Jupyter notebooks for experimentation
examples/          # Script-based demos and sweep utilities
requirements.txt   # Minimal dependency set (NumPy, Matplotlib, JupyterLab)
```

## Next steps

- Extend the channel model with measured S-parameters or FIR tap sets.
- Integrate external DSP libraries or GPU acceleration for massive sweeps.
- Expand the compliance toolkit with COM/JTOL-style sweeps directly in
  notebooks.
