# SerDes Simulation Toolkit

This repository provides a Python-based SerDes simulation environment with
behavioral models and plotting utilities. The toolkit covers NRZ and PAM4
signaling, behavioral Tx/Rx blocks, IBIS-AMI model generation, and both
statistical and bit-by-bit channel simulation flows.

## Features

- **Signaling** – Generate NRZ and PAM4 waveforms with arbitrary symbol
  patterns, symbol rates, and amplitudes.
- **Equalization models** – Parameterized feed-forward equalizers (FFEs) and
  continuous-time linear equalizers (CTLEs).
- **IBIS-AMI** – Build LTI and NLTV IBIS-AMI models, including jitter
  specifications and IBIS/AMI corner sweeps.
- **S-parameters** – Represent Tx/Rx buffers, packages, and channels via
  S-parameters and convert them into time-domain impulse responses.
- **Channel simulation** – Statistical and bit-by-bit modes with optional
  multi-core acceleration supporting data rates up to 100 Gbps.
- **Visualization** – Eye-density, BER, PDF, bathtub, and contour-related
  metrics through a plotting helper.
- **Remote execution** – Serialize dedicated remote server reservations and
  stage commands for user-exclusive simulation servers.

## Quick start

```python
import numpy as np

from serdessim import (
    SignalSpec, SignalType, FeedForwardEqualizer,
    ContinuousTimeLinearEqualizer, SParameterBlock, ChannelChain,
    SimulationConfiguration, Simulator
)

# Describe the transmitted waveform
spec = SignalSpec(
    type=SignalType.PAM4,
    symbol_rate=56e9,
    samples_per_symbol=32,
    amplitude=0.7,
)

# Configure equalization blocks
ffe = FeedForwardEqualizer(taps=[0.2, 1.0, -0.1])
ctle = ContinuousTimeLinearEqualizer(zeros=[-2e9], poles=[-40e9, -1e9], dc_gain=1.2)

# Construct a simple pass-through channel from S-parameters
unity = SParameterBlock(frequency=np.array([1.0, 10.0]), sdd21=np.array([1.0, 1.0]))
chain = ChannelChain(unity, unity, unity, unity, unity)

config = SimulationConfiguration(
    signal=spec,
    tx_ffe=ffe,
    rx_ctle=ctle,
    channel=chain,
    num_symbols=2048,
    ncores=4,
)

result = Simulator(config).run()
print("Estimated BER:", result.statistical.ber if result.statistical else "n/a")
```

## Plotting

The :mod:`serdessim.plots` module contains helpers to visualize simulation
outputs. For example:

```python
from serdessim.plots import Plotter

plotter = Plotter()
fig = plotter.eye_density(result.statistical.artifacts)
fig.show()
```

## Remote simulation servers

Dedicated hardware reservations can be recorded via :mod:`serdessim.server`:

```python
from serdessim.server import RemoteServer, ServerConfig

server = RemoteServer(ServerConfig(host="serdes-sim.example.com", user="alice"))
server.serialize("reservation.json")
submission = server.launch_simulation("run_serdes_sim --config config.yaml")
print(submission)
```

## Development

The code base relies on NumPy, SciPy, and Matplotlib. Install dependencies with
`pip install -r requirements.txt` (if desired) and execute your own scripts or
notebooks to explore the simulation capabilities.
