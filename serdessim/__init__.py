"""SerDes simulation toolkit.

The :mod:`serdessim` package bundles behavioral building blocks that allow
lightweight serializer/deserializer (SerDes) experimentation directly from
Jupyter notebooks.  The modules intentionally keep their dependencies
minimal—NumPy is required while SciPy and Matplotlib are optional.  When SciPy
is not installed the package provides pure NumPy fallbacks so that unit tests
and basic demonstrations remain functional in constrained environments.
"""

from .signal import SignalSpec, SignalType, generate_waveform, prbs7
from .filters import FeedForwardEqualizer, ContinuousTimeLinearEqualizer
from .ibis import (
    AmiCorner,
    IbisAmiModel,
    IbisCorner,
    JitterSpec,
    LtiAmiModel,
    NltvAmiModel,
    TxRxModelFactory,
)
from .channel import (
    BitByBitSimulator,
    Channel,
    ChannelChain,
    SParameterBlock,
    SimulationArtifacts,
    SimulationResult,
    StatisticalSimulator,
)
from .simulation import SimulationConfiguration, SimulationResultContainer, Simulator
from .plots import PlotConfig, Plotter
from .server import RemoteServer, ServerConfig

__all__ = [
    "SignalSpec",
    "SignalType",
    "generate_waveform",
    "prbs7",
    "FeedForwardEqualizer",
    "ContinuousTimeLinearEqualizer",
    "AmiCorner",
    "IbisAmiModel",
    "IbisCorner",
    "JitterSpec",
    "LtiAmiModel",
    "NltvAmiModel",
    "TxRxModelFactory",
    "BitByBitSimulator",
    "Channel",
    "ChannelChain",
    "SParameterBlock",
    "SimulationArtifacts",
    "SimulationResult",
    "StatisticalSimulator",
    "SimulationConfiguration",
    "SimulationResultContainer",
    "Simulator",
    "PlotConfig",
    "Plotter",
    "RemoteServer",
    "ServerConfig",
]
