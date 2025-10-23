"""Top-level package for the SerDes simulation toolkit.

This package provides behavioral models and simulation utilities for
serializer/deserializer (SerDes) design exploration.
"""

from .signal import SignalSpec, SignalType, generate_waveform, prbs7
from .filters import FeedForwardEqualizer, ContinuousTimeLinearEqualizer
from .ibis import (
    IbisCorner, AmiCorner, IbisAmiModel, LtiAmiModel, NltvAmiModel,
    TxRxModelFactory, JitterSpec
)
from .channel import (
    SParameterBlock, Channel, ChannelChain, StatisticalSimulator,
    BitByBitSimulator, SimulationArtifacts, SimulationResult
)
from .simulation import (
    SimulationConfiguration, SimulationResultContainer, Simulator
)
from .plots import Plotter, PlotConfig
from .server import RemoteServer, ServerConfig

__all__ = [
    "SignalSpec",
    "SignalType",
    "generate_waveform",
    "prbs7",
    "FeedForwardEqualizer",
    "ContinuousTimeLinearEqualizer",
    "IbisCorner",
    "AmiCorner",
    "IbisAmiModel",
    "LtiAmiModel",
    "NltvAmiModel",
    "TxRxModelFactory",
    "JitterSpec",
    "SParameterBlock",
    "Channel",
    "ChannelChain",
    "StatisticalSimulator",
    "BitByBitSimulator",
    "SimulationArtifacts",
    "SimulationConfiguration",
    "SimulationResult",
    "SimulationResultContainer",
    "Simulator",
    "Plotter",
    "PlotConfig",
    "RemoteServer",
    "ServerConfig",
]
