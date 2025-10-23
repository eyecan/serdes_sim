"""Interactive SerDes simulation framework."""

from .config import (
    SignalConfig,
    TxEqualizerConfig,
    ChannelConfig,
    RxEqualizerConfig,
    JitterConfig,
    CdrConfig,
    EyeConfig,
)
from .pipeline import SerDesPipeline, PipelineResult
from .analysis import EyeDiagram, BathtubCurve

__all__ = [
    "SignalConfig",
    "TxEqualizerConfig",
    "ChannelConfig",
    "RxEqualizerConfig",
    "JitterConfig",
    "CdrConfig",
    "EyeConfig",
    "SerDesPipeline",
    "PipelineResult",
    "EyeDiagram",
    "BathtubCurve",
]
