"""IBIS-AMI modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List

import numpy as np

from .filters import ContinuousTimeLinearEqualizer, FeedForwardEqualizer


class IbisCorner(str, Enum):
    """Supported IBIS corner cases."""

    TYPICAL = "typical"
    MIN = "min"
    MAX = "max"


class AmiCorner(str, Enum):
    """Supported AMI corner cases."""

    TYPICAL = "typical"
    SLOW = "slow"
    FAST = "fast"


@dataclass
class JitterSpec:
    """Jitter settings consistent with IBIS 7.0-style definitions."""

    random_rms: float = 0.0
    deterministic_pp: float = 0.0
    periodic: Dict[float, float] = field(default_factory=dict)

    def sample(self, num_samples: int, sample_rate: float) -> np.ndarray:
        """Generate a jitter waveform."""

        rng = np.random.default_rng()
        time = np.arange(num_samples) / sample_rate
        random_component = rng.normal(scale=self.random_rms, size=num_samples)
        deterministic_component = np.zeros(num_samples)
        for freq, amplitude in self.periodic.items():
            deterministic_component += amplitude * np.sin(2 * np.pi * freq * time)
        bounded_component = (
            self.deterministic_pp / 2
            * np.sin(2 * np.pi * time / (num_samples / sample_rate))
            if self.deterministic_pp
            else 0.0
        )
        return random_component + deterministic_component + bounded_component


@dataclass
class IbisAmiModel:
    """Base class for IBIS-AMI models."""

    name: str
    corner: IbisCorner = IbisCorner.TYPICAL
    ami_corner: AmiCorner = AmiCorner.TYPICAL
    jitter: JitterSpec = field(default_factory=JitterSpec)

    def impulse_response(self, sample_rate: float, num_points: int) -> np.ndarray:
        raise NotImplementedError

    def sample_parameters(self) -> Dict[str, float]:
        """Return parameters reflecting the configured corners."""

        return {
            "corner": self.corner.value,
            "ami_corner": self.ami_corner.value,
            "random_rms": self.jitter.random_rms,
            "deterministic_pp": self.jitter.deterministic_pp,
        }

    def as_ami_dict(self) -> Dict[str, float]:
        data = {"Model_Name": self.name}
        data.update(self.sample_parameters())
        return data


@dataclass
class LtiAmiModel(IbisAmiModel):
    """Linear time-invariant AMI model built from an impulse response."""

    impulse: np.ndarray = field(default_factory=lambda: np.zeros(1))

    def impulse_response(self, sample_rate: float, num_points: int) -> np.ndarray:  # noqa: ARG002
        return self.impulse[:num_points]


@dataclass
class NltvAmiModel(IbisAmiModel):
    """Non-linear time-variant AMI model composed of behavioral blocks."""

    ffe: FeedForwardEqualizer | None = None
    ctle: ContinuousTimeLinearEqualizer | None = None

    def impulse_response(self, sample_rate: float, num_points: int) -> np.ndarray:
        impulse = np.zeros(num_points)
        impulse[0] = 1.0
        if self.ffe is not None:
            impulse = self.ffe.filter(impulse)
        if self.ctle is not None:
            impulse = self.ctle.filter(impulse, sample_rate)
        return impulse


class TxRxModelFactory:
    """Factory helpers to derive AMI models from behavioral models."""

    @staticmethod
    def from_equalizers(
        name: str,
        ffe: FeedForwardEqualizer | None,
        ctle: ContinuousTimeLinearEqualizer | None,
        *,
        corner: IbisCorner = IbisCorner.TYPICAL,
        ami_corner: AmiCorner = AmiCorner.TYPICAL,
        jitter: JitterSpec | None = None,
    ) -> NltvAmiModel:
        return NltvAmiModel(
            name=name,
            corner=corner,
            ami_corner=ami_corner,
            jitter=jitter or JitterSpec(),
            ffe=ffe,
            ctle=ctle,
        )

    @staticmethod
    def from_impulse(
        name: str,
        impulse: Iterable[float],
        *,
        corner: IbisCorner = IbisCorner.TYPICAL,
        ami_corner: AmiCorner = AmiCorner.TYPICAL,
        jitter: JitterSpec | None = None,
    ) -> LtiAmiModel:
        impulse_arr = np.asarray(list(impulse), dtype=float)
        return LtiAmiModel(
            name=name,
            corner=corner,
            ami_corner=ami_corner,
            jitter=jitter or JitterSpec(),
            impulse=impulse_arr,
        )


def sweep_corners(model: IbisAmiModel) -> List[IbisAmiModel]:
    """Generate model variants for all AMI and IBIS corners."""

    variants: List[IbisAmiModel] = []
    for corner in IbisCorner:
        for ami_corner in AmiCorner:
            jitter = JitterSpec(
                random_rms=model.jitter.random_rms,
                deterministic_pp=model.jitter.deterministic_pp,
                periodic=dict(model.jitter.periodic),
            )
            variant = type(model)(
                name=f"{model.name}_{corner.value}_{ami_corner.value}",
                corner=corner,
                ami_corner=ami_corner,
                jitter=jitter,
            )
            if isinstance(model, NltvAmiModel):
                variant.ffe = model.ffe
                variant.ctle = model.ctle
            elif isinstance(model, LtiAmiModel):
                variant.impulse = model.impulse
            variants.append(variant)
    return variants
