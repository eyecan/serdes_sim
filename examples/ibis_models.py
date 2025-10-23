from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from serdessim import (
    AmiCorner,
    ContinuousTimeLinearEqualizer,
    FeedForwardEqualizer,
    IbisCorner,
    JitterSpec,
    TxRxModelFactory,
)
from serdessim.ibis import sweep_corners


def main() -> None:
    ffe = FeedForwardEqualizer(taps=[0.2, 1.0, -0.05])
    ctle = ContinuousTimeLinearEqualizer(zeros=[-3e9], poles=[-40e9, -2e9], dc_gain=1.1)
    jitter = JitterSpec(random_rms=0.002, deterministic_pp=0.01)

    model = TxRxModelFactory.from_equalizers(
        name="TxExample",
        ffe=ffe,
        ctle=ctle,
        corner=IbisCorner.TYPICAL,
        ami_corner=AmiCorner.TYPICAL,
        jitter=jitter,
    )

    print("Base AMI dictionary:")
    print(model.as_ami_dict())

    print("\nCorner sweeps:")
    for variant in sweep_corners(model):
        params = variant.sample_parameters()
        print(f"- {variant.name}: {params['corner']} / {params['ami_corner']}")


if __name__ == "__main__":
    main()
