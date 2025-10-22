"""Remote execution harness for dedicated simulation servers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ServerConfig:
    """Definition of a remote simulation host."""

    host: str
    port: int = 22
    user: str | None = None
    workdir: str = "/tmp/serdessim"


class RemoteServer:
    """Simple façade for a dedicated simulation server resource."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config

    def reservation_summary(self) -> Dict[str, Any]:
        return {
            "host": self.config.host,
            "port": self.config.port,
            "user": self.config.user or "(default)",
            "workdir": self.config.workdir,
        }

    def serialize(self, path: str | Path) -> None:
        data = self.reservation_summary()
        with open(path, "w", encoding="utf8") as fh:
            json.dump(data, fh, indent=2)

    def launch_simulation(self, command: str) -> Dict[str, Any]:
        """Pretend to launch a simulation on the remote machine."""

        return {
            "command": command,
            "status": "submitted",
            "server": self.reservation_summary(),
        }
