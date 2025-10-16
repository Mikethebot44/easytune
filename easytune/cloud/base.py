from __future__ import annotations

from typing import Optional, Protocol


class RemoteBackend(Protocol):
    def train(
        self,
        *,
        model: str,
        task: str,
        data_dir: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        out_dir: str,
        gpu: Optional[str] = None,
        timeout_s: int = 3600,
    ) -> None:
        ...


