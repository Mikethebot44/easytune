"""Optional cloud backends and convenience helpers."""

from .base import RemoteBackend  # re-export for convenience

__all__ = ["RemoteBackend", "train_on_modal"]


def train_on_modal(model: str, task: str, data_dir: str, out_dir: str, **kwargs) -> None:
    """One-liner to run remote training on Modal.

    Example:
        from easytune.cloud import train_on_modal
        train_on_modal("facebook/dinov2-base", "image-similarity", data_dir, out_dir, epochs=5)
    """
    try:
        from .modal_backend import ModalBackend  # local import to keep optional
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Modal backend is unavailable in this environment") from exc

    from ..finetuner import FineTuner

    FineTuner(model=model, task=task).train_remote(
        ModalBackend(), data_dir=data_dir, out_dir=out_dir, **kwargs
    )

