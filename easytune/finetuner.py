from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import datasets as ds
from .losses import NTXentLoss
from .models import (
    AdapterConfig,
    BaseAdapter,
    ImageAdapter,
    TextAdapter,
    load_adapter,
    load_image_model,
    load_text_model,
    save_adapter,
)
from .utils import (
    ensure_int_labels,
    print_info,
    print_warning,
    save_json,
    stratified_split,
    warn_if_imbalanced,
    validate_files_exist,
)

if TYPE_CHECKING:  # typing-only to avoid runtime dependency on cloud subpackage
    from .cloud import RemoteBackend


TaskType = Literal["image-similarity", "text-similarity"]
DeviceType = Literal["auto", "cpu", "cuda"]


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-5
    validation_split: float = 0.2
    temperature: float = 0.07
    seed: int = 42


class FineTuner:
    """Main user-facing API for EasyTune.

    Example:
        from easytune import FineTuner
        tuner = FineTuner(model="facebook/dinov2-base", task="image-similarity")
        tuner.add_from_folder("/path/to/folder")
        tuner.train(epochs=5)
        tuner.save("fine_tuned_model")
    """

    def __init__(
        self,
        model: str,
        task: TaskType = "image-similarity",
        device: DeviceType = "auto",
        adapter: Optional[BaseAdapter] = None,
        projection_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        self.model_name = model
        self.task: TaskType = task
        self.device_str: DeviceType = device
        self.device = self._resolve_device(device)
        self._projection_dim: Optional[int] = projection_dim
        self._dropout: float = dropout

        # Adapter (model + projection head)
        if adapter is not None:
            # Dependency injection path (useful for tests)
            self.adapter = adapter.to(self.device)
            self._adapter_type = "image" if self.task == "image-similarity" else "text"
            # Note: caller must ensure dataloaders provide appropriate batch keys
        else:
            if self.task == "image-similarity":
                m, processor = load_image_model(
                    self.model_name, projection_dim=self._projection_dim, dropout=self._dropout
                )
                self._adapter_type = "image"
                self._processor = processor  # HF ImageProcessor
            elif self.task == "text-similarity":
                m, tokenizer = load_text_model(
                    self.model_name, projection_dim=self._projection_dim, dropout=self._dropout
                )
                self._adapter_type = "text"
                self._tokenizer = tokenizer  # HF Tokenizer
            else:
                raise ValueError(
                    "Unsupported task. Use 'image-similarity' or 'text-similarity'."
                )
            self.adapter: BaseAdapter = m.to(self.device)

        # Data storage
        self._items: List = []
        self._labels: List[int] = []

        # Training bookkeeping
        self._best_val_loss: Optional[float] = None
        self._last_train_loss: Optional[float] = None
        self._last_val_loss: Optional[float] = None
        self._train_history: List[Dict] = []

    # ---------------------- Public API ----------------------
    def add_data(
        self,
        *,
        image_paths: Optional[Sequence[str]] = None,
        texts: Optional[Sequence[str]] = None,
        labels: Sequence[int] = (),
        validation_split: float = 0.2,
    ) -> None:
        """Add supervised samples.

        Exactly one of (image_paths, texts) must be provided.
        """
        if (image_paths is None) == (texts is None):
            raise ValueError("Provide exactly one of image_paths or texts")

        int_labels = ensure_int_labels(labels)
        if image_paths is not None:
            if self._adapter_type != "image":
                raise ValueError("Image data provided for a text model. Set task='image-similarity'.")
            image_paths = validate_files_exist(image_paths)
            if len(image_paths) != len(int_labels):
                raise ValueError("Number of image_paths and labels must match")
            self._items.extend(list(image_paths))
            self._labels.extend(list(int_labels))
        else:
            if self._adapter_type != "text":
                raise ValueError("Text data provided for an image model. Set task='text-similarity'.")
            texts = list(texts or [])
            if len(texts) != len(int_labels):
                raise ValueError("Number of texts and labels must match")
            self._items.extend(texts)
            self._labels.extend(list(int_labels))

        warn_if_imbalanced(self._labels)
        self._validation_split = validation_split
        print_info(f"Loaded {len(self._items)} samples; {len(set(self._labels))} classes")

    def add_from_folder(self, folder_path: str, validation_split: float = 0.2) -> None:
        if self._adapter_type != "image":
            raise ValueError("add_from_folder is only valid for image models")
        bundle = ds.load_from_folder(folder_path)
        self._items.extend(bundle.items)
        self._labels.extend(bundle.labels)
        warn_if_imbalanced(self._labels)
        self._validation_split = validation_split
        print_info(
            f"Loaded {len(bundle.items)} images from folder; total {len(self._items)} samples"
        )

    def add_from_huggingface(
        self, dataset_name: str, split: str = "train", max_samples: Optional[int] = None, validation_split: float = 0.2
    ) -> None:
        bundle = ds.load_from_huggingface(dataset_name, split=split, max_samples=max_samples)
        # Infer data type
        if len(bundle.items) == 0:
            raise ValueError("Loaded dataset is empty")
        sample = bundle.items[0]
        is_text = isinstance(sample, str)
        if is_text and self._adapter_type != "text":
            raise ValueError("Text dataset loaded for an image model. Set task='text-similarity'.")
        if (not is_text) and self._adapter_type != "image":
            raise ValueError("Image dataset loaded for a text model. Set task='text-similarity'.")
        self._items.extend(bundle.items)
        self._labels.extend(bundle.labels)
        warn_if_imbalanced(self._labels)
        self._validation_split = validation_split
        print_info(f"Loaded {len(bundle.items)} samples from HF '{dataset_name}' (split={split})")

    def train(
        self,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        validation_split: Optional[float] = None,
        temperature: float = 0.07,
        seed: int = 42,
    ) -> "FineTuner":
        if len(self._items) == 0:
            raise ValueError("No data added. Use add_data/add_from_folder/add_from_huggingface first.")

        val_ratio = validation_split if validation_split is not None else getattr(self, "_validation_split", 0.2)
        cfg = TrainConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=val_ratio,
            temperature=temperature,
            seed=seed,
        )

        # Build datasets and loaders
        (X_train, y_train), (X_val, y_val) = ds.build_splits(self._items, self._labels, cfg.validation_split, seed=cfg.seed)
        train_loader, val_loader = self._build_dataloaders(X_train, y_train, X_val, y_val, batch_size=cfg.batch_size)

        # Optimizer and loss
        self.adapter.train()
        optimizer = AdamW(self.adapter.parameters(), lr=cfg.learning_rate)
        criterion = NTXentLoss(temperature=cfg.temperature)

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val_loss: Optional[float] = None

        for epoch in range(cfg.epochs):
            epoch_loss = 0.0
            num_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
            for batch, labels in pbar:
                batch = self._move_batch_to_device(batch, self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                embeddings = self.adapter(batch)
                loss = criterion(embeddings, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss = epoch_loss / max(1, num_batches)
            self._last_train_loss = train_loss

            val_loss = None
            if val_loader is not None and len(val_loader) > 0:
                self.adapter.eval()
                with torch.no_grad():
                    vloss = 0.0
                    vsteps = 0
                    for vbatch, vlabels in val_loader:
                        vbatch = self._move_batch_to_device(vbatch, self.device)
                        vlabels = vlabels.to(self.device)
                        vemb = self.adapter(vbatch)
                        vloss += criterion(vemb, vlabels).item()
                        vsteps += 1
                val_loss = vloss / max(1, vsteps)
                self.adapter.train()

            self._last_val_loss = val_loss
            self._train_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss is not None:
                print_info(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.adapter.state_dict().items()}
            else:
                print_info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")

        # Restore best
        if best_state is not None:
            self.adapter.load_state_dict(best_state)
            self._best_val_loss = best_val_loss

        return self

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        # Save adapter (base model + projection head + tokenizer/processor)
        save_adapter(self.adapter, path)
        # Save config and metadata
        meta = {
            "model_name": self.model_name,
            "task": self.task,
            "adapter_type": self._adapter_type,
            "embedding_dim": getattr(self.adapter, "hidden_size", None),
            "projection_dim": getattr(self.adapter, "projection_dim", self._projection_dim),
            "train_history": self._train_history,
            "best_val_loss": self._best_val_loss,
            "saved_at": int(time.time()),
            "easytune_version": "0.1.0",
        }
        cfg = AdapterConfig(
            base_model_name=self.model_name,
            adapter_type=self._adapter_type,
            projection_dim=getattr(self.adapter, "projection_dim", self._projection_dim),
        )
        save_json(os.path.join(path, "adapter_config.json"), asdict(cfg))
        save_json(os.path.join(path, "metadata.json"), meta)
        print_info(f"Model saved to: {path}")

    @staticmethod
    def load(path: str, device: DeviceType = "auto") -> "FineTuner":
        # Load adapter config
        import json

        with open(os.path.join(path, "adapter_config.json"), "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = AdapterConfig(**cfg_dict)
        # Create a new tuner and override adapter with loaded weights
        task: TaskType = "image-similarity" if cfg.adapter_type == "image" else "text-similarity"
        tuner = FineTuner(model=cfg.base_model_name, task=task, device=device)
        loaded_adapter = load_adapter(cfg, path)
        tuner.adapter = loaded_adapter.to(tuner.device)
        print_info(f"Model loaded from: {path}")
        return tuner

    # ---------------------- Helpers ----------------------
    def train_remote(
        self,
        backend: "RemoteBackend",  # type: ignore[name-defined]
        data_dir: str,
        out_dir: str,
        *,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        gpu: Optional[str] = None,
        timeout_s: int = 3600,
    ) -> None:
        """Delegate training to a remote backend.

        The backend is responsible for executing training in a remote environment
        and materializing artifacts to the provided out_dir on the local machine.
        """
        if backend is None:
            raise ValueError("Remote backend is required")
        backend.train(
            model=self.model_name,
            task=self.task,
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            out_dir=out_dir,
            gpu=gpu,
            timeout_s=timeout_s,
        )

    def _build_dataloaders(
        self, X_train: Sequence, y_train: Sequence[int], X_val: Sequence, y_val: Sequence[int], batch_size: int
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        if self._adapter_type == "image":
            train_ds = ds.ImageLabelDataset(X_train, y_train, transform=self._processor)
            val_ds = ds.ImageLabelDataset(X_val, y_val, transform=self._processor) if len(y_val) > 0 else None
        else:
            train_ds = ds.TextDataset(X_train, y_train, tokenizer=self._tokenizer)
            val_ds = ds.TextDataset(X_val, y_val, tokenizer=self._tokenizer) if len(y_val) > 0 else None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None
        return train_loader, val_loader

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _resolve_device(self, device: DeviceType) -> torch.device:
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            print_warning("CUDA requested but not available; falling back to CPU")
            return torch.device("cpu")
        # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
