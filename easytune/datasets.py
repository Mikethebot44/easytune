import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from .utils import stratified_split, validate_files_exist


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


@dataclass
class DataBundle:
    items: List
    labels: List[int]


class ImageLabelDataset(Dataset):
    def __init__(
        self,
        images: Sequence[Union[str, Image.Image]],
        labels: Sequence[int],
        transform: Optional[Callable] = None,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        self.images = list(images)
        self.labels = list(int(l) for l in labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        item = self.images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if isinstance(item, str):
            with Image.open(item) as img:
                img = img.convert("RGB")
                image = img
        else:
            image = item.convert("RGB")

        if self.transform is not None:
            batch = self.transform(images=image, return_tensors="pt")
            # Transform returns batch dimension; squeeze it
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) == 1:
                    batch[k] = v.squeeze(0)
        else:
            # Minimal fallback: convert to tensor HWC->CHW
            import torchvision.transforms as T

            to_tensor = T.ToTensor()
            batch = {"pixel_values": to_tensor(image)}
        return batch, label


class TextDataset(Dataset):
    def __init__(
        self, texts: Sequence[str], labels: Sequence[int], tokenizer: Optional[Callable] = None
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.texts = list(texts)
        self.labels = list(int(l) for l in labels)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for TextDataset.__getitem__")
        batch = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch, label


def load_from_folder(path: str) -> DataBundle:
    class_names: List[str] = []
    image_paths: List[str] = []
    labels: List[int] = []

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Folder not found: {path}")

    for entry in sorted(os.listdir(path)):
        class_dir = os.path.join(path, entry)
        if not os.path.isdir(class_dir):
            continue
        class_index = len(class_names)
        class_names.append(entry)
        for root, _, files in os.walk(class_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in IMG_EXTS:
                    image_paths.append(os.path.join(root, fname))
                    labels.append(class_index)

    image_paths = validate_files_exist(image_paths)
    return DataBundle(items=image_paths, labels=labels)


def load_from_huggingface(
    name: str, split: str = "train", max_samples: Optional[int] = None
) -> DataBundle:
    from datasets import load_dataset

    ds = load_dataset(name, split=split)

    # Heuristics: prefer columns named 'image' or 'text'
    col_names = ds.column_names
    if "image" in col_names:
        items = [ex["image"] for ex in ds]
        labels = [int(ex.get("label", 0)) for ex in ds]
    elif "text" in col_names:
        items = [ex["text"] for ex in ds]
        labels = [int(ex.get("label", 0)) for ex in ds]
    else:
        # pick first string-like column as text, and 'label' as labels
        text_col = None
        for c in col_names:
            if isinstance(ds[0][c], str):
                text_col = c
                break
        if text_col is None:
            raise ValueError(
                f"Dataset {name} does not have an 'image' or obvious text column"
            )
        items = [ex[text_col] for ex in ds]
        labels = [int(ex.get("label", 0)) for ex in ds]

    if max_samples is not None:
        items = items[:max_samples]
        labels = labels[:max_samples]

    return DataBundle(items=list(items), labels=list(labels))


# ---------------------- Convenience helpers for Quickstart UX ----------------------
def detect_class_root(path: Union[str, Path]) -> str:
    """Detect a directory containing class subfolders with images.

    Handles either:
    - root containing class subfolders directly, or
    - root containing split subfolders (e.g., train/val/test) each with class dirs
    Returns the detected directory path as a string.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    def has_images(dirpath: Path) -> bool:
        for f in dirpath.glob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                return True
        return False

    # Case 1: root already contains class subfolders with images
    class_dirs = [d for d in p.iterdir() if d.is_dir() and has_images(d)]
    if class_dirs:
        return str(p)

    # Case 2: root contains split subfolders with class dirs
    for split in p.iterdir():
        if not split.is_dir():
            continue
        class_dirs = [d for d in split.iterdir() if d.is_dir() and has_images(d)]
        if class_dirs:
            return str(split)

    raise RuntimeError(
        f"Could not detect a folder containing class subfolders with images under: {p}"
    )


def kaggle_download(dataset_id: str) -> str:
    """Download a Kaggle dataset via kagglehub and return local directory.

    Requires `kagglehub` to be installed by the user environment.
    """
    try:
        import kagglehub  # type: ignore
    except Exception as e:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "kagglehub is required to download Kaggle datasets: pip install kagglehub[hf-datasets]"
        ) from e
    return kagglehub.dataset_download(dataset_id)


def kaggle_download_and_detect(dataset_id: str) -> str:
    """Download a Kaggle dataset and detect a class-root usable by EasyTune."""
    d = kaggle_download(dataset_id)
    return detect_class_root(d)


def build_splits(
    items: Sequence, labels: Sequence[int], val_ratio: float, seed: int = 42
) -> Tuple[Tuple[List, List[int]], Tuple[List, List[int]]]:
    indices = list(range(len(labels)))
    train_idx, val_idx = stratified_split(indices, labels, val_ratio, seed)
    def subset(seq, idxs):
        return [seq[i] for i in idxs]
    X_train = subset(items, train_idx)
    y_train = subset(labels, train_idx)
    X_val = subset(items, val_idx)
    y_val = subset(labels, val_idx)
    return (X_train, y_train), (X_val, y_val)
