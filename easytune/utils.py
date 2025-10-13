import os
import json
import random
import warnings
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def print_warning(message: str) -> None:
    warnings.warn(message)


def print_info(message: str) -> None:
    print(message)


def validate_files_exist(paths: Iterable[str]) -> List[str]:
    valid_paths = []
    missing = []
    for p in paths:
        if os.path.isfile(p) and os.access(p, os.R_OK):
            valid_paths.append(p)
        else:
            missing.append(p)
    if missing:
        print_warning(f"Skipped {len(missing)} missing/unreadable files. First few: {missing[:5]}")
    if not valid_paths:
        raise FileNotFoundError("No valid files found.")
    return valid_paths


def ensure_int_labels(labels: Sequence) -> List[int]:
    try:
        int_labels = [int(l) for l in labels]
    except Exception as exc:
        raise ValueError("Labels must be integers or convertible to integers.") from exc
    return int_labels


def label_stats(labels: Sequence[int]) -> Counter:
    return Counter(labels)


def warn_if_imbalanced(labels: Sequence[int], threshold: float = 0.9) -> None:
    counts = label_stats(labels)
    total = sum(counts.values())
    if not total:
        return
    max_frac = max(c / total for c in counts.values())
    if max_frac >= threshold:
        print_warning(
            f"Highly imbalanced classes detected: majority class fraction {max_frac:.2f}"
        )


def stratified_split(
    indices: Sequence[int], labels: Sequence[int], val_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[int], List[int]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    per_label_indices = {}
    for idx, label in zip(indices, labels):
        per_label_indices.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for label, idxs in per_label_indices.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio))) if len(idxs) > 1 else 0
        val_indices.extend(idxs[:n_val])
        train_indices.extend(idxs[n_val:])

    # If some classes had only one sample, ensure we don't end up with empty train/val
    if not train_indices or not val_indices:
        # fallback simple split while keeping random seed
        all_indices = list(indices)
        rng.shuffle(all_indices)
        n_val = max(1, int(round(len(all_indices) * val_ratio)))
        val_indices = all_indices[:n_val]
        train_indices = all_indices[n_val:]

    return train_indices, val_indices


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
