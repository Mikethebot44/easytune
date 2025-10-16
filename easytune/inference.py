from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .finetuner import FineTuner, DeviceType


AdapterKind = Literal["image", "text"]


@dataclass
class EasyModel:
    """Minimal, beginner-friendly inference wrapper for trained EasyTune models.

    Usage:
        m = EasyModel.load("./cloud_artifacts/latest")
        vecs = m.embed_images(["a.jpg", "b.jpg"])  # if image adapter
    """

    tuner: FineTuner
    adapter_type: AdapterKind

    @staticmethod
    def load(path: str, device: DeviceType = "auto") -> "EasyModel":
        tuner = FineTuner.load(path, device=device)
        adapter_type: AdapterKind = "image" if tuner._adapter_type == "image" else "text"
        return EasyModel(tuner=tuner, adapter_type=adapter_type)

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        if self.adapter_type != "image":
            raise ValueError("This model is a text adapter; use embed_texts().")
        # Build a tiny one-off DataLoader-like loop
        from PIL import Image
        processor = getattr(self.tuner, "_processor")
        self.tuner.adapter.eval()
        embs: List[np.ndarray] = []
        with torch.no_grad():
            for p in image_paths:
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    batch = processor(images=img, return_tensors="pt")
                    batch = {k: v.to(self.tuner.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    vec = self.tuner.adapter(batch)  # (1, D)
                    embs.append(vec.detach().cpu().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, getattr(self.tuner.adapter, "hidden_size", 0)))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self.adapter_type != "text":
            raise ValueError("This model is an image adapter; use embed_images().")
        tokenizer = getattr(self.tuner, "_tokenizer")
        self.tuner.adapter.eval()
        embs: List[np.ndarray] = []
        with torch.no_grad():
            for t in texts:
                batch = tokenizer(t, padding=True, truncation=True, return_tensors="pt")
                batch = {k: v.to(self.tuner.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                vec = self.tuner.adapter(batch)  # (1, D)
                embs.append(vec.detach().cpu().numpy())
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, getattr(self.tuner.adapter, "hidden_size", 0)))

    def embed_any(self, items: Sequence[Union[str, "Image.Image"]]) -> np.ndarray:
        if self.adapter_type == "image":
            # items are file paths or PIL Images
            from PIL import Image
            processor = getattr(self.tuner, "_processor")
            self.tuner.adapter.eval()
            embs: List[np.ndarray] = []
            with torch.no_grad():
                for it in items:
                    if isinstance(it, str):
                        with Image.open(it) as img:
                            img = img.convert("RGB")
                            batch = processor(images=img, return_tensors="pt")
                    else:
                        batch = processor(images=it.convert("RGB"), return_tensors="pt")
                    batch = {k: v.to(self.tuner.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    vec = self.tuner.adapter(batch)
                    embs.append(vec.detach().cpu().numpy())
            return np.concatenate(embs, axis=0) if embs else np.zeros((0, getattr(self.tuner.adapter, "hidden_size", 0)))
        else:
            return self.embed_texts([str(x) for x in items])


class SimpleIndex:
    """Tiny cosine-similarity index using NumPy only.

    This is intentionally minimal for ease-of-use. For larger datasets, users can
    swap in FAISS if installed; we will auto-detect and use it if present.
    """

    def __init__(self, embeddings: np.ndarray, ids: Sequence[Any]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D (N, D)")
        if embeddings.shape[0] != len(ids):
            raise ValueError("embeddings rows must match number of ids")
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.ids = list(ids)
        self._faiss = None
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(self.embeddings.shape[1])
            # normalize for cosine as dot product
            normed = self._l2_normalize(self.embeddings)
            index.add(normed)
            self._faiss = (faiss, index)
            self._faiss_normed = normed
        except Exception:
            self._faiss = None

        # Precompute normalized if not using FAISS
        if self._faiss is None:
            self._normed = self._l2_normalize(self.embeddings)

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=eps)
        return x / norms

    @staticmethod
    def from_embeddings(embeddings: np.ndarray, ids: Sequence[Any]) -> "SimpleIndex":
        return SimpleIndex(embeddings, ids)

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> List[List[Tuple[Any, float]]]:
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[None, :]
        if self._faiss is not None:
            faiss, index = self._faiss
            q = self._l2_normalize(query_embeddings.astype(np.float32, copy=False))
            scores, idxs = index.search(q, k)
            results: List[List[Tuple[Any, float]]] = []
            for row_scores, row_idxs in zip(scores, idxs):
                row: List[Tuple[Any, float]] = []
                for j, s in zip(row_idxs, row_scores):
                    if j == -1:
                        continue
                    row.append((self.ids[int(j)], float(s)))
                results.append(row)
            return results

        # NumPy fallback
        q = self._l2_normalize(query_embeddings.astype(np.float32, copy=False))
        # cosine = dot product when vectors are L2-normalized
        sims = q @ self._normed.T  # (Q, N)
        results_np: List[List[Tuple[Any, float]]] = []
        for row in sims:
            topk = np.argpartition(-row, kth=min(k, row.shape[0]-1))[:k]
            topk_sorted = topk[np.argsort(-row[topk])]
            results_np.append([(self.ids[int(i)], float(row[int(i)])) for i in topk_sorted])
        return results_np



