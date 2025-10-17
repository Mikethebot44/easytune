import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
)


@dataclass
class AdapterConfig:
    base_model_name: str
    adapter_type: str  # "image" or "text"
    projection_dim: Optional[int] = None


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseAdapter(nn.Module):
    def __init__(self, base_model: nn.Module, projection_head: ProjectionHead):
        super().__init__()
        self.base_model = base_model
        self.projection_head = projection_head

    def get_embedding_dim(self) -> int:
        raise NotImplementedError

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode(batch)


# ---------------------- Helper utilities ----------------------
def _infer_hidden_size(cfg) -> int:
    """Best-effort inference of hidden size from a HF config.

    Tries common keys across encoder models and their sub-configs
    (e.g., CLIP text/vision configs).
    """
    # Direct top-level keys
    for key in [
        "hidden_size",
        "embed_dim",
        "d_model",
        "word_embedding_dimension",
        "word_embedding_dim",  # some configs use this shorter name
        "projection_dim",
    ]:
        val = getattr(cfg, key, None)
        if isinstance(val, int) and val > 0:
            return val

    # Sub-configs for dual encoders like CLIP/SigLIP
    for sub in ["text_config", "vision_config"]:
        subcfg = getattr(cfg, sub, None)
        if subcfg is not None:
            for key in [
                "hidden_size",
                "embed_dim",
                "d_model",
                "word_embedding_dimension",
                "word_embedding_dim",
                "projection_dim",
            ]:
                val = getattr(subcfg, key, None)
                if isinstance(val, int) and val > 0:
                    return val

    raise ValueError("Unable to infer hidden size from model config")


def _pool_text(outputs, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Prefer model-provided pooled embeddings; fallback to masked mean pooling."""
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        return outputs.text_embeds

    token_embeddings: torch.Tensor = outputs.last_hidden_state  # (B, S, D)
    if mask is None:
        return token_embeddings.mean(dim=1)
    m = mask.unsqueeze(-1).type_as(token_embeddings)
    summed = (token_embeddings * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _pool_image(outputs) -> torch.Tensor:
    """Prefer model-provided pooled embeddings; fallback to mean over tokens."""
    for attr in ["image_embeds", "pooler_output"]:
        if hasattr(outputs, attr):
            val = getattr(outputs, attr)
            if val is not None:
                return val
    return outputs.last_hidden_state.mean(dim=1)


class ImageAdapter(BaseAdapter):
    def __init__(self, model_name: str, projection_dim: Optional[int] = None, dropout: float = 0.0):
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name)

        hidden = _infer_hidden_size(getattr(base_model, "config", base_model))
        head_out = projection_dim or hidden
        projection_head = ProjectionHead(hidden, head_out, dropout=dropout)

        super().__init__(base_model=base_model, projection_head=projection_head)
        self.image_processor = image_processor
        # Expose output embedding dimension for downstream consumers/metadata
        self.hidden_size = head_out
        self.projection_dim = head_out

    def get_embedding_dim(self) -> int:
        return self.hidden_size

    @torch.no_grad()
    def preprocess(self, images):
        # images: List[PIL.Image.Image] or tensor
        return self.image_processor(images=images, return_tensors="pt")

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.base_model(pixel_values=batch["pixel_values"])  # type: ignore[arg-type]
        pooled = _pool_image(outputs)
        return self.projection_head(pooled)


class TextAdapter(BaseAdapter):
    def __init__(self, model_name: str, projection_dim: Optional[int] = None, dropout: float = 0.0):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        base_model = AutoModel.from_pretrained(model_name)

        hidden = _infer_hidden_size(getattr(base_model, "config", base_model))
        head_out = projection_dim or hidden
        projection_head = ProjectionHead(hidden, head_out, dropout=dropout)

        super().__init__(base_model=base_model, projection_head=projection_head)
        self.tokenizer = tokenizer
        # Expose output embedding dimension for downstream consumers/metadata
        self.hidden_size = head_out
        self.projection_dim = head_out

    def get_embedding_dim(self) -> int:
        return self.hidden_size

    @torch.no_grad()
    def tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")
        )
        pooled = _pool_text(outputs, batch.get("attention_mask"))
        return self.projection_head(pooled)


def load_image_model(name: str, projection_dim: Optional[int] = None, dropout: float = 0.0) -> Tuple[ImageAdapter, AutoImageProcessor]:
    adapter = ImageAdapter(name, projection_dim=projection_dim, dropout=dropout)
    return adapter, adapter.image_processor


def load_text_model(name: str, projection_dim: Optional[int] = None, dropout: float = 0.0) -> Tuple[TextAdapter, AutoTokenizer]:
    adapter = TextAdapter(name, projection_dim=projection_dim, dropout=dropout)
    return adapter, adapter.tokenizer


def save_adapter(adapter: BaseAdapter, path: str) -> None:
    # Save base model and processor/tokenizer if available
    import os
    from transformers.utils import WEIGHTS_NAME

    os.makedirs(path, exist_ok=True)
    base_path = os.path.join(path, "base")
    adapter.base_model.save_pretrained(base_path)

    if isinstance(adapter, TextAdapter):
        tok_path = os.path.join(path, "tokenizer")
        adapter.tokenizer.save_pretrained(tok_path)
    elif isinstance(adapter, ImageAdapter):
        proc_path = os.path.join(path, "processor")
        adapter.image_processor.save_pretrained(proc_path)

    # Save projection head separately
    torch.save(adapter.projection_head.state_dict(), os.path.join(path, "head.pt"))


def load_adapter(config: AdapterConfig, path: str) -> BaseAdapter:
    # Recreate adapter and load weights
    if config.adapter_type == "text":
        adapter = TextAdapter(config.base_model_name, projection_dim=config.projection_dim)
    elif config.adapter_type == "image":
        adapter = ImageAdapter(config.base_model_name, projection_dim=config.projection_dim)
    else:
        raise ValueError(f"Unknown adapter type: {config.adapter_type}")

    base_path = f"{path}/base"
    # Reload base model/tokenizer/processor from saved directories to ensure parity
    adapter.base_model = AutoModel.from_pretrained(base_path)
    if isinstance(adapter, TextAdapter):
        tok_path = f"{path}/tokenizer"
        if os.path.isdir(tok_path):
            adapter.tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    elif isinstance(adapter, ImageAdapter):
        proc_path = f"{path}/processor"
        if os.path.isdir(proc_path):
            adapter.image_processor = AutoImageProcessor.from_pretrained(proc_path)

    # Load projection head weights
    head_state = torch.load(f"{path}/head.pt", map_location="cpu")
    adapter.projection_head.load_state_dict(head_state)
    return adapter
