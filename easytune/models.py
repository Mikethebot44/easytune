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


class ImageAdapter(BaseAdapter):
    def __init__(self, model_name: str):
        # facebook/dinov2-base
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name)

        # Determine embedding dimension heuristically
        # Many vision Transformers expose last_hidden_state; pool with mean over tokens
        sample_hidden_size = base_model.config.hidden_size
        projection_head = ProjectionHead(sample_hidden_size)

        super().__init__(base_model=base_model, projection_head=projection_head)
        self.image_processor = image_processor
        self.hidden_size = sample_hidden_size

    def get_embedding_dim(self) -> int:
        return self.hidden_size

    @torch.no_grad()
    def preprocess(self, images):
        # images: List[PIL.Image.Image] or tensor
        return self.image_processor(images=images, return_tensors="pt")

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.base_model(pixel_values=batch["pixel_values"])  # type: ignore[arg-type]
        hidden = outputs.last_hidden_state  # (B, S, D)
        pooled = hidden.mean(dim=1)  # simple mean pooling across tokens
        return self.projection_head(pooled)


class TextAdapter(BaseAdapter):
    def __init__(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        base_model = AutoModel.from_pretrained(model_name)
        hidden_size = getattr(base_model.config, "hidden_size", None) or getattr(
            base_model.config, "word_embedding_dim", None
        )
        if hidden_size is None:
            raise ValueError("Unable to infer hidden size from base model config")
        projection_head = ProjectionHead(hidden_size)
        super().__init__(base_model=base_model, projection_head=projection_head)
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size

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
        token_embeddings = outputs.last_hidden_state  # (B, S, D)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            # Mean pool over tokens using attention mask
            mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
        else:
            pooled = token_embeddings.mean(dim=1)
        return self.projection_head(pooled)


def load_image_model(name: str) -> Tuple[ImageAdapter, AutoImageProcessor]:
    adapter = ImageAdapter(name)
    return adapter, adapter.image_processor


def load_text_model(name: str) -> Tuple[TextAdapter, AutoTokenizer]:
    adapter = TextAdapter(name)
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
        adapter = TextAdapter(config.base_model_name)
    elif config.adapter_type == "image":
        adapter = ImageAdapter(config.base_model_name)
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
