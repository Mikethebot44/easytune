from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Supervised NT-Xent (normalized temperature-scaled cross-entropy) loss.

    Encourages embeddings with the same label to have higher cosine similarity
    than those with different labels. Works with batches containing at least
    two samples per class for the strongest signal but will gracefully handle
    small batches as well.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be (batch, dim)")
        if embeddings.size(0) != labels.size(0):
            raise ValueError("embeddings and labels batch dimension must match")

        # Normalize to unit vectors for cosine similarity
        embeddings = F.normalize(embeddings, dim=1)

        similarity = torch.mm(embeddings, embeddings.t())  # (B, B)
        logits = similarity / self.temperature

        # Mask out self-similarity using a very negative number (avoid -inf -> NaNs)
        batch_size = embeddings.size(0)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        logits = logits.masked_fill(self_mask, -1e9)

        # For each anchor i, positives are j where labels[j] == labels[i]
        labels = labels.view(-1)
        matches = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        matches = matches & (~self_mask)

        # Compute log-softmax over all non-self entries
        log_probs = F.log_softmax(logits, dim=1)

        # Compute mean log prob over positives for each anchor
        # If an anchor has no positive, skip it in the average
        positive_counts = matches.sum(dim=1)  # (B,)
        # Avoid division by zero; we'll mask these out later
        positive_counts_clamped = positive_counts.clamp(min=1)

        positive_log_probs = (log_probs * matches).sum(dim=1) / positive_counts_clamped

        # Mask anchors with no positives
        valid_mask = positive_counts > 0
        if valid_mask.any():
            loss = -positive_log_probs[valid_mask].mean()
        else:
            # Fallback: if no positives exist in the batch, minimize uniformity
            # by pushing embeddings apart (maximize entropy)
            loss = -(log_probs[~self_mask].mean())

        return loss


def contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    return NTXentLoss(temperature=temperature)(embeddings, labels)
