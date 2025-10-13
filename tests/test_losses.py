import torch
from easytune.losses import NTXentLoss


def test_ntxent_loss_runs():
    # Two classes in a batch of 6
    labels = torch.tensor([0,0,0,1,1,1])
    # Make class 0 embeddings similar, class 1 similar, cross-class different
    emb0 = torch.randn(3, 16)
    emb1 = torch.randn(3, 16) + 2.0  # shift to separate
    embeddings = torch.cat([emb0, emb1], dim=0)
    loss = NTXentLoss(temperature=0.1)(embeddings, labels)
    assert torch.isfinite(loss).item() is True
    assert loss.dim() == 0
