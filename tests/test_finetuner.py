import types
import torch
from torch.utils.data import DataLoader, Dataset

from easytune.finetuner import FineTuner


class TinyAdapter(torch.nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.encoder = torch.nn.Linear(4, dim)
        self.hidden_size = dim

    def forward(self, batch):
        return self.encoder(batch["pixel_values"])  # (B,4)->(B,dim)

    def parameters(self, recurse=True):
        return super().parameters(recurse=recurse)


class TinyDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, idx):
        x = torch.randn(4)
        y = torch.tensor(0 if idx < 10 else 1, dtype=torch.long)
        return {"pixel_values": x}, y


def test_train_loop_smoke(monkeypatch):
    # Initialize tuner with injected tiny adapter to avoid downloads
    tuner = FineTuner(model="dummy", task="image-similarity", device="cpu", adapter=TinyAdapter())

    def fake_build_dataloaders(X_train, y_train, X_val, y_val, batch_size):
        train = DataLoader(TinyDataset(), batch_size=4, shuffle=True)
        val = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
        return train, val

    tuner._items = list(range(20))
    tuner._labels = [0]*10 + [1]*10
    monkeypatch.setattr(tuner, "_build_dataloaders", fake_build_dataloaders)

    out = tuner.train(epochs=1, batch_size=4, learning_rate=1e-3)
    assert out is tuner
