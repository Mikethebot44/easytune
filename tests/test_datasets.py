from PIL import Image
import torch

from easytune.datasets import ImageLabelDataset, TextDataset


class DummyTokenizer:
    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        max_len = max(len(t) for t in texts)
        input_ids = []
        attn = []
        for t in texts:
            l = len(t)
            ids = list(range(1, l+1)) + [0]*(max_len-l)
            mask = [1]*l + [0]*(max_len-l)
            input_ids.append(ids)
            attn.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def test_image_label_dataset(tmp_path):
    # Create two small images
    img1 = Image.new("RGB", (8, 8), color=(255, 0, 0))
    img2 = Image.new("RGB", (8, 8), color=(0, 255, 0))
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    img1.save(p1)
    img2.save(p2)

    # Minimal transform that returns 'pixel_values'
    def transform(images, return_tensors="pt"):
        import numpy as np
        if hasattr(images, "size") and not isinstance(images, torch.Tensor):
            arr = np.array(images, dtype="float32") / 255.0  # HWC
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
        else:
            tensor = images
        return {"pixel_values": tensor.unsqueeze(0)}

    ds = ImageLabelDataset([str(p1), str(p2)], [0, 1], transform=transform)
    batch, label = ds[0]
    assert "pixel_values" in batch
    assert label.item() in (0, 1)


def test_text_dataset():
    tok = DummyTokenizer()
    ds = TextDataset(["hello", "world!"], [0, 1], tokenizer=tok)
    batch, label = ds[1]
    assert "input_ids" in batch and "attention_mask" in batch
    assert label.item() in (0, 1)
