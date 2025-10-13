# easytune

EasyTune is a lightweight Python library that abstracts away the complexity of fine-tuning embedding models (image and text). The MVP provides a simple API for loading a model, adding labeled data, training with a contrastive objective, and saving/loading the resulting model.

## Quickstart

```python
from easytune import FineTuner

# Image similarity
tuner = FineTuner(model="facebook/dinov2-base", task="image-similarity")
tuner.add_from_folder("./dataset/")  # expects class subfolders
model = tuner.train(epochs=5, batch_size=16, learning_rate=1e-5)
model.save("fine_tuned_dinov2")

# Text similarity
tuner = FineTuner(model="sentence-transformers/all-MiniLM-L6-v2", task="text-similarity")
tuner.add_data(texts=["hello", "hi there", "goodbye"], labels=[0,0,1])
model = tuner.train(epochs=3)
model.save("fine_tuned_miniLM")
```

## Installation

```bash
pip install -e .
```

## Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.x
- TQDM, Pillow, NumPy, Pandas, Torchvision

## Notes
- Uses a supervised NT-Xent contrastive loss.
- Automatic train/validation split with stratification.
- Runs on CPU or GPU automatically.
