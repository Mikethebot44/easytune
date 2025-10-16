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

## Inference (2 lines)

```python
from easytune import EasyModel, SimpleIndex
m = EasyModel.load('./cloud_artifacts/latest')             # or a local save dir
E = m.embed_images(['a.jpg','b.jpg','c.jpg'])             # or m.embed_texts([...])
idx = SimpleIndex.from_embeddings(E, ids=['a','b','c'])
print(idx.search(m.embed_images(['query.jpg']), k=3))
```

## Cloud (Optional)

You can run training on a cloud GPU without changing your code using the optional Modal backend (no AWS required).

1) Install Modal CLI and authenticate:

```bash
pip install modal
modal token new
```

2) Create or point to a local dataset directory (for image tasks it expects class subfolders).

3) Run via the provided example (downloads artifacts to `./cloud_artifacts`):

```bash
modal run examples/remote/modal_easytune.py --data-dir ./dataset --out-dir ./cloud_artifacts --model facebook/dinov2-base --task image-similarity
```

Or invoke programmatically through the SDK convenience method:

```python
from easytune.finetuner import FineTuner
from easytune.cloud.modal_backend import ModalBackend

tuner = FineTuner(model="facebook/dinov2-base", task="image-similarity")
tuner.train_remote(ModalBackend(), data_dir="./dataset", out_dir="./cloud_artifacts", epochs=5, batch_size=16, learning_rate=1e-5)
```

Notes:
- Modal is optional and not installed by default; the backend uses the `modal` CLI.
- To change GPU type/cost, pass `--gpu` to the example or `gpu="A10G"` to `ModalBackend`.