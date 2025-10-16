from __future__ import annotations

import modal
 


def build_image():
    return (
        modal.Image.debian_slim()
        .pip_install(
            "torch",
            "torchvision",
            "torchaudio",
            "transformers",
            "datasets",
            "tqdm",
            "pillow",
            "numpy",
        )
    )


app = modal.App("easytune-cloud")
artifacts = modal.Volume.from_name("easytune-artifacts", create_if_missing=True)
src_volume = modal.Volume.from_name("easytune-src", create_if_missing=True)
data_volume = modal.Volume.from_name("easytune-dataset", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=60 * 60,
    image=build_image(),
    volumes={
        "/artifacts": artifacts,
        "/workspace": src_volume,
        "/data": data_volume,
    },
)
def train_remote(
    model: str,
    task: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    data_dir: str,
):
    import sys
    import os
    # Be resilient to how the source is uploaded into the Volume
    # Prefer /workspace/easytune; otherwise, search for a parent that contains the package
    if os.path.isdir("/workspace/easytune"):
        sys.path.insert(0, "/workspace")
    else:
        for root, dirs, files in os.walk("/workspace"):
            if "easytune" in dirs and os.path.isfile(os.path.join(root, "easytune", "__init__.py")):
                sys.path.insert(0, root)
                break
    from easytune.finetuner import FineTuner

    tuner = FineTuner(model=model, task=task, device="cuda")
    if task == "image-similarity":
        # Resolve dataset path flexibly depending on how it was uploaded
        candidates = []
        if data_dir:
            candidates.append(data_dir)
        # common default layouts when uploading without remote path
        candidates.extend(["/data/dataset", "/data"])  
        dataset_path = None
        for p in candidates:
            if os.path.isdir(p):
                dataset_path = p
                break
        if dataset_path is None:
            raise FileNotFoundError("No dataset directory found under /data. Upload your dataset Volume or set --data-dir.")
        tuner.add_from_folder(dataset_path, validation_split=0.2)
    else:
        raise ValueError("Only image-similarity example is implemented in this script.")

    tuner.train(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=0.07,
        seed=42,
    )
    tuner.save("/artifacts/latest")


@app.local_entrypoint()
def main(
    model: str = "facebook/dinov2-base",
    task: str = "image-similarity",
    data_dir: str = "",
    out_dir: str = "",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    gpu: str = "A10G",
    timeout_s: int = 3600,
):
    if not out_dir:
        raise SystemExit("--out-dir is required")

    handle = train_remote.spawn(
        model=model,
        task=task,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        data_dir=(data_dir or "/data"),
    )
    handle.get()

    # Download artifacts to local path
    artifacts.download(out_dir)
    print(f"Artifacts downloaded to: {out_dir}")


