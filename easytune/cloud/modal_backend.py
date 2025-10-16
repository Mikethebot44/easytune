from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .base import RemoteBackend


class ModalBackend(RemoteBackend):
    """Run EasyTune training remotely via Modal CLI with auto-upload/download.

    - Automatically uploads source code to a persistent Volume (`easytune-src`)
    - Automatically uploads dataset directory to a persistent Volume (`easytune-dataset`)
    - Launches the remote training function
    - Downloads artifacts from `easytune-artifacts` Volume into the requested local folder
    """

    def __init__(
        self,
        *,
        modal_entry: str = "examples/remote/modal_easytune.py",
        artifacts_volume: str = "easytune-artifacts",
        src_volume: str = "easytune-src",
        dataset_volume: str = "easytune-dataset",
        reuse_uploads: bool = True,
        stream_logs: bool = True,
    ) -> None:
        self.modal_entry = modal_entry
        self.artifacts_volume = artifacts_volume
        self.src_volume = src_volume
        self.dataset_volume = dataset_volume
        self.reuse_uploads = reuse_uploads
        self.stream_logs = stream_logs

    # ---------------------- Internal helpers ----------------------
    def _modal_cmd(self) -> list[str]:
        exe = shutil.which("modal")
        return [exe] if exe else [sys.executable, "-m", "modal"]

    def _run_modal(self, args: list[str], *, cwd: Optional[Path] = None) -> None:
        cmd = self._modal_cmd() + args
        subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)

    def _run_modal_stream(self, args: list[str], *, cwd: Optional[Path] = None) -> None:
        """Run modal command and stream stdout/stderr for live feedback (e.g., in notebooks)."""
        cmd = self._modal_cmd() + args
        with subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"Command failed: {' '.join(cmd)} (exit {ret})")

    def _resolve_repo_root(self) -> Path:
        # Prefer the current working directory upwards
        cwd = Path.cwd()
        for p in [cwd, *cwd.parents]:
            if (p / "setup.py").exists() and (p / "easytune").exists():
                return p
        # Fallback: derive from this file's location (easytune/easytune/cloud/)
        here = Path(__file__).resolve()
        for p in [here.parent.parent.parent, here.parent.parent.parent.parent]:
            if p.is_dir() and (p / "setup.py").exists() and (p / "easytune").exists():
                return p
        # Last resort: cwd
        return cwd

    def _ensure_dir(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    # ---------------------- Public API ----------------------
    def train(
        self,
        *,
        model: str,
        task: str,
        data_dir: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        out_dir: str,
        gpu: Optional[str] = None,  # kept for API parity; not all CLIs support overriding via flags
        timeout_s: int = 3600,       # kept for API parity
    ) -> None:
        # Validate dataset path if it's local (no leading "/")
        is_remote_data = data_dir.startswith("/")
        if not is_remote_data and not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        repo_root = self._resolve_repo_root()

        # 1) Upload source code into persistent volume (skip if reuse requested and volume already has files)
        src_has_content = self._volume_has_content(self.src_volume)
        if not (self.reuse_uploads and src_has_content):
            self._run_modal(["volume", "put", self.src_volume, ".", "/"], cwd=repo_root)

        # 2) Upload dataset into persistent volume unless a remote path is provided
        if is_remote_data:
            remote_data = data_dir  # e.g., "/data" or "/data/train"
        else:
            dataset_path = Path(data_dir).resolve()
            data_has_content = self._volume_has_content(self.dataset_volume)
            if not (self.reuse_uploads and data_has_content):
                self._run_modal(["volume", "put", self.dataset_volume, ".", "/"], cwd=dataset_path)
            remote_data = "/data"

        # 3) Run the cloud training job; artifacts stored into easytune-artifacts Volume
        modal_args = [
            "run",
            self.modal_entry,
            "--model",
            model,
            "--task",
            task,
            "--out-dir",
            out_dir,
            "--data-dir",
            remote_data,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
        ]
        # Not all Modal versions accept these flags at runtime; prefer decorator defaults
        # if gpu:
        #     modal_args.extend(["--gpu", gpu])
        # modal_args.extend(["--timeout-s", str(timeout_s)])

        if self.stream_logs:
            self._run_modal_stream(modal_args, cwd=repo_root)
        else:
            self._run_modal(modal_args, cwd=repo_root)

        # 4) Download artifacts into out_dir from the artifacts volume
        self._ensure_dir(out_dir)
        self._run_modal(["volume", "get", self.artifacts_volume, "/latest", out_dir], cwd=repo_root)

    def _volume_has_content(self, volume_name: str) -> bool:
        """Best-effort check: true if listing the root of the volume returns any entries."""
        try:
            cmd = self._modal_cmd() + ["volume", "ls", volume_name, "/"]
            out = subprocess.run(cmd, capture_output=True, text=True)
            # modal returns 0 even if empty; look for at least one non-empty line
            if out.returncode != 0:
                return False
            lines = [ln.strip() for ln in (out.stdout or "").splitlines()]
            return any(lines)
        except Exception:
            return False


