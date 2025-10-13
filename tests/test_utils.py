import os
import tempfile
from easytune.utils import stratified_split, ensure_int_labels, validate_files_exist


def test_ensure_int_labels():
    assert ensure_int_labels(["0", 1, 2.0]) == [0, 1, 2]


def test_stratified_split_distribution():
    labels = [0]*8 + [1]*2
    idx = list(range(len(labels)))
    train_idx, val_idx = stratified_split(idx, labels, val_ratio=0.2, seed=123)
    # Check non-empty
    assert len(train_idx) > 0 and len(val_idx) > 0
    # Check no overlap
    assert set(train_idx).isdisjoint(set(val_idx))


def test_validate_files_exist(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("ok")
    valid = validate_files_exist([str(f1), str(f2)])
    assert str(f1) in valid
