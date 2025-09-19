import pytest

from train.utils import ensure_dir, ensure_parent_dir, flatten_config


def test_flatten_config_merges_nested_keys():
    config = {
        "architecture": "deepseekv3",
        "training": {"lr": 3e-4, "batch_size": 8},
        "model": {"num_layers": 4},
    }

    flat = flatten_config(config)

    assert flat["architecture"] == "deepseekv3"
    assert flat["lr"] == pytest.approx(3e-4)
    assert flat["batch_size"] == 8
    assert flat["num_layers"] == 4


def test_ensure_dir_creates_directory(tmp_path):
    target = tmp_path / "logs"
    assert not target.exists()

    ensure_dir(str(target))

    assert target.exists()
    assert target.is_dir()


def test_ensure_parent_dir_makes_parent(tmp_path):
    file_path = tmp_path / "checkpoints" / "model.pt"
    assert not file_path.parent.exists()

    ensure_parent_dir(str(file_path))

    assert file_path.parent.exists()
    assert file_path.parent.is_dir()
