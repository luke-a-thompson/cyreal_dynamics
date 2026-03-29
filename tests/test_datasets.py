"""Tests covering dataset utilities and disk sources."""
from __future__ import annotations

import gzip
import pickle
import struct
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cyreal.datasets.ppg_dalia as ppg_dalia_module
from cyreal.transforms import BatchTransform
from cyreal.datasets import (
    CelebADataset,
    CIFAR10Dataset,
    CIFAR100Dataset,
    EMNISTDataset,
    FashionMNISTDataset,
    KMNISTDataset,
    MNISTDataset,
    PPGDaliaDataset,
)


def _write_idx_images(path, images):
    with gzip.open(path, "wb") as f:
        num, rows, cols = images.shape
        f.write(struct.pack(">IIII", 2051, num, rows, cols))
        f.write(images.tobytes())


def _write_idx_labels(path, labels):
    with gzip.open(path, "wb") as f:
        num = labels.shape[0]
        f.write(struct.pack(">II", 2049, num))
        f.write(labels.tobytes())


def _seed_fake_cifar10(tmp_path, split="train", samples_per_batch=1):
    archive_path = tmp_path / "cifar-10-python.tar.gz"
    archive_path.write_bytes(b"")
    batches_dir = tmp_path / "cifar-10-batches-py"
    batches_dir.mkdir(parents=True, exist_ok=True)

    if split == "train":
        names = [f"data_batch_{i}" for i in range(1, 6)]
    elif split == "test":
        names = ["test_batch"]
    else:
        raise ValueError("split must be 'train' or 'test'.")

    current_label = 0
    for name in names:
        data = []
        labels = []
        for _ in range(samples_per_batch):
            pixel_value = current_label % 256
            image = np.full((3, 32, 32), pixel_value, dtype=np.uint8)
            data.append(image.reshape(-1))
            labels.append(current_label)
            current_label += 1
        batch = {"data": np.stack(data, axis=0), "labels": labels}
        with open(batches_dir / name, "wb") as f:
            pickle.dump(batch, f, protocol=2)


def _seed_fake_cifar100(tmp_path, split="train", samples=4):
    archive_path = tmp_path / "cifar-100-python.tar.gz"
    archive_path.write_bytes(b"")
    target_dir = tmp_path / "cifar-100-python"
    target_dir.mkdir(parents=True, exist_ok=True)

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")

    data = []
    fine_labels = []
    coarse_labels = []
    for idx in range(samples):
        pixel_value = idx % 256
        image = np.full((3, 32, 32), pixel_value, dtype=np.uint8)
        data.append(image.reshape(-1))
        fine_labels.append(idx)
        coarse_labels.append(idx % 5)
    batch = {
        "data": np.stack(data, axis=0),
        "fine_labels": fine_labels,
        "coarse_labels": coarse_labels,
    }
    with open(target_dir / split, "wb") as f:
        pickle.dump(batch, f, protocol=2)


def _seed_fake_celeba(tmp_path: Path):
    pil = pytest.importorskip("PIL.Image")
    image_module = pil

    image_dir = tmp_path / "img_align_celeba"
    image_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("000001.jpg", 0, np.array([1, -1, 1], dtype=np.int8), 10),
        ("000002.jpg", 1, np.array([-1, 1, -1], dtype=np.int8), 40),
        ("000003.jpg", 2, np.array([1, 1, -1], dtype=np.int8), 90),
    ]

    for name, _, _, pixel in samples:
        image = np.full((4, 5, 3), pixel, dtype=np.uint8)
        image_module.fromarray(image, mode="RGB").save(image_dir / name)

    with (tmp_path / "list_eval_partition.txt").open("w", encoding="utf-8") as f:
        f.write(f"{len(samples)}\n")
        for name, partition, _, _ in samples:
            f.write(f"{name} {partition}\n")

    with (tmp_path / "list_attr_celeba.txt").open("w", encoding="utf-8") as f:
        f.write(f"{len(samples)}\n")
        f.write("Smiling Young Male\n")
        for name, _, attrs, _ in samples:
            values = " ".join(str(int(v)) for v in attrs)
            f.write(f"{name} {values}\n")


def _seed_fake_ppg_dalia_outer_archive(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    inner_archive_path = raw_dir / "fixture_inner.zip"
    with zipfile.ZipFile(inner_archive_path, "w") as inner_zip:
        for subject_id in range(1, 16):
            time_steps = 512
            acc_steps = time_steps // 2
            slow_steps = time_steps // 16
            label_steps = 80

            offset = float(subject_id)
            payload = {
                "signal": {
                    "wrist": {
                        "ACC": np.stack(
                            [
                                np.linspace(offset, offset + 1.0, acc_steps, dtype=np.float32),
                                np.linspace(offset + 1.0, offset + 2.0, acc_steps, dtype=np.float32),
                                np.linspace(offset + 2.0, offset + 3.0, acc_steps, dtype=np.float32),
                            ],
                            axis=1,
                        ),
                        "BVP": np.linspace(offset, offset + 2.0, time_steps, dtype=np.float32),
                        "EDA": np.linspace(offset, offset + 0.5, slow_steps, dtype=np.float32),
                        "TEMP": np.linspace(offset + 0.25, offset + 0.75, slow_steps, dtype=np.float32),
                    }
                },
                "label": np.linspace(offset, offset + 1.0, label_steps, dtype=np.float32),
            }
            inner_zip.writestr(
                f"PPG_FieldStudy/S{subject_id}/S{subject_id}.pkl",
                pickle.dumps(payload, protocol=2),
            )

    outer_archive_path = raw_dir / "ppg+dalia.zip"
    with zipfile.ZipFile(outer_archive_path, "w") as outer_zip:
        outer_zip.write(inner_archive_path, arcname="nested/data.zip")

    inner_archive_path.unlink()


MNIST_LIKE_DATASETS = [
    pytest.param(MNISTDataset, {}, id="mnist"),
    pytest.param(FashionMNISTDataset, {}, id="fashion"),
    pytest.param(KMNISTDataset, {}, id="kmnist"),
    pytest.param(EMNISTDataset, {"subset": "letters"}, id="emnist-letters"),
]


@pytest.mark.parametrize("dataset_cls,extra_kwargs", MNIST_LIKE_DATASETS)
def test_idx_dataset_reads_cached_idx(tmp_path, dataset_cls, extra_kwargs):
    num, rows, cols = 3, 2, 2
    images = np.arange(num * rows * cols, dtype=np.uint8).reshape(num, rows, cols)
    labels = np.arange(num, dtype=np.uint8)

    images_path = tmp_path / "train_images.gz"
    labels_path = tmp_path / "train_labels.gz"
    _write_idx_images(images_path, images)
    _write_idx_labels(labels_path, labels)

    dataset = dataset_cls(split="train", cache_dir=tmp_path, **extra_kwargs)
    assert len(dataset) == num
    example = dataset[0]
    img = example["image"]
    label = example["label"]
    assert img.shape == (rows, cols, 1)
    assert img.dtype == np.uint8
    assert label == 0
    np.testing.assert_array_equal(img[..., 0], images[0])


@pytest.mark.parametrize("dataset_cls,extra_kwargs", MNIST_LIKE_DATASETS)
def test_idx_disk_sources_stream_from_disk(tmp_path, dataset_cls, extra_kwargs):
    num, rows, cols = 3, 2, 2
    images = np.arange(num * rows * cols, dtype=np.uint8).reshape(num, rows, cols)
    labels = np.arange(num, dtype=np.uint8)

    images_path = tmp_path / "train_images.gz"
    labels_path = tmp_path / "train_labels.gz"
    _write_idx_images(images_path, images)
    _write_idx_labels(labels_path, labels)

    common_kwargs = {
        "split": "train",
        "cache_dir": tmp_path,
        "ordering": "sequential",
        "prefetch_size": 2,
    }
    common_kwargs.update(extra_kwargs)
    source = dataset_cls.make_disk_source(**common_kwargs)

    batched = BatchTransform(
        batch_size=2,
        pad_last_batch=True,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, state = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch["image"][..., 0]), images[:2])
    np.testing.assert_array_equal(np.asarray(batch["label"]), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))

    batch2, mask2, _ = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch2["image"][0, ..., 0]), images[2])
    np.testing.assert_array_equal(np.asarray(batch2["label"][:1]), np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask2), np.array([True, False]))


def test_cifar10_disk_source_streams_from_disk(tmp_path):
    _seed_fake_cifar10(tmp_path, split="test", samples_per_batch=3)

    source = CIFAR10Dataset.make_disk_source(
        split="test",
        cache_dir=tmp_path,
        ordering="sequential",
        prefetch_size=2,
    )
    batched = BatchTransform(
        batch_size=2,
        pad_last_batch=True,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, state = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch["label"]), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(batch["image"][0, 0, 0]),
        np.array([0, 0, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(batch["image"][1, 0, 0]),
        np.array([1, 1, 1], dtype=np.uint8),
    )
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))

    batch2, mask2, _ = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch2["label"][:1]), np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask2), np.array([True, False]))
    np.testing.assert_array_equal(
        np.asarray(batch2["image"][1]),
        np.zeros((32, 32, 3), dtype=np.uint8),
    )


def test_cifar100_disk_source_streams_from_disk(tmp_path):
    _seed_fake_cifar100(tmp_path, split="test", samples=3)

    source = CIFAR100Dataset.make_disk_source(
        split="test",
        cache_dir=tmp_path,
        ordering="sequential",
        prefetch_size=2,
    )

    batched = BatchTransform(
        batch_size=2,
        pad_last_batch=True,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, state = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch["label"]), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(batch["coarse_label"]), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(batch["image"][0, 0, 0]),
        np.array([0, 0, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(batch["image"][1, 0, 0]),
        np.array([1, 1, 1], dtype=np.uint8),
    )
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))

    batch2, mask2, _ = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch2["label"][:1]), np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(batch2["coarse_label"][:1]), np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask2), np.array([True, False]))


def test_celeba_dataset_reads_local_layout(tmp_path):
    _seed_fake_celeba(tmp_path)

    dataset = CelebADataset(split="valid", data_dir=tmp_path)
    assert len(dataset) == 1

    sample = dataset[0]
    assert sample["image"].shape == (4, 5, 3)
    assert sample["image"].dtype == np.uint8
    np.testing.assert_array_equal(
        np.asarray(sample["attributes"]),
        np.array([0, 1, 0], dtype=np.int32),
    )
    assert dataset.attribute_names == ("Smiling", "Young", "Male")


def test_celeba_disk_source_streams_from_disk(tmp_path):
    _seed_fake_celeba(tmp_path)

    source = CelebADataset.make_disk_source(
        split="test",
        data_dir=tmp_path,
        ordering="sequential",
        prefetch_size=2,
    )
    batched = BatchTransform(
        batch_size=2,
        pad_last_batch=True,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, _ = batched.next(state)
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, False]))
    np.testing.assert_array_equal(
        np.asarray(batch["attributes"][0]),
        np.array([1, 1, 0], dtype=np.int32),
    )
    assert np.asarray(batch["image"])[0].shape == (4, 5, 3)


def test_ppg_dalia_dataset_processes_nested_zip_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ppg_dalia_module, "_INPUT_WINDOW_SIZE", 64)
    monkeypatch.setattr(ppg_dalia_module, "_INPUT_WINDOW_STEP", 16)
    monkeypatch.setattr(ppg_dalia_module, "_OUTPUT_WINDOW_SIZE", 8)
    monkeypatch.setattr(ppg_dalia_module, "_OUTPUT_WINDOW_STEP", 4)

    _seed_fake_ppg_dalia_outer_archive(tmp_path)

    dataset = PPGDaliaDataset(split="train", cache_dir=tmp_path)
    assert len(dataset) > 0
    assert dataset.multirate_spec.driver_length == 64
    assert dataset.multirate_spec.solution_length == 8
    assert dataset.multirate_spec.downsample_factor == 8

    sample = dataset[0]
    assert sample["driver"].shape == (64, 6)
    assert sample["driver"].dtype == np.float32
    assert sample["solution"].shape == (8,)
    assert sample["solution"].dtype == np.float32

    raw_dir = tmp_path / "raw"
    assert (raw_dir / "data.zip").exists()
    assert not (raw_dir / "ppg+dalia.zip").exists()

    processed_dir = tmp_path / "processed"
    assert (processed_dir / "X_train.npy").exists()
    assert (processed_dir / "y_train.npy").exists()


def test_ppg_dalia_local_files_only_raises_when_archives_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="local_files_only=True"):
        PPGDaliaDataset(
            split="train",
            cache_dir=tmp_path,
            local_files_only=True,
        )


def test_ppg_dalia_local_files_only_uses_local_outer_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(ppg_dalia_module, "_INPUT_WINDOW_SIZE", 64)
    monkeypatch.setattr(ppg_dalia_module, "_INPUT_WINDOW_STEP", 16)
    monkeypatch.setattr(ppg_dalia_module, "_OUTPUT_WINDOW_SIZE", 8)
    monkeypatch.setattr(ppg_dalia_module, "_OUTPUT_WINDOW_STEP", 4)

    _seed_fake_ppg_dalia_outer_archive(tmp_path)

    def _download_should_not_run(_url, _path):
        raise AssertionError("download_archive should not be called in local_files_only mode")

    monkeypatch.setattr(ppg_dalia_module, "download_archive", _download_should_not_run)

    dataset = PPGDaliaDataset(
        split="train",
        cache_dir=tmp_path,
        local_files_only=True,
    )
    assert len(dataset) > 0


def test_ppg_dalia_bad_outer_zip_reports_resolved_cache_path(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    bad_archive = raw_dir / "ppg+dalia.zip"
    bad_archive.write_bytes(b"this is not a zip")

    with pytest.raises(zipfile.BadZipFile) as exc_info:
        PPGDaliaDataset(
            split="train",
            cache_dir=tmp_path,
            local_files_only=True,
        )

    msg = str(exc_info.value)
    assert str(bad_archive.resolve()) in msg
    assert str(tmp_path.resolve()) in msg
    assert "local_files_only=True" in msg


def test_ppg_dalia_disk_source_streams_processed_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(ppg_dalia_module, "_INPUT_WINDOW_SIZE", 64)
    monkeypatch.setattr(ppg_dalia_module, "_INPUT_WINDOW_STEP", 16)
    monkeypatch.setattr(ppg_dalia_module, "_OUTPUT_WINDOW_SIZE", 8)
    monkeypatch.setattr(ppg_dalia_module, "_OUTPUT_WINDOW_STEP", 4)

    _seed_fake_ppg_dalia_outer_archive(tmp_path)

    dataset = PPGDaliaDataset(split="val", cache_dir=tmp_path)
    source = PPGDaliaDataset.make_disk_source(
        split="val",
        cache_dir=tmp_path,
        ordering="sequential",
        prefetch_size=2,
    )
    assert source.multirate_spec.driver_length == 64
    assert source.multirate_spec.solution_length == 8
    assert source.multirate_spec.downsample_factor == 8

    batched = BatchTransform(
        batch_size=2,
        pad_last_batch=True,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, _ = batched.next(state)
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))
    np.testing.assert_allclose(np.asarray(batch["driver"][0]), np.asarray(dataset[0]["driver"]))
    np.testing.assert_allclose(
        np.asarray(batch["solution"][0]), np.asarray(dataset[0]["solution"])
    )
