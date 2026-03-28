"""PPG-DaLiA dataset utilities with nested-zip caching."""

from __future__ import annotations

import json
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _sliding_window_view

from .dataset_protocol import DatasetProtocol
from .utils import (
    download_archive,
    ensure_zip_member_extracted,
    resolve_cache_dir,
    to_host_jax_array as _to_host_jax_array,
)
from ..sources import DiskSource

PPG_DALIA_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"

_DEFAULT_CACHE_NAME = "cyreal_ppg_dalia"
_OUTER_ARCHIVE_NAME = "ppg+dalia.zip"
_INNER_ARCHIVE_NAME = "data.zip"
_PROCESSING_VERSION = "v1"
_DEFAULT_VARIANT_SEED = 0
_SUBJECT_IDS = tuple(range(1, 16))
_VARIANT_COUNT = 6
_RAW_SUBJECT_PREFIX = "data/PPG_FieldStudy"

_INPUT_WINDOW_SIZE = 49_920
_INPUT_WINDOW_STEP = 4_992
_OUTPUT_WINDOW_SIZE = 390
_OUTPUT_WINDOW_STEP = 39


def _normalize_signed(array: np.ndarray) -> np.ndarray:
    array_np = np.asarray(array, dtype=np.float32)
    min_value = float(np.min(array_np))
    max_value = float(np.max(array_np))
    if max_value <= min_value:
        return np.zeros_like(array_np, dtype=np.float32)
    scaled = 2.0 * (array_np - min_value) / (max_value - min_value) - 1.0
    return np.asarray(scaled, dtype=np.float32)


def _as_column(array: np.ndarray) -> np.ndarray:
    array_np = np.asarray(array, dtype=np.float32)
    if array_np.ndim == 1:
        return array_np[:, None]
    return array_np


def _subject_member_name(subject_id: int) -> str:
    return f"{_RAW_SUBJECT_PREFIX}/S{subject_id}/S{subject_id}.pkl"


def _find_inner_archive_member(outer_archive_path: Path) -> str:
    with zipfile.ZipFile(outer_archive_path, "r") as zf:
        candidates = [
            info.filename
            for info in zf.infolist()
            if not info.is_dir() and Path(info.filename).name == _INNER_ARCHIVE_NAME
        ]
    if not candidates:
        raise FileNotFoundError(f"Could not find '{_INNER_ARCHIVE_NAME}' inside '{outer_archive_path}'.")
    if len(candidates) > 1:
        candidates.sort()
    return candidates[0]


def _ensure_inner_archive(base_dir: Path) -> Path:
    raw_dir = base_dir / "raw"
    inner_archive_path = raw_dir / _INNER_ARCHIVE_NAME
    if inner_archive_path.exists():
        return inner_archive_path

    outer_archive_path = raw_dir / _OUTER_ARCHIVE_NAME
    download_archive(PPG_DALIA_URL, outer_archive_path)
    member_name = _find_inner_archive_member(outer_archive_path)
    ensure_zip_member_extracted(
        outer_archive_path,
        raw_dir,
        member_name,
        target_name=_INNER_ARCHIVE_NAME,
    )
    if outer_archive_path.exists():
        outer_archive_path.unlink()
    return inner_archive_path


def _load_subject_payload(inner_archive_path: Path, subject_id: int) -> dict:
    member_name = _subject_member_name(subject_id)
    with zipfile.ZipFile(inner_archive_path, "r") as zf:
        try:
            with zf.open(member_name, "r") as f:
                return pickle.load(f, encoding="latin1")
        except KeyError as exc:
            raise FileNotFoundError(
                f"Missing archive member '{member_name}' in '{inner_archive_path}'."
            ) from exc


def _prepare_subject_arrays(subject_payload: dict) -> tuple[np.ndarray, np.ndarray]:
    wrist = subject_payload["signal"]["wrist"]

    acc = np.repeat(_as_column(wrist["ACC"]), 2, axis=0)
    bvp = _as_column(wrist["BVP"])
    eda = np.repeat(_as_column(wrist["EDA"]), 16, axis=0)
    temp = np.repeat(_as_column(wrist["TEMP"]), 16, axis=0)

    inputs = np.concatenate(
        [
            _normalize_signed(acc),
            _normalize_signed(bvp),
            _normalize_signed(eda),
            _normalize_signed(temp),
        ],
        axis=1,
    )

    labels = np.asarray(subject_payload["label"], dtype=np.float32).reshape(-1)
    if labels.size == 0:
        raise ValueError("Subject label array cannot be empty.")
    labels = np.concatenate((np.repeat(labels[:1], 3), labels), axis=0)
    labels = _normalize_signed(labels)
    return inputs, labels


def _slice_fraction(array: np.ndarray, start_fraction: float, end_fraction: float | None = None) -> np.ndarray:
    start = int(start_fraction * len(array))
    end = len(array) if end_fraction is None else int(end_fraction * len(array))
    return array[start:end]


def _split_subject_variant(
    inputs: np.ndarray,
    labels: np.ndarray,
    variant: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if variant == 0:
        return (
            _slice_fraction(inputs, 0.0, 0.7),
            _slice_fraction(labels, 0.0, 0.7),
            _slice_fraction(inputs, 0.7, 0.85),
            _slice_fraction(labels, 0.7, 0.85),
            _slice_fraction(inputs, 0.85, None),
            _slice_fraction(labels, 0.85, None),
        )
    if variant == 1:
        return (
            _slice_fraction(inputs, 0.0, 0.7),
            _slice_fraction(labels, 0.0, 0.7),
            _slice_fraction(inputs, 0.85, None),
            _slice_fraction(labels, 0.85, None),
            _slice_fraction(inputs, 0.7, 0.85),
            _slice_fraction(labels, 0.7, 0.85),
        )
    if variant == 2:
        return (
            _slice_fraction(inputs, 0.15, 0.85),
            _slice_fraction(labels, 0.15, 0.85),
            _slice_fraction(inputs, 0.0, 0.15),
            _slice_fraction(labels, 0.0, 0.15),
            _slice_fraction(inputs, 0.85, None),
            _slice_fraction(labels, 0.85, None),
        )
    if variant == 3:
        return (
            _slice_fraction(inputs, 0.15, 0.85),
            _slice_fraction(labels, 0.15, 0.85),
            _slice_fraction(inputs, 0.85, None),
            _slice_fraction(labels, 0.85, None),
            _slice_fraction(inputs, 0.0, 0.15),
            _slice_fraction(labels, 0.0, 0.15),
        )
    if variant == 4:
        return (
            _slice_fraction(inputs, 0.30, None),
            _slice_fraction(labels, 0.30, None),
            _slice_fraction(inputs, 0.0, 0.15),
            _slice_fraction(labels, 0.0, 0.15),
            _slice_fraction(inputs, 0.15, 0.30),
            _slice_fraction(labels, 0.15, 0.30),
        )
    if variant == 5:
        return (
            _slice_fraction(inputs, 0.30, None),
            _slice_fraction(labels, 0.30, None),
            _slice_fraction(inputs, 0.15, 0.30),
            _slice_fraction(labels, 0.15, 0.30),
            _slice_fraction(inputs, 0.0, 0.15),
            _slice_fraction(labels, 0.0, 0.15),
        )
    raise ValueError(f"Unsupported PPG-DaLiA split variant '{variant}'.")


def _window_inputs(inputs: np.ndarray) -> np.ndarray:
    if inputs.shape[0] < _INPUT_WINDOW_SIZE:
        raise ValueError(
            f"Input split length {inputs.shape[0]} is shorter than window size {_INPUT_WINDOW_SIZE}."
        )
    windows = _sliding_window_view(inputs, _INPUT_WINDOW_SIZE, axis=0)[::_INPUT_WINDOW_STEP]
    return np.asarray(np.swapaxes(windows, 1, 2), dtype=np.float32)


def _window_targets(labels: np.ndarray) -> np.ndarray:
    if labels.shape[0] < _OUTPUT_WINDOW_SIZE:
        raise ValueError(
            f"Target split length {labels.shape[0]} is shorter than window size {_OUTPUT_WINDOW_SIZE}."
        )
    windows = _sliding_window_view(labels, _OUTPUT_WINDOW_SIZE, axis=0)[::_OUTPUT_WINDOW_STEP]
    return np.asarray(windows, dtype=np.float32)


def _paired_windows(inputs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    context_windows = _window_inputs(inputs)
    target_windows = _window_targets(labels)
    pair_count = min(int(context_windows.shape[0]), int(target_windows.shape[0]))
    if pair_count <= 0:
        raise ValueError("PPG-DaLiA split produced no aligned context/target windows.")
    return context_windows[:pair_count], target_windows[:pair_count]


def _processed_paths(processed_dir: Path, split: Literal["train", "val", "test"]) -> tuple[Path, Path]:
    return processed_dir / f"X_{split}.npy", processed_dir / f"y_{split}.npy"


def _processed_cache_ready(processed_dir: Path) -> bool:
    required = [processed_dir / "metadata.json"]
    for split in ("train", "val", "test"):
        required.extend(_processed_paths(processed_dir, split))
    return all(path.exists() for path in required)


def _write_metadata(
    processed_dir: Path,
    *,
    variant_seed: int,
    subject_variants: dict[str, int],
) -> None:
    metadata = {
        "processing_version": _PROCESSING_VERSION,
        "variant_seed": int(variant_seed),
        "input_window_size": _INPUT_WINDOW_SIZE,
        "input_window_step": _INPUT_WINDOW_STEP,
        "output_window_size": _OUTPUT_WINDOW_SIZE,
        "output_window_step": _OUTPUT_WINDOW_STEP,
        "subject_variants": subject_variants,
    }
    (processed_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_processed_cache(base_dir: Path, *, variant_seed: int = _DEFAULT_VARIANT_SEED) -> Path:
    processed_dir = base_dir / "processed" / _PROCESSING_VERSION
    if _processed_cache_ready(processed_dir):
        return processed_dir

    inner_archive_path = _ensure_inner_archive(base_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(variant_seed)
    subject_variants = {
        f"S{subject_id}": int(variant)
        for subject_id, variant in zip(
            _SUBJECT_IDS,
            rng.integers(0, _VARIANT_COUNT, size=len(_SUBJECT_IDS)),
            strict=True,
        )
    }

    train_contexts = []
    val_contexts = []
    test_contexts = []
    train_targets = []
    val_targets = []
    test_targets = []

    for subject_id in _SUBJECT_IDS:
        subject_payload = _load_subject_payload(inner_archive_path, subject_id)
        inputs, labels = _prepare_subject_arrays(subject_payload)
        variant = subject_variants[f"S{subject_id}"]
        (
            train_inputs,
            train_labels,
            val_inputs,
            val_labels,
            test_inputs,
            test_labels,
        ) = _split_subject_variant(inputs, labels, variant)

        train_context, train_target = _paired_windows(train_inputs, train_labels)
        val_context, val_target = _paired_windows(val_inputs, val_labels)
        test_context, test_target = _paired_windows(test_inputs, test_labels)

        train_contexts.append(train_context)
        val_contexts.append(val_context)
        test_contexts.append(test_context)
        train_targets.append(train_target)
        val_targets.append(val_target)
        test_targets.append(test_target)

    split_arrays = {
        "train": (
            np.concatenate(train_contexts, axis=0),
            np.concatenate(train_targets, axis=0),
        ),
        "val": (
            np.concatenate(val_contexts, axis=0),
            np.concatenate(val_targets, axis=0),
        ),
        "test": (
            np.concatenate(test_contexts, axis=0),
            np.concatenate(test_targets, axis=0),
        ),
    }

    for split, (contexts, targets) in split_arrays.items():
        contexts_path, targets_path = _processed_paths(processed_dir, split)
        np.save(contexts_path, np.asarray(contexts, dtype=np.float32))
        np.save(targets_path, np.asarray(targets, dtype=np.float32))

    _write_metadata(
        processed_dir,
        variant_seed=variant_seed,
        subject_variants=subject_variants,
    )
    return processed_dir


@dataclass
class PPGDaliaDataset(DatasetProtocol):
    """Processed PPG-DaLiA dataset with cached train/val/test splits."""

    split: Literal["train", "val", "test"] = "train"
    cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        base_dir = resolve_cache_dir(self.cache_dir, default_name=_DEFAULT_CACHE_NAME)
        processed_dir = _build_processed_cache(base_dir)
        contexts_path, targets_path = _processed_paths(processed_dir, self.split)
        self._contexts = _to_host_jax_array(np.load(contexts_path))
        self._targets = _to_host_jax_array(np.load(targets_path))

    def __len__(self) -> int:
        return int(self._contexts.shape[0])

    def __getitem__(self, index: int):
        return {
            "context": self._contexts[index],
            "target": self._targets[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        return {
            "context": self._contexts,
            "target": self._targets,
        }

    @classmethod
    def make_disk_source(
        cls,
        *,
        split: Literal["train", "val", "test"] = "train",
        cache_dir: str | Path | None = None,
        ordering: Literal["sequential", "shuffle"] = "shuffle",
        prefetch_size: int = 64,
    ) -> DiskSource:
        base_dir = resolve_cache_dir(cache_dir, default_name=_DEFAULT_CACHE_NAME)
        processed_dir = _build_processed_cache(base_dir)
        contexts_path, targets_path = _processed_paths(processed_dir, split)

        contexts_memmap = np.load(contexts_path, mmap_mode="r")
        targets_memmap = np.load(targets_path, mmap_mode="r")
        if contexts_memmap.shape[0] != targets_memmap.shape[0]:
            raise ValueError("PPG-DaLiA context and target counts do not match.")

        def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
            idx = int(np.asarray(index))
            return {
                "context": np.asarray(contexts_memmap[idx], dtype=np.float32),
                "target": np.asarray(targets_memmap[idx], dtype=np.float32),
            }

        sample_spec = {
            "context": jax.ShapeDtypeStruct(
                shape=tuple(int(x) for x in contexts_memmap.shape[1:]),
                dtype=jnp.float32,
            ),
            "target": jax.ShapeDtypeStruct(
                shape=tuple(int(x) for x in targets_memmap.shape[1:]),
                dtype=jnp.float32,
            ),
        }

        return DiskSource(
            length=int(contexts_memmap.shape[0]),
            sample_fn=_read_sample,
            sample_spec=sample_spec,
            ordering=ordering,
            prefetch_size=prefetch_size,
        )


__all__ = ["PPGDaliaDataset", "PPG_DALIA_URL"]
