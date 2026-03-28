"""PPG-DaLiA dataset utilities with a fixed cached layout."""

from __future__ import annotations

import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _sliding_window_view

from ..sources import DiskSource
from .dataset_protocol import DatasetProtocol
from .utils import (
    download_archive,
    ensure_zip_member_extracted,
    resolve_cache_dir,
)
from .utils import (
    to_host_jax_array as _to_host_jax_array,
)

PPG_DALIA_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"

_DEFAULT_CACHE_NAME = "cyreal_ppg_dalia"
_SUBJECT_IDS = tuple(range(1, 16))
_SUBJECT_VARIANTS = (5, 3, 3, 1, 1, 0, 0, 0, 1, 4, 3, 5, 3, 3, 5)

_INPUT_WINDOW_SIZE = 49_920
_INPUT_WINDOW_STEP = 4_992
_OUTPUT_WINDOW_SIZE = 390
_OUTPUT_WINDOW_STEP = 39

_SEGMENT_FRACTIONS: dict[str, tuple[float, float | None]] = {
    "front70": (0.0, 0.7),
    "mid15_a": (0.7, 0.85),
    "tail15": (0.85, None),
    "center70": (0.15, 0.85),
    "head15": (0.0, 0.15),
    "from30": (0.30, None),
    "mid15_b": (0.15, 0.30),
}

_VARIANT_SEGMENT_ORDER: dict[int, tuple[str, str, str]] = {
    0: ("front70", "mid15_a", "tail15"),
    1: ("front70", "tail15", "mid15_a"),
    2: ("center70", "head15", "tail15"),
    3: ("center70", "tail15", "head15"),
    4: ("from30", "head15", "mid15_b"),
    5: ("from30", "mid15_b", "head15"),
}


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
    return f"data/PPG_FieldStudy/S{subject_id}/S{subject_id}.pkl"


def _find_inner_archive_member(outer_archive_path: Path) -> str:
    with zipfile.ZipFile(outer_archive_path, "r") as zf:
        candidates = [
            info.filename
            for info in zf.infolist()
            if not info.is_dir() and Path(info.filename).name == "data.zip"
        ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find 'data.zip' inside '{outer_archive_path}'."
        )
    if len(candidates) > 1:
        candidates.sort()
    return candidates[0]


def _bad_zip_error(
    *,
    kind: Literal["outer", "inner"],
    archive_path: Path,
    base_dir: Path,
    exc: zipfile.BadZipFile,
    local_files_only: bool | None = None,
) -> zipfile.BadZipFile:
    base_msg = (
        f"Invalid PPG-DaLiA {kind} archive at '{archive_path.resolve()}' "
        f"(cache_dir resolved to '{base_dir.resolve()}'"
    )
    if local_files_only is not None:
        base_msg += f", local_files_only={local_files_only}"
    return zipfile.BadZipFile(f"{base_msg}). {exc}")


def _ensure_inner_archive(base_dir: Path, *, local_files_only: bool = False) -> Path:
    inner_archive_path = base_dir / "data.zip"
    if inner_archive_path.exists():
        return inner_archive_path

    outer_archive_path = base_dir / "ppg+dalia.zip"
    if not outer_archive_path.exists():
        if local_files_only:
            raise FileNotFoundError(
                "PPG-DaLiA local_files_only=True but no local archive was found. "
                f"Expected either '{inner_archive_path}' or '{outer_archive_path}'."
            )
        download_archive(PPG_DALIA_URL, outer_archive_path)
    try:
        member_name = _find_inner_archive_member(outer_archive_path)
        ensure_zip_member_extracted(
            outer_archive_path,
            base_dir,
            member_name,
            target_name="data.zip",
        )
    except zipfile.BadZipFile as exc:
        raise _bad_zip_error(
            kind="outer",
            archive_path=outer_archive_path,
            base_dir=base_dir,
            local_files_only=local_files_only,
            exc=exc,
        ) from exc
    if outer_archive_path.exists():
        outer_archive_path.unlink()
    return inner_archive_path


def _load_subject_payload(inner_archive_path: Path, subject_id: int) -> dict:
    member_name = _subject_member_name(subject_id)
    try:
        with zipfile.ZipFile(inner_archive_path, "r") as zf:
            try:
                with zf.open(member_name, "r") as f:
                    return pickle.load(f, encoding="latin1")
            except KeyError as exc:
                raise FileNotFoundError(
                    f"Missing archive member '{member_name}' in '{inner_archive_path}'."
                ) from exc
    except zipfile.BadZipFile as exc:
        raise _bad_zip_error(
            kind="inner",
            archive_path=inner_archive_path,
            base_dir=inner_archive_path.parent.parent,
            exc=exc,
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


def _slice_fraction(
    array: np.ndarray, start_fraction: float, end_fraction: float | None = None
) -> np.ndarray:
    start = int(start_fraction * len(array))
    end = len(array) if end_fraction is None else int(end_fraction * len(array))
    return array[start:end]


def _split_subject_variant(
    inputs: np.ndarray,
    labels: np.ndarray,
    variant: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    segment_order = _VARIANT_SEGMENT_ORDER.get(variant)
    if segment_order is None:
        raise ValueError(f"Unsupported PPG-DaLiA split variant '{variant}'.")

    sliced_inputs = []
    sliced_labels = []
    for segment_name in segment_order:
        start_fraction, end_fraction = _SEGMENT_FRACTIONS[segment_name]
        sliced_inputs.append(_slice_fraction(inputs, start_fraction, end_fraction))
        sliced_labels.append(_slice_fraction(labels, start_fraction, end_fraction))

    return (
        sliced_inputs[0],
        sliced_labels[0],
        sliced_inputs[1],
        sliced_labels[1],
        sliced_inputs[2],
        sliced_labels[2],
    )


def _window_inputs(inputs: np.ndarray) -> np.ndarray:
    if inputs.shape[0] < _INPUT_WINDOW_SIZE:
        raise ValueError(
            f"Input split length {inputs.shape[0]} is shorter than window size {_INPUT_WINDOW_SIZE}."
        )
    windows = _sliding_window_view(inputs, _INPUT_WINDOW_SIZE, axis=0)[
        ::_INPUT_WINDOW_STEP
    ]
    return np.asarray(np.swapaxes(windows, 1, 2), dtype=np.float32)


def _window_targets(labels: np.ndarray) -> np.ndarray:
    if labels.shape[0] < _OUTPUT_WINDOW_SIZE:
        raise ValueError(
            f"Target split length {labels.shape[0]} is shorter than window size {_OUTPUT_WINDOW_SIZE}."
        )
    windows = _sliding_window_view(labels, _OUTPUT_WINDOW_SIZE, axis=0)[
        ::_OUTPUT_WINDOW_STEP
    ]
    return np.asarray(windows, dtype=np.float32)


def _paired_windows(
    inputs: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    context_windows = _window_inputs(inputs)
    target_windows = _window_targets(labels)
    pair_count = min(int(context_windows.shape[0]), int(target_windows.shape[0]))
    if pair_count <= 0:
        raise ValueError("PPG-DaLiA split produced no aligned context/target windows.")
    return context_windows[:pair_count], target_windows[:pair_count]


def _processed_paths(
    processed_dir: Path, split: Literal["train", "val", "test"]
) -> tuple[Path, Path]:
    return processed_dir / f"X_{split}.npy", processed_dir / f"y_{split}.npy"


def _processed_cache_ready(processed_dir: Path) -> bool:
    required = []
    for split in ("train", "val", "test"):
        required.extend(_processed_paths(processed_dir, split))
    return all(path.exists() for path in required)


def _build_processed_cache(base_dir: Path, *, local_files_only: bool = False) -> Path:
    processed_dir = base_dir / "processed"
    if _processed_cache_ready(processed_dir):
        return processed_dir

    inner_archive_path = _ensure_inner_archive(
        base_dir,
        local_files_only=local_files_only,
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    split_contexts: dict[str, list[np.ndarray]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    split_targets: dict[str, list[np.ndarray]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for subject_id in _SUBJECT_IDS:
        subject_payload = _load_subject_payload(inner_archive_path, subject_id)
        inputs, labels = _prepare_subject_arrays(subject_payload)
        variant = _SUBJECT_VARIANTS[subject_id - 1]
        (
            train_inputs,
            train_labels,
            val_inputs,
            val_labels,
            test_inputs,
            test_labels,
        ) = _split_subject_variant(inputs, labels, variant)

        split_inputs_labels = {
            "train": (train_inputs, train_labels),
            "val": (val_inputs, val_labels),
            "test": (test_inputs, test_labels),
        }
        for split, (split_inputs, split_labels) in split_inputs_labels.items():
            contexts, targets = _paired_windows(split_inputs, split_labels)
            split_contexts[split].append(contexts)
            split_targets[split].append(targets)

    for split in ("train", "val", "test"):
        contexts = np.concatenate(split_contexts[split], axis=0)
        targets = np.concatenate(split_targets[split], axis=0)
        contexts_path, targets_path = _processed_paths(processed_dir, split)
        np.save(contexts_path, np.asarray(contexts, dtype=np.float32))
        np.save(targets_path, np.asarray(targets, dtype=np.float32))
    return processed_dir


@dataclass
class PPGDaliaDataset(DatasetProtocol):
    """Processed PPG-DaLiA dataset with cached train/val/test splits."""

    split: Literal["train", "val", "test"] = "train"
    cache_dir: str | Path | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        base_dir = resolve_cache_dir(self.cache_dir, default_name=_DEFAULT_CACHE_NAME)
        processed_dir = _build_processed_cache(
            base_dir,
            local_files_only=self.local_files_only,
        )
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
        local_files_only: bool = False,
        ordering: Literal["sequential", "shuffle"] = "shuffle",
        prefetch_size: int = 64,
    ) -> DiskSource:
        base_dir = resolve_cache_dir(cache_dir, default_name=_DEFAULT_CACHE_NAME)
        processed_dir = _build_processed_cache(
            base_dir,
            local_files_only=local_files_only,
        )
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
