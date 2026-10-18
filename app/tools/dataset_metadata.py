"""Readers and validators for dataset-level metadata and IDX image files."""

from __future__ import annotations

import csv
import gzip
import struct
from collections import Counter
from pathlib import Path
from typing import Any


IDX_FILES = {
    "train": (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ),
    "test": (
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ),
}


def profile_idx_image_dataset(path: str | Path) -> dict[str, Any] | None:
    """Read standard gzip-compressed IDX image/label pairs without loading pixels.

    Returns None when the directory does not contain a recognized IDX dataset.
    """
    root = Path(path).expanduser()
    if not root.is_dir():
        return None

    present = {
        split: (root / image_name, root / label_name)
        for split, (image_name, label_name) in IDX_FILES.items()
    }
    recognized = any(
        image_path.exists() or label_path.exists()
        for image_path, label_path in present.values()
    )
    if not recognized:
        return None

    split_counts: dict[str, int] = {}
    class_counts: Counter[str] = Counter()
    dimensions: dict[str, list[int]] = {}
    total = 0

    for split, (image_path, label_path) in present.items():
        if not image_path.is_file() or not label_path.is_file():
            raise ValueError(
                f"Incomplete IDX {split} split: expected both "
                f"{image_path.name} and {label_path.name}."
            )

        with gzip.open(image_path, "rb") as handle:
            header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"Truncated IDX image header: {image_path.name}")

        magic, image_count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(
                f"Invalid IDX image magic number in {image_path.name}: {magic}"
            )
        if image_count <= 0 or rows <= 0 or columns <= 0:
            raise ValueError(f"Invalid IDX image dimensions/count: {image_path.name}")

        with gzip.open(label_path, "rb") as handle:
            label_header = handle.read(8)
            if len(label_header) != 8:
                raise ValueError(f"Truncated IDX label header: {label_path.name}")
            label_magic, label_count = struct.unpack(">II", label_header)

            if label_magic != 2049:
                raise ValueError(
                    f"Invalid IDX label magic number in {label_path.name}: "
                    f"{label_magic}"
                )
            if label_count != image_count:
                raise ValueError(
                    f"IDX count mismatch for {split}: {image_count} images, "
                    f"{label_count} labels."
                )

            labels = handle.read(label_count)
            if len(labels) != label_count:
                raise ValueError(f"Truncated IDX labels: {label_path.name}")

        # Standard Fashion-MNIST/MNIST labels are integers 0 through 9.
        invalid_labels = sorted(set(labels) - set(range(10)))
        if invalid_labels:
            raise ValueError(
                f"Invalid class labels in {label_path.name}: {invalid_labels}"
            )

        split_counts[split] = image_count
        class_counts.update(str(label) for label in labels)
        dimensions[split] = [rows, columns]
        total += image_count

    return {
        "modality": "image",
        "dataset_format": "gzip_idx",
        "files_discovered": 4,
        "valid_count": total,
        "invalid_count": 0,
        "invalid_files": [],
        "items": [],
        "split_counts": split_counts,
        "class_counts": dict(sorted(class_counts.items())),
        "dimension_counts": {
            f"{rows}x{columns}": count
            for (rows, columns), count in [
                (tuple(dimensions["train"]), split_counts["train"]),
                (tuple(dimensions["test"]), split_counts["test"]),
            ]
        },
        "dimensions_by_split": dimensions,
        "label_count": total,
        "notes": [
            "Pixel values were not decoded; binary headers and label arrays were validated.",
            "Split names are taken from the recognized IDX filenames.",
            "Class IDs are numeric labels; human-readable class names are not assumed.",
        ],
    }


def find_and_join_audio_metadata(
    audio_path: str | Path,
    audio_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Join audio items to a nearby CSV using a filename column.

    Searches the audio directory and nearby metadata/parent directories.
    Requires a filename key and uses category/label and fold fields when present.
    """
    root = Path(audio_path).expanduser()
    audio_dir = root if root.is_dir() else root.parent

    candidates = [
        *audio_dir.glob("*.csv"),
        *audio_dir.glob("meta/*.csv"),
        *audio_dir.parent.glob("*.csv"),
        *audio_dir.parent.glob("meta/*.csv"),
    ]
    seen: set[Path] = set()
    candidates = [
        candidate for candidate in candidates
        if not (candidate.resolve() in seen or seen.add(candidate.resolve()))
    ]

    item_by_name = {str(item["name"]): item for item in audio_items}
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue

                columns = {column.strip().lower(): column for column in reader.fieldnames}
                filename_col = next(
                    (
                        columns[key]
                        for key in ("filename", "file_name", "file", "audio_file", "name")
                        if key in columns
                    ),
                    None,
                )
                if filename_col is None:
                    continue

                rows = list(reader)
        except (OSError, UnicodeError, csv.Error):
            continue

        filenames = [str(row.get(filename_col, "")).strip() for row in rows]
        if not filenames or any(not name for name in filenames):
            continue
        if len(filenames) != len(set(filenames)):
            raise ValueError(f"Duplicate filenames in metadata CSV: {candidate}")
        if not (set(filenames) & set(item_by_name)):
            continue

        matched = set(filenames) & set(item_by_name)
        if matched != set(item_by_name):
            continue

        label_col = next(
            (
                columns[key]
                for key in ("category", "class", "label", "target", "class_name")
                if key in columns
            ),
            None,
        )
        fold_col = next(
            (
                columns[key]
                for key in ("fold", "split", "partition", "subset")
                if key in columns
            ),
            None,
        )

        labels: Counter[str] = Counter()
        folds: Counter[str] = Counter()
        for row, filename in zip(rows, filenames):
            item = item_by_name[filename]
            if label_col and row.get(label_col) not in (None, ""):
                label = str(row[label_col]).strip()
                item["label"] = label
                labels[label] += 1
            if fold_col and row.get(fold_col) not in (None, ""):
                fold = str(row[fold_col]).strip()
                item["fold"] = fold
                folds[fold] += 1

        return {
            "metadata_file": str(candidate),
            "metadata_rows": len(rows),
            "matched_files": len(matched),
            "unmatched_metadata_rows": len(set(filenames) - set(item_by_name)),
            "label_column": label_col,
            "fold_column": fold_col,
            "class_counts": dict(sorted(labels.items())),
            "fold_counts": dict(sorted(folds.items())),
        }

    return None
