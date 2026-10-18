"""Deterministic profiling and exploratory analysis for image and audio datasets."""

from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}
AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".ogg", ".aiff", ".aif"
}


def _try_idx_reader(path: str | Path) -> dict[str, Any] | None:
    """Try to read an IDX-format image dataset (e.g., MNIST/Fashion-MNIST)."""
    from app.tools.dataset_metadata import profile_idx_image_dataset
    return profile_idx_image_dataset(path)


def _files(path: str | Path, extensions: set[str]) -> list[Path]:
    root = Path(path).expanduser()

    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    if root.is_file():
        if root.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported file format: {root.suffix}")
        return [root]

    files = sorted(
        item for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in extensions
    )

    if not files:
        raise ValueError(
            f"No supported files found in {root}. "
            f"Supported extensions: {', '.join(sorted(extensions))}"
        )

    return files


def profile_image_dataset(path: str | Path) -> dict[str, Any]:
    """Inspect image dimensions, formats, modes, sizes, and unreadable files.

    First tries to read IDX-format datasets (e.g., MNIST/Fashion-MNIST).
    Falls back to standard image file extensions (PNG, JPEG, etc.).
    """
    from PIL import Image, UnidentifiedImageError

    # Try IDX reader first for compressed datasets like MNIST/Fashion-MNIST
    idx_profile = _try_idx_reader(path)
    if idx_profile is not None:
        return idx_profile

    files = _files(path, IMAGE_EXTENSIONS)
    items = []
    invalid_files = []

    for file in files:
        try:
            with Image.open(file) as image:
                image.verify()

            with Image.open(file) as image:
                width, height = image.size
                items.append({
                    "name": file.name,
                    "format": image.format or file.suffix.lstrip(".").upper(),
                    "width": int(width),
                    "height": int(height),
                    "mode": image.mode,
                    "file_size_bytes": file.stat().st_size,
                    "aspect_ratio": round(width / height, 4) if height else None,
                })
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            invalid_files.append({
                "name": file.name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    if not items:
        raise ValueError(
            f"No readable images found among {len(files)} supported files."
        )

    return {
        "modality": "image",
        "files_discovered": len(files),
        "valid_count": len(items),
        "invalid_count": len(invalid_files),
        "invalid_files": invalid_files,
        "items": items,
        "format_counts": dict(Counter(item["format"] for item in items)),
        "mode_counts": dict(Counter(item["mode"] for item in items)),
        "width_range": [
            min(item["width"] for item in items),
            max(item["width"] for item in items),
        ],
        "height_range": [
            min(item["height"] for item in items),
            max(item["height"] for item in items),
        ],
        "file_size_bytes": {
            "min": min(item["file_size_bytes"] for item in items),
            "max": max(item["file_size_bytes"] for item in items),
            "mean": round(
                sum(item["file_size_bytes"] for item in items) / len(items), 2
            ),
        },
    }


def perform_image_eda(path: str | Path, output_dir: str | Path = "reports") -> dict:
    """Return image-level distributions suitable for reports and recommendations."""
    profile = profile_image_dataset(path)

    # Handle IDX format (e.g., MNIST/Fashion-MNIST) which has different structure
    if profile.get("dataset_format") == "gzip_idx":
        # Properly combine dimension counts from splits (same dimensions may appear in both)
        dimension_counts = {}
        for split, count in profile.get("split_counts", {}).items():
            dims = profile.get("dimensions_by_split", {}).get(split, [])
            if len(dims) == 2:
                key = f"{dims[0]}x{dims[1]}"
                dimension_counts[key] = dimension_counts.get(key, 0) + count

        return {
            "modality": "image",
            "dataset_format": "gzip_idx",
            "item_count": profile["valid_count"],
            "items": profile.get("items", []),
            "format_counts": {"IDX/GZIP": profile["valid_count"]},
            "mode_counts": {},
            "dimension_counts": dimension_counts,
            "aspect_ratio_range": [1.0, 1.0],  # Square images for MNIST
            "invalid_count": profile["invalid_count"],
            "split_counts": profile.get("split_counts", {}),
            "class_counts": profile.get("class_counts", {}),
            "dimensions_by_split": profile.get("dimensions_by_split", {}),
        }

    items = profile["items"]

    return {
        "modality": "image",
        "item_count": len(items),
        "items": items,
        "format_counts": profile["format_counts"],
        "mode_counts": profile["mode_counts"],
        "dimension_counts": dict(Counter(
            f'{item["width"]}x{item["height"]}' for item in items
        )),
        "aspect_ratio_range": [
            min(item["aspect_ratio"] for item in items),
            max(item["aspect_ratio"] for item in items),
        ],
        "invalid_count": profile["invalid_count"],
    }


def profile_audio_dataset(path: str | Path) -> dict[str, Any]:
    """Inspect audio metadata and calculate streaming amplitude statistics."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Audio analysis requires soundfile. Install it with "
            "'python -m pip install soundfile'."
        ) from exc

    files = _files(path, AUDIO_EXTENSIONS)
    items = []
    invalid_files = []

    for file in files:
        try:
            with sf.SoundFile(str(file)) as audio:
                sample_rate = int(audio.samplerate)
                channels = int(audio.channels)
                frames = int(audio.frames)

                if sample_rate <= 0 or channels <= 0 or frames <= 0:
                    raise ValueError("Audio has invalid or empty metadata.")

                peak = 0.0
                sum_squares = 0.0
                sample_count = 0
                clipped_count = 0

                while True:
                    block = audio.read(
                        frames=sample_rate,
                        dtype="float32",
                        always_2d=True,
                    )
                    if block.size == 0:
                        break

                    absolute = abs(block)
                    peak = max(peak, float(absolute.max()))
                    sum_squares += float((block * block).sum())
                    sample_count += int(block.size)
                    clipped_count += int((absolute >= 0.999).sum())

                items.append({
                    "name": file.name,
                    "format": file.suffix.lstrip(".").upper(),
                    "sample_rate_hz": sample_rate,
                    "channels": channels,
                    "frames": frames,
                    "duration_seconds": round(frames / sample_rate, 4),
                    "subtype": audio.subtype,
                    "file_size_bytes": file.stat().st_size,
                    "peak_amplitude": round(peak, 6),
                    "rms_amplitude": round(
                        (sum_squares / sample_count) ** 0.5, 6
                    ) if sample_count else 0.0,
                    "clipped_sample_count": clipped_count,
                    "clipped_sample_percentage": round(
                        clipped_count / sample_count * 100, 6
                    ) if sample_count else 0.0,
                })
        except Exception as exc:
            invalid_files.append({
                "name": file.name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    if not items:
        raise ValueError(
            f"No readable audio files found among {len(files)} supported files."
        )

    durations = [item["duration_seconds"] for item in items]
    sample_rates = [item["sample_rate_hz"] for item in items]

    return {
        "modality": "audio",
        "files_discovered": len(files),
        "valid_count": len(items),
        "invalid_count": len(invalid_files),
        "invalid_files": invalid_files,
        "items": items,
        "format_counts": dict(Counter(item["format"] for item in items)),
        "sample_rate_counts": dict(Counter(sample_rates)),
        "channel_counts": dict(Counter(item["channels"] for item in items)),
        "duration_seconds": {
            "min": round(min(durations), 4),
            "max": round(max(durations), 4),
            "mean": round(sum(durations) / len(durations), 4),
        },
    }


def perform_audio_eda(path: str | Path, output_dir: str | Path = "reports") -> dict:
    """Return audio-level distributions suitable for reports and recommendations."""
    profile = profile_audio_dataset(path)
    items = profile["items"]

    return {
        "modality": "audio",
        "item_count": len(items),
        "items": items,
        "format_counts": profile["format_counts"],
        "sample_rate_counts": profile["sample_rate_counts"],
        "channel_counts": profile["channel_counts"],
        "durations_seconds": [item["duration_seconds"] for item in items],
        "rms_amplitudes": [item["rms_amplitude"] for item in items],
        "peak_amplitudes": [item["peak_amplitude"] for item in items],
        "clipped_sample_percentages": [
            item["clipped_sample_percentage"] for item in items
        ],
        "invalid_count": profile["invalid_count"],
    }
