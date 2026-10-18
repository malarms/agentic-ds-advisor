import wave

import numpy as np
import pytest
from PIL import Image

from app.tools.multimodal import (
    profile_image_dataset,
    profile_audio_dataset,
    perform_image_eda,
    perform_audio_eda,
)


def test_image_profile_and_eda(tmp_path):
    Image.new("RGB", (32, 24), (120, 80, 40)).save(tmp_path / "one.png")
    Image.new("L", (16, 16), 128).save(tmp_path / "two.png")
    (tmp_path / "broken.png").write_bytes(b"not an image")

    profile = profile_image_dataset(tmp_path)
    eda = perform_image_eda(tmp_path)

    assert profile["valid_count"] == 2
    assert profile["invalid_count"] == 1
    assert profile["width_range"] == [16, 32]
    assert eda["item_count"] == 2
    assert sum(eda["dimension_counts"].values()) == 2


def test_audio_profile_and_eda(tmp_path):
    pytest.importorskip("soundfile")

    path = tmp_path / "tone.wav"
    sample_rate = 8000
    samples = (np.sin(
        2 * np.pi * 440 * np.arange(sample_rate) / sample_rate
    ) * 0.5).astype(np.float32)

    import soundfile as sf
    sf.write(path, samples, sample_rate)

    profile = profile_audio_dataset(path)
    eda = perform_audio_eda(path)

    assert profile["valid_count"] == 1
    assert profile["items"][0]["sample_rate_hz"] == sample_rate
    assert profile["items"][0]["duration_seconds"] == pytest.approx(1.0, abs=0.001)
    assert profile["items"][0]["rms_amplitude"] > 0
    assert eda["item_count"] == 1


def test_image_profile_rejects_folder_without_images(tmp_path):
    with pytest.raises(ValueError, match="No supported files"):
        profile_image_dataset(tmp_path)
