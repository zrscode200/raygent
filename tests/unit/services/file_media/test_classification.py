from __future__ import annotations

import os
from pathlib import Path

import pytest

from raygent_harness.services.file_media import (
    FILE_MEDIA_SAMPLE_BYTES,
    FileMediaClassification,
    classify_file_extension,
    classify_file_media,
    classify_magic_bytes,
    detect_supported_image_media_type,
    is_blocked_device_path,
)


def test_classifies_native_image_by_magic_even_when_extension_is_mislabeled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mislabeled.txt"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 10_000)

    result = classify_file_media(str(path))

    assert result == FileMediaClassification(
        path=str(path),
        extension="txt",
        kind="native_image",
        source="magic",
        media_type="image/png",
        size_bytes=10_008,
        exists=True,
        is_regular_file=True,
        sample_size_bytes=FILE_MEDIA_SAMPLE_BYTES,
    )


def test_classifies_pdf_and_notebook_without_requiring_existing_file(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "report.pdf"
    notebook = tmp_path / "analysis.ipynb"

    pdf_result = classify_file_media(str(pdf))
    notebook_result = classify_file_media(str(notebook))

    assert pdf_result.kind == "pdf"
    assert pdf_result.source == "extension"
    assert pdf_result.media_type == "application/pdf"
    assert pdf_result.exists is False
    assert notebook_result.kind == "notebook"
    assert notebook_result.source == "extension"
    assert notebook_result.media_type == "application/x-ipynb+json"
    assert notebook_result.exists is False


def test_extension_only_classification_does_not_touch_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "image.png"

    def fail_stat(_path: str) -> os.stat_result:
        raise AssertionError("extension-only classification must not stat")

    monkeypatch.setattr("os.stat", fail_stat)

    result = classify_file_extension(str(path))
    unknown = classify_file_extension(str(tmp_path / "plain.txt"))

    assert result.kind == "native_image"
    assert result.source == "extension"
    assert result.media_type == "image/png"
    assert result.exists is False
    assert result.sample_size_bytes == 0
    assert unknown.kind == "unknown"
    assert unknown.source == "fallback"


def test_classifies_unsupported_images_and_binary_extensions(tmp_path: Path) -> None:
    bmp = tmp_path / "image.bmp"
    archive = tmp_path / "archive.zip"

    bmp_result = classify_file_media(str(bmp))
    archive_result = classify_file_media(str(archive))

    assert bmp_result.kind == "unsupported_image"
    assert bmp_result.media_type == "image/bmp"
    assert archive_result.kind == "binary"
    assert archive_result.binary_reason == "extension"


def test_classifies_binary_samples_directories_and_special_files(tmp_path: Path) -> None:
    binary = tmp_path / "payload.txt"
    binary.write_bytes(b"abc\x00def")

    directory_result = classify_file_media(str(tmp_path))
    binary_result = classify_file_media(str(binary))

    assert directory_result.kind == "directory"
    assert directory_result.source == "stat"
    assert directory_result.is_regular_file is False
    assert binary_result.kind == "binary"
    assert binary_result.source == "magic"
    assert binary_result.binary_reason == "sample"

    dev_null = Path("/dev/null")
    if dev_null.exists():
        special_result = classify_file_media(str(dev_null))
        assert special_result.kind == "special"
        assert special_result.source == "stat"


def test_classification_sample_is_bounded_and_metadata_only(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_bytes(b"a" * (FILE_MEDIA_SAMPLE_BYTES * 3))

    result = classify_file_media(str(path))

    assert result.kind == "text"
    assert result.sample_size_bytes == FILE_MEDIA_SAMPLE_BYTES
    assert result.size_bytes == FILE_MEDIA_SAMPLE_BYTES * 3
    assert not hasattr(result, "sample")
    assert not hasattr(result, "content")


def test_magic_byte_classifier_and_blocked_device_helper() -> None:
    assert detect_supported_image_media_type(b"\xff\xd8\xffpayload") == "image/jpeg"
    pdf_result = classify_magic_bytes(b"%PDF-1.4", path="x.bin")
    assert pdf_result is not None
    assert pdf_result.kind == "pdf"
    assert classify_magic_bytes(b"plain text", path="x.txt") is None
    assert is_blocked_device_path("/dev/zero") is True
    assert is_blocked_device_path(os.path.join("/tmp", "zero")) is False


@pytest.mark.parametrize(
    ("payload", "media_type"),
    (
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"GIF89a", "image/gif"),
        (b"RIFFxxxxWEBP", "image/webp"),
    ),
)
def test_supported_image_magic_types(payload: bytes, media_type: str) -> None:
    assert detect_supported_image_media_type(payload) == media_type
