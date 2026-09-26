"""
Image validation and decoding.

Never trusts the declared MIME type alone — decoded bytes are checked
with Pillow, which parses the actual image structure.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
SUPPORTED_MIME_TYPES = set(SUPPORTED_FORMATS.values())


class ImageValidationError(Exception):
    def __init__(self, message: str, code: str = "invalid_image"):
        self.message = message
        self.code = code
        super().__init__(message)


class ImageTooLargeError(ImageValidationError):
    def __init__(self, message: str):
        super().__init__(message, code="request_too_large")


class TooManyImagesError(ImageValidationError):
    def __init__(self, message: str):
        super().__init__(message, code="invalid_request_error")


@dataclass
class DecodedImage:
    mime_type: str
    data: bytes


def _parse_data_url(url: str) -> tuple[str | None, str]:
    """Returns (declared_mime_type_or_None, base64_payload)."""
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        if not payload:
            raise ImageValidationError("Malformed data URL: missing base64 payload.")
        mime_type = None
        if ";base64" in header:
            mime_type = header[len("data:"):header.index(";base64")] or None
        return mime_type, payload
    raise ImageValidationError(
        "Only base64 data URLs (data:image/...;base64,...) are supported for image_url.url."
    )


def decode_and_validate_image(url: str, max_size_mb: int) -> DecodedImage:
    declared_mime, payload = _parse_data_url(url)

    try:
        raw_bytes = base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ImageValidationError(f"Invalid base64 image data: {exc}") from exc

    max_bytes = max_size_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise ImageTooLargeError(f"Image exceeds maximum allowed size of {max_size_mb}MB.")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.verify()  # structural check
        # Re-open after verify() (which leaves the file unusable) to read format.
        with Image.open(io.BytesIO(raw_bytes)) as img2:
            actual_format = img2.format
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(f"Could not decode image data: {exc}") from exc

    if actual_format not in SUPPORTED_FORMATS:
        raise ImageValidationError(
            f"Unsupported image format '{actual_format}'. Supported: png, jpeg, webp."
        )

    actual_mime = SUPPORTED_FORMATS[actual_format]

    if declared_mime and declared_mime not in SUPPORTED_MIME_TYPES:
        raise ImageValidationError(f"Unsupported declared MIME type '{declared_mime}'.")

    return DecodedImage(mime_type=actual_mime, data=raw_bytes)


def validate_image_count(count: int, max_images: int) -> None:
    if count > max_images:
        raise TooManyImagesError(f"Request contains {count} images; maximum allowed is {max_images}.")
