"""Every request limit defaults to 0 = unlimited.

These prove the caps that used to refuse a request are actually gone in
production configuration, not merely configurable — and that the code
path behind each one still works once the cap is out of the way."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from tests.conftest import make_test_image_data_url

FAKE_SOLUTION = "## Code\n```cpp\nint main(){}\n```"


def _set_limits(monkeypatch, **limits):
    """Overrides limits and forces Settings to be rebuilt from the
    environment — app_client has already built the app and cached them."""
    import app.config as config_module

    for name, value in limits.items():
        monkeypatch.setenv(name, str(value))
    config_module.get_settings.cache_clear()


def _mock_solver(monkeypatch):
    import app.api.chat_completions as cc

    async def fake_solve(settings, text, client=None, max_tokens=None):
        return FAKE_SOLUTION

    monkeypatch.setattr(cc, "solve_problem", fake_solve)


def _mock_ocr(monkeypatch, text="extracted text"):
    import app.api.chat_completions as cc
    import app.api.ocr as ocr_module
    from app.models.solver import OcrResult

    async def fake_ocr(settings, images, client=None):
        return [OcrResult(index=i, text=f"{text} {i}") for i in range(len(images))]

    monkeypatch.setattr(cc, "run_ocr_on_images", fake_ocr)
    monkeypatch.setattr(ocr_module, "run_ocr_on_images", fake_ocr)


def _noisy_data_url(seed=0):
    """A JPEG large enough that two of them exceed the old 3 MB batch cap."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (1200, 1600, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_no_prompt_character_limit(app_client, backend_api_key, monkeypatch):
    _set_limits(monkeypatch, MAX_PROMPT_CHARS=0)
    _mock_solver(monkeypatch)

    # 300k characters is well past the old 250k cap.
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [{"role": "user", "content": "Solve"}],
            "ocr_results": [{"index": 0, "text": "x" * 300_000}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == FAKE_SOLUTION


def test_no_screenshot_count_limit(app_client, backend_api_key, monkeypatch):
    _set_limits(monkeypatch, MAX_IMAGES=0)
    _mock_solver(monkeypatch)

    # 21 is past the old MAX_IMAGES cap of 20.
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [{"role": "user", "content": "Solve"}],
            "ocr_results": [{"index": i, "text": f"s {i}"} for i in range(21)],
        },
    )
    assert resp.status_code == 200


def test_no_inline_image_byte_limit(app_client, backend_api_key, monkeypatch):
    _set_limits(monkeypatch, MAX_OCR_BATCH_BYTES=0)
    _mock_ocr(monkeypatch)
    _mock_solver(monkeypatch)

    # Two ~2 MB images total more than the old 3 MB inline cap.
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _noisy_data_url(1)}},
                        {"type": "image_url", "image_url": {"url": _noisy_data_url(2)}},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200


def test_no_ocr_batch_image_count_limit(app_client, backend_api_key, monkeypatch):
    _set_limits(monkeypatch, MAX_OCR_BATCH_IMAGES=0)
    _mock_ocr(monkeypatch)
    img = make_test_image_data_url("PNG")

    # 6 is past the old MAX_OCR_BATCH_IMAGES cap of 4.
    resp = app_client.post(
        "/v1/ocr",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={"images": [img] * 6, "start_index": 0},
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 6


def test_no_ocr_batch_byte_limit(app_client, backend_api_key, monkeypatch):
    _set_limits(monkeypatch, MAX_OCR_BATCH_BYTES=0)
    _mock_ocr(monkeypatch)

    resp = app_client.post(
        "/v1/ocr",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={"images": [_noisy_data_url(3), _noisy_data_url(4)], "start_index": 0},
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 2


def test_no_per_image_size_limit():
    """The gate is what changed, not the decoding: the same payload is
    refused with a positive cap and accepted with 0."""
    from app.utils.images import ImageTooLargeError, decode_and_validate_image

    url = _noisy_data_url(5)

    with pytest.raises(ImageTooLargeError):
        decode_and_validate_image(url, 1)

    decoded = decode_and_validate_image(url, 0)
    assert decoded.mime_type == "image/jpeg"
    assert decoded.data


def test_no_image_count_gate():
    from app.utils.images import validate_image_count

    validate_image_count(10_000, 0)  # must not raise
