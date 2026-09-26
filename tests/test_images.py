import base64

from tests.conftest import make_test_image_data_url


def _mock_solver(monkeypatch):
    import app.api.chat_completions as cc

    async def fake_solve(settings, text, client=None, max_tokens=None):
        return "## Problem\n...\n## Code\n```cpp\nint main(){}\n```"

    monkeypatch.setattr(cc, "solve_problem", fake_solve)


def _mock_ocr(monkeypatch):
    import app.api.chat_completions as cc
    from app.models.solver import OcrResult

    async def fake_ocr(settings, images, client=None):
        return [OcrResult(index=i, text="x") for i in range(len(images))]

    monkeypatch.setattr(cc, "run_ocr_on_images", fake_ocr)


def test_invalid_base64_image_rejected(app_client, backend_api_key, monkeypatch):
    _mock_solver(monkeypatch)
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,not-valid-base64!!"}}]}
            ],
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_malformed_image_bytes_rejected(app_client, backend_api_key, monkeypatch):
    _mock_solver(monkeypatch)
    junk = base64.b64encode(b"this is not an image").decode()
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{junk}"}}]}],
        },
    )
    assert resp.status_code == 400


def test_oversized_image_rejected(app_client, backend_api_key, monkeypatch):
    _mock_solver(monkeypatch)
    # tests/conftest.py pins MAX_IMAGE_SIZE_MB to 10 so this limit stays
    # exercised (production default is 0 = unlimited). Craft a bigger payload.
    huge_junk = base64.b64encode(b"0" * (11 * 1024 * 1024)).decode()
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{huge_junk}"}}]}],
        },
    )
    assert resp.status_code == 413


def test_too_many_images_rejected(app_client, backend_api_key, monkeypatch):
    _mock_ocr(monkeypatch)
    _mock_solver(monkeypatch)
    img = make_test_image_data_url("PNG")
    # tests/conftest.py pins MAX_IMAGES to 20 (production default is 0 =
    # unlimited); 21 inline images must be refused before OCR runs.
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={
            "model": "dsa-solver",
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}} for _ in range(21)]}
            ],
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "too_many_images"


def test_valid_png_jpeg_webp_accepted(app_client, backend_api_key, monkeypatch):
    _mock_ocr(monkeypatch)
    _mock_solver(monkeypatch)
    for fmt in ("PNG", "JPEG", "WEBP"):
        img = make_test_image_data_url(fmt)
        resp = app_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {backend_api_key}"},
            json={
                "model": "dsa-solver",
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}}]}],
            },
        )
        assert resp.status_code == 200, f"{fmt} should be accepted"
