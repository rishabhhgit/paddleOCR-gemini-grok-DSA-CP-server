import base64
import io
import os
import shutil

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def tmp_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("APP_ADMIN_PASSWORD", "test-admin-pw")
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://test.example.com")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "fake-gemini-model")
    monkeypatch.setenv("GROK_API_KEY", "fake-grok-key")
    monkeypatch.setenv("GROK_MODEL", "fake-grok-model")
    # Every limit defaults to 0 = unlimited in app/config.py, so tests
    # that exist to prove a limit still refuses a request have to pin
    # one. Nothing here changes what production does.
    monkeypatch.setenv("MAX_IMAGES", "20")
    monkeypatch.setenv("MAX_IMAGE_SIZE_MB", "10")
    monkeypatch.setenv("MAX_OCR_BATCH_IMAGES", "4")
    monkeypatch.setenv("MAX_OCR_BATCH_BYTES", "3000000")
    monkeypatch.setenv("MAX_PROMPT_CHARS", "250000")
    monkeypatch.setenv("SOLVER_MAX_TOKENS", "32768")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "540")
    monkeypatch.setenv("ALWAYS_VERIFY", "false")
    # Skip the real PaddleOCR model load at app startup: tests monkeypatch
    # run_ocr_on_images directly and never need the real local engine, and
    # loading/downloading actual model weights would make every test run
    # slow and network-dependent.
    monkeypatch.setenv("PRELOAD_OCR_ON_STARTUP", "false")
    yield data_dir
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture()
def app_client(tmp_data_dir):
    # Fresh settings/singletons per test since app.config caches settings
    # and app.security modules cache module-level singletons.
    import app.config as config_module
    import app.security.api_keys as api_keys_module

    config_module.get_settings.cache_clear()
    api_keys_module._store = None
    api_keys_module._store_key_signature = None

    from app.main import app

    with TestClient(app) as client:
        yield client

    config_module.get_settings.cache_clear()
    api_keys_module._store = None
    api_keys_module._store_key_signature = None


@pytest.fixture()
def admin_client(app_client):
    resp = app_client.post("/admin/login", json={"password": "test-admin-pw"})
    assert resp.status_code == 200
    return app_client


@pytest.fixture()
def backend_api_key(admin_client):
    resp = admin_client.get("/admin/provider", params={"reveal": True})
    assert resp.status_code == 200
    return resp.json()["api_key"]


def make_test_image_data_url(fmt="PNG") -> str:
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[fmt]
    return f"data:{mime};base64,{b64}"
