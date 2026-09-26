def test_admin_login_wrong_password(app_client):
    resp = app_client.post("/admin/login", json={"password": "wrong"})
    assert resp.status_code == 401


def test_admin_login_correct_password(app_client):
    resp = app_client.post("/admin/login", json={"password": "test-admin-pw"})
    assert resp.status_code == 200
    assert "dsa_admin_session" in resp.cookies


def test_admin_provider_requires_session(app_client):
    resp = app_client.get("/admin/provider")
    assert resp.status_code == 401


def test_admin_provider_accessible_after_login(admin_client):
    resp = admin_client.get("/admin/provider")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "multi-model-dsa"
    assert body["model"] == "dsa-solver"


def test_admin_page_serves_html(app_client):
    resp = app_client.get("/admin")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_admin_test_backend_requires_session(app_client):
    resp = app_client.post("/admin/test")
    assert resp.status_code == 401


def test_admin_test_backend_reports_gemini_grok_and_paddle_ocr(admin_client, monkeypatch):
    import app.api.admin as admin_module
    from app.models.solver import OcrResult

    async def fake_gemini(settings, prompt, content, client=None):
        return "int main(){}"

    async def fake_grok(settings, prompt, content, client=None):
        return "int main(){}"

    async def fake_ocr(settings, images, client=None):
        return [OcrResult(index=0, text="OCR TEST 12345")]

    monkeypatch.setattr(admin_module, "call_gemini", fake_gemini)
    monkeypatch.setattr(admin_module, "call_grok", fake_grok)
    monkeypatch.setattr(admin_module, "run_ocr_on_images", fake_ocr)

    resp = admin_client.post("/admin/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["gemini_connectivity"] == "ok"
    assert body["gemini_success"] is True
    assert body["grok_connectivity"] == "ok"
    assert body["grok_success"] is True
    assert body["paddle_ocr_connectivity"] == "ok"
    assert body["paddle_ocr_success"] is True


def test_admin_test_backend_reports_paddle_ocr_failure(admin_client, monkeypatch):
    import app.api.admin as admin_module
    from app.services.paddle_ocr import PaddleOcrError

    async def fake_gemini(settings, prompt, content, client=None):
        return "int main(){}"

    async def fake_grok(settings, prompt, content, client=None):
        return "int main(){}"

    async def failing_ocr(settings, images, client=None):
        raise PaddleOcrError("simulated OCR failure")

    monkeypatch.setattr(admin_module, "call_gemini", fake_gemini)
    monkeypatch.setattr(admin_module, "call_grok", fake_grok)
    monkeypatch.setattr(admin_module, "run_ocr_on_images", failing_ocr)

    resp = admin_client.post("/admin/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["gemini_connectivity"] == "ok"
    assert body["grok_connectivity"] == "ok"
    assert body["paddle_ocr_connectivity"] == "failed"
    assert body["paddle_ocr_success"] is False
    assert "paddle_ocr_error" in body
