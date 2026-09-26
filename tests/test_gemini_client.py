from app.config import Settings
from app.services.gemini_client import _endpoint


def test_default_base_url_targets_the_public_api():
    url = _endpoint(Settings(GEMINI_MODEL="gemini-2.0-flash"))
    assert url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def test_base_url_override_is_used_and_trailing_slash_is_stripped():
    url = _endpoint(Settings(GEMINI_BASE_URL="http://127.0.0.1:8199/", GEMINI_MODEL="fake-model"))
    assert url == "http://127.0.0.1:8199/v1beta/models/fake-model:generateContent"
