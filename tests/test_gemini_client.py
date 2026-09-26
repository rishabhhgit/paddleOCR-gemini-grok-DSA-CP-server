from app.config import Settings
from app.services.gemini_client import _endpoint, _headers
from app.services.grok_client import _endpoint as _grok_endpoint


def test_default_base_url_targets_the_openai_compatible_surface():
    url = _endpoint(Settings(GEMINI_MODEL="gemini-2.0-flash"))
    assert url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def test_base_url_override_is_used_and_trailing_slash_is_stripped():
    url = _endpoint(Settings(GEMINI_BASE_URL="http://127.0.0.1:8199/", GEMINI_MODEL="fake-model"))
    assert url == "http://127.0.0.1:8199/chat/completions"


def test_gemini_authenticates_with_a_bearer_key():
    headers = _headers(Settings(GEMINI_API_KEY="secret"))
    assert headers == {"Authorization": "Bearer secret"}


def test_grok_default_base_url_targets_groq():
    url = _grok_endpoint(Settings(GROK_MODEL="llama-3.3-70b-versatile"))
    assert url == "https://api.groq.com/openai/v1/chat/completions"
