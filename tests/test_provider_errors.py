"""Diagnostics shared by both provider clients.

The rule under test: an error that reaches the logs (and, via the admin
panel, the admin) must be a structured provider message, never the raw
response body."""
import httpx

from app.services.provider_errors import error_excerpt, is_budget_rejection, rejected_budget_ceiling


def test_budget_rejection_is_recognised_in_a_json_body():
    resp = httpx.Response(400, json={"error": {"message": "maxOutputTokens must be <= 8192"}})
    assert is_budget_rejection(resp)


def test_budget_rejection_is_recognised_in_a_plain_body():
    resp = httpx.Response(400, text="max_tokens must be <= 16384")
    assert is_budget_rejection(resp)


def test_budget_rejection_is_recognised_in_a_groq_style_body():
    resp = httpx.Response(
        400, json={"error": {"message": "max_completion_tokens must be less than or equal to 32768"}}
    )
    assert is_budget_rejection(resp)


def test_an_unrelated_error_is_not_treated_as_a_budget_rejection():
    resp = httpx.Response(400, json={"error": {"message": "API key not valid. Please pass a valid API key."}})
    assert not is_budget_rejection(resp)


def test_gemini_shaped_error_message_is_extracted():
    resp = httpx.Response(400, json={"error": {"code": 400, "message": "Invalid argument", "status": "INVALID_ARGUMENT"}})
    assert error_excerpt(resp) == ": Invalid argument"


def test_openai_and_groq_shaped_error_message_is_extracted():
    resp = httpx.Response(400, json={"error": {"message": "model not found", "type": "invalid_request_error"}})
    assert error_excerpt(resp) == ": model not found"


def test_a_string_error_field_is_extracted():
    resp = httpx.Response(400, json={"error": "max_tokens must be <= 16384"})
    assert error_excerpt(resp) == ": max_tokens must be <= 16384"


def test_fastapi_detail_is_extracted():
    resp = httpx.Response(422, json={"detail": "field required"})
    assert error_excerpt(resp) == ": field required"


def test_a_long_message_is_truncated():
    resp = httpx.Response(400, json={"error": {"message": "x" * 1000}})
    excerpt = error_excerpt(resp, limit=50)
    assert excerpt == ": " + "x" * 50


def test_a_non_json_body_is_never_echoed():
    """HTML/proxy bodies are unvetted provider text; only the status code
    should survive, or an auth header could reach the log verbatim."""
    resp = httpx.Response(502, text="<html><body>Bearer sk-abc123 secret</body></html>")
    assert error_excerpt(resp) == "."


def test_a_json_body_without_a_known_message_field_is_not_echoed():
    resp = httpx.Response(400, json={"unexpected": "Authorization: Bearer sk-abc123"})
    assert error_excerpt(resp) == "."


def test_an_empty_body_is_not_echoed():
    assert error_excerpt(httpx.Response(500, text="")) == "."


def test_whitespace_in_a_message_is_normalised():
    resp = httpx.Response(400, json={"error": {"message": "line one\n\n   line two"}})
    assert error_excerpt(resp) == ": line one line two"


# --- budget rejections ---


def test_ceiling_is_read_from_a_json_error_message():
    resp = httpx.Response(400, json={"error": {"message": "maxOutputTokens must be <= 8192"}})
    assert rejected_budget_ceiling(resp) == 8192


def test_ceiling_is_read_from_a_plain_text_body():
    resp = httpx.Response(400, text="max_tokens must be <= 16384")
    assert rejected_budget_ceiling(resp) == 16384


def test_ceiling_is_read_from_a_spelled_out_comparison():
    resp = httpx.Response(
        400, json={"error": {"message": "max_completion_tokens must be less than or equal to 32768"}}
    )
    assert rejected_budget_ceiling(resp) == 32768


def test_ceiling_is_zero_when_the_provider_does_not_name_one():
    resp = httpx.Response(400, json={"error": {"message": "maxOutputTokens is not supported"}})
    assert rejected_budget_ceiling(resp) == 0
