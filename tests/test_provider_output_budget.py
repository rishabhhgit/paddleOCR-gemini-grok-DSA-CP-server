"""Output-budget behaviour shared by both provider clients.

Both clients speak the OpenAI chat-completions dialect now — Gemini
through Google's OpenAI-compatible surface, Grok through Groq — so what
is under test is the regression where a model hitting its own output cap
had its half-written solution returned as if it were complete: no budget
was ever sent, and finish_reason was reported as "stop" regardless."""
import json

import httpx
import pytest
import respx

from app.config import Settings
from app.services import gemini_client, grok_client
from app.services.gemini_client import GeminiError, call_gemini
from app.services.grok_client import GrokError, call_grok
from app.services.provider_errors import clear_budget_ceilings

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GROK_URL = "https://api.groq.com/openai/v1/chat/completions"


@pytest.fixture(autouse=True)
def _isolated_ceiling_cache():
    """A ceiling discovered for `test-model` in one test must not decide
    the budget of the next one."""
    clear_budget_ceilings()
    yield
    clear_budget_ceilings()


def _gemini_settings(**overrides):
    base = dict(GEMINI_API_KEY="k", GEMINI_MODEL="test-model", SOLVER_MAX_TOKENS=32768)
    base.update(overrides)
    return Settings(**base)


def _grok_settings(**overrides):
    base = dict(GROK_API_KEY="k", GROK_MODEL="test-model", SOLVER_MAX_TOKENS=32768)
    base.update(overrides)
    return Settings(**base)


def _gemini_response(finish_reason="stop", text="int main(){}"):
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
        },
    )


def _grok_response(finish_reason="stop", content="int main(){}"):
    return httpx.Response(
        200,
        json={"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}]},
    )


def _budget_of(request) -> int:
    return json.loads(request.content)["max_tokens"]


def _grok_budget_of(request) -> int:
    return json.loads(request.content)["max_completion_tokens"]


# --- Gemini ---


@respx.mock
async def test_gemini_request_carries_the_configured_output_budget():
    route = respx.post(GEMINI_URL).mock(return_value=_gemini_response())

    text = await call_gemini(_gemini_settings(SOLVER_MAX_TOKENS=12345), "sys", "problem")

    assert text == "int main(){}"
    payload = json.loads(route.calls[0].request.content)
    assert payload["max_tokens"] == 12345
    assert payload["model"] == "test-model"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["messages"][1] == {"role": "user", "content": "problem"}
    assert route.calls[0].request.headers["Authorization"] == "Bearer k"


@respx.mock
async def test_gemini_unlimited_budget_asks_for_the_model_maximum():
    """0 = no limit. Omitting the field would NOT be unlimited: the
    provider would fall back to a model-dependent default far below its
    real output limit, which is exactly what used to cut answers off."""
    route = respx.post(GEMINI_URL).mock(return_value=_gemini_response())

    await call_gemini(_gemini_settings(SOLVER_MAX_TOKENS=0), "sys", "problem")

    assert _budget_of(route.calls[0].request) == gemini_client.UNLIMITED_BUDGET


@respx.mock
async def test_gemini_truncation_is_reported_instead_of_returned():
    """A half-written solution must never leave the server as an answer."""
    respx.post(GEMINI_URL).mock(
        return_value=_gemini_response(finish_reason="length", text="int solve() { /* cut mid-func")
    )

    with pytest.raises(GeminiError) as exc:
        await call_gemini(_gemini_settings(), "sys", "problem")

    assert "half-written" in str(exc.value)
    assert "SOLVER_MAX_TOKENS" in str(exc.value)


@respx.mock
async def test_gemini_retries_at_the_ceiling_the_model_names():
    """Models capped below SOLVER_MAX_TOKENS reject the field with a 400
    naming their real maximum. Retry there rather than hard-failing, and
    keep an explicit budget so truncation stays detectable."""
    route = respx.post(GEMINI_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "max_tokens must be <= 8192"}}),
            _gemini_response(),
        ]
    )

    text = await call_gemini(_gemini_settings(), "sys", "problem")

    assert text == "int main(){}"
    assert route.call_count == 2
    assert _budget_of(route.calls[0].request) == 32768
    assert _budget_of(route.calls[1].request) == 8192


@respx.mock
async def test_gemini_drops_the_budget_when_the_model_refuses_without_naming_a_ceiling():
    route = respx.post(GEMINI_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "max_tokens is not supported"}}),
            _gemini_response(),
        ]
    )

    await call_gemini(_gemini_settings(), "sys", "problem")

    assert route.call_count == 2
    assert "max_tokens" not in json.loads(route.calls[1].request.content)


@respx.mock
async def test_gemini_budget_rejection_still_fails_when_the_retry_also_fails():
    respx.post(GEMINI_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "max_tokens must be <= 8192"}})
    )

    with pytest.raises(GeminiError) as exc:
        await call_gemini(_gemini_settings(), "sys", "problem")

    assert "400" in str(exc.value)
    assert "max_tokens" in str(exc.value)
    assert "max_tokens must be <= 8192" in str(exc.value)


@respx.mock
async def test_gemini_ignores_an_unrelated_400():
    route = respx.post(GEMINI_URL).mock(return_value=httpx.Response(400, json={"error": {"message": "API key not valid"}}))

    with pytest.raises(GeminiError):
        await call_gemini(_gemini_settings(), "sys", "problem")

    assert route.call_count == 1


@respx.mock
async def test_gemini_blocked_finish_reason_with_no_text_is_an_error():
    respx.post(GEMINI_URL).mock(
        return_value=_gemini_response(finish_reason="content_filter", text="")
    )

    with pytest.raises(GeminiError) as exc:
        await call_gemini(_gemini_settings(), "sys", "problem")

    assert "content_filter" in str(exc.value)


@respx.mock
async def test_gemini_accepts_content_returned_as_a_list_of_parts():
    respx.post(GEMINI_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "int main(){}"}],
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )
    )

    assert await call_gemini(_gemini_settings(), "sys", "problem") == "int main(){}"


# --- Grok ---


@respx.mock
async def test_grok_request_carries_the_configured_output_budget():
    route = respx.post(GROK_URL).mock(return_value=_grok_response())

    text = await call_grok(_grok_settings(SOLVER_MAX_TOKENS=12345), "sys", "problem")

    assert text == "int main(){}"
    payload = json.loads(route.calls[0].request.content)
    assert payload["max_completion_tokens"] == 12345
    assert payload["model"] == "test-model"
    assert route.calls[0].request.headers["Authorization"] == "Bearer k"


@respx.mock
async def test_grok_unlimited_budget_asks_for_the_model_maximum():
    route = respx.post(GROK_URL).mock(return_value=_grok_response())

    await call_grok(_grok_settings(SOLVER_MAX_TOKENS=0), "sys", "problem")

    assert _grok_budget_of(route.calls[0].request) == grok_client.UNLIMITED_BUDGET


@respx.mock
async def test_grok_truncation_is_reported_instead_of_returned():
    respx.post(GROK_URL).mock(return_value=_grok_response(finish_reason="length", content="int solve() { /* cut"))

    with pytest.raises(GrokError) as exc:
        await call_grok(_grok_settings(), "sys", "problem")

    assert "half-written" in str(exc.value)
    assert "SOLVER_MAX_TOKENS" in str(exc.value)


@respx.mock
async def test_grok_retries_at_the_ceiling_the_model_names():
    route = respx.post(GROK_URL).mock(
        side_effect=[
            httpx.Response(
                400,
                json={"error": {"message": "max_completion_tokens must be less than or equal to 16384"}},
            ),
            _grok_response(),
        ]
    )

    text = await call_grok(_grok_settings(), "sys", "problem")

    assert text == "int main(){}"
    assert route.call_count == 2
    assert _grok_budget_of(route.calls[0].request) == 32768
    assert _grok_budget_of(route.calls[1].request) == 16384


@respx.mock
async def test_grok_drops_the_budget_when_the_model_refuses_without_naming_a_ceiling():
    route = respx.post(GROK_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "max_completion_tokens not supported"}}),
            _grok_response(),
        ]
    )

    await call_grok(_grok_settings(), "sys", "problem")

    assert route.call_count == 2
    assert "max_completion_tokens" not in json.loads(route.calls[1].request.content)


@respx.mock
async def test_grok_inlined_reasoning_is_stripped_from_the_answer():
    """Groq defaults to reasoning_format="raw", so a reasoning model
    returns its thoughts inside `content`. Only the answer may reach the
    arbiter."""
    # Assembled at runtime: a literal opening/closing tag in source is
    # easy to lose to a careless tool, and the whole point is the shape.
    opening, closing = "<" + "think>", "</" + "think>"
    respx.post(GROK_URL).mock(
        return_value=_grok_response(
            content=(
                f"{opening}I should double check the base case.{closing}"
                f"{opening}Yes.{closing}answer"
            )
        )
    )

    assert await call_grok(_grok_settings(), "sys", "problem") == "answer"


# --- Client-supplied budget ---


@respx.mock
async def test_client_budget_cannot_exceed_the_server_ceiling():
    route = respx.post(GROK_URL).mock(return_value=_grok_response())

    await call_grok(_grok_settings(SOLVER_MAX_TOKENS=4096), "sys", "problem", max_tokens=999_999)

    assert _grok_budget_of(route.calls[0].request) == 4096


@respx.mock
async def test_client_budget_can_only_lower_the_ceiling():
    route = respx.post(GROK_URL).mock(return_value=_grok_response())

    await call_grok(_grok_settings(SOLVER_MAX_TOKENS=32768), "sys", "problem", max_tokens=1024)

    assert _grok_budget_of(route.calls[0].request) == 1024


# --- ceiling discovery is remembered ---


@respx.mock
async def test_a_discovered_ceiling_is_reused_instead_of_rejected_again():
    """Discovery costs one extra round trip per model, not one per request."""
    route = respx.post(GEMINI_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "max_tokens must be <= 8192"}}),
            _gemini_response(),
            _gemini_response(),
        ]
    )
    settings = _gemini_settings(SOLVER_MAX_TOKENS=0)

    await call_gemini(settings, "sys", "problem")
    await call_gemini(settings, "sys", "problem")

    assert route.call_count == 3
    assert _budget_of(route.calls[0].request) == gemini_client.UNLIMITED_BUDGET
    assert _budget_of(route.calls[1].request) == 8192
    # Second request goes straight to the known ceiling — no rejection.
    assert _budget_of(route.calls[2].request) == 8192


@respx.mock
async def test_a_discovered_ceiling_still_narrows_an_explicit_request():
    route = respx.post(GROK_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "max_completion_tokens must be <= 4096"}}),
            _grok_response(),
            _grok_response(),
        ]
    )
    settings = _grok_settings()

    await call_grok(settings, "sys", "problem", max_tokens=32768)
    await call_grok(settings, "sys", "problem", max_tokens=32768)

    assert _grok_budget_of(route.calls[2].request) == 4096


# --- timeouts ---


def test_zero_timeout_means_no_timeout_at_all():
    settings = Settings(GEMINI_API_KEY="k", GEMINI_MODEL="m", REQUEST_TIMEOUT_SECONDS=0)
    timeout = gemini_client._timeout(settings)
    assert timeout.connect is None
    assert timeout.read is None
    assert timeout.write is None
    assert timeout.pool is None


def test_positive_timeout_caps_every_httpx_phase():
    settings = Settings(GROK_API_KEY="k", GROK_MODEL="m", REQUEST_TIMEOUT_SECONDS=540)
    timeout = grok_client._timeout(settings)
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (540.0, 540.0, 540.0, 540.0)


def test_timeout_defaults_to_no_limit():
    settings = Settings(GEMINI_API_KEY="k", GEMINI_MODEL="m")
    assert settings.REQUEST_TIMEOUT_SECONDS == 0
    assert gemini_client._timeout(settings).read is None


# --- budget defaults ---


def test_output_budget_defaults_to_the_provider_maximum():
    gemini = Settings(GEMINI_API_KEY="k", GEMINI_MODEL="m")
    grok = Settings(GROK_API_KEY="k", GROK_MODEL="m")
    assert gemini.SOLVER_MAX_TOKENS == 0
    assert grok.SOLVER_MAX_TOKENS == 0
    assert gemini_client._budget(gemini, None) == gemini_client.UNLIMITED_BUDGET
    assert grok_client._budget(grok, None) == grok_client.UNLIMITED_BUDGET


@respx.mock
async def test_truncation_at_the_provider_maximum_says_so_instead_of_promising_a_config_change():
    respx.post(GEMINI_URL).mock(
        return_value=_gemini_response(finish_reason="length", text="int solve() { /* cut")
    )

    with pytest.raises(GeminiError) as exc:
        await call_gemini(_gemini_settings(SOLVER_MAX_TOKENS=0), "sys", "problem")

    message = str(exc.value)
    assert "hard limit" in message
    assert "Raise SOLVER_MAX_TOKENS" not in message


def test_all_request_limits_default_to_unlimited():
    settings = Settings(GEMINI_API_KEY="k", GEMINI_MODEL="m")
    assert settings.MAX_IMAGES == 0
    assert settings.MAX_IMAGE_SIZE_MB == 0
    assert settings.MAX_OCR_BATCH_IMAGES == 0
    assert settings.MAX_OCR_BATCH_BYTES == 0
    assert settings.MAX_PROMPT_CHARS == 0
    assert settings.ALWAYS_VERIFY is True


def test_base_urls_target_the_openai_compatible_endpoints():
    settings = Settings(GEMINI_API_KEY="k", GEMINI_MODEL="m", GROK_API_KEY="k", GROK_MODEL="m")
    assert gemini_client._endpoint(settings) == GEMINI_URL
    assert grok_client._endpoint(settings) == GROK_URL
