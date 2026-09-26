"""
Thin Grok client, speaking the OpenAI chat-completions dialect.

Groq's API is OpenAI-compatible (POST {base_url}/chat/completions with
{model, messages}, bearer auth), so this mirrors gemini_client.py
exactly. Exposes a single generic `call_grok` used both for the
independent "solve" step and (when Grok is configured as the tiebreaker)
the final arbitration step. The model name (GROK_MODEL) is only ever
used internally and is never echoed back to the client.
"""
from __future__ import annotations

import re

import httpx

from app.config import Settings
from app.services.provider_errors import (
    BUDGET_CEILINGS,
    error_excerpt,
    is_budget_rejection,
    rejected_budget_ceiling,
)

# Largest output budget the API documents for `max_completion_tokens`.
# Used when Settings.SOLVER_MAX_TOKENS is 0, i.e. "no limit": the API
# demands a number, so the closest thing to an uncapped generation is
# asking for the biggest one it accepts. Omitting the field would NOT be
# unlimited — providers are free to fall back to a much smaller default.
# A model with a smaller maximum rejects this with a 400 naming its own
# ceiling, which _post retries at (see provider_errors.py).
GROK_URL_TEMPLATE = "{base}/chat/completions"

UNLIMITED_BUDGET = 128000

# finish_reason values that mean "we hit the output cap", across the
# spellings OpenAI-compatible endpoints actually use.
TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens"})

# Groq returns reasoning models' thoughts inline in `content` by default
# (reasoning_format defaults to "raw"). Their answers are parsed after
# ours, so the thinking has to come back out before the text is used.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class GrokError(Exception):
    """Raised when the upstream Grok provider fails or is unreachable."""


def _build_payload(model: str, system_prompt: str, user_content: str, max_tokens: int) -> dict:
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
    }
    if max_tokens > 0:
        payload["max_completion_tokens"] = max_tokens
    return payload


def _endpoint(settings: Settings) -> str:
    return GROK_URL_TEMPLATE.format(base=settings.GROK_BASE_URL.rstrip("/"))


def _headers(settings: Settings) -> dict:
    return {"Authorization": f"Bearer {settings.GROK_API_KEY}"}


def _budget(settings: Settings, max_tokens: int | None) -> int:
    """The output budget for one call: the server's SOLVER_MAX_TOKENS
    (or the largest Groq accepts when that is 0 = "no limit"), narrowed
    by a client-supplied `max_tokens` if it asks for less. A client can
    only ever lower the budget, never raise it."""
    ceiling = settings.SOLVER_MAX_TOKENS if settings.SOLVER_MAX_TOKENS > 0 else UNLIMITED_BUDGET
    if max_tokens is None:
        return ceiling
    return max(0, min(max_tokens, ceiling))


def _timeout(settings: Settings) -> httpx.Timeout:
    """REQUEST_TIMEOUT_SECONDS of 0 (or less) = no timeout at all — the
    call waits however long Groq takes. A positive value caps every
    httpx phase (connect/read/write/pool) at that many seconds."""
    if settings.REQUEST_TIMEOUT_SECONDS <= 0:
        return httpx.Timeout(None)
    return httpx.Timeout(settings.REQUEST_TIMEOUT_SECONDS)


def _ceiling_key(settings: Settings) -> str:
    return f"{settings.GROK_BASE_URL}|{settings.GROK_MODEL}"


async def _post(
    client: httpx.AsyncClient,
    settings: Settings,
    system_prompt: str,
    user_content: str,
    budget: int,
) -> tuple[httpx.Response, int]:
    """POSTs once, retrying a single time at the ceiling the model will
    accept if it rejects ours. Returns the response together with the
    budget that response was actually produced under."""
    url = _endpoint(settings)
    headers = _headers(settings)
    key = _ceiling_key(settings)
    known = BUDGET_CEILINGS.get(key)
    if known is not None:
        budget = min(budget, known)

    async def attempt(tokens: int) -> httpx.Response:
        return await client.post(
            url,
            json=_build_payload(settings.GROK_MODEL, system_prompt, user_content, tokens),
            headers=headers,
            timeout=_timeout(settings),
        )

    try:
        resp = await attempt(budget)
        if budget > 0 and resp.status_code == 400 and is_budget_rejection(resp):
            # Retry once at the ceiling the provider just named, or with no
            # budget at all if it did not say, and remember the answer so
            # only this model's first request pays for the discovery.
            # Either way the response is still checked for truncation
            # before it is returned.
            ceiling = rejected_budget_ceiling(resp)
            BUDGET_CEILINGS[key] = ceiling
            budget = min(budget, ceiling) if ceiling > 0 else 0
            resp = await attempt(budget)
    except httpx.HTTPError as exc:
        raise GrokError(f"Failed to reach Grok: {exc}") from exc
    return resp, budget


def _truncation_error(budget: int) -> GrokError:
    if budget <= 0:
        cap = "the provider's default output cap"
        hint = "The provider's own default is too small for this problem."
    elif budget >= UNLIMITED_BUDGET:
        # We already asked for the largest budget Groq documents, so
        # there is no configuration left that would give the answer more
        # room — this is the model hitting its own hard limit.
        cap = f"its own {budget}-token hard limit"
        hint = (
            "The model was already given the largest budget Groq documents, "
            "so this is its own hard limit on one response."
        )
    else:
        cap = f"its {budget}-token output cap"
        hint = f"Raise SOLVER_MAX_TOKENS above {budget} (0 = no limit)."
    return GrokError(
        f"Grok stopped at {cap} before finishing the answer, so it "
        f"would have returned a half-written solution. {hint}"
    )


def _strip_reasoning(content: str) -> str:
    """Removes `<think>...` blocks Groq inlines into `content` for
    reasoning models, leaving only the answer. A no-op for models that
    do not reason (and when the API is already hiding the reasoning)."""
    if "<think" not in content:
        return content
    return _THINK_BLOCK.sub("", content).strip()


def _parse(resp: httpx.Response, budget: int) -> str:
    if resp.status_code >= 400:
        raise GrokError(f"Grok returned HTTP {resp.status_code}{error_excerpt(resp)}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise GrokError("Grok returned a non-JSON response.") from exc

    if not isinstance(data, dict):
        raise GrokError("Unexpected Grok response format: expected an object.")

    choices = data.get("choices") or []
    finish_reason = None
    text = ""
    if choices and isinstance(choices[0], dict):
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message") or {}
        if isinstance(message, dict):
            text = _strip_reasoning(message.get("content") or "")

    if finish_reason in TRUNCATION_FINISH_REASONS:
        raise _truncation_error(budget)

    if not text:
        raise GrokError("Grok returned an empty response.")

    return text


async def call_grok(
    settings: Settings,
    system_prompt: str,
    user_content: str,
    client: httpx.AsyncClient | None = None,
    max_tokens: int | None = None,
) -> str:
    """Sends a single-turn request to Grok and returns the text content.

    `max_tokens` narrows this call's budget below `Settings.SOLVER_MAX_TOKENS`
    and is never allowed to exceed it. Truncated responses raise instead of
    being returned.
    """
    if not settings.GROK_MODEL:
        raise GrokError("GROK_MODEL is not configured on the server.")
    if not settings.GROK_API_KEY:
        raise GrokError("GROK_API_KEY is not configured on the server.")

    budget = _budget(settings, max_tokens)

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()
    try:
        resp, budget = await _post(client, settings, system_prompt, user_content, budget)
    finally:
        if owns_client:
            await client.aclose()

    return _parse(resp, budget)
