"""
Shared diagnostics for the two provider clients.

gemini_client.py and grok_client.py deliberately mirror each other, but
they must agree on two rules that are easy to get subtly wrong:

1. Whether a 4xx is the provider rejecting OUR output budget. That is the
   only rejection worth a second attempt — retrying an auth or validation
   error would just turn a fast failure into a slow one. When the
   provider names the ceiling it will accept, the retry uses that number
   so an explicit cap is still in force; otherwise it falls back to the
   provider's own default. The ceiling is remembered per model so only
   the first request pays for that discovery.
2. How to turn a response into a short, loggable excerpt. The raw body is
   never used: it is unvetted, provider-controlled text headed straight
   for the logs (see app/utils/logging.py), so only structured message
   fields are trusted and everything else falls back to the status code.

Keeping both here stops the clients drifting apart on either rule.
"""
from __future__ import annotations

import re

import httpx

# Fragments a provider uses when it is complaining about the output
# budget it was asked for, e.g. "maxOutputTokens must be <= 8192",
# "max_tokens must be <= 16384", or Groq's
# "max_completion_tokens must be less than or equal to 32768". All four
# spellings of the field exist across OpenAI-compatible endpoints, and
# none of them contains another as a substring.
BUDGET_REJECTION_MARKERS = (
    "maxoutputtokens",
    "max_output_tokens",
    "max_completion_tokens",
    "max_tokens",
    "max tokens",
)

# The number a provider gives us when refusing a budget above its own
# maximum: either the compact "must be <= 8192" or the spelled-out
# "must be less than or equal to 32768".
_CEILING_PATTERN = re.compile(r"(?:<=|less than or equal to)\s*([0-9]+)", re.IGNORECASE)

# Ceiling each model has actually accepted, keyed by `base_url|model`.
# Learned from a budget rejection so the discovery costs one extra
# round trip per model instead of one per request: we ask for the
# largest budget the provider documents (see UNLIMITED_BUDGET in the
# client modules), and a model with a smaller maximum tells us once
# what it can take.
BUDGET_CEILINGS: dict[str, int] = {}


def clear_budget_ceilings() -> None:
    """Forgets every discovered ceiling. Tests use this so one model's
    discovered maximum can't leak into a request made under another."""
    BUDGET_CEILINGS.clear()


def _safe_json(resp: httpx.Response):
    try:
        return resp.json()
    except ValueError:
        return None


def _structured_message(data) -> str | None:
    """Pulls the human-readable message out of the error shapes providers
    actually use: `{"error": {"message": ...}}` (Gemini/OpenAI/Groq),
    `{"error": "..."}`, `{"detail": ...}` (FastAPI), or `{"message": ...}`."""
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if isinstance(error, dict):
        candidate = error.get("message")
    elif isinstance(error, str):
        candidate = error
    else:
        candidate = data.get("message") or data.get("detail")
    if isinstance(candidate, str) and candidate.strip():
        return " ".join(candidate.split())
    return None


def _rejection_message(resp: httpx.Response) -> str | None:
    message = _structured_message(_safe_json(resp))
    if message:
        return message
    # Budget rejections are also matched against the raw body, so a
    # provider replying in plain text is still readable here — this value
    # is used for pattern matching only, never logged verbatim.
    text = " ".join(resp.text.split())
    return text or None


def is_budget_rejection(resp: httpx.Response) -> bool:
    """True when a 4xx means "the cap you requested is above this model's
    maximum" rather than "this request is malformed"."""
    body = resp.text.lower()
    return any(marker in body for marker in BUDGET_REJECTION_MARKERS)


def rejected_budget_ceiling(resp: httpx.Response) -> int:
    """The largest budget this model will accept, when the provider named
    one; 0 when it refused without saying (meaning "use the provider's
    own default").

    Retrying with the number it gave us beats dropping the budget
    entirely: an explicit cap is what makes truncation detectable and
    reportable, and this value is by definition one the model accepts.
    """
    message = _rejection_message(resp)
    match = _CEILING_PATTERN.search(message) if message else None
    return int(match.group(1)) if match else 0


def error_excerpt(resp: httpx.Response, limit: int = 300) -> str:
    """`": <provider message>"` for a diagnosable error, or `"."` when
    there is nothing safe to say — never the raw response body."""
    message = _structured_message(_safe_json(resp))
    if message is None:
        return "."
    return f": {message[:limit]}"
