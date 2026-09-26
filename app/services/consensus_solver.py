"""
Two-model consensus solver.

Pipeline for every problem:

  1. SOLVE  — Gemini and Grok independently solve the same problem in
     parallel, from the same reconstructed problem text, using the
     identical SOLVER_SYSTEM_PROMPT (see prompts.py) so their outputs
     are directly comparable.

  2. AGREE? — If both answers are equivalent after whitespace/formatting
     normalization, that agreement is itself strong evidence of
     correctness, and it's returned immediately without a further
     verification round-trip (set ALWAYS_VERIFY=true to disable this
     shortcut and always run step 3 instead).

  3. VERIFY — If they disagree, BOTH models are asked to arbitrate:
     each is shown the problem plus both candidate solutions (labeled
     anonymously as Candidate A / Candidate B) and told to check each
     one against the problem's constraints and edge cases, then produce
     one final correct, optimal, submission-ready solution — not just
     pick one, but actually re-derive/fix as needed. This runs on both
     models concurrently.

  4. AGREE AGAIN? — If both arbiters land on an equivalent final
     answer, that's returned — now double cross-checked by both
     providers.

  5. TIEBREAK — If the two arbiters still disagree (rare — genuinely
     ambiguous or under-specified problems), `TIEBREAKER_PROVIDER`
     (default: "grok") decides, since at that point endless further
     rounds have diminishing returns.

Any single upstream failure degrades gracefully to whichever model(s)
are still working, rather than failing the whole request outright.
"""
from __future__ import annotations

import asyncio
import re

import httpx

from app.config import Settings
from app.services.gemini_client import GeminiError, call_gemini
from app.services.grok_client import GrokError, call_grok
from app.services.prompts import ARBITER_SYSTEM_PROMPT, SOLVER_SYSTEM_PROMPT


class ConsensusSolverError(Exception):
    """Raised when neither Gemini nor Grok could produce a usable answer."""


def _normalize(text: str) -> str:
    """Loose equality check used only to decide whether two answers
    already agree — never used to alter what's actually returned."""
    stripped = text.strip()
    stripped = re.sub(r"^```[a-zA-Z0-9+_-]*\n?", "", stripped)
    stripped = re.sub(r"\n?```$", "", stripped)
    stripped = re.sub(r"[ \t]+", " ", stripped)
    stripped = re.sub(r"\n{2,}", "\n", stripped)
    return stripped.strip().lower()


def _build_arbiter_content(problem_text: str, candidate_a: str, candidate_b: str) -> str:
    return (
        f"{problem_text}\n\n"
        "---\n"
        "CANDIDATE A:\n"
        f"{candidate_a}\n\n"
        "---\n"
        "CANDIDATE B:\n"
        f"{candidate_b}"
    )


async def _solve_both(settings: Settings, problem_text: str, client: httpx.AsyncClient, max_tokens: int | None):
    results = await asyncio.gather(
        call_gemini(settings, SOLVER_SYSTEM_PROMPT, problem_text, client=client, max_tokens=max_tokens),
        call_grok(settings, SOLVER_SYSTEM_PROMPT, problem_text, client=client, max_tokens=max_tokens),
        return_exceptions=True,
    )
    return results  # [gemini_result_or_exc, grok_result_or_exc]


async def _arbitrate_both(
    settings: Settings,
    arbiter_content: str,
    client: httpx.AsyncClient,
    max_tokens: int | None,
):
    results = await asyncio.gather(
        call_gemini(settings, ARBITER_SYSTEM_PROMPT, arbiter_content, client=client, max_tokens=max_tokens),
        call_grok(settings, ARBITER_SYSTEM_PROMPT, arbiter_content, client=client, max_tokens=max_tokens),
        return_exceptions=True,
    )
    return results  # [gemini_result_or_exc, grok_result_or_exc]


def _pick_tiebreak(settings: Settings, gemini_value, grok_value):
    """Resolves to whichever provider is configured as the tiebreaker,
    falling back to the other if that one is unavailable/errored."""
    gemini_ok = not isinstance(gemini_value, Exception)
    grok_ok = not isinstance(grok_value, Exception)
    prefer_grok = settings.TIEBREAKER_PROVIDER.lower() == "grok"

    if prefer_grok:
        if grok_ok:
            return grok_value
        if gemini_ok:
            return gemini_value
    else:
        if gemini_ok:
            return gemini_value
        if grok_ok:
            return grok_value
    return None


async def solve_problem(
    settings: Settings,
    problem_text: str,
    client: httpx.AsyncClient | None = None,
    max_tokens: int | None = None,
) -> str:
    """`max_tokens` is an optional per-call output budget. Each provider
    clamps it to `Settings.SOLVER_MAX_TOKENS`, and if the model rejects
    the value it retries once at the ceiling the model names."""
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()
    try:
        # --- Step 1: independent solves ---
        gemini_solution, grok_solution = await _solve_both(settings, problem_text, client, max_tokens)
        gemini_ok = not isinstance(gemini_solution, (Exception,))
        grok_ok = not isinstance(grok_solution, (Exception,))

        if not gemini_ok and not grok_ok:
            raise ConsensusSolverError(
                f"Both solver providers failed. Gemini: {gemini_solution}. Grok: {grok_solution}."
            )
        if not grok_ok:
            return gemini_solution
        if not gemini_ok:
            return grok_solution

        # --- Step 2: quick agreement check ---
        if not settings.ALWAYS_VERIFY and _normalize(gemini_solution) == _normalize(grok_solution):
            return gemini_solution

        # --- Step 3: cross-verification / arbitration by both models ---
        arbiter_content = _build_arbiter_content(problem_text, gemini_solution, grok_solution)
        gemini_arb, grok_arb = await _arbitrate_both(settings, arbiter_content, client, max_tokens)
        gemini_arb_ok = not isinstance(gemini_arb, (Exception,))
        grok_arb_ok = not isinstance(grok_arb, (Exception,))

        if not gemini_arb_ok and not grok_arb_ok:
            # Both arbiters failed (e.g. transient upstream issue) — fall
            # back to the tiebreaker's own first-pass solution rather than
            # failing the whole request.
            fallback = _pick_tiebreak(settings, gemini_solution, grok_solution)
            if fallback is not None:
                return fallback
            raise ConsensusSolverError(
                f"Both arbiter calls failed. Gemini: {gemini_arb}. Grok: {grok_arb}."
            )
        if not grok_arb_ok:
            return gemini_arb
        if not gemini_arb_ok:
            return grok_arb

        # --- Step 4: do the two arbiters agree? ---
        if _normalize(gemini_arb) == _normalize(grok_arb):
            return gemini_arb

        # --- Step 5: last-resort tiebreak ---
        tiebroken = _pick_tiebreak(settings, gemini_arb, grok_arb)
        if tiebroken is not None:
            return tiebroken

        raise ConsensusSolverError("Unable to resolve a final answer between Gemini and Grok.")
    except (GeminiError, GrokError) as exc:
        raise ConsensusSolverError(str(exc)) from exc
    finally:
        if owns_client:
            await client.aclose()
