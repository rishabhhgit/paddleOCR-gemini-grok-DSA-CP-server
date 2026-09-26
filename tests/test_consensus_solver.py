import pytest

from app.config import Settings
from app.services import consensus_solver as cs
from app.services.gemini_client import GeminiError
from app.services.grok_client import GrokError


def _settings(**overrides):
    base = dict(
        GEMINI_API_KEY="k",
        GEMINI_MODEL="m",
        GROK_API_KEY="k",
        GROK_MODEL="m",
        TIEBREAKER_PROVIDER="grok",
        ALWAYS_VERIFY=False,
    )
    base.update(overrides)
    return Settings(**base)


def _patch_solve(monkeypatch, gemini_value, grok_value):
    async def fake_gemini(settings, prompt, content, client=None, max_tokens=None):
        if isinstance(gemini_value, Exception):
            raise gemini_value
        return gemini_value

    async def fake_grok(settings, prompt, content, client=None, max_tokens=None):
        if isinstance(grok_value, Exception):
            raise grok_value
        return grok_value

    monkeypatch.setattr(cs, "call_gemini", fake_gemini)
    monkeypatch.setattr(cs, "call_grok", fake_grok)


@pytest.mark.asyncio
async def test_matching_solutions_returned_without_arbitration(monkeypatch):
    calls = {"count": 0}

    async def fake_gemini(settings, prompt, content, client=None, max_tokens=None):
        calls["count"] += 1
        return "int main(){return 0;}"

    async def fake_grok(settings, prompt, content, client=None, max_tokens=None):
        calls["count"] += 1
        return "int main(){return 0;}"

    monkeypatch.setattr(cs, "call_gemini", fake_gemini)
    monkeypatch.setattr(cs, "call_grok", fake_grok)

    result = await cs.solve_problem(_settings(), "Two sum problem")
    assert result == "int main(){return 0;}"
    # Only the first-pass solve calls, no arbitration round.
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_disagreeing_solutions_trigger_arbitration_and_agree(monkeypatch):
    solve_step = {"n": 0}

    async def fake_gemini(settings, prompt, content, client=None, max_tokens=None):
        if "CANDIDATE A" in content:
            return "FINAL_ANSWER"
        solve_step["n"] += 1
        return "gemini_solution_A"

    async def fake_grok(settings, prompt, content, client=None, max_tokens=None):
        if "CANDIDATE A" in content:
            return "FINAL_ANSWER"
        return "grok_solution_B"

    monkeypatch.setattr(cs, "call_gemini", fake_gemini)
    monkeypatch.setattr(cs, "call_grok", fake_grok)

    result = await cs.solve_problem(_settings(), "Some hard DP problem")
    assert result == "FINAL_ANSWER"


@pytest.mark.asyncio
async def test_disagreement_after_arbitration_uses_tiebreaker(monkeypatch):
    async def fake_gemini(settings, prompt, content, client=None, max_tokens=None):
        if "CANDIDATE A" in content:
            return "gemini_arbiter_final"
        return "gemini_solution"

    async def fake_grok(settings, prompt, content, client=None, max_tokens=None):
        if "CANDIDATE A" in content:
            return "grok_arbiter_final"
        return "grok_solution"

    monkeypatch.setattr(cs, "call_gemini", fake_gemini)
    monkeypatch.setattr(cs, "call_grok", fake_grok)

    result = await cs.solve_problem(_settings(TIEBREAKER_PROVIDER="grok"), "Ambiguous problem")
    assert result == "grok_arbiter_final"

    result = await cs.solve_problem(_settings(TIEBREAKER_PROVIDER="gemini"), "Ambiguous problem")
    assert result == "gemini_arbiter_final"


@pytest.mark.asyncio
async def test_one_solver_failing_falls_back_to_the_other(monkeypatch):
    _patch_solve(monkeypatch, GeminiError("down"), "grok_only_solution")
    result = await cs.solve_problem(_settings(), "Some problem")
    assert result == "grok_only_solution"

    _patch_solve(monkeypatch, "gemini_only_solution", GrokError("down"))
    result = await cs.solve_problem(_settings(), "Some problem")
    assert result == "gemini_only_solution"


@pytest.mark.asyncio
async def test_both_solvers_failing_raises(monkeypatch):
    _patch_solve(monkeypatch, GeminiError("down"), GrokError("down"))
    with pytest.raises(cs.ConsensusSolverError):
        await cs.solve_problem(_settings(), "Some problem")


@pytest.mark.asyncio
async def test_always_verify_forces_arbitration_even_on_agreement(monkeypatch):
    async def fake_gemini(settings, prompt, content, client=None, max_tokens=None):
        if "CANDIDATE A" in content:
            return "verified_final"
        return "same_solution"

    async def fake_grok(settings, prompt, content, client=None, max_tokens=None):
        if "CANDIDATE A" in content:
            return "verified_final"
        return "same_solution"

    monkeypatch.setattr(cs, "call_gemini", fake_gemini)
    monkeypatch.setattr(cs, "call_grok", fake_grok)

    result = await cs.solve_problem(_settings(ALWAYS_VERIFY=True), "Trivial problem")
    assert result == "verified_final"
