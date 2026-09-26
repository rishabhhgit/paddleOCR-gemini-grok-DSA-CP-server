def test_missing_api_key_returns_401(app_client):
    resp = app_client.post("/v1/chat/completions", json={"model": "dsa-solver", "messages": []})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_invalid_api_key_returns_401(app_client):
    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer not-a-real-key"},
        json={"model": "dsa-solver", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_valid_api_key_passes_auth(app_client, backend_api_key, monkeypatch):
    # Mock the solver so we isolate auth behavior from provider calls.
    import app.api.chat_completions as cc

    async def fake_solve(settings, text, client=None):
        return "## Problem\n...\n## Code\n```cpp\nint main(){}\n```"

    monkeypatch.setattr(cc, "solve_problem", fake_solve)

    resp = app_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={"model": "dsa-solver", "messages": [{"role": "user", "content": "Solve: sum of array"}]},
    )
    assert resp.status_code == 200
