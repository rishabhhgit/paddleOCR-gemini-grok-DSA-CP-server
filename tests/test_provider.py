def test_provider_config_masked_by_default(admin_client):
    resp = admin_client.get("/admin/provider")
    assert resp.status_code == 200
    key = resp.json()["api_key"]
    assert "*" in key


def test_provider_config_reveal(admin_client):
    resp = admin_client.get("/admin/provider", params={"reveal": True})
    assert resp.status_code == 200
    key = resp.json()["api_key"]
    assert key.startswith("dsa_sk_")
    assert "*" not in key


def test_api_key_persists_across_requests(admin_client):
    resp1 = admin_client.get("/admin/provider", params={"reveal": True})
    resp2 = admin_client.get("/admin/provider", params={"reveal": True})
    assert resp1.json()["api_key"] == resp2.json()["api_key"]


def test_regenerate_invalidates_old_key(admin_client, backend_api_key):
    # old key currently works for auth
    import app.api.chat_completions as cc

    resp = admin_client.post("/admin/provider/regenerate")
    assert resp.status_code == 200

    new_config = admin_client.get("/admin/provider", params={"reveal": True}).json()
    assert new_config["api_key"] != backend_api_key

    # old key should now fail auth on the chat endpoint
    auth_resp = admin_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_api_key}"},
        json={"model": "dsa-solver", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert auth_resp.status_code == 401


def test_public_provider_config_endpoint_requires_admin(app_client):
    resp = app_client.get("/api/provider/config")
    assert resp.status_code == 401


def test_public_provider_config_endpoint(admin_client):
    resp = admin_client.get("/api/provider/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["endpoint"] == "https://test.example.com/v1/chat/completions"
    assert body["status"] == "active"


def test_provider_api_key_env_override_is_used_directly(app_client, monkeypatch):
    # When PROVIDER_API_KEY is set, it should be returned as-is instead of
    # a disk-generated key -- this is what keeps the key stable across
    # redeploys on hosts with no persistent disk (e.g. Render free tier).
    import app.config as config_module
    import app.security.api_keys as api_keys_module

    monkeypatch.setenv("PROVIDER_API_KEY", "dsa_sk_fixed-test-key-value")
    config_module.get_settings.cache_clear()
    api_keys_module._store = None
    api_keys_module._store_key_signature = None

    resp = app_client.post("/admin/login", json={"password": "test-admin-pw"})
    assert resp.status_code == 200

    resp1 = app_client.get("/admin/provider", params={"reveal": True})
    resp2 = app_client.get("/admin/provider", params={"reveal": True})
    assert resp1.json()["api_key"] == "dsa_sk_fixed-test-key-value"
    assert resp2.json()["api_key"] == "dsa_sk_fixed-test-key-value"

    config_module.get_settings.cache_clear()
    api_keys_module._store = None
    api_keys_module._store_key_signature = None


def test_provider_api_key_env_override_survives_simulated_redeploy(app_client, monkeypatch, tmp_data_dir):
    # Simulate a redeploy on an ephemeral disk: wipe DATA_DIR and reset the
    # in-process store, the way a fresh Render container would start up.
    import shutil

    import app.config as config_module
    import app.security.api_keys as api_keys_module

    monkeypatch.setenv("PROVIDER_API_KEY", "dsa_sk_stable-across-redeploys")
    config_module.get_settings.cache_clear()
    api_keys_module._store = None
    api_keys_module._store_key_signature = None

    app_client.post("/admin/login", json={"password": "test-admin-pw"})
    first = app_client.get("/admin/provider", params={"reveal": True}).json()["api_key"]

    # Wipe disk + reset singletons, as a fresh container/deploy would do.
    shutil.rmtree(tmp_data_dir, ignore_errors=True)
    tmp_data_dir.mkdir()
    config_module.get_settings.cache_clear()
    api_keys_module._store = None
    api_keys_module._store_key_signature = None

    app_client.post("/admin/login", json={"password": "test-admin-pw"})
    second = app_client.get("/admin/provider", params={"reveal": True}).json()["api_key"]

    assert first == second == "dsa_sk_stable-across-redeploys"

    config_module.get_settings.cache_clear()
    api_keys_module._store = None
    api_keys_module._store_key_signature = None


def test_regenerate_rejected_when_env_key_is_set(app_client, monkeypatch):
    import app.config as config_module
    import app.security.api_keys as api_keys_module

    monkeypatch.setenv("PROVIDER_API_KEY", "dsa_sk_fixed-test-key-value")
    config_module.get_settings.cache_clear()
    api_keys_module._store = None
    api_keys_module._store_key_signature = None

    app_client.post("/admin/login", json={"password": "test-admin-pw"})
    resp = app_client.post("/admin/provider/regenerate")
    assert resp.status_code == 409

    config_module.get_settings.cache_clear()
    api_keys_module._store = None
    api_keys_module._store_key_signature = None


def test_default_key_derived_from_admin_session_secret_survives_disk_wipe(app_client, tmp_data_dir):
    # Core guarantee: with NO PROVIDER_API_KEY set at all, the key must
    # still be identical across a simulated Render redeploy (disk wiped,
    # process singletons reset), as long as ADMIN_SESSION_SECRET (a
    # required, already-set env var) doesn't change.
    import shutil

    import app.config as config_module
    import app.security.api_keys as api_keys_module

    app_client.post("/admin/login", json={"password": "test-admin-pw"})
    first = app_client.get("/admin/provider", params={"reveal": True}).json()["api_key"]
    assert first.startswith("dsa_sk_")

    # Simulate a redeploy on an ephemeral disk: wipe DATA_DIR and reset
    # the in-process singletons, exactly like a fresh container boot.
    shutil.rmtree(tmp_data_dir, ignore_errors=True)
    tmp_data_dir.mkdir()
    config_module.get_settings.cache_clear()
    api_keys_module._store = None
    api_keys_module._store_key_signature = None

    app_client.post("/admin/login", json={"password": "test-admin-pw"})
    second = app_client.get("/admin/provider", params={"reveal": True}).json()["api_key"]

    assert first == second


def test_different_admin_session_secret_yields_different_default_key():
    # Sanity check that the derivation is actually a function of the
    # secret (not a coincidence / hardcoded value), tested directly
    # against the derivation function used by FileAPIKeyStore.
    from app.security.api_keys import _derive_key

    key_a = _derive_key("secret-one")
    key_b = _derive_key("a-totally-different-secret")
    assert key_a != key_b
    assert key_a.startswith("dsa_sk_")
    assert key_b.startswith("dsa_sk_")


def test_data_dir_redirects_to_tmp_on_vercel(monkeypatch):
    # On Vercel, only /tmp is writable at request time. If DATA_DIR is
    # left at its Docker/local default ("data") and VERCEL=1 is present
    # (set automatically by the platform), settings should redirect it
    # to /tmp so the app doesn't crash trying to create an unwritable
    # directory.
    import app.config as config_module

    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("APP_ADMIN_PASSWORD", "test-admin-pw")
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-session-secret")
    config_module.get_settings.cache_clear()

    settings = config_module.get_settings()
    assert settings.DATA_DIR.startswith("/tmp")

    config_module.get_settings.cache_clear()


def test_data_dir_untouched_when_not_on_vercel(monkeypatch):
    # Off Vercel (VERCEL unset), the default "data" dir should be left
    # alone -- this is what Docker/local/Render deployments rely on.
    import app.config as config_module

    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setenv("APP_ADMIN_PASSWORD", "test-admin-pw")
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-session-secret")
    config_module.get_settings.cache_clear()

    settings = config_module.get_settings()
    assert settings.DATA_DIR == "data"

    config_module.get_settings.cache_clear()
