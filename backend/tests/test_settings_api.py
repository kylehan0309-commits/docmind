async def test_get_settings_shape(client):
    body = (await client.get("/settings")).json()
    assert body["provider"] == "local"
    assert set(body["available"]) == {"local", "anthropic", "gemini"}
    assert body["available"] == {"local": True, "anthropic": False, "gemini": False}
    assert "gemini" in body["models"] and "anthropic" in body["models"]
    assert "chat_model" in body["models"]["local"]


async def test_put_rejects_provider_without_key(client):
    assert (await client.put("/settings", json={"provider": "gemini"})).status_code == 400
    assert (await client.put("/settings", json={"provider": "anthropic"})).status_code == 400


async def test_put_rejects_unknown_provider(client):
    assert (await client.put("/settings", json={"provider": "banana"})).status_code == 400


async def test_put_switches_to_local(client):
    resp = await client.put("/settings", json={"provider": "local"})
    assert resp.status_code == 200 and resp.json()["provider"] == "local"


async def test_put_switches_to_anthropic_when_key_is_present(client, monkeypatch):
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "sk-test")

    resp = await client.put("/settings", json={"provider": "anthropic"})
    assert resp.status_code == 200 and resp.json()["provider"] == "anthropic"
    assert (await client.get("/settings")).json()["provider"] == "anthropic"


async def test_provider_choice_survives_a_simulated_restart(client, monkeypatch):
    """PUT /settings should durably persist the choice, not just flip the
    in-memory flag - simulate a restart by resetting _active and reloading."""
    from app.services import provider

    monkeypatch.setattr("app.config.settings.gemini_api_key", "g-test")
    resp = await client.put("/settings", json={"provider": "gemini"})
    assert resp.status_code == 200

    # a fresh process would start _active from settings.llm_provider ("local")
    provider._active = "local"
    assert provider.get_provider() == "local"

    await provider.load_persisted_provider()
    assert provider.get_provider() == "gemini"


async def test_load_persisted_provider_falls_back_if_key_is_gone(client, monkeypatch):
    """A provider persisted in an earlier run whose key has since been removed
    from .env must not break startup - just keep the .env default."""
    from app.services import provider

    monkeypatch.setattr("app.config.settings.gemini_api_key", "g-test")
    await client.put("/settings", json={"provider": "gemini"})

    provider._active = "local"
    monkeypatch.setattr("app.config.settings.gemini_api_key", "")  # key revoked

    await provider.load_persisted_provider()  # must not raise
    assert provider.get_provider() == "local"
