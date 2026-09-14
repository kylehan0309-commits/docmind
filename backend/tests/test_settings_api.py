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
