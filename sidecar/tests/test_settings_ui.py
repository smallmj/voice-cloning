"""UI preference persistence (issue #21): theme + engine selection."""

from __future__ import annotations

import httpx


def test_ui_prefs_default(client: httpx.Client):
    r = client.get("/settings/ui")
    assert r.status_code == 200
    assert r.json() == {"theme": "system", "engine_id": None}


def test_ui_prefs_round_trip(client: httpx.Client):
    r = client.put("/settings/ui", json={"theme": "light", "engine_id": "fake"})
    assert r.status_code == 200
    assert r.json() == {"theme": "light", "engine_id": "fake"}

    r = client.get("/settings/ui")
    assert r.json() == {"theme": "light", "engine_id": "fake"}

    # Partial updates merge; theme survives an engine-only write.
    r = client.put("/settings/ui", json={"engine_id": None})
    assert r.status_code == 200
    assert r.json() == {"theme": "light", "engine_id": None}


def test_ui_prefs_rejects_bad_theme(client: httpx.Client):
    r = client.put("/settings/ui", json={"theme": "sepia"})
    assert r.status_code == 422


def test_ui_prefs_rejects_empty_body(client: httpx.Client):
    r = client.put("/settings/ui", json={})
    assert r.status_code == 422


def test_ui_prefs_requires_auth(sidecar):
    r = httpx.put(
        f"{sidecar['base_url']}/settings/ui", json={"theme": "dark"}, timeout=10
    )
    assert r.status_code == 401
