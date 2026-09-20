"""UI preference persistence (issue #21): theme + engine selection.

Issue #23 / ADR-0018 decision 6: the same store keeps per-engine last-used
generation parameters (engine_params).
"""

from __future__ import annotations

import httpx


def test_ui_prefs_default(client: httpx.Client):
    r = client.get("/settings/ui")
    assert r.status_code == 200
    assert r.json() == {"theme": "system", "engine_id": None, "engine_params": {}}


def test_ui_prefs_round_trip(client: httpx.Client):
    r = client.put("/settings/ui", json={"theme": "light", "engine_id": "fake"})
    assert r.status_code == 200
    assert r.json() == {"theme": "light", "engine_id": "fake", "engine_params": {}}

    r = client.get("/settings/ui")
    assert r.json() == {"theme": "light", "engine_id": "fake", "engine_params": {}}

    # Partial updates merge; theme survives an engine-only write.
    r = client.put("/settings/ui", json={"engine_id": None})
    assert r.status_code == 200
    assert r.json() == {"theme": "light", "engine_id": None, "engine_params": {}}


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


def test_engine_params_round_trip(client: httpx.Client):
    # Issue #23: generation parameters are remembered per engine and ride
    # the same library-scoped store (backup/restore covers them).
    r = client.put(
        "/settings/ui",
        json={"engine_params": {"fake": {"fake_mode": "fast"}, "bad-key": "oops"}},
    )
    assert r.status_code == 200
    assert r.json()["engine_params"] == {"fake": {"fake_mode": "fast"}}

    # Merge: a later engine-only write keeps the stored params.
    r = client.put("/settings/ui", json={"engine_id": "fake"})
    assert r.json()["engine_params"] == {"fake": {"fake_mode": "fast"}}

    # Values are normalized to strings; non-dict payloads are rejected.
    r = client.put("/settings/ui", json={"engine_params": []})
    assert r.status_code == 422
