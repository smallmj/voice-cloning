"""UI preference persistence (issue #21): theme + engine selection.

Issue #23 / ADR-0018 decision 6: the same store keeps per-engine last-used
generation parameters (engine_params).
"""

from __future__ import annotations

import httpx


def test_ui_prefs_default(client: httpx.Client):
    r = client.get("/settings/ui")
    assert r.status_code == 200
    assert r.json() == {
        "theme": "system",
        "engine_id": None,
        "engine_params": {},
        # Issue #36: shared right sidebar state (open by default, 360px wide).
        "sidebar_open": True,
        "sidebar_width": 360,
    }


def test_ui_prefs_sidebar_round_trip(client: httpx.Client):
    # Issue #36: the sidebar's visibility and width survive a restart —
    # they ride the same /settings/ui store as theme/engine selection.
    r = client.put("/settings/ui", json={"sidebar_open": False, "sidebar_width": 480})
    assert r.status_code == 200
    assert r.json()["sidebar_open"] is False
    assert r.json()["sidebar_width"] == 480

    # Widths outside the draggable range are clamped, not rejected.
    r = client.put("/settings/ui", json={"sidebar_width": 10})
    assert r.json()["sidebar_width"] == 260
    r = client.put("/settings/ui", json={"sidebar_width": 99999})
    assert r.json()["sidebar_width"] == 640

    # Non-bool open / non-numeric width are rejected.
    assert client.put("/settings/ui", json={"sidebar_open": "yes"}).status_code == 422
    assert client.put("/settings/ui", json={"sidebar_width": "wide"}).status_code == 422


def test_ui_prefs_round_trip(client: httpx.Client):
    r = client.put("/settings/ui", json={"theme": "light", "engine_id": "fake"})
    assert r.status_code == 200
    assert r.json()["theme"] == "light"
    assert r.json()["engine_id"] == "fake"
    assert isinstance(r.json()["sidebar_open"], bool)

    r = client.get("/settings/ui")
    assert r.json()["theme"] == "light"
    assert r.json()["engine_id"] == "fake"
    assert isinstance(r.json()["sidebar_open"], bool)

    # Partial updates merge; theme survives an engine-only write.
    r = client.put("/settings/ui", json={"engine_id": None})
    assert r.status_code == 200
    assert r.json()["theme"] == "light"
    assert r.json()["engine_id"] is None


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
