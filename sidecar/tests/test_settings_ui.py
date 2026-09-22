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
        # Issue #44: software-level display & log behavior settings.
        "ui_scale": 1.0,
        "font_size": None,
        "log_buffer": 500,
        # Issue #65: updater preferences live in the same settings store.
        "updater_channel_mode": "auto",
        "updater_mirror_prefix": "https://gh-proxy.com/",
        "updater_skipped_tag": None,
    }


def test_ui_prefs_updater_round_trip(client: httpx.Client):
    # Issue #65: 更新渠道 / 镜像前缀 / 跳过此版本 persist with the library.
    r = client.put(
        "/settings/ui",
        json={
            "updater_channel_mode": "mirror",
            "updater_mirror_prefix": "https://my-proxy.example/",
            "updater_skipped_tag": "v0.2.0",
        },
    )
    assert r.status_code == 200
    assert r.json()["updater_channel_mode"] == "mirror"
    assert r.json()["updater_mirror_prefix"] == "https://my-proxy.example/"
    assert r.json()["updater_skipped_tag"] == "v0.2.0"

    # Corrupt/invalid values are rejected or degrade, never stored raw.
    r = client.put("/settings/ui", json={"updater_channel_mode": "nightly"})
    assert r.status_code == 422
    r = client.put("/settings/ui", json={"updater_mirror_prefix": 42})
    assert r.status_code == 422
    r = client.put("/settings/ui", json={"updater_skipped_tag": ""})
    assert r.status_code == 422


def test_ui_prefs_scale_font_log_round_trip(client: httpx.Client):
    # Issue #44: UI scale, font-size override and log buffer count ride the
    # same /settings/ui store; out-of-range values are clamped, not rejected.
    r = client.put(
        "/settings/ui", json={"ui_scale": 1.25, "font_size": 16, "log_buffer": 2000}
    )
    assert r.status_code == 200
    assert r.json()["ui_scale"] == 1.25
    assert r.json()["font_size"] == 16
    assert r.json()["log_buffer"] == 2000

    r = client.put(
        "/settings/ui",
        json={"ui_scale": 99, "font_size": 99, "log_buffer": 999999},
    )
    assert r.json()["ui_scale"] == 1.5
    assert r.json()["font_size"] == 24
    assert r.json()["log_buffer"] == 5000

    r = client.put("/settings/ui", json={"ui_scale": 0.1, "font_size": 1, "log_buffer": 1})
    assert r.json()["ui_scale"] == 0.85
    assert r.json()["font_size"] == 11
    assert r.json()["log_buffer"] == 100

    # font_size null clears the override; non-numeric values are rejected.
    assert client.put("/settings/ui", json={"font_size": None}).json()["font_size"] is None
    assert client.put("/settings/ui", json={"ui_scale": "big"}).status_code == 422
    assert client.put("/settings/ui", json={"font_size": "big"}).status_code == 422
    assert client.put("/settings/ui", json={"log_buffer": 1.5}).status_code == 422


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
