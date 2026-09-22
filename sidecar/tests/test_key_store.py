"""Unit tests for the BYOK key store (issue #9, ADR-0003)."""

from __future__ import annotations

import pytest

from voiceclone_sidecar.key_store import (
    KeyStore,
    KeyStoreError,
    MemoryBackend,
)


@pytest.fixture()
def store() -> KeyStore:
    return KeyStore(backend=MemoryBackend())


def test_set_get_roundtrip(store):
    store.set("cloud-a", " sk-secret-123 ")
    # Values are stripped; the exact value is readable only by engines.
    assert store.get("cloud-a") == "sk-secret-123"


def test_unset_engine_reads_none(store):
    assert store.get("cloud-a") is None


def test_delete(store):
    store.set("cloud-a", "k")
    store.delete("cloud-a")
    assert store.get("cloud-a") is None
    # Deleting a missing entry is a no-op.
    store.delete("cloud-a")


def test_empty_key_is_rejected(store):
    with pytest.raises(KeyStoreError):
        store.set("cloud-a", "   ")


def test_overlong_key_is_rejected(store):
    with pytest.raises(KeyStoreError):
        store.set("cloud-a", "x" * 5000)


def test_status_reports_flags_never_values(store):
    store.set("cloud-a", "the-actual-secret")
    status = store.status(["cloud-a", "cloud-b"])
    assert status == [
        {"engine_id": "cloud-a", "configured": True},
        {"engine_id": "cloud-b", "configured": False},
    ]
    assert "the-actual-secret" not in str(status)


def test_memory_backend_is_isolated_per_instance():
    a, b = MemoryBackend(), MemoryBackend()
    a.set("svc", "acc", "value")
    assert b.get("svc", "acc") is None
