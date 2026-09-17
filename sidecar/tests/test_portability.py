"""Contract tests for portability (issue #14): voice packages + whole-library
backup/restore. Both are manual actions over the self-contained data dir;
nothing here is automatic or cloud-synced (ADR-0005).
"""

from __future__ import annotations

import io
import json
import struct
import wave
import zipfile

import httpx


def wav_bytes(seconds: float = 5.0, rate: int = 16000) -> bytes:
    frames = int(rate * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(8000 * (i % 100) / 100)) for i in range(frames)
        ))
    return buf.getvalue()


def create_voice(client: httpx.Client, name: str = "测试音色", **kwargs) -> dict:
    files = {"file": ("ref.wav", wav_bytes(kwargs.pop("seconds", 5.0)), "audio/wav")}
    data = {"name": name, "description": "一段说明"}
    avatar = kwargs.pop("avatar", None)
    if avatar is not None:
        files["avatar"] = ("avatar.png", avatar, "image/png")
    r = client.post("/voices", data=data, files=files)
    assert r.status_code == 200, r.text
    return r.json()


def zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


# --- voice package -----------------------------------------------------------


def test_export_voice_package_contains_record_and_files(client, sidecar):
    png = b"\x89PNG\r\n\x1a\nfakepng"
    voice = create_voice(client, avatar=png)
    r = client.get(f"/voices/{voice['id']}/export")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["kind"] == "voice-clone-package"
        assert manifest["version"] == 1
        exported = manifest["voice"]
        assert exported["name"] == voice["name"]
        assert exported["reference"]["sha256"] == voice["reference"]["sha256"]
        # Bindings travel as information only; the importer drops them.
        assert exported["bindings"] == voice["bindings"]
        assert zf.read(f"files/{voice['reference']['filename']}") == wav_bytes()
        assert zf.read(f"files/{voice['avatar']['filename']}") == png


def test_import_voice_package_roundtrip_drops_bindings(client):
    voice = create_voice(client)
    # Bind it somewhere first: the package must not carry this over.
    r = client.post(f"/voices/{voice['id']}/bindings/fake")
    assert r.status_code == 200, r.text
    pkg = client.get(f"/voices/{voice['id']}/export").content

    r = client.post("/voices/import", files={"file": ("v.zip", pkg, "application/zip")})
    assert r.status_code == 200, r.text
    imported = r.json()["voice"]
    assert imported["id"] != voice["id"]  # fresh id, no collision
    assert imported["name"] == voice["name"]
    assert imported["description"] == voice["description"]
    assert imported["origin"] == voice["origin"]
    assert imported["bindings"] == {}  # rebuilt on demand, never imported
    ref = imported["reference"]
    assert ref["sha256"] == voice["reference"]["sha256"]
    assert ref["filename"] == voice["reference"]["filename"]

    # The imported reference file actually exists and is byte-identical.
    original = client.get(f"/voices/{voice['id']}/reference").content
    copy = client.get(f"/voices/{imported['id']}/reference").content
    assert copy == original


def test_import_rejects_corrupt_package(client):
    r = client.post(
        "/voices/import", files={"file": ("v.zip", b"not a zip", "application/zip")}
    )
    assert r.status_code == 422
    r = client.post("/voices/import", files={"file": ("v.zip", b"PK\x03\x04garbage", "application/zip")})
    assert r.status_code == 422


def test_import_rejects_tampered_reference(client, sidecar):
    voice = create_voice(client)
    pkg = client.get(f"/voices/{voice['id']}/export").content
    # Flip a byte inside the packaged reference audio: checksum must fail.
    with zipfile.ZipFile(io.BytesIO(pkg)) as zin, io.BytesIO() as buf:
        ref_name = next(n for n in zin.namelist() if n.startswith("files/"))
        with zipfile.ZipFile(buf, "w") as zout:
            for info in zin.infolist():
                data = bytearray(zin.read(info.filename))
                if info.filename == ref_name:
                    data[len(data) // 2] ^= 0xFF
                zout.writestr(info.filename, bytes(data))
        tampered = buf.getvalue()
    r = client.post(
        "/voices/import", files={"file": ("v.zip", bytes(tampered), "application/zip")}
    )
    assert r.status_code == 422
    assert "校验" in r.json()["detail"]


def test_import_package_missing_reference_file_is_rejected(client):
    wav = wav_bytes()
    import hashlib
    manifest = {
        "kind": "voice-clone-package",
        "version": 1,
        "created_at": "2026-01-01T00:00:00Z",
        "voice": {
            "id": "x", "name": "导入", "description": "", "origin": "cloned",
            "reference": {"filename": "ref.wav", "sha256": hashlib.sha256(wav).hexdigest()},
            "avatar": None, "bindings": {},
        },
    }
    pkg = zip_bytes({"manifest.json": json.dumps(manifest).encode()})  # no files/
    r = client.post("/voices/import", files={"file": ("v.zip", pkg, "application/zip")})
    assert r.status_code == 422
    assert "缺少参考音频" in r.json()["detail"]


# --- library backup / restore -------------------------------------------------


def test_backup_restore_roundtrip_restores_voices_history_and_settings(client, sidecar):
    voice = create_voice(client)
    r = client.post("/generations", json={"engine_id": "fake", "text": "备份历史测试"})
    assert r.status_code == 200, r.text
    generation_total = client.get("/generations").json()["total"]
    assert generation_total >= 1

    backup = client.post("/backup")
    assert backup.status_code == 200, backup.text
    assert backup.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(backup.content)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["kind"] == "voice-clone-library-backup"
        names = zf.namelist()
        assert "voices.json" in names
        assert "generations.json" in names
        assert any(n.endswith(".wav") for n in names)  # audio artifacts travel with the library

    # Wipe the library: delete the voice and its generation history.
    assert client.delete(f"/voices/{voice['id']}").status_code == 200
    assert voice["id"] not in {v["id"] for v in client.get("/voices").json()["voices"]}
    for record in client.get("/generations", params={"limit": 100}).json()["records"]:
        client.delete(f"/generations/{record['id']}")
    assert client.get("/generations").json()["total"] == 0

    # Restore: every durable store must reload from the swapped-in data dir.
    r = client.post(
        "/restore", files={"file": ("backup.zip", backup.content, "application/zip")}
    )
    assert r.status_code == 200, r.text
    assert r.json()["restored"] is True
    assert r.json()["generations"] == generation_total

    restored_voices = {v["id"]: v for v in client.get("/voices").json()["voices"]}
    assert restored_voices[voice["id"]]["name"] == voice["name"]

    ref = client.get(f"/voices/{voice['id']}/reference")
    assert ref.status_code == 200
    assert client.get("/generations").json()["total"] == generation_total


def test_restore_rejects_wrong_kind_and_corrupt_zip(client):
    manifest = json.dumps({"kind": "voice-clone-package", "version": 1}).encode()
    r = client.post(
        "/restore",
        files={"file": ("x.zip", zip_bytes({"manifest.json": manifest}), "application/zip")},
    )
    assert r.status_code == 422
    r = client.post("/restore", files={"file": ("x.zip", b"PKjunk", "application/zip")})
    assert r.status_code == 422


def test_restore_rejects_unsupported_format_version(client):
    manifest = json.dumps({"kind": "voice-clone-library-backup", "version": 99}).encode()
    r = client.post(
        "/restore",
        files={"file": ("x.zip", zip_bytes({"manifest.json": manifest}), "application/zip")},
    )
    assert r.status_code == 422


def test_data_dir_is_self_contained(client, sidecar):
    """Every byte the app needs lives under the data root: indexes, audio
    artifacts and voice files. Nothing points outside."""
    voice = create_voice(client)
    data_dir = sidecar["data_dir"]
    # Indexes are created eagerly by their stores; compare/settings only
    # once first used, so only require the always-present ones.
    for name in ("voices.json", "generations.json"):
        assert (data_dir / name).is_file(), f"{name} missing from data dir"
    assert (data_dir / "voices" / voice["id"]).is_dir()
