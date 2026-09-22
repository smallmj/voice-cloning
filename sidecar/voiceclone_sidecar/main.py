"""Sidecar entry: FastAPI app factory + server bootstrap (issue #20).

``create_app`` used to be a single 1700-line closure (88 nested defs, 52
handlers) — no handler could be imported, so every backend fix landed in
one file and nothing was unit-testable. The routes now live in
:mod:`voiceclone_sidecar.routers` (one ``APIRouter`` per domain) over a
shared :class:`voiceclone_sidecar.context.AppContext`, and the synthesis
pipeline is importable at :mod:`voiceclone_sidecar.pipeline`. This module
is the composition root: app factory, middleware, and the ``main()`` server
bootstrap.

The sidecar binds 127.0.0.1 on a dynamic port (port 0) and enforces bearer
auth on every route — the sole exception being the media routes (audio
files, voice reference/avatar images, the log WebSocket), which also accept
a short-TTL, media-scoped query token so URL-bound media elements never
need the raw sidecar token (issue #19). Once listening it prints a single
JSON handshake line on stdout, e.g.:

    {"event": "ready", "port": 54321, "pid": 1234}

The desktop shell reads that line to learn where to connect.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
from pathlib import Path

from fastapi import FastAPI

from ._version import SIDECAR_VERSION
from .context import build_context
from .key_store import KeyStore
from .registry import Registry, default_registry
from .routers import install_routers
from .runtime import paths


def create_app(
    registry: Registry,
    token: str,
    audio_dir: Path,
    data_dir: Path | None = None,
    runtime_root: Path | None = None,
    key_store: KeyStore | None = None,
    rate_limit_attempts: int = 4,
    rate_limit_backoff: float = 1.0,
) -> FastAPI:
    from fastapi.middleware.cors import CORSMiddleware

    # Issue #19: the default docs routes (/openapi.json, /docs, /redoc) carry
    # no auth dependency, so they would publish the full API surface to any
    # local process or web page. There is no human-facing API browser here —
    # they stay off entirely.
    app = FastAPI(
        title="voiceclone-sidecar",
        version=SIDECAR_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    # Renderer runs on a dev origin in development; the sidecar is a local
    # loopback service guarded by bearer auth, so origins stay permissive.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ctx = build_context(
        registry=registry,
        token=token,
        audio_dir=audio_dir,
        data_dir=data_dir,
        runtime_root=runtime_root,
        key_store=key_store,
        rate_limit_attempts=rate_limit_attempts,
        rate_limit_backoff=rate_limit_backoff,
    )
    install_routers(app, ctx)
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="voiceclone-sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = dynamic port")
    parser.add_argument("--token", default=os.environ.get("SIDECAR_TOKEN") or secrets.token_urlsafe(32))
    parser.add_argument("--audio-dir", default=os.environ.get("SIDECAR_AUDIO_DIR") or "data/audio")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SIDECAR_DATA_DIR") or None,
        help="voice library root; defaults to the audio dir's parent",
    )
    args = parser.parse_args(argv)
    audio_dir = Path(args.audio_dir)
    data_dir = Path(args.data_dir) if args.data_dir else None

    import uvicorn

    from .engine_config import load_full_settings

    app = create_app(
        # ADR-0015: the registry merges the process environment's
        # VOICECLONE_* / UV_* variables over the built-in defaults, plus any
        # per-engine overrides persisted in the settings storage, and injects
        # the result into every engine — so documented env vars and settings
        # both actually reach the engines instead of dying at module
        # constants. ADR-0016: the same store's global download-source
        # preferences (sources block) ride along on the same seam.
        default_registry(
            output_dir=audio_dir,
            key_store=KeyStore(),
            settings=load_full_settings(data_dir or audio_dir.parent),
        ),
        token=args.token,
        audio_dir=audio_dir,
        data_dir=data_dir,
        runtime_root=paths.runtime_root(),
    )

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)

    async def serve_and_announce() -> None:
        serve_task = asyncio.create_task(server.serve())
        # Announce only once the socket is actually listening — the shell
        # connects right after reading this line.
        while not server.started:
            if serve_task.done():
                break
            await asyncio.sleep(0.01)
        if server.started:
            port = None
            for s in server.servers:
                if s.sockets:
                    port = s.sockets[0].getsockname()[1]
                    break
            print(json.dumps({"event": "ready", "port": port, "pid": os.getpid()}), flush=True)
            sys.stdout.flush()
        await serve_task

    asyncio.run(serve_and_announce())


if __name__ == "__main__":
    main()
