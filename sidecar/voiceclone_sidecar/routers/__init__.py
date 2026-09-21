"""Per-domain APIRouters extracted from the former ``create_app`` closure
(issue #20). Each module exposes a ``build_router(ctx)`` factory over the
shared :class:`~voiceclone_sidecar.context.AppContext`."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI

from ..context import AppContext
from . import (
    compare,
    consent,
    diagnostics,
    engines,
    generations,
    jobs,
    meta,
    portability,
    regression,
    settings,
    voice_design,
    voices,
)


def install_routers(app: FastAPI, ctx: AppContext) -> None:
    """Register every router in the same order the routes used to be
    declared inside ``create_app``."""
    modules = (
        meta,
        consent,
        engines,
        voices,
        portability,
        diagnostics,
        voice_design,
        generations,
        jobs,
        compare,
        regression,
        settings,
    )
    for module in modules:
        router: APIRouter = module.build_router(ctx)
        app.include_router(router)


__all__ = ["install_routers"]
