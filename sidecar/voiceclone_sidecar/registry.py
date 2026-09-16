"""Engine protocol and registry.

The registry is the ONLY extension point: adding an engine must not require
touching any generation logic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from .capabilities import Capabilities


@dataclass(frozen=True)
class GenerationRequest:
    generation_id: str
    text: str
    params: dict


@dataclass(frozen=True)
class GenerationResult:
    """Path is the engine-written audio file; the sidecar serves it."""

    audio_path: str
    sample_rate: int


class Engine(abc.ABC):
    """A synthesizing backend: a cloud API or a local model."""

    engine_id: str
    display_name: str

    @abc.abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abc.abstractmethod
    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        """Run one text -> audio generation. Use ``log(msg)`` to emit
        progress lines; they are streamed to the UI over WebSocket."""


class InstallableEngine(Engine):
    """An engine whose environment/weights must be installed before use.

    Implement this (in addition to Engine) for engines backed by a bundled
    runtime: the sidecar exposes GET /engines/{id}/status and
    POST /engines/{id}/install from exactly this surface.
    """

    @abc.abstractmethod
    def is_installed(self) -> bool: ...

    @abc.abstractmethod
    def install_state(self) -> dict:
        """{"installed": bool, "steps": {step_id: {status, error, ...}}}"""

    @abc.abstractmethod
    def install(self, log, progress=None) -> dict: ...


class Registry:
    """Real engine registry. Engines register here; nothing else knows them."""

    def __init__(self) -> None:
        self._engines: dict[str, Engine] = {}

    def register(self, engine: Engine) -> None:
        if engine.engine_id in self._engines:
            raise ValueError(f"engine already registered: {engine.engine_id}")
        self._engines[engine.engine_id] = engine

    def get(self, engine_id: str) -> Engine | None:
        return self._engines.get(engine_id)

    def list(self) -> list[Engine]:
        return list(self._engines.values())


def default_registry(output_dir=None) -> Registry:
    """Registry with the engines shipped in this package.

    The fake engine is a built-in: it runs the exact same code path as real
    engines, keeps the app usable without models/keys, and is the only
    backend the contract tests exercise.

    Real local engines register only where they can run (platform +
    architecture); a cloud-API engine would always be registered.
    """
    from .engines.fake import FakeEngine

    registry = Registry()
    registry.register(FakeEngine(output_dir=output_dir))

    import os
    import sys

    if sys.platform == "darwin" and os.uname().machine == "arm64":
        from .engines.qwen3_tts import Qwen3TtsMlxEngine

        registry.register(Qwen3TtsMlxEngine(output_dir=output_dir))
    return registry
