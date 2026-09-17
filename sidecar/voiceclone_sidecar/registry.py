"""Engine protocol and registry.

The registry is the ONLY extension point: adding an engine must not require
touching any generation logic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from .capabilities import Capabilities, ParamSpec


@dataclass(frozen=True)
class GenerationRequest:
    generation_id: str
    text: str
    params: dict


@dataclass(frozen=True)
class GenerationResult:
    """Path is the engine-written audio file; the sidecar serves it.

    ``model_version`` and ``cost`` are lineage facts the engine itself knows
    best; engines that cannot report them leave the defaults and the record
    stores null (the contract keeps the keys present either way).
    """

    audio_path: str
    sample_rate: int
    model_version: str | None = None
    cost: float | None = None


class Engine(abc.ABC):
    """A synthesizing backend: a cloud API or a local model."""

    engine_id: str
    display_name: str
    # BYOK engines need a per-user API key from the OS key store (ADR-0003);
    # the sidecar refuses generation while the key is not configured.
    requires_key: bool = False
    # Billing and data-usage disclosure shown verbatim in the settings page.
    # The billing note must state the practical Chinese unit price (issue #9).
    billing_note: str | None = None
    data_usage_note: str | None = None

    @abc.abstractmethod
    def capabilities(self) -> Capabilities: ...

    def param_specs(self) -> list[ParamSpec]:
        """The full, closed list of user-facing parameters this engine
        accepts. The UI renders exactly this list — nothing else."""
        return []

    @abc.abstractmethod
    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        """Run one text -> audio generation. Use ``log(msg)`` to emit
        progress lines; they are streamed to the UI over WebSocket."""

    def bind_reference(self, ref_path, ref_text: str | None, log) -> dict:
        """Prepare the reference on this engine and return binding extras.

        Local engines hand the file over at generation time and don't need
        this; cloud engines enroll the sample and return e.g. a voice_id.
        Failures surface the vendor's error message to the user.
        """
        raise NotImplementedError(
            f"engine {self.engine_id} does not support explicit reference binding"
        )


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


def default_registry(output_dir=None, key_store=None) -> Registry:
    """Registry with the engines shipped in this package.

    The fake engine is a built-in: it runs the exact same code path as real
    engines, keeps the app usable without models/keys, and is the only
    backend the contract tests exercise.

    Real local engines register only where they can run (platform +
    architecture); cloud-API engines are always registered — they are usable
    the moment the user brings their own API key (ADR-0003).
    """
    from .engines.fake import FakeEngine

    registry = Registry()
    registry.register(FakeEngine(output_dir=output_dir))

    import os
    import sys

    if key_store is None:
        from .secrets import KeyStore

        key_store = KeyStore()

    from .engines.qwen_tts_cloud import Qwen3TtsVcCloudEngine

    registry.register(Qwen3TtsVcCloudEngine(output_dir=output_dir, key_store=key_store))

    if sys.platform == "darwin" and os.uname().machine == "arm64":
        from .engines.qwen3_tts import Qwen3TtsMlxEngine

        registry.register(Qwen3TtsMlxEngine(output_dir=output_dir))

    if sys.platform == "win32":
        from .engines.indextts25 import IndexTts25CudaEngine

        registry.register(IndexTts25CudaEngine(output_dir=output_dir))

    if os.environ.get("VOICECLONE_TEST_ENGINES") == "1":
        # Contract-test seam only: exercises the reference-text auto-fill
        # path without any real ASR/synthesis backend.
        from .engines.fake import FakeRefTextEngine

        registry.register(FakeRefTextEngine(output_dir=output_dir))

        # Contract-test seam: a BYOK cloud-shaped engine for the issue-#9
        # surface (keys, billing disclosure, cloud bindings, voice_id).
        from .engines.fake import FakeKeyEngine

        registry.register(FakeKeyEngine(output_dir=output_dir))

        # Contract-test seams (issue #10): two output-level extremes. The
        # comparison feature must bring both to the same LUFS.
        from .engines.fake import FakeLoudEngine, FakeQuietEngine

        registry.register(FakeLoudEngine(output_dir=output_dir))
        registry.register(FakeQuietEngine(output_dir=output_dir))
    return registry
