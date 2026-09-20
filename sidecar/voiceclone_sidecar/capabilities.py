"""Engine capability declaration.

Every engine must declare what it supports. A missing declaration is treated
as "not supported" — never as "silently ignore the parameter".

ADR-0018: ``ParamSpec`` must be able to express the REAL parameter space of
an engine, and "not supported" must be DATA on the spec itself — not an
absence a reviewer has to remember. The UI renders exactly the exposed
specs for the selected engine; parameters a spec does not declare are never
shown (issue #9), and specs marked ``exposed=False`` are hidden/disabled
WITH their ``not_exposed_reason`` instead of silently dropped (issue #23).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# ADR-0018 decision 3: the closed vocabulary of "why is this parameter not
# exposed". Anything outside this set means the declaration is lying.
NOT_EXPOSED_REASONS = (
    "no-op",  # accepted (or merely printed) by the engine but never used
    "server-injected",  # the sidecar/vendor injects it: ref_audio, voice_id, model
    "wrong-mode",  # only valid in a mode this engine is not serving here
    "paid-tier",  # requires a plan/tier the BYOK user may not have
    "breaks-pipeline",  # would break this app's pipeline (WAV format, normalization, …)
    "unverified",  # exists upstream but unverified on our pinned version — not exposed until proven
)

PARAM_KINDS = (
    "select",  # one of `choices`
    "text",  # short single-line text
    "textarea",  # long multi-line text, optional `max_length`
    "number",  # min/max/step, `integer` distinguishes int/float
    "bool",  # checkbox
    "output",  # read-only display of an engine-produced value (e.g. CoT gen_text)
)


@dataclass(frozen=True)
class AppliesTo:
    """What the declaration was verified against (ADR-0018 decision 3):
    the engine id, the model/version string, and the serving mode."""

    engine: str | None = None
    model: str | None = None
    mode: str | None = None

    def to_dict(self) -> dict:
        return {"engine": self.engine, "model": self.model, "mode": self.mode}


@dataclass(frozen=True)
class ObjectField:
    """One field inside an object-array item (fixed order, ADR-0018):
    e.g. MiniMax ``timbre_weights[{voice_id, weight}]``."""

    name: str
    label: str
    kind: str = "number"
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "min": self.min,
            "max": self.max,
            "choices": list(self.choices),
        }


@dataclass(frozen=True)
class ParamSpec:
    """One user-facing parameter an engine accepts.

    ``layer`` splits the two UI sections (ADR-0018 decision 1):
      - ``"canonical"`` — cross-engine concepts with stable value semantics
        (speed, language, pronunciation). Every canonical spec MUST be paired
        with a per-engine adapter (unit conversion / inverse mapping) wired
        into ``synthesize``; the canonical value is what the user sees, the
        wire value is what the engine receives, and the two may differ
        (``to_wire``).
      - ``"engine"`` — engine-specific parameters with no cross-engine
        meaning. Never invent a canonical slot for them.

    The UI renders EXACTLY the exposed specs of the selected engine. Specs
    with ``exposed=False`` carry ``not_exposed_reason`` (data, not habit):
    the UI hides or disables them and states the reason.
    """

    name: str
    label: str
    kind: str  # one of PARAM_KINDS
    default: str | int | float | bool | None = None
    choices: tuple[str, ...] = ()
    help: str = ""
    # Layering / grouping.
    layer: str = "engine"  # "canonical" | "engine"
    group: str = ""  # nested group label, e.g. "voice_setting" (a wire path)
    wire_path: str = ""  # where the mapped value lands in the engine payload; "" = top level
    # Number presentation (ADR-0018 decision 2).
    unit: str = ""  # "x" | "s" | "ms" | "%" | "Hz" | "kbps" | "semitone" | …
    min: float | None = None
    max: float | None = None
    step: float | None = None
    integer: bool = False  # True: integral values only (distinct from float)
    min_open: bool = False  # open interval bound: value must be > min, not >=
    max_open: bool = False  # e.g. MiniMax vol is (0, 10]
    max_from: str | None = None  # dynamic upper bound, resolved at runtime (e.g. checkpoint)
    max_length: int | None = None  # textarea/text character cap
    # Object arrays with fixed order (emo_vector[8], timbre_weights ≤4, …).
    items: tuple[ObjectField, ...] = ()
    max_items: int | None = None
    # "Not supported is data" (ADR-0018 decision 3).
    exposed: bool = True
    not_exposed_reason: str | None = None  # one of NOT_EXPOSED_REASONS
    applies_to: AppliesTo | None = None
    ignored_when: tuple[str, ...] = ()  # predicates under which the engine ignores the value
    # User value -> wire payload mapping ("用户看到的 ≠ 引擎收到的").
    to_wire: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"ParamSpec {self.name}: unknown kind {self.kind!r}")
        if not self.exposed and self.not_exposed_reason not in NOT_EXPOSED_REASONS:
            raise ValueError(
                f"ParamSpec {self.name}: exposed=False requires not_exposed_reason "
                f"in {NOT_EXPOSED_REASONS}"
            )
        if self.exposed and self.applies_to is None:
            # ADR-0018 decision 3/5: an exposed parameter must state what it
            # was verified against — otherwise the UI is making a claim no
            # evidence backs.
            raise ValueError(f"ParamSpec {self.name}: exposed specs require applies_to")
        if self.layer not in ("canonical", "engine"):
            raise ValueError(f"ParamSpec {self.name}: layer must be canonical|engine")
        if self.layer == "canonical" and self.to_wire is None:
            # Decision 1: every canonical parameter must ship with its
            # per-engine adapter; without a mapping the raw user value would
            # reach the engine unconverted (the silent-wrong-audio failure).
            raise ValueError(
                f"ParamSpec {self.name}: canonical specs require a to_wire adapter"
            )

    def to_dict(self) -> dict:
        """JSON-safe projection. ``to_wire`` is code, not data; the UI only
        needs to know a mapping exists (``wire_map=True``)."""
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "choices": list(self.choices),
            "help": self.help,
            "layer": self.layer,
            "group": self.group,
            "wire_path": self.wire_path,
            "unit": self.unit,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "integer": self.integer,
            "min_open": self.min_open,
            "max_open": self.max_open,
            "max_from": self.max_from,
            "max_length": self.max_length,
            "items": [i.to_dict() for i in self.items],
            "max_items": self.max_items,
            "exposed": self.exposed,
            "not_exposed_reason": self.not_exposed_reason,
            "applies_to": self.applies_to.to_dict() if self.applies_to else None,
            "ignored_when": list(self.ignored_when),
            "wire_map": self.to_wire is not None,
        }


@dataclass(frozen=True)
class Capabilities:
    """What an engine supports. Every field defaults to "not supported"."""

    languages: tuple[str, ...] = ()
    voice_cloning: bool = False
    voice_design: bool = False
    pronunciation_control: bool = False
    emotion: bool = False
    commercial_license: bool = False
    cross_device_use: bool = False
    upload_used_for_training: bool = False
    api_closed_loop: bool = False
    # Engines whose cloning quality/behavior depends on the reference text
    # declare it; the sidecar then auto-fills ref_text from the voice's
    # transcript (transcribing on demand) instead of making the user type it.
    requires_reference_text: bool = False
    # Over-length text is auto-segmented at this many characters per request
    # (issue #13). None means the engine declares no limit and long text is
    # sent as one request — segmentation never silently rewrites behavior.
    max_chars_per_request: int | None = None

    def to_dict(self) -> dict:
        return {
            "languages": list(self.languages),
            "voice_cloning": self.voice_cloning,
            "voice_design": self.voice_design,
            "pronunciation_control": self.pronunciation_control,
            "emotion": self.emotion,
            "commercial_license": self.commercial_license,
            "cross_device_use": self.cross_device_use,
            "upload_used_for_training": self.upload_used_for_training,
            "api_closed_loop": self.api_closed_loop,
            "requires_reference_text": self.requires_reference_text,
            "max_chars_per_request": self.max_chars_per_request,
        }
