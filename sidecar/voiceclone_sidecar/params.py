"""Canonical (规范层) parameter definitions and per-engine adapters.

ADR-0018 decision 1: the canonical layer is DELIBERATELY thin — only
concepts whose value semantics are genuinely identical across engines get a
canonical slot, and every canonical parameter must ship with a per-engine
adapter (``to_wire``) that maps what the user sees to what the engine
receives. Research showed forcing the layer wide produces silently wrong
audio (IndexTTS ``duration_factor`` is a duration multiplier — the INVERSE
of speed; pitch has four incompatible units; …).

The engines keep the adapters honest: a canonical param whose adapter is
missing is a construction-time error (see ParamSpec.__post_init__).
"""

from __future__ import annotations

from .capabilities import AppliesTo, ParamSpec

# ---------------------------------------------------------------------------
# speed — the one canonical control where the mapping is famously non-obvious.
# IndexTTS takes `duration_factor` (0.5–2.0, >1 = SLOWER); everything else
# that speaks speed takes a rate (>1 = faster). The adapter is the inverse.
# ---------------------------------------------------------------------------


def speed_to_duration_factor(value):
    """User speed (rate, >1 = faster) -> IndexTTS duration_factor (>1 = slower)."""
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if rate <= 0:
        return None
    return round(1.0 / rate, 4)


# Kept as the canonical adapter under its historical name too.
_speed_to_duration_factor = speed_to_duration_factor


def canonical_speed(engine_id: str, model: str, help: str = "") -> ParamSpec:
    """Canonical speed control.

    For IndexTTS-family engines pass this factory and wire the mapped value
    into the worker payload as ``duration_factor``; the same factory cannot
    be reused verbatim by rate-native engines, which override ``to_wire``
    (identity) — the mapping is per-engine BY DESIGN.
    """
    return ParamSpec(
        name="speed",
        label="语速",
        kind="number",
        default=1.0,
        min=0.5,
        max=2.0,
        step=0.05,
        unit="x",
        layer="canonical",
        help=help or "1.0 为原速；>1 更快，<1 更慢。",
        applies_to=AppliesTo(engine=engine_id, model=model, mode="cloning"),
        to_wire=_speed_to_duration_factor,
    )


def canonical_language(engine_id: str, model: str, choices, wire_name: str,
                       default=None, mode: str = "cloning") -> ParamSpec:
    """Canonical language control mapped onto the engine's own wire key.

    Six engines use six different language schemes and one has no parameter
    at all — so the canonical slot only exists where the engine actually
    has one, and ``to_wire`` carries the per-engine value mapping.
    """
    def to_wire(value):
        if value is None or value == "":
            return None
        return value if value in choices else None

    return ParamSpec(
        name="language",
        label="语种",
        kind="select",
        default=default if default in choices else choices[0],
        choices=tuple(choices),
        layer="canonical",
        wire_path=wire_name,
        help="发音语言。",
        applies_to=AppliesTo(engine=engine_id, model=model, mode=mode),
        to_wire=to_wire,
    )


def canonical_pronunciation(engine_id: str, model: str, grammar_help: str) -> ParamSpec:
    """Canonical pronunciation-annotation entry (issue #23).

    The user types the canonical ``汉字=拼音`` block; the engine adapter
    rewrites the text into the engine's own grammar before synthesis. The
    three grammars are mutually incompatible — this canonical entry is the
    first real use case proving the canonical+adapter shape works.
    """
    return ParamSpec(
        name="pronunciation",
        label="发音标注",
        kind="textarea",
        max_length=2000,
        layer="canonical",
        help=grammar_help,
        applies_to=AppliesTo(engine=engine_id, model=model, mode="cloning"),
        to_wire=lambda value: value or "",
    )
