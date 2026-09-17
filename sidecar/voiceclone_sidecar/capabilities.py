"""Engine capability declaration.

Every engine must declare what it supports. A missing declaration is treated
as "not supported" — never as "silently ignore the parameter".
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParamSpec:
    """One user-facing parameter an engine accepts.

    The UI renders EXACTLY this list for the selected engine — parameters a
    spec does not declare are never shown (issue #9: unsupported parameters
    are not rendered, not silently ignored)."""

    name: str
    label: str
    kind: str  # "select" | "text" | "number"
    default: str | int | float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "choices": list(self.choices),
            "help": self.help,
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
        }
