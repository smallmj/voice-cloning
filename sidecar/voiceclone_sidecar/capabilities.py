"""Engine capability declaration.

Every engine must declare what it supports. A missing declaration is treated
as "not supported" — never as "silently ignore the parameter".
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
        }
