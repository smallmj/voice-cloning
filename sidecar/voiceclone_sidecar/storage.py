"""Shared storage helper: atomic JSON index writes.

Both the voice library and the generation history persist as JSON indexes;
a crash mid-write must never lose the whole file, so every save goes through
one temp-file + os.replace path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
