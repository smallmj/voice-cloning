"""FireRedTTS3 PoC 补丁 CLI（issue #26；自 issue #25 起委托到引擎包内实现）。

补丁表与 apply 逻辑的唯一真源在
``sidecar/voiceclone_sidecar/engines/fireredtts3_patch.py``（ADR-0019 决策 2：
PoC 与产品共用同一补丁面，永不漂移）。本文件只保留命令行入口，供 PoC 环境
（引擎 venv，未装 voiceclone_sidecar）使用——此时从仓库 checkout 直接加载包模块。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_patch_module():
    try:
        from voiceclone_sidecar.engines import fireredtts3_patch as mod

        return mod
    except ImportError:
        # PoC venv has no voiceclone_sidecar: load it straight from the repo
        # checkout this script ships in.
        repo_sidecar = Path(__file__).resolve().parents[2] / "sidecar"
        sys.path.insert(0, str(repo_sidecar))
        from voiceclone_sidecar.engines import fireredtts3_patch as mod

        return mod


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="FireRedTTS3 upstream checkout directory")
    args = parser.parse_args()
    mod = _load_patch_module()
    try:
        for line in mod.apply_patches(args.upstream.resolve()):
            print(line)
    except mod.PatchAnchorError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
