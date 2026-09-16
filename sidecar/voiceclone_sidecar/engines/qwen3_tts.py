"""Qwen3-TTS 0.6B-Base on MLX — the first real local engine.

Torch-free by construction: the engine venv installs ``mlx-audio`` only.
Weights come from the mlx-community conversion of Qwen/Qwen3-TTS-12Hz-0.6B-Base
(bf16), downloaded through a source list (Hugging Face first, hf-mirror as the
China mirror), resumable at byte granularity. Once weights are local, the
worker runs with HF_HUB_OFFLINE=1 — generation is fully offline.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from ..capabilities import Capabilities
from ..registry import GenerationRequest, GenerationResult, InstallableEngine
from ..runtime import downloader, installer, paths, uvman

REPO = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"

WEIGHTS_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
]

# Mirror order matters: HF first, hf-mirror.com as the China fallback. Both
# templates keep {path} for repo-file paths; other full URLs pass through.
SOURCES = [
    f"https://huggingface.co/{REPO}/resolve/main/{{path}}",
    f"https://hf-mirror.com/{REPO}/resolve/main/{{path}}",
]

PACKAGES = ["mlx-audio>=0.5.4"]
PYTHON_SPEC = "3.12"


class Qwen3TtsMlxEngine(InstallableEngine):
    """Local engine. Also carries the InstallableEngine surface."""

    engine_id = "qwen3-tts-mlx"
    display_name = "Qwen3-TTS 0.6B Base（本地 · MLX）"

    def __init__(self, output_dir: Path | None = None, root: Path | None = None, env: dict | None = None) -> None:
        self.output_dir = output_dir
        self.env = env if env is not None else dict(_runtime_env())
        self.root = root if root is not None else paths.runtime_root(self.env)

    # -- capabilities ------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            languages=("zh", "en", "ja", "ko", "de", "fr"),
            voice_cloning=True,
            voice_design=False,
            pronunciation_control=False,
            emotion=False,
            commercial_license=False,
            cross_device_use=False,
            upload_used_for_training=False,
            api_closed_loop=False,
        )

    # -- install surface ----------------------------------------------------

    def _ctx(self) -> installer.InstallContext:
        return installer.InstallContext(engine_id=self.engine_id, root=self.root, env=self.env)

    def is_installed(self) -> bool:
        state = installer.load_state(self._ctx())
        return bool(state.get("installed"))

    def install_state(self) -> dict:
        state = installer.load_state(self._ctx())
        return {"installed": bool(state.get("installed")), "steps": state.get("steps", {})}

    def install_steps(self) -> list[installer.InstallStep]:
        weights_dir = paths.engine_weights_dir(self.root, self.engine_id)
        ctx = self._ctx()

        def find_uv(log):
            return uvman.find_uv(self.root, env=self.env, log=log)

        def step_python(log, progress):
            uv = find_uv(log)
            uvman.ensure_python(uv, PYTHON_SPEC, env=self.env, log=log)

        def step_venv(log, progress):
            uvman.create_venv(find_uv(log), self.root, self.engine_id, PYTHON_SPEC, env=self.env, log=log)

        def step_packages(log, progress):
            uv = find_uv(log)
            venv = uvman.create_venv(uv, self.root, self.engine_id, PYTHON_SPEC, env=self.env, log=log)
            uvman.pip_install(uv, venv, PACKAGES, env=self.env, log=log)

        def step_weights(log, progress):
            specs = [
                downloader.DownloadSpec(path=f, dest_name=f)
                for f in WEIGHTS_FILES
            ]
            for spec in specs:
                log(f"weights: {spec.dest_name}")
                downloader.download_file(
                    spec, weights_dir, SOURCES,
                    progress=lambda name, done, total, _s=spec: progress(
                        f"weights:{name}", done, total
                    ),
                    log=log,
                )
            log(f"weights: all {len(specs)} files ready in {weights_dir}")

        return [
            installer.InstallStep("python", f"安装 uv 托管的 Python {PYTHON_SPEC}", step_python),
            installer.InstallStep("venv", "创建引擎独立 venv", step_venv, artifact=uvman.engine_venv_dir(self.root, self.engine_id)),
            installer.InstallStep("packages", f"安装 {', '.join(PACKAGES)}", step_packages, artifact=uvman.engine_venv_dir(self.root, self.engine_id)),
            installer.InstallStep("weights", f"下载引擎权重 {REPO}", step_weights, artifact=weights_dir),
        ]

    def install(self, log, progress=None) -> dict:
        return installer.run_install(self._ctx(), self.install_steps(), log, progress)

    # -- generation ---------------------------------------------------------

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        ctx = self._ctx()
        if not self.is_installed():
            raise RuntimeError(
                "engine is not installed yet — call POST /engines/qwen3-tts-mlx/install first"
            )
        venv = paths.engine_venv_dir(self.root, self.engine_id)
        python = uvman.venv_python(venv)
        if not python.exists():
            raise RuntimeError(f"engine venv is broken (no python at {python}); reinstall the engine")

        out_dir = (self.output_dir or Path.cwd() / "data" / "audio").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / f"{request.generation_id or uuid.uuid4().hex}.wav").resolve()

        payload = {
            "weights_dir": str(paths.engine_weights_dir(self.root, self.engine_id)),
            "text": request.text,
            "output": str(out_path),
            "language": (request.params or {}).get("language", "Chinese"),
        }
        if request.params.get("ref_audio"):
            payload["ref_audio"] = request.params["ref_audio"]
            payload["ref_text"] = request.params.get("ref_text")

        worker = Path(__file__).with_name("qwen3_tts_worker.py")
        log(f"qwen3-tts-mlx: starting worker {python.name} -u {worker.name}")
        proc = subprocess.Popen(
            [str(python), "-u", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(self.root),
        )
        result = {}

        def pump():
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith("RESULT: "):
                    result.update(json.loads(line[len("RESULT: "):]))
                elif line.startswith("WORKER_ERROR: "):
                    log(f"qwen3-tts-mlx: {line[len('WORKER_ERROR: '):]}")
                else:
                    log(f"qwen3-tts-mlx: {line}")

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.close()
        code = proc.wait()
        reader.join(timeout=5)
        if code != 0 or "audio_path" not in result:
            raise RuntimeError(f"qwen3-tts-mlx worker exited with code {code}")
        return GenerationResult(audio_path=result["audio_path"], sample_rate=result["sample_rate"])


def _runtime_env() -> dict:
    """China-friendly defaults; users can override any of them."""
    return {
        "UV_PYTHON_INSTALL_MIRROR": "https://ghfast.top/https://github.com/indygreg/python-build-standalone/releases/download",
    }
