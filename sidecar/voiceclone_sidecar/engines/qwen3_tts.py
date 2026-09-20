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

from ..capabilities import AppliesTo, Capabilities, ParamSpec
from ..engine_config import EngineConfig, resolve_seam
from ..registry import GenerationRequest, GenerationResult, InstallableEngine
from ..runtime import downloader, installer, paths, uvman
from .. import params as params_mod
from .. import sources

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

# ModelScope counterpart (verified 2026-09: the official mlx-community mirror
# exists and resolves; ADR-0016 decision 4 closes the "qwen3-tts-mlx has no
# ModelScope source" coverage gap).
MS_REPO = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"

PACKAGES = ["mlx-audio>=0.5.4"]
PYTHON_SPEC = "3.12"


class Qwen3TtsMlxEngine(InstallableEngine):
    """Local engine. Also carries the InstallableEngine surface."""

    engine_id = "qwen3-tts-mlx"
    display_name = "Qwen3-TTS 0.6B Base（本地 · MLX）"

    def __init__(self, output_dir: Path | None = None, root: Path | None = None,
                 env: dict | None = None, config: EngineConfig | None = None) -> None:
        output_dir, env = resolve_seam(config, self.engine_id, output_dir, env)
        self.output_dir = output_dir
        self.env = env
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
            requires_reference_text=True,  # the sidecar auto-fills ref_text from the transcript
            max_chars_per_request=500,  # local VRAM ceiling; keep requests small (issue #13)
        )

    # Languages the Qwen3-TTS model itself speaks; the worker already sends
    # the canonical `language` value verbatim.
    LANGUAGE_CHOICES = ("Chinese", "English", "Japanese", "Korean", "German", "French")

    def param_specs(self) -> list[ParamSpec]:
        specs = [
            params_mod.canonical_language(
                self.engine_id, REPO, self.LANGUAGE_CHOICES,
                wire_name="language", default="Chinese",
            ),
        ]
        # ADR-0018 decision 3 — "不支持是数据": mlx-audio ACCEPTS a speed
        # argument, prints `Speed: {speed}x`, and never uses it (audit §4,
        # verified 2026-09-18 on our pinned mlx-audio). Declaring it as
        # no-op data is exactly what stops the UI from offering a slider
        # that silently lies.
        specs.append(
            ParamSpec(
                name="speed",
                label="语速",
                kind="number",
                min=0.5,
                max=2.0,
                step=0.05,
                unit="x",
                exposed=False,
                not_exposed_reason="no-op",
                applies_to=AppliesTo(engine=self.engine_id, model=REPO, mode="cloning"),
                help="mlx-audio 接受 speed 参数但从不使用（实测证实）；暴露即撒谎，故不暴露。",
            )
        )
        return specs

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
            # ADR-0016: the preferred weight source is read from the INJECTED
            # env at install time (settings change → registry rebuild → new
            # chain); the remaining sources stay as a silent fallback.
            chain = sources.weight_sources(
                REPO, ms_repo=MS_REPO,
                preferred=sources.weight_pref(self.env),
            )
            specs = [
                downloader.DownloadSpec(path=f, dest_name=f)
                for f in WEIGHTS_FILES
            ]
            for spec in specs:
                log(f"weights: {spec.dest_name}")
                downloader.download_file(
                    spec, weights_dir, chain,
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
        return GenerationResult(
            audio_path=result["audio_path"],
            sample_rate=result["sample_rate"],
            model_version=REPO,
            cost=0.0,  # local synthesis has no per-run cost
        )
