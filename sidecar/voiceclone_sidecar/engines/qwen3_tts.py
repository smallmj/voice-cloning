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
import threading
import uuid
from pathlib import Path

from .. import params as params_mod
from .. import sources
from ..capabilities import AppliesTo, Capabilities, ParamSpec
from ..engine_config import EngineConfig, resolve_seam
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
            languages=("zh", "en", "ja", "ko", "de", "fr", "it", "pt", "ru", "es"),
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

    # Languages the Qwen3-TTS model itself speaks (issue #47): the weight
    # config's talker_config.codec_language_id keys in canonical (capitalized)
    # form; the wire adapter lowercases them for `lang_code`. "Auto" means
    # "let the model decide" — expressed on the wire by NOT sending
    # lang_code at all (upstream auto behavior).
    LANGUAGE_CHOICES = (
        "Auto", "Chinese", "English", "French", "German", "Italian",
        "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
    )

    # mlx-audio qwen3 sampling parameters the upstream `generate` actually
    # consumes (verified in the engine venv source: make_sampler /
    # make_logits_processors, audit §2.1, issue #48). Defaults are the
    # official weight generation_config.json values — always sent, see
    # param_specs. Engine-specific folded area.
    SAMPLING_SPECS = (
        ("temperature", "Temperature", 0.9, 0.0, 2.0, 0.05, False,
         "采样温度（官方 generation_config 默认 0.9）。"),
        ("top_p", "Top-p", 1.0, 0.0, 1.0, 0.05, False,
         "核采样阈值（官方默认 1.0）。"),
        ("top_k", "Top-k", 50, 1, 200, 1, True,
         "Top-k 采样（官方默认 50）。"),
        ("repetition_penalty", "重复惩罚", 1.05, 1.0, 2.0, 0.05, False,
         "重复惩罚系数（官方默认 1.05）。"),
    )

    def param_specs(self) -> list[ParamSpec]:
        # Issue #47: the wire bug is fixed (`language=` was silently dropped
        # into **kwargs; the correct upstream key is `lang_code`, lowercase to
        # match the weight codec_language_id keys). BUT V3 real-machine
        # verification (2026-09-22, Mac MLX, evidence in capability_matrix)
        # proved mlx-audio's qwen3 Model.generate never CONSUMES lang_code:
        # Chinese text + lang_code="japanese" still produced Chinese, and
        # upstream source (0.5.4 AND latest main) never reads it. The model
        # follows the TEXT language automatically — all 10 languages verified
        # working that way. Per audit decision 3 ("验证未通过转 not_exposed")
        # and the speed precedent, the selector stays honest DATA, not a
        # lying dropdown. The to_wire mapping is kept so a future upstream
        # that consumes lang_code flips this back with one line.
        specs = [
            params_mod.canonical_language(
                self.engine_id, REPO, self.LANGUAGE_CHOICES,
                wire_name="lang_code", default="Auto",
                wire_transform=lambda v: None if v == "Auto" else v.lower(),
                exposed=False, not_exposed_reason="no-op",
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
        # Issue #48: engine-specific sampling parameters in the folded area.
        # The four values are ALWAYS sent: the official generation_config
        # (0.9/1.0/50/1.05) is what the UI shows, but mlx-audio's generate()
        # hardcodes DIFFERENT silent defaults (0.6/0.8/-1/1.3), so the
        # issue-38 "default not sent" convention would make the engine do
        # something else than what the user sees. Sending the declared
        # defaults restores "what the user sees = what the engine receives".
        for name, label, default, lo, hi, step, integer, help_text in self.SAMPLING_SPECS:
            specs.append(
                ParamSpec(
                    name=name,
                    label=label,
                    kind="number",
                    default=default,
                    min=lo,
                    max=hi,
                    step=step,
                    integer=integer,
                    layer="engine",
                    help=help_text,
                    applies_to=AppliesTo(engine=self.engine_id, model=REPO, mode="cloning"),
                    to_wire=lambda v, _d=default, _int=integer: (
                        _d if v in (None, "")
                        else int(float(v)) if _int else float(v)
                    ),
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
        self._ctx()  # ensure the engine context is initialized (pure side-effect call)

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

    def build_worker_payload(self, request: GenerationRequest,
                             weights_dir: str, out_path: Path) -> dict:
        """Map the user-facing request onto the worker JSON (issues #47/#48).

        The wire payload is built from the param specs' ``to_wire`` adapters —
        the user-facing value and the engine value may differ. The four
        sampling parameters are ALWAYS sent (mlx-audio's silent defaults
        differ from the official generation_config; see param_specs), and
        "Auto" language means the lang_code key is absent (upstream auto
        behavior), never an empty string.
        """
        spec_by_name = {s.name: s for s in self.param_specs()}
        params = request.params or {}
        payload = {
            "weights_dir": str(weights_dir),
            "text": request.text,
            "output": str(out_path),
        }
        lang_code = spec_by_name["language"].to_wire(params.get("language"))
        if lang_code:
            payload["lang_code"] = lang_code
        for name, _label, _default, _lo, _hi, _step, _int, _help in self.SAMPLING_SPECS:
            value = spec_by_name[name].to_wire(params.get(name))
            payload[name] = value  # always sent — see param_specs
        if params.get("ref_audio"):
            payload["ref_audio"] = params["ref_audio"]
            payload["ref_text"] = params.get("ref_text")
        return payload

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        self._ctx()  # ensure the engine context is initialized (pure side-effect call)
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

        payload = self.build_worker_payload(
            request, paths.engine_weights_dir(self.root, self.engine_id), out_path,
        )
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
