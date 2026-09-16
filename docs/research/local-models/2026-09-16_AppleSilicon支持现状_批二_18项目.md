# Apple Silicon (M1–M5) status of 18 open-source TTS / voice-cloning projects

**Research date: 2026-09-16.** Every finding carries its own 信息日期 (date of the evidence). Anything I could not confirm from a primary source is marked **未核实**. No number, RTF, VRAM figure or license below is invented.

**Tooling note:** `web_fetch` was unusable (non-public-IP errors) as warned. All GitHub issue/PR bodies and comments were pulled with `curl` + a local extractor over the pages' embedded JSON; GitHub REST API was rate-limited (core 0/60), so discovery used the GitHub HTML search endpoint (`blackbirdSearchRoute`) plus `web_search`.

---

## 0. The single most important structural finding

For this batch, **MLX (`mlx-audio` / `mlx-community`) — not PyTorch MPS — is the reliable Apple Silicon route.** PyTorch MPS is a minefield here: it is broken or degraded for **Dia, Zonos, OpenVoice, Chatterbox, Qwen3-TTS, MOSS-TTSD and VibeVoice**, and the recurring root causes are the same three:

1. **GQA + `scaled_dot_product_attention` on MPS** → `loc("mps_matmul"...): error: incompatible dimensions` / `invalid shape` → `LLVM ERROR: Failed to infer result type(s)`. Cited verbatim in Dia #136/#197, Zonos #44.
2. **`torch.compile` / `max-autotune` / Triton is CUDA-only** (PyTorch issue #150121), and **flash-attn has no MPS build** (Kimi-Audio #15, Qwen3-TTS #69, MOSS-TTSD #22, Step-Audio README).
3. **`torch.load` without `map_location`** hard-fails on MPS-only machines (Chatterbox #85, #357).

---

## 1. Dia-1.6B (nari-labs)

**(a) Runs at all / backend** — Yes, four distinct ways: PyTorch **MPS** (after two merged fixes + torch ≥2.7), PyTorch **CPU** fallback, **MLX**, **ONNX**.
- MPS: PR [#198](https://github.com/nari-labs/dia/pull/198) *“Enable Dia 1.6B Model to Run on Apple Silicon GPUs (MPS Backend)”* — **MERGED** (2025-05-12/13); closes issue [#197](https://github.com/nari-labs/dia/issues/197). Earlier CPU-only fallback: PR [#167](https://github.com/nari-labs/dia/pull/167) *“Add Apple Silicon-compatible example script for Dia model”* — **MERGED** 2025-05-12, adds `example/simple-mac.py`.
- MLX: `mlx-community/Dia-1.6B` + `-fp16/-4bit/-3bit/-6bit`; supported in [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) (`docs/models/tts/dia.md`).
- ONNX: [onnx-community/Dia-1.6B-0626-ONNX](https://huggingface.co/onnx-community/Dia-1.6B-0626-ONNX) (2025-06-30).

**(b) Crashes / garbage / leaks, quoted**
- Issue [#166](https://github.com/nari-labs/dia/issues/166) (2025-05-02): *“the existing example in /example/simple.py doesn't run on Apple Silicon (M-series) Macs… It uses `use_torch_compile=True`, which is incompatible with the MPS backend. It doesn't explicitly move the model or tensors to CPU, causing runtime errors like `incompatible dimensions`, `invalid shape`, or MPS kernel crashes (`mps_matmul`).”*
- Issue [#136](https://github.com/nari-labs/dia/issues/136) (2025-04-26): `loc("mps_matmul"…): error: incompatible dimensions` / `invalid shape` / `LLVM ERROR: Failed to infer result type(s)` / `zsh: abort python3 app.py`. Independently reproduced by **Mac Mini M4**, **M1 Max**, **M4 48GB Mac Pro**, **M1 Air** in the same thread. Workaround confirmed 2025-05-01: `pip3 install torch torchaudio torch-stoi --upgrade` (2.6.0 → 2.7.0).
- Issue [#197](https://github.com/nari-labs/dia/issues/197) (2025-05-12) root-causes it: *“a shape mismatch in the attention mechanism when Grouped Query Attention (GQA) is enabled. The MPS backend doesn't handle tensor broadcasting as flexibly as CUDA or CPU.”*
- **MLX hard crash** — mlx-audio issue [#101](https://github.com/Blaizzy/mlx-audio/issues/101) (2025-04-28): on a **MacBook Pro M4 Pro 48GB**, *“it crashed partway through. The MacBook Pro screen and external screen went off. Keyboard light remained on. I held down the power button to reboot.”* Maintainer Blaizzy (2025-05-24): *“We patched all models to fix memory spikes here #164.”*

**(c) Measured numbers (hardware named)**
| Hardware | Backend | RTF | tokens/s | Source, 信息日期 |
|---|---|---|---|---|
| MacBook Pro **M3 Pro 36GB** | CPU only, fp32 | 0.029–0.030x | 2.52–2.61 | [#166](https://github.com/nari-labs/dia/issues/166) 2025-05-02 (total 301.628 s, 767 steps) |
| **M3 Pro 36GB** | **MPS**, fp32 | **0.10x** (≈10× slower than realtime) | — | [#203](https://github.com/nari-labs/dia/issues/203) 2025-05-13 |
| **M3 Pro 36GB** | CPU only, fp32 | 0.03x | — | #203 |
| 32GB **M2 Pro** | — | 0.1x | — | #203 |
| MacBook Pro **M4 Pro 48GB** | **MLX** | **0.32x** | 26.8–31.8 it/s | mlx-audio [#101](https://github.com/Blaizzy/mlx-audio/issues/101) 2025-04-28 (23.90 s, peak 9.04 GB) |
| "Apple Silicon" (unspecified) | MPS, fp32, `use_torch_compile=False` | **0.154x** | **13.2** | PR [#288](https://github.com/nari-labs/dia/pull/288) 2025-12-11 — **PR is still OPEN (not merged)**; its CPU column claims 0.017x / 1.5 tok/s / 1771.6 s |
| 80GB A100 (reference) | CUDA fp16 | 2.53x | 217.9 | #203 2025-05-15 |
| RTX 3090 (reference) | CUDA fp16 | ~2.16x | 185.6 | #203 2025-05-15 |

Note the MPS 0.10x (#203) vs the unmerged 0.154x (#288) discrepancy — treat 0.154x as an unmerged claim.

**(d) Non-PyTorch macOS route** — **`mlx-audio`** (`pip install mlx-audio`, MIT): `mlx_audio.tts.generate --model mlx-community/Dia-1.6B-fp16`. GGUF via **CrispStrobe/CrispASR** (macOS build, *“Metal GPU support built in”*; TTS engine list includes Dia). ONNX via onnx-community.

**(e) License (primary source)** — **Apache-2.0.** HF [`nari-labs/Dia-1.6B`](https://huggingface.co/nari-labs/Dia-1.6B) → `license:apache-2.0`; README: *“This project is licensed under the Apache License 2.0.”*

---

## 2. Higgs Audio v2 (boson-ai)

**(a) Backend** — PyTorch **MPS** is explicitly handled but *requires a CPU round-trip* for audio decoding; **MLX** is supported and documented with RTF.
- PR [#98](https://github.com/boson-ai/higgs-audio/pull/98) *“Use CUDA for audio decoding unless you are on 'mps' and need to use CPU”* — **MERGED** (sxjscience). Quote: *“Previously, both CUDA and MPS tensors were being moved to CPU for audio decoding. The CPU transfer was only necessary for MPS devices due to embedding operation limitations in quantization layers.”* (2025)
- MLX: `mlx-community/higgs-audio-v2-3B-mlx-q8` / `-q6` (`library_name: mlx-audio`, last modified 2026-04-18). mlx-audio model README: *“Llama-3.2-3B-backed TTS … with real-time voice cloning **on Apple Silicon**.”*

**(b) Evidence** — Issue [#96](https://github.com/boson-ai/higgs-audio/issues/96) (2025-08-02, **still OPEN**), macOS Sequoia 15.6: `Using device: mps` then `Generating audio chunks: 0%| | 0/1 [00:00<?, ?it/s]` — **generation hangs at 0%, produces nothing**. Issue [#181](https://github.com/boson-ai/higgs-audio/issues/181) (2026-06-15, OPEN) *“MPS Support for Higgs Audio v3?”*: *“SGLang does not support MPS.”*

**(c) RTF — the best-documented Apple numbers in this batch** (mlx-audio `mlx_audio/tts/models/higgs_audio/README.md`, header explicitly **RTF (M5 Max)**):

| Model | Params | Format | RTF (M5 Max) | Memory |
|---|---|---|---|---|
| `bosonai/higgs-audio-v2-generation-3B-base` | 3B | bf16 | **0.60×** | 6.8 GB |
| `mlx-community/higgs-audio-v2-3B-mlx-q8` | 3B | 8-bit | **0.36×** | 6.18 GB |
| `mlx-community/higgs-audio-v2-3B-mlx-q6` | 3B | 6-bit | **0.33×** | 4.75 GB |

**(d) Non-PyTorch macOS route** — **`mlx-audio`** + `mlx-community/higgs-audio-v2-3B-mlx-q8|q6` + required `mlx-community/higgs-audio-v2-tokenizer`. For **v3** (4B) there is a community native-MPS FastAPI server: `prefixus/tts_local_api` (*“native PyTorch MPS inference on Apple Silicon”*, macOS 13+, **64 GB unified memory recommended**, 4B fp16 ≈ 8 GB).

**(e) License — primary source, with a conflict to flag**
- **v2 weights:** `bosonai/higgs-tts-2-3b-base` (the `higgs-audio-v2-generation-3B-base` ID 307-redirects here) → `license: other`. The LICENSE file fetched from that repo is titled **“BOSON HIGGS AUDIO 2 COMMUNITY LICENSE AGREEMENT”**, *“Boson Higgs Audio 2 Version Release Date: June 20, 2025”*, *“based upon the Meta Llama 3 Community License Agreement as of April 18, 2024”*. So: **not Apache/MIT** — a Llama-3-derived community licence.
- ⚠️ **Conflict:** mlx-audio's Higgs Audio v2 README states *“Higgs Audio v2 is released under the Apache 2.0 License”* and links the repo LICENSE. That contradicts the community-licence text above. **Do not rely on the Apache-2.0 claim.**
- **v3:** repo README: *“Higgs Audio v3 is released under the **Boson Higgs Audio v3 Research and Non-Commercial License**. Production / hosted / revenue-generating use requires a separate commercial license.”*

---

## 3. VibeVoice (microsoft)

**(a) Backend** — PyTorch **MPS** (needs the SDPA fix), **MLX** (`mlx-audio`), **GGUF/ggml** (`cstr`, `audio-cpp`, `VibeASR.cpp`), **ONNX**, plus a community ANE effort.

**(b) Crashes / garbage — extensively documented**
- Issue [#312](https://github.com/microsoft/VibeVoice/issues/312) (2026-04-02, closed) / fix PR [#303](https://github.com/microsoft/VibeVoice/pull/303) (**MERGED**), *“MockCacheLayer.get_mask_sizes must include query_length (breaks SDPA on MPS/CPU)”*: *“Streaming TTS inference crashes immediately on any device using **SDPA attention** — this includes **Apple Silicon (MPS)** — all Mac users … The model loads successfully but crashes on the first text-window forward pass during generation.”*
- Issue [#119](https://github.com/microsoft/VibeVoice/issues/119) *“Apple silicone”* (2025-12-05 → 2026-01-19, **still OPEN**) — the key thread:
  - scalar27 (2025-12-05), **M1 Max 64 GB**: *“I just tested the new smaller model (realtime) with my M1 Max 64 Gb. It runs but audio is very choppy.”*
  - dxcore35 (2025-12-06): *“I always have same problem on apple silicon it produce garbage…”*
  - scalar27 (2025-12-06): *“It's not garbage per se. It is just that the audio is not smooth… if quants can be made of this one that should help a lot.”*
  - YaoyaoChang (**Microsoft collaborator**, 2025-12-07): *“Our code runs well on Mac M4. Playback smoothness may vary depending on the hardware, and the final generated audio is always correct.”* (i.e. vendor denies the quality issue)
  - Namitjain07 (2025-12-11): *“Voice is very choppy on m1 processors”*
  - dnsdi (2026-01-19): *“more often than not I get crackling sound on **m4 chip with 24 gb ram** when played real-time, downloaded files are fine though.”*
- Issue [#373](https://github.com/microsoft/VibeVoice/issues/373) (2026-04-28, OPEN) — **MLX garbage output on M5**: `mlx-community/VibeVoice-ASR-4bit` via `mlx-audio` on **macOS 26.4.1 arm64 (Apple Silicon, M5)**, `mlx 0.31.2`, `mlx-metal 0.31.2`, `mlx-audio 0.4.2`: *“the autoregressive decoder degenerates near the end of the recording into an infinite repetition of the token sequence `Yes. Yes. Yes. …`”* (24 of 25 other recordings clean).

**(c) Numbers** — Issue #119, dxcore35 (2025-12-07): **Device: Macbook apple silicon M1 Max 64GB — “For 59 second audio it takes 72 seconds to generate.”** → RTF ≈ **1.22**. No tokens/s published for M-series. Microsoft's own edge engine: `VibeASR.cpp` (*“RTF < 1 on 3+ CPU threads — no GPU required”*, release note 2026-07-23) — CPU, not MPS.

**(d) Non-PyTorch macOS route**
- **MLX:** [`mlx-community/VibeVoice-ASR-bf16`](https://huggingface.co/mlx-community/VibeVoice-ASR-bf16) (`mlx-audio`, 2026-01-23); `appautomaton/vibevoice-mlx` (int8) via **`appautomaton/mlx-speech`**.
- **GGUF/ggml:** [`cstr/vibevoice-realtime-0.5b-GGUF`](https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF) (2026-08-02); [`audio-cpp/VibeVoice-ASR-Streaming-7B-GGUF`](https://huggingface.co/audio-cpp/VibeVoice-ASR-Streaming-7B-GGUF) (2026-09-09); `cstr/vibevoice-asr-GGUF`, `cstr/vibevoice-asr-bitnet-GGUF` via **CrispASR**; [`microsoft/VibeVoice-ASR-BitNet`](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet) + **`microsoft/VibeASR.cpp`**.
- **ONNX:** `onnx-community/VibeVoice-Realtime-0.5B-Onnx` (2026-09-11); `christopherthompson81/vibevoice-asr-streaming-1.5b-onnx`.

**(e) License** — **MIT** for the model cards (`microsoft/VibeVoice-1.5B`, `-Realtime-0.5B`, `-ASR`, `-ASR-Streaming-1.5B/7B`: `license:mit`). Caveat quoted from the Realtime card: *“The VibeVoice-Realtime model is limited to research purposes…”* and it enumerates prohibited uses.

---

## 4. Zonos (Zyphra)

**(a) Backend** — **Zonos v0.1: CPU-only on macOS** (MPS not supported upstream). **ZONOS2 (new, 2026-06): real MLX + GGUF routes exist.**
- Issue [#185](https://github.com/Zyphra/Zonos/issues/185) *“MPS Support”* (2025-03-10, **still OPEN**) quotes the upstream code: *“# MPS breaks for whatever reason. Uncomment when it's working.”*
- PR [#190](https://github.com/Zyphra/Zonos/pull/190) *“Add MPS support”* by 12v — **still OPEN**, *“12v wants to merge 1 commit”* (2025-03).

**(b) Crashes / garbage / slow — quoted**
- Issue [#1](https://github.com/Zyphra/Zonos/issues/1) *“Apple Silicon Support?”* (2025-02-10): *“mamba_ssm library used here does not support apple silicon yet.”* — cchance27: *“it's got custom cuda cpp kernels.”* — **Zyphra team (gabrielclark3330):** *“We are not planning to ship support for the hybrid models on apple silicon at this moment.”* — abaddouh: *“the mamba team seems not at all interested in supporting mac… this is going to be a hard no-go.”*
- Issue [#44](https://github.com/Zyphra/Zonos/issues/44) *“macos support”* (2025-02-12→16): *“CPU works on MacOS if you use the torch-backbone branch… However, it is kind of slow, and I was not able to get MPS to work, which might be due to a mismatch between num_heads and num_heads_kv, which causes problems with scaled_dot_product_attention.”* Verbatim MPS error: `loc("mps_matmul"…): error: incompatible dimensions` / `error: invalid shape` / `LLVM ERROR: Failed to infer result type(s)`. Then: *“The project maintainers have merged that branch into main, so MacOS (or any CPU-only platform supported by Torch) is technically supported now.”*
- **Garbage audio on MPS** — PR #190 author 12v: *“The recording generated on MPS doesn't sound as good as the recording generated on CPU.”* A commenter rebuts (2025): *“the quality loss happens on CUDA too. My guess would simply be floating point precision is less precise.”*
- Issue [#166](https://github.com/Zyphra/Zonos/issues/166) *“Not working properly on Apple Silicon chip”* (2025-02-28): **Mac mini m4** — *“it stops at 4% while generating”*, `113/2588 [00:33<12:21, 3.34it/s]`.

**(c) Numbers** — Issue #44 comment (2025): *“inference is insanely slow on mac (I am using an **m4 Max** and a few seconds of audio takes minutes.)”* → after applying PR #190: *“the transformer is massively faster on MPS.”* Mac mini M4: **3.34 it/s** (#166). Exact RTF 未核实.

**(d) Non-PyTorch macOS route** — **ZONOS2** is the real fix: [`mlx-community/Zyphra-ZONOS2`](https://huggingface.co/mlx-community/Zyphra-ZONOS2) (`library_name: mlx-audio`, 2026-06-13), `shraey/zonos2-mlx`; GGUF [`Zyphra/ZONOS2-GGUF`](https://huggingface.co/Zyphra/ZONOS2-GGUF) (2026-06-22, 7 858 dl) and [`Zyphra/ZONOS1-GGUF`](https://huggingface.co/Zyphra/ZONOS1-GGUF) (2026-07-15); `cstr/zonos-v0.1-transformer-GGUF` (ggml).

**(e) License** — **Apache-2.0.** HF `Zyphra/Zonos-v0.1-transformer`, `-hybrid`, `Zyphra/ZONOS2`, `Zyphra/ZONOS2-GGUF` all → `license:apache-2.0`.

---

## 5. Llasa (zhaochenyang20 / HKUST)

**(a) Backend** — **No official Apple Silicon path.** Architecture is Llama-based LLaMA (1B/3B/8B) + XCodec2 speech tokens, so the realistic non-PyTorch route is **GGUF/llama.cpp**.
- Official training repo is **[zhenye234/LLaSA_training](https://github.com/zhenye234/LLaSA_training)** — note the batch prompt's `zhaochenyang20/Llasa` path returns **404**; same for `zhaochenyang20/Llasa-3B` and `HKUSTAudio/Llasa`.
- I searched `repo:zhaochenyang20/Llasa mac`, `repo:zhenye234/LLaSA_training mac`, `… mps` → **0 results**, and the README contains no `mac`/`mps`/`metal`/`mlx` guidance.

**(b)** **未核实** — no MPS crash, garbage-audio or leak reports found for Llasa.
**(c)** **未核实** — no M-series RTF/tokens-s figures found.

**(d) Non-PyTorch macOS route (unofficial, third-party only)**
- **GGUF:** [`NikolayKozloff/Llasa-1B-Q8_0-GGUF`](https://huggingface.co/NikolayKozloff/Llasa-1B-Q8_0-GGUF), `NikolayKozloff/Llasa-3B-Q8_0-GGUF`, `srinivasbilla/llasa-3b-Q4_K_M-GGUF`, `srinivasbilla/llasa-3b-Q8_0-GGUF` (all 2025-01).
- **MLX:** `srinivasbilla/llasa-3b-Q4-mlx` (2025-01-21, `library_name: mlx`) — **not** from `mlx-community`, single small repo, **未核实** as production-usable.
- `CrispTTS` search snippet lists *“LLaSA (hybrid, …)”* as an engine, but `CrispStrobe/CrispTTS` README returned **404** on both `main` and `master` → **未核实**.

**(e) License (primary source)** — **CC-BY-NC-4.0.** HF `HKUSTAudio/Llasa-1B`, `Llasa-3B`, `Llasa-8B` → `license:cc-by-nc-4.0`; card front-matter: `license: cc-by-nc-4.0`, `base_model: meta-llama/Llama-3.2-3B-Instruct`. **Non-commercial.** Note: `NikolayKozloff/Llasa-3B-Q8_0-GGUF` self-labels `license:cc-by-4.0` — that conflicts with upstream; trust upstream.

---

## 6. Step-Audio (stepfun-ai)

**(a) Backend** — **CUDA-only upstream.** README is explicit: *“**An NVIDIA GPU with CUDA support is required.**”* Uses vLLM and a **custom flash-attention library** (`“Because our attention mechanism is a variant of ALIBI, the official flash attention library is not compatible.”`). Apple Silicon only via a third-party MLX port of the *EditX* variant.

**(b)** Issue [#46](https://github.com/stepfun-ai/Step-Audio/issues/46) *“Only support CUDA？”* (2025-02-18, closed): *“I'd like to use it on Mac with M Series Chip and MPS. Could I make it?”* No MPS implementation resulted.
**(c)** **未核实** — no M-series figures.
**(d) Non-PyTorch macOS route** — [`appautomaton/step-audio-editx-8bit-mlx`](https://huggingface.co/appautomaton/step-audio-editx-8bit-mlx) exposed by **[appautomaton/mlx-speech](https://github.com/appautomaton/mlx-speech)** (selector `step-audio`; MIT licence on the runtime; *“Requires an Apple Silicon Mac (M1 or later) and Python 3.13+”*). This is **Step-Audio-EditX**, not the 130B Step-Audio-Chat.
**(e) License (primary source)** — Code: *“The code in this open-source repository is licensed under the Apache 2.0 License.”* Weights: *“The use of weights for Step Audio related models requires following license in Step-Audio-Chat, Step-Audio-Tokenizer and Step-Audio-TTS-3B.”* **Step-Audio 2 mini / mini Base / mini Think: Apache 2.0** (HF card + Step-Audio2 README).

---

## 7. Kimi-Audio (MoonshotAI)

**(a) Backend** — Partial. Mac requires **removing flash-attn** (community patch); a **real MLX port** exists.
**(b) Evidence** — Issue [#15](https://github.com/MoonshotAI/Kimi-Audio/issues/15) *“建议下一个版本支持一下mac，花了我好长时间来适配mac”* (2025-04-27, OPEN): *“mac没有cuda版本的flash attn，下个版本移除吧。代码我修改好了。”* (= "Mac has no CUDA version of flash-attn; remove it in the next version. I've already modified the code.") — the attached diff deletes `from flash_attn import …`, changes `flash_attention: bool = True` → `= False`, and replaces the fused-attention calls with standard PyTorch attention. Issue [#50](https://github.com/MoonshotAI/Kimi-Audio/issues/50) (2025-04-29) *“您好 请问如何在Mac上运行该项目？”* remains unanswered.
**(c)** **未核实** — no M-series RTF/tokens-s.
**(d) Non-PyTorch macOS route** — [`mlx-community/kimi-audio-7b`](https://huggingface.co/mlx-community/kimi-audio-7b) (`library_name: kimi-audio`, **2026-04-05**, `license:mit`) — the only confirmed Apple-native route. A secondary blog (codersera.com) claims the 7B runs ASR on Apple Silicon via MLX-LM but that *speech generation still depends on CUDA-only kernels* — **secondary source, 未核实**.
**(e) License (primary source)** — README: *“Code derived from Qwen2.5-7B is licensed under the Apache 2.0 License. Other parts of the code are licensed under the MIT License.”* HF `moonshotai/Kimi-Audio-7B-Instruct` → `license:mit`.

---

## 8. Qwen3-TTS **and** Qwen2.5-Omni (QwenLM)

### 8a. Qwen3-TTS
**(a) Backend** — **Yes, strongly.** MLX (three independent providers), **CoreML** (incl. Neural Engine), **GGUF**, and PyTorch MPS via an open community PR.
**(b) Evidence**
- Issue [#69](https://github.com/QwenLM/Qwen3-TTS/issues/69) (2026-01-25, **closed as NOT_PLANNED**): *“[Qwen3-TTS-12Hz-1.7B-VoiceClone] Doesn't work on Mac M4Pro chip”* — `OS: MacOS 15.6`, `device_map="cpu"`, `dtype=torch.float16`; log shows *“Warning: flash-attn is not installed. Will only run the manual PyTorch version.”*; reporter: *“it keep running for about 20 minutes and doesn't stop, I don't know it's stucked or it's normal.”*
- PR [#124](https://github.com/QwenLM/Qwen3-TTS/pull/124) *“macOS/M-Series Support for Qwen3-TTS”* by evuori (**OPEN**, 2026-01-28): adds `get_optimal_device()` → *“Auto-detects MPS > CUDA > CPU”*, `get_attention_implementation()` → *“Auto-skips FlashAttention on non-CUDA”*, `device_synchronize()` replacing `torch.cuda.synchronize()`, and doc sections *“macOS / Apple Silicon (M1/M2/M3/M4) Support”*. **Upstream has not merged it.**
- mlx-audio issue [#851](https://github.com/Blaizzy/mlx-audio/issues/851) (2026-07-20): *“Two calls to `mx.clear_cache()` inside `_generate_icl` fire on the streaming hot path. Removing them gives ~13% RTF and ~7% TTFA on **M1 Pro**.”* — a real MLX streaming-performance bug (memory-management on the Apple path).

**(c) Numbers (hardware named)**
- **M1 Pro 16GB**, `Qwen3-TTS-12Hz-0.6B` **6-bit affine**, `streaming_interval=0.1`, one warmup + 6 sentences: **RTF avg 0.682 → 0.596** with `mx.clear_cache` no-op'd (max 0.690 → 0.601). Reproduced on mlx-audio 0.4.5 and 0.4.2. Source: mlx-audio [#851](https://github.com/Blaizzy/mlx-audio/issues/851), 2026-07-20.
- mlx-audio docs batch table (**6-bit, short prompt**) — **hardware not stated → 未核实**:

  | Batch | TPS | Throughput | Avg TTFB | Memory |
  |---|---|---|---|---|
  | 1 | 20.8 | 1.67x | 84.8 ms | 3.88 GB |
  | 2 | 34.7 | 2.78x | 78.0 ms | 3.92 GB |
  | 4 | 53.2 | 4.26x | 99.9 ms | 3.98 GB |
  | 8 | 68.1 | 5.45x | 140.5 ms | 4.10 GB |

- HF discussion [#15](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/discussions/15) (2026-01-26, stockbeaver): *“Does this model work with MPS? … I'm generating audio in an **M1 iMac with 16GB**. A little slow, but it works!”* — same post references `pip install mps-flash-attn` / `from mps_flash_attn import replace_sdpa`.

**(d) Non-PyTorch macOS routes (all confirmed to exist on HF)**
- **MLX:** `mlx-community/Qwen3-TTS-12Hz-{0.6B,1.7B}-{Base,CustomVoice,VoiceDesign}` in bf16/4/5/6/8-bit (all 2026-01-25/26; `1.7B-Base-bf16` has **13 355** downloads); `aufklarer/Qwen3-TTS-12Hz-{0.6B,1.7B}-{Base,CustomVoice}-MLX-{4,8}bit`; third-party `odiak/Qwen3-TTS-MLX` (*“MLX on Apple Silicon (Experimental)”*).
- **CoreML / ANE:** `TheStageAI/Qwen3-TTS-12Hz-0.6B-Base` (`library_name: coreml`, 2026-07-29); `erjigit17/Qwen3-TTS-0.6B-ANE` (`library_name: coremltools`, 2026-09-15, base = `Qwen3-TTS-12Hz-0.6B-CustomVoice`).
- **GGUF:** [`Serveurperso/Qwen3-TTS-GGUF`](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF) (`library_name: gguf`, 2026-09-09, **475 066 downloads**); [`ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF`](https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF) (2026-08-03); `cstr/qwen3-tts-voices-GGUF`; CrispASR lists Qwen3-TTS among 52 TTS engines.
- **ONNX:** `pltobing/Qwen3-TTS-Streaming-ONNX`, `Arm/qwen3-tts-0-6b-*-mix-precision` (also `litert`).

**(e) License (primary source)** — **Apache-2.0.** `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, `-VoiceDesign`, `-1.7B-Base`, `-0.6B-Base`, `-0.6B-CustomVoice`, `Qwen/Qwen3-TTS-Tokenizer-12Hz` all → `license:apache-2.0`.

### 8b. Qwen2.5-Omni
**(a) Backend** — PyTorch MPS works for multimodal understanding but **not for direct speech output**.
**(b) Evidence** — HF discussion [#19](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/discussions/19) *“Add llama.cpp support”*: *“On Apple Silicon **M2 Max 32GB** under `torch` I can run most of the multimodals (slowly) but not — so far — direct audio output (but it does save an `output.wav` successfully). Requires limiting the parameters and not trying to do more than one mode at once. Audio output creates an intelligible `output.wav` file but so far has not worked in direct mode.”*
- README stresses **FlashAttention-2** (*“we recommend enabling flash_attention_2”*) which is CUDA-only, and notes *“We are currently planning to develop a version that can perform inference with lower resource consumption requirements so that Qwen2.5-Omni can run on most platforms.”*
**(c)** **未核实** — only the qualitative *“slowly”* above.
**(d) Non-PyTorch macOS route** — No official MLX/GGUF/CoreML from Qwen found. Community: HF Space `Jimmi42/Qwen2.5-Omni-Apple-silicon` (claims *“2-5x performance improvements, Apple Silicon optimization”* — **未核实**, self-reported). `mlx-community/Qwen2-Audio-7B-Instruct-4bit` exists but is the *older Qwen2-Audio*, not Qwen2.5-Omni.
**(e) License** — `Qwen/Qwen2.5-Omni-3B` and `-7B` → **`license:other`** on HF. Exact terms **未核实** (I did not fetch the licence text).

---

## 9. MOSS-TTSD / MOSS-TTS (OpenMOSS)

**(a) Backend** — Upstream is **CUDA / SGLang only**. Apple Silicon is served entirely by third-party **MLX** ports.
**(b) Evidence** — Issue [#22](https://github.com/OpenMOSS/MOSS-TTSD/issues/22) (2025-07-15): `“You are attempting to use Flash Attention 2.0 with a model not initialized on GPU”` followed by `RuntimeError: CUDA error: device kernel image is invalid`. README requires `git clone OpenMOSS/sglang -b moss-ttsd-v1.0-with-cat` and warns about *“VRAM fragmentation”* — a CUDA-shaped workflow.
**(c)** **未核实** — no M-series RTF/tokens-s published. (Upstream speed claim is *“up to 16x”* from SGLang on GPU, 2025-09-09 — not Apple.)
**(d) Non-PyTorch macOS routes**
- **MLX (best):** [`appautomaton/openmoss-ttsd-mlx`](https://huggingface.co/appautomaton/openmoss-ttsd-mlx) — *“MLX-native int8 conversion of OpenMOSS TTSD for multi-speaker dialogue generation **on Apple Silicon**”* (base = `OpenMOSS-Team/MOSS-TTSD-v1.0`, 2026-09-11), consumed via **[appautomaton/mlx-speech](https://github.com/appautomaton/mlx-speech)** selectors `moss-ttsd`, `moss-local`, `moss-sound-effect`.
- **mlx-community:** `mlx-community/MOSS-TTSD-v1.0-MLX-8bit` (2026-03-31), `MOSS-TTS-8B-8bit` (2026-03-24), `MOSS-TTS-Nano-100M` (2026-04-24, 563 dl), `MOSS-TTS-Local-Transformer-v1.5-{bf16,4bit,8bit}`.
- **ONNX:** `OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX` (2026-04-17).
- **GGUF/ggml:** CrispASR carries `cstr/MOSS-Audio-4B-Instruct-GGUF`, `moss-transcribe`, `moss-diarize`.
- mlx-audio supports **MOSS-TTS** and **MOSS-TTS-Nano** natively.
**(e) License (primary source)** — **Apache-2.0.** README: *“MOSS-TTSD is released under the Apache 2.0 license.”* HF `OpenMOSS-Team/MOSS-TTSD-v1.0`, `MOSS-TTS`, `MOSS-TTS-v1.5`, `MOSS-TTS-Nano-100M`, `MOSS-TTS-Realtime`, `MOSS-TTS-Local-Transformer(-v1.5)` → `license:apache-2.0`.

---

## 10. Chatterbox (resemble-ai) — incl. chatterbox-gguf / llama.cpp ports

**(a) Backend** — **The best-supported model in this batch on Apple Silicon.** PyTorch **MPS** is first-class in the README (`device = "cuda"  # or "cpu" / "mps"`), and there are **three** independent non-PyTorch Apple routes: `ggml`/Metal (`chatterbox.cpp`), GGUF via CrispASR/audio.cpp, and MLX.

**(b) Crashes — quoted**
- Issue [#85](https://github.com/resemble-ai/chatterbox/issues/85) *“torch.load() Issue on Mac | M1 Pro”* (2025-05-30): `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False.` The MPS device-detection block passes, but *“all `torch.load()` in `tts.py` needed to be refactored”* to add `map_location=device`. `example_for_mac.py` is the community fix; a commenter asked *“Why does an official example require patching the torch module?”*
- Issue [#357](https://github.com/resemble-ai/chatterbox/issues/357) *“Chatterbox doesn't work on Apple silicon due to the lack of CUDA.”* (2025-11-16, **still OPEN**): fix is `torch.load(ckpt_dir / "s3gen.pt", map_location=device, weights_only=True)` at `mtl_tts.py:179`.
- **Resolution confirmed** — codehearted (2025-12-29): *“as of today, I can run Chatterbox Turbo TTS and Chatterbox TTS by switching the part of the line in the sample code to say `(device="mps")` instead of `(device="cuda")` — on my **M1 Air** running 15.7, with python 3.11.4 — and that's all it takes.”*

**(c) Numbers — `chatterbox.cpp` benchmark table (primary source, repo README)**
End-to-end, short sentence, voice cloning from an 11 s reference, warm runs, excludes model load:

| Backend | Wall | RTF | vs real-time |
|---|---|---|---|
| **Turbo** — Vulkan RTX 5090 Q4_0 | 463 ms | 0.07 | 14.2× |
| **Turbo** — **Metal (Mac Studio M3 Ultra, Q4_0)** | **985 ms** | **0.16** | 6.4× |
| **Turbo** — CPU AMD Ryzen 9 9950X Q4_0 | 5 397 ms | 0.82 | 1.2× |
| **Turbo** — **CPU (Mac Studio M3 Ultra, NEON)** | 7 568 ms | 1.05 | 0.96× |
| **Multilingual** — **Metal M3 Ultra Q4_0 `--cfm-steps 7`** | **1.05 s** | **0.30** | 3.3× |
| **Multilingual** — Metal M3 Ultra Q4_0 | 1.22 s | 0.35 | 2.9× |
| **Multilingual** — Metal M3 Ultra F16 | 1.41 s | 0.38 | 2.6× |
| **Multilingual** — **Metal M4 Q4_0** | **3.0 s** | **1.37** | 0.73× |
| **Multilingual** — Metal M4 F16 | 4.0 s | 1.65 | 0.61× |
| **Multilingual** — **CPU M4, 4t NEON, Q4_0** | 10.7 s | 4.32 | 0.23× |
| **Multilingual** — CPU M4, 4t NEON, F16 | 17.1 s | 6.70 | 0.15× |
| Reference ONNX Runtime CPU 4t Q4 | 31.7 s | 14.55 | 0.07× |

Also from the Resemble README (2026): **Chatterbox-Nano 110M** *“running **3x faster than realtime on 8 CPU cores**”* — CPU claim, not MPS.

**(d) Non-PyTorch macOS route (name the exact repos)**
- **GGML/Metal:** **[gianni-cor/chatterbox.cpp](https://github.com/gianni-cor/chatterbox.cpp)** — *“Pure C++/ggml inference on CPU / Metal / CUDA / Vulkan, no runtime dependency on Python or PyTorch”*, one binary auto-detecting Turbo vs Multilingual from GGUF metadata. **This is the strongest Apple route for Chatterbox.**
- **GGUF repos:** [`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF) (2026-08-13, 5 387 dl), `cstr/chatterbox-turbo-GGUF`, `cstr/chatterbox-nano-GGUF`, [`calcuis/chatterbox-gguf`](https://huggingface.co/calcuis/chatterbox-gguf) (2025-09-18), `hans00/Chatterbox-TTS-GGUF`, `BricksDisplay/Chatterbox-Multilingual-TTS-GGUF`.
- **MLX:** `mlx-community/chatterbox-fp16` + `chatterbox-multilingual-v3` (2026-07-31, `license:mit`); `mlx-community/Chatterbox-TTS-{fp16,4bit,8bit}` (2025-12-05) and `Chatterbox-Turbo-TTS-*` (2025-12-22).
- **Other ggml runtimes:** `0xShug0/audio.cpp` (Metal; release note 2026-07-31 mentions Metal optimisations), CrispASR (macOS build, Metal).

**(e) License (primary source)** — **MIT.** HF `ResembleAI/chatterbox` → `license: mit`, `cardData.license = mit`, 1 866 331 downloads. `chatterbox.cpp` README: *“Resemble AI, MIT-licensed zero-shot text-to-speech.”* `mlx-community/chatterbox-multilingual-v3` → `license:mit`.

---

## 11. NeuTTS Air (neuphonic)

**(a) Backend** — **llama.cpp GGUF** on Apple Silicon (`llama-cpp-python`), with an optional ONNX decoder. Not an MPS model.
**(b)** **未核实** — no MPS crash/garbage/leak reports (the Apple path bypasses MPS entirely).
**(c) Numbers — hardware named (primary source, repo README)**
Q4_0 quantisations, `llama-bench`, 500 prefill / 250 output tokens:

| Device | NeuTTS-Air | NeuTTS-Nano |
|---|---|---|
| Galaxy A25 5G (CPU only) | 20 tokens/s | 45 tokens/s |
| AMD Ryzen 9 HX 370 (CPU only) | 119 tokens/s | 221 tokens/s |
| **iMac M4 16 GB (CPU only)** | **111 tokens/s** | **195 tokens/s** |
| RTX 4090 | 16 194 tokens/s | 19 268 tokens/s |

Threads used: 14 prefill / 16 decode on the M4. ⚠️ The README explicitly caveats: *“these benchmarks only include the Speech Language Model and do not include the Codec which is needed for a full audio generation pipeline.”* So 111 tok/s is **not** end-to-end RTF.

**(d) Non-PyTorch macOS route** — **llama.cpp / `llama-cpp-python`** with GGUF. README **macOS (Apple Silicon)** section recommends Accelerate over Metal for CPU: `CMAKE_ARGS="-DGGML_METAL=OFF -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple" pip install "neutts[llama]" --force-reinstall --no-cache-dir`, and notes *“If you have a dedicated GPU (Nvidia/CUDA, AMD/ROCm, **M-Series Mac/Metal**)…”*. Weights: [`neuphonic/neutts-air-q4-gguf`](https://huggingface.co/neuphonic/neutts-air-q4-gguf) (8 244 dl), `neutts-air-q8-gguf`, `neutts-air`, [`Mungert/neutts-air-GGUF`](https://huggingface.co/Mungert/neutts-air-GGUF) (34 619 dl), `mradermacher/neutts-air-GGUF`. ONNX: `neuphonic/neutts-air-onnx`.

**(e) License (primary source)** — README licence table: **NeuTTS-Air = Apache 2.0**; NeuTTS-Nano and neutts-2e = *“NeuTTS Open License 1.0”*. HF `neuphonic/neutts-air`, `neutts-air-q4-gguf`, `neutts-air-q8-gguf` → `license:apache-2.0`.

---

## 12. dots.tts (rednote-hilab)

**(a) Backend** — Upstream PyTorch targets **CUDA**; Apple Silicon is served by **two independent pure-MLX ports** (one Python, one native Swift) plus ggml/GGUF.
**(b) Evidence — quoted**
- `sb1992/dots-tts-mlx` README: *“**Apple Silicon only** — MLX is Metal-only (the upstream PyTorch model targets CUDA).”*
- Upstream README: *“`--optimize` triggers a one-shot `torch.compile` warmup that walks every DiT compile bucket + KvPrefill + vocoder chunk sizes. Cold start takes **~3 minutes** on H800”* — a `torch.compile` dependency that is the same class of problem as Dia's; on MPS this is not the fast path.

**(c) Numbers — M-series, hardware named**
- `sammcj/mlx-swift-dots-tts` README: *“On an **M5 Max**, the AR backbone decodes **~2x faster** in MLX than PyTorch-MPS at fp32 and **~3.8x at int4** (it's memory-bandwidth-bound). The flow-matching DiT — the dominant cost — runs **~2–2.8x faster** in MLX fp32 (compute-bound, so quantisation there saves memory, not time). Quantisation shrinks the **8.2 GB fp32 core to ~2–3 GB**.”*
- `sb1992/dots-tts-mlx` README: MeanFlow *“reaches full quality in 4 single-pass steps with no CFG… Measured here at **~1.9–2.2× faster** per clip on reference cloning (EN/HI/ZH).”*
- Upstream (reference, **H800, not Apple**): `dots.tts-soar` RTF p50 **0.20 / 0.18** and first-chunk 225 ms / 69 ms; `dots.tts-mf` **0.15 / 0.13** and 204 ms / 68 ms (README, 2026-07).

**(d) Non-PyTorch macOS route (exact repos)**
- **[sb1992/dots-tts-mlx](https://github.com/sb1992/dots-tts-mlx)** — *“A pure-MLX port of rednote-hilab/dots.tts … running natively on Apple Silicon.”* Requires Python ≥3.10 on Apple Silicon; port code Apache-2.0.
- **[sammcj/mlx-swift-dots-tts](https://github.com/sammcj/mlx-swift-dots-tts)** — *“Native MLX Swift port … no Python runtime.”* Weights: [`smcleod/dots.tts-soar-mlx`](https://huggingface.co/smcleod/dots.tts-soar-mlx), `smcleod/dots.tts-base-bf16`. ⚠️ README caveat: *“the SwiftPM debug/test binary crashes on the first MLX op because the Metal kernels aren't compiled into it”* — use `xcodebuild`.
- **mlx-community:** `mlx-community/dots-tts-mlx` (2026-07-16) + `-int4/-int8/-mf-int4/-mf-int8` (2026-07-17).
- **appautomaton/mlx-speech:** selectors `dots-tts-soar` / `dots-tts-mf` → `appautomaton/dots-tts-mlx`.
- **GGUF/ggml:** `cstr/dots-tts-soar-GGUF` (crispasr, 2026-08-02); `EvoAwaken-Workshop/dots-tts-base-gguf` and `-edit-gguf` (2026-09-08); `0xShug0/audio.cpp`; CrispASR.

**(e) License (primary source)** — **Apache-2.0.** README: *“dots.tts code and released checkpoints are licensed under Apache-2.0.”* HF `dots-studio/dots.tts-soar`, `dots.tts-base`, `dots.tts-mf` → `license:apache-2.0`. ⚠️ **`dots-studio/dots.tts.edit` → `license:other`** (different terms). `sb1992/dots-tts-mlx` README stresses the port is a derivative and *“You must comply with the upstream licenses for the model weights.”*

---

## 13. Kani-TTS (nineninesix)

**(a) Backend** — **MLX-first for Apple Silicon** (official), plus GGUF.
**(b)** **未核实** — no MPS crash/garbage/leak reports.
**(c) Number — hardware named** — `nineninesix-ai/kani-mlx` README: *“Time to first chunk (25 frames) on **Macbook Air with M2 Silicon and 8GB RAM** is around **2-3 sec**. Tested on DuckDuckGo browser.”* Each chunk = 25 frames ≈ 2.0 s new audio with 15 lookback frames ≈ 1.2 s context.
**(d) Non-PyTorch macOS route** — **[nineninesix-ai/kani-mlx](https://github.com/nineninesix-ai/kani-mlx)** *“Kani TTS for Apple Silicon — A high-quality text-to-speech (TTS) system powered by MLX”*; prerequisites *“macOS 15.0 or higher (for MLX support)”*; ships standalone script + FastAPI server + web client. Weights: [`nineninesix/kani-tts-370m-MLX`](https://huggingface.co/nineninesix/kani-tts-370m-MLX) (`library_name: mlx`) + `nineninesix/nemo-nano-codec-22khz-0.6kbps-12.5fps-MLX`. Main repo README: *“For Apple Silicon users, we provide an optimized KaniTTS-MLX that takes full advantage of the unified memory architecture and Neural Engine on M1/M2/M3 chips.”* GGUF: `mradermacher/kani-tts-370m-GGUF`, `-450m-0.1-ft-GGUF`, `Mungert/kani-tts-450m-0.1-pt-GGUF`, `niobures/Kani-TTS`.

**(e) License (primary source) — note the split**
- **Code: Apache-2.0** — `kani-tts` README: *“Apache 2. See LICENSE file for details.”* Badge `License-Apache_2.0`. HF `nineninesix/kani-tts-370m-MLX` → `license:apache-2.0`.
- ⚠️ **The codec is separately licensed.** Main README license line for the codec component: *“**License:** NVIDIA Open Model License”* (links to NVIDIA's June-2024 open model licence agreement).
- ⚠️ Newer weights changed terms: `nineninesix/kani-tts-450m-0.1-ft`, `kani-tts-370m`, `kani-tts-400m-*` → **`license:other`** (2026-02-18), while `kani-tts-370m-MLX` (2025-10-08) still shows `apache-2.0`.

---

## 14. OuteTTS (edwko) — incl. llama.cpp/GGUF

**(a) Backend** — Yes, via **llama.cpp Metal (GGUF)** and **MLX**; **PyTorch MPS is explicitly unsupported**.
**(b) Evidence — quoted**
- Issue [#51](https://github.com/edwko/OuteTTS/issues/51) *“Inference do not support MPS”* (2025-01-06, closed): *“The operator 'aten::col2im' is not currently supported on the MPS backend and will fall back to run on the CPU. This may have performance implications.”*
- Issue [#75](https://github.com/edwko/OuteTTS/issues/75) *“Speed is slow”* (2025-04-09, **still OPEN**), on **M1 Mac + llama.cpp**: *“A 30 words sentence, need about 6000+ tokens on M1 mac with llama.cpp. Why it is so slow, and **sometimes the words will missing and pass**”* — i.e. dropped/missing words = garbage-audio symptom on the **Apple llama.cpp** path.
- Issue [#78](https://github.com/edwko/OuteTTS/issues/78) (2025-04-11) *“jupyter on mac cant find OUteTTS”* after `CMAKE_ARGS= -DGGML_METAL=on pip install outetts --upgrade`.

**(c)** Only the qualitative M1 token-count above; **no RTF/x-realtime figure found → 未核实**.

**(d) Non-PyTorch macOS route (exact)**
- **llama.cpp Metal:** README section *“Transformers + llama.cpp Metal (Apple Silicon/Mac)”* → `CMAKE_ARGS="-DGGML_METAL=on" pip install outetts --upgrade`.
- **MLX:** `mlx-community/OuteTTS-1.0-0.6B-fp16`; mlx-audio's supported-TTS table lists **OuteTTS** (`mlx-community/OuteTTS-1.0-0.6B-fp16`); OuteTTS README itself lists *“MLX-Audio | Python (MLX) | 1.0”*.
- **GGUF repos:** [`OuteAI/Llama-OuteTTS-1.0-1B-GGUF`](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B-GGUF) (2 223 dl), `OuteAI/OuteTTS-0.1-350M-GGUF`, `-0.2-500M-GGUF`, `-0.3-500M-GGUF`, `-0.3-1B-GGUF`, `second-state/*`, `gaianet/*`, `mradermacher/*`.
- **ONNX:** `onnx-community/OuteTTS-0.2-500M`, `OuteAI/Llama-OuteTTS-1.0-1B-ONNX`.
- CrispTTS lists *“OuteTTS (LlamaCPP or HF backend)”* (repo README 404'd → treat that specific claim as 未核实).

**(e) License — per-checkpoint, primary source (HF)**
| Checkpoint | License |
|---|---|
| `OuteAI/OuteTTS-0.1-350M` / `-GGUF` | **cc-by-4.0** |
| `OuteAI/OuteTTS-0.2-500M` / `-GGUF` | **cc-by-nc-4.0** |
| `OuteAI/OuteTTS-0.3-500M` / `-GGUF` | **cc-by-sa-4.0** |
| `OuteAI/OuteTTS-0.3-1B` / `-GGUF` | **cc-by-nc-sa-4.0** |
| `OuteAI/Llama-OuteTTS-1.0-1B` / `-GGUF` / `-ONNX` | **cc-by-nc-sa-4.0** |

⚠️ `mlx-community/OuteTTS-1.0-0.6B-fp16` is tagged `license:apache-2.0` — **inconsistent with the upstream cc-by-nc-sa-4.0**; treat the MLX repo's tag as unreliable. Also note the naming mismatch (0.6B in the MLX repo vs 1B upstream).

---

## 15. Muyan-TTS (MYZY-AI / Muyan-TTS)

**(a) Backend** — **未核实.** The GitHub repo is actually **[MYZY-AI/Muyan-TTS](https://github.com/MYZY-AI/Muyan-TTS)** (the batch prompt's `Muyan-TTS/Muyan-TTS` did not resolve). README states: *“All the inference process ran on a single **NVIDIA A100 (40G, PCIe)** GPU”* — no Apple Silicon claim anywhere.
**(b)** **未核实** — GitHub search `repo:MYZY-AI/Muyan-TTS mac` and `… mps` both returned **0 results**. No MPS/Apple thread exists.
**(c)** **未核实** — no M-series figures.
**(d) Non-PyTorch macOS route** — **No MLX, CoreML or official llama.cpp port found.** Only third-party GGUF quantisations exist: [`mradermacher/Muyan-TTS-GGUF`](https://huggingface.co/mradermacher/Muyan-TTS-GGUF) (256 dl), `mradermacher/Muyan-TTS-SFT-GGUF`, `NikolayKozloff/Muyan-TTS-Q8_0-GGUF`, `NikolayKozloff/Muyan-TTS-SFT-Q8_0-GGUF`. Since the backbone is Qwen2.5-0.5B, GGUF is plausible but **the full pipeline (BiCodec decoder) is 未核实 on Apple Silicon**.
**(e) License** — HF `MYZY-AI/Muyan-TTS` and `MYZY-AI/Muyan-TTS-SFT` → **`license:apache-2.0`**; GitHub repo lists *“Apache License”* and `mradermacher/Muyan-TTS-GGUF` → `license:apache-2.0`.

---

## 16. VoiceCraft (jasonppy)

**(a) Backend** — Runs on Apple Silicon **with manual patching** (CPU, or MPS "where MPS isn't an option" in the reporter's words). No official Apple support.
**(b) Evidence — quoted** — Issue [#33](https://github.com/jasonppy/VoiceCraft/issues/33) (2024-03-30), **macOS Sonoma 14.3.1, M1 Max 64GB**: *“As part of adapting the code for Apple Silicon, I replaced CUDA references with MPS (or CPU where MPS isn't an option). However, I encountered a runtime error related to `espeak` not being recognized by the system despite being installed.”* → `RuntimeError: espeak not installed on your system` at `TextTokenizer(backend="espeak")`. Fix (2024-03-31): set the phonemizer library path explicitly — `EspeakWrapper.set_library('/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib')`.
**(c)** **未核实** — no RTF/tokens-s on M-series.
**(d) Non-PyTorch macOS route** — **No official MLX/GGUF/CoreML port found.** Related: `zhisheng01/VoiceCraft-X` (tagged `onnx`, 2025-07-17) — **未核实** whether it targets macOS. `Manoghn/voicecraft-mistral-7b-gguf` exists but is a different (Mistral-based) architecture, not VoiceCraft's EnCodec+transformer stack.
**(e) License (primary source)** — README: *“The codebase is under **CC BY-NC-SA 4.0** ([LICENSE-CODE]), and the model weights are under **Coqui Public Model License 1.0.0** ([LICENSE-MODEL]).”* HF `pyp1/VoiceCraft` → `license:cc-by-nc-sa-4.0`. **Non-commercial.**

---

## 17. Seed-VC (Plachtaa)

**(a) Backend** — Yes, PyTorch **MPS** (voice conversion); README *“Suggested python 3.10 on Windows, **Mac M Series (Apple Silicon)** or Linux”*, changelog *“Added Mac M Series (Apple Silicon) support”*. But training is CUDA-only.
**(b) Evidence — quoted**
- Issue [#121](https://github.com/Plachtaa/Seed-VC/issues/121) (2025-02-08): user *“support m3 chip mac”* → author **Plachtaa**: *“I look forward someone to sponsor me a m3 mac so it would be possible for me to work on it😇”* — i.e. at that date M3 support was **not** implemented.
- Issue [#27](https://github.com/Plachtaa/Seed-VC/issues/27) (2024-10-14): *“现在只有cuda版本训练代码，由于本人没用过Mac所以这个可能需要你自己去改代码”* (= "only CUDA training code exists now; I've never used a Mac so you may need to modify it yourself").
- README: *“**It is strongly recommended to use a GPU for real-time voice conversion.**”* and a macOS gotcha: *“On Mac — running `real-time-gui.py` might raise an error `ModuleNotFoundError: No module named '_tkinter'`.”*

**(c)** **未核实** on M-series specifically. README's latency claim is hardware-agnostic: *“algorithm delay of ~300ms and device side delay of ~100ms”*.
**(d) Non-PyTorch macOS route** — **未核实.** No MLX/GGUF/CoreML port found.
**(e) License (primary source)** — HF `Plachta/Seed-VC` → **`license:gpl-3.0`**; the many community forks on HF carry the same `gpl-3.0` tag (e.g. `litagin/Seed-VC-duplicated`, `wuzhanliai21/Seed-VC`, `binant/Seed-VC`).

---

## 18. OpenVoice v2 (myshell-ai)

**(a) Backend** — Yes, PyTorch **MPS** with `PYTORCH_ENABLE_MPS_FALLBACK=1`. **No official MPS support or blessing from the maintainers.**
**(b) Evidence — quoted (maintainer position first)**
- Issue [#18](https://github.com/myshell-ai/OpenVoice/issues/18) *“How to use this project on Apple's M1 chip.”* (2024-01-02) — maintainer **Zengyi-Qin**: *“If you wanted to try a demo, please use the lepton or myshell link, instead of running on M1… **In neither situation we recommend running on M1.**”*
- Issue [#133](https://github.com/myshell-ai/OpenVoice/issues/133) *“please support mac m1/m2/m3 mps metal”* (2024-02-12, closed) — **Zengyi-Qin**: *“This is not a single-point problem. The repo has many dependencies that hasn't been adapted for apple chips.”*
- Same thread, MPS crash — hk6an6 (2024-03-02): *“The demo ipynb sets the device to "cuda:0" and when I change that to "mps" it breaks.”* → `IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)` at `openvoice/attentions.py:350` in `_get_relative_embeddings`.
- Also reported: `RuntimeError: Placeholder storage has not been allocated on MPS device!` (issue #18 thread), and a commenter's CPU reference point *“CPU： Apple M3 Max”*.
- **Working recipe (community-confirmed, twice):** ehartford (2024-02-16): *“it works pretty much out of the box. Just need to install pytorch for mps and need to `pip install torch` `brew install ffmpeg` and `pip install chardet`.”* — AI-Guru (2024-03-28): *“got OpenVoice on MPS up and running in no time. The only thing that I had to do was setting **PYTORCH_ENABLE_MPS_FALLBACK to 1**.”* — hk6an6's reply explains why: *“Apple's Metal does not yet have support for every operation in PyTorch. You are supplementing the gaps with your CPU.”* — smartexpert (2024-05-26): *“Can confirm I was able to run it successfully on **M3** using these instructions.”*

**(c)** **未核实** — no RTF/tokens-s on M-series.
**(d) Non-PyTorch macOS route** — **No MLX port found.** ONNX-adjacent: `seasonstudio/openvoice_tone_clone_onnx` (tagged `onnx`, 2024-10-27) — **未核实** whether it is a complete macOS-runnable pipeline.
**(e) License (primary source)** — **MIT.** README: *“Starting from April 2024, both V2 and V1 are released under MIT License. Free for commercial use.”* and *“OpenVoice V1 and V2 are MIT Licensed. Free for both commercial and research use.”* HF `myshell-ai/OpenVoiceV2` → `license:mit`.

---

## 19. MLX ecosystem for TTS — specifically investigated

### 19.1 mlx-audio (Blaizzy) — the central Apple Silicon TTS runtime
- Repo: [github.com/Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio). **License: MIT** (README badge + `## License` → *“[MIT License](LICENSE)”*).
- **Requirements section (verbatim):** *“Python 3.10+ / **Apple Silicon Mac (M1/M2/M3/M4)** / MLX framework / ffmpeg”*. **信息日期:** README on `main` retrieved 2026-09-16; the newest artefacts it references date it to **on/after 2026-03-27** (it lists `mlx-community/Voxtral-4B-TTS-2603-mlx-bf16`, HF `lastModified` 2026-03-27, and `mlx-community/MOSS-TTS-Local-Transformer-v1.5-*`). A sibling Swift project exists: `Blaizzy/mlx-audio-swift` for iOS/macOS on-device TTS.
- **TTS models with a real MLX port in mlx-audio (verbatim table row → HF repo):** Kokoro; KittenTTS; **Qwen3-TTS** (`mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`); **Higgs Audio v3** (`bosonai/higgs-audio-v3-tts-4b`); OmniVoice (`mlx-community/OmniVoice-bf16`, 646+ languages); CSM / MisoTTS; **Dia** (`mlx-community/Dia-1.6B-fp16`); **OuteTTS** (`mlx-community/OuteTTS-1.0-0.6B-fp16`); Spark; **Chatterbox** (*“Expressive multilingual TTS (v2/v3)”*, `mlx-community/chatterbox-multilingual-v3`, `chatterbox-fp16`); Soprano; Ming Omni TTS (BailingMM) 16.8B-A3B and 0.5B; KugelAudio; **Voxtral TTS**; rumik-oss 1; VoxCPM2; LongCat-AudioDiT; MeloTTS; **MOSS-TTS**, **MOSS-TTS-Nano**; **Higgs Audio v2** (`mlx-community/higgs-audio-v2-3B-mlx-q8|q6`). Also STT incl. **VibeVoice-ASR**, Qwen3-ASR, Parakeet, Voxtral; STS incl. MiMo-Audio, MossFormer2 SE, DeepFilterNet.
- Documented per-model RTF tables exist only for **Higgs Audio v2** (RTF on **M5 Max**: 0.60×/0.36×/0.33×) — see §2. Dia, Chatterbox, MOSS-TTS, Qwen3-TTS, Higgs v3 docs carry no RTF table.

### 19.2 mlx-community HF TTS ports (dates = HF `lastModified`, checked 2026-09-16)
Confirmed present, most relevant to this batch:
`mlx-community/Dia-1.6B-fp16` (2025-04-27) · `Dia-1.6B` `-4bit` `-3bit` `-6bit` (2025-04-23/24) · `higgs-audio-v2-3B-mlx-q8` (2026-04-18) · `higgs-audio-v2-3B-mlx-q6` · `chatterbox-fp16` / `chatterbox-multilingual-v3` (2026-07-31) · `Chatterbox-TTS-{fp16,4bit,8bit}` (2025-12-05) · `Chatterbox-Turbo-TTS-*` (2025-12-22) · `Qwen3-TTS-12Hz-0.6B-{Base,CustomVoice}-*` and `1.7B-{Base,CustomVoice,VoiceDesign}-*` (2026-01-25/26) · `OuteTTS-1.0-0.6B-fp16` (2025-05-19) · `MOSS-TTS-8B-8bit` (2026-03-24) · `MOSS-TTSD-v1.0-MLX-8bit` (2026-03-31) · `MOSS-TTS-Nano-100M` (2026-04-24) · `MOSS-TTS-Local-Transformer-v1.5-{bf16,4bit,8bit}` · `Zyphra-ZONOS2` (2026-06-13) · `dots-tts-mlx` (2026-07-16) + `-int4/-int8/-mf-int4/-mf-int8` (2026-07-17) · `kimi-audio-7b` (2026-04-05) · `VibeVoice-ASR-bf16` (2026-01-23) · Voxtral-4B-TTS-2603 variants (2026-03-27) · MiniMax-Music3 (7 quant levels) · `Spark-TTS-0.5B-*`, `Kokoro-82M-*`, `kitten-tts-*`, `Irodori-TTS-*`, `MeloTTS-English-MLX`, `Soprano-1.1-80M-bf16`.
**Not in mlx-community at all (searched 2026-09-16):** Llasa, Muyan-TTS, VoiceCraft, Seed-VC, OpenVoice, NeuTTS Air, Step-Audio. Kani-TTS has an MLX port but under its own org (`nineninesix/kani-tts-370m-MLX`), not mlx-community.

### 19.3 Other Apple-native (non-PyTorch) TTS providers worth naming
- **[appautomaton/mlx-speech](https://github.com/appautomaton/mlx-speech)** — *“Pure-MLX speech synthesis, voice cloning, dialogue, sound-effects, and ASR for **Apple Silicon**: Fish S2 Pro, VibeVoice, LongCat, MOSS, Step-Audio, Cohere ASR.”* **License MIT**; *“Requires an Apple Silicon Mac (M1 or later) and Python 3.13+.”* Relevant selectors: `vibevoice` (→ `appautomaton/vibevoice-mlx`), `moss-local`, `moss-ttsd`, `moss-sound-effect`, `step-audio` (→ `appautomaton/step-audio-editx-8bit-mlx`), `dots-tts-soar`/`dots-tts-mf`, `longcat`, `fish-s2-pro`, `dramabox`. ⚠️ README notes some bundled weights *“carry the NSCLv1 non-commercial license.”*
- **[aufklarer](https://huggingface.co/aufklarer)** (HF org) — real **MLX** *and* **CoreML** ports: `Qwen3-TTS-12Hz-0.6B-{Base,CustomVoice}-MLX-4bit/8bit`, `Qwen3-TTS-12Hz-1.7B-Base-MLX-4bit/8bit` (all `license:apache-2.0`), `CosyVoice3-0.5B-MLX-4bit`, `Kokoro-82M-CoreML`, `Qwen3-ASR-CoreML`, `Qwen3-ForcedAligner-0.6B-CoreML-INT4/INT8`, `DeepFilterNet3-CoreML`, `Silero-VAD-v6.2.1-CoreML`, `Parakeet-TDT-v3-CoreML-INT8`, `PersonaPlex-7B-MLX-4bit/8bit` (`cc-by-nc-4.0`).
- **[soniqo](https://huggingface.co/soniqo)** (HF org) — **ONNX + LiteRT** (`VoxCPM-0.5B-ONNX`, `VoxCPM2-ONNX`, `VoxCPM2-LiteRT(-INT8)`, all apache-2.0; plus ASR/VAD). LiteRT, not CoreML.
- **ggml / GGUF runtimes (Metal-capable):** [gianni-cor/chatterbox.cpp](https://github.com/gianni-cor/chatterbox.cpp) (Chatterbox; strongest Metal numbers in this report); [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) (*“Runs on Windows, Linux, and macOS, with support for NVIDIA, AMD, **Apple Silicon**, and CPU-only machines”*, GGUF-first, release 0.8.0 dated 2026-09-15, v0.5 note 2026-07-31 claims *“Metal optimizations with tested VoxCPM2 runs up to 2.56x faster on Apple Silicon”*); [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR) (*“macOS | crispasr-macos.tar.gz | **Metal GPU support built in**”*, 52 TTS engines incl. Qwen3-TTS, VibeVoice, dots.tts, Chatterbox, Dia, Zonos, Kokoro, CSM, Piper, MeloTTS) backing the `cstr/*-GGUF` HF repos; [microsoft/VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp).
- **CoreML TTS:** `mattmireles/kokoro-coreml` (Kokoro CoreML; README reports per-call warm wall times measured June 2026 on **M2 Studio 64 GB / M2 Air 24 GB / M1 Mini 16 GB** — a *not-in-this-batch* model, listed only as evidence that a CoreML TTS route with real M-series numbers exists); aufklarer CoreML repos above; `TheStageAI/*` and `erjigit17/*` CoreML/ANE for Qwen3-TTS.

---

## 20. Summary table — backend by model

Legend: **✅** confirmed working · **⚠️** works with caveats/patches · **❌** not available/blocked · **?** 未核实

| # | Model | PyTorch MPS | CPU-only | ONNX / CoreML | MLX | llama.cpp / GGUF-ggml | Best non-PyTorch macOS route | License (primary source) |
|---|---|---|---|---|---|---|---|---|
| 1 | **Dia-1.6B** | ⚠️ merged fixes ([PR#198](https://github.com/nari-labs/dia/pull/198)), needs torch ≥2.7; 0.10x RTF | ✅ 0.03x | ✅ ONNX (`onnx-community/Dia-1.6B-0626-ONNX`) | ✅ `mlx-community/Dia-1.6B-fp16`, **RTF 0.32x on M4 Pro 48GB** | ✅ CrispASR (Metal) | **mlx-audio** | **Apache-2.0** |
| 2 | **Higgs Audio v2** | ⚠️ works, needs MPS CPU-roundtrip for decode ([PR#98](https://github.com/boson-ai/higgs-audio/pull/98)); **hangs at 0% on MPS** ([#96](https://github.com/boson-ai/higgs-audio/issues/96)) | ? | ? | ✅ **RTF 0.33–0.60× on M5 Max** | — | **mlx-audio + `higgs-audio-v2-3B-mlx-q6`** | **Boson Higgs Audio 2 Community License** (`license:other`, Llama-3-derived) — ⚠️ mlx-audio wrongly says Apache-2.0 |
| 3 | **VibeVoice** | ⚠️ SDPA crash fixed by merged [PR#303](https://github.com/microsoft/VibeVoice/pull/303); choppy/crackling on M1/M4 | ✅ VibeASR.cpp RTF<1 @3+ threads | ✅ ONNX | ✅ MLX (`VibeVoice-ASR-bf16`); ⚠️ M5 degenerate `"Yes."` loop [#373](https://github.com/microsoft/VibeVoice/issues/373) | ✅ `cstr/vibevoice-realtime-0.5b-GGUF`, `audio-cpp/...-GGUF` | **mlx-audio / mlx-speech int8** | **MIT** (research-purpose caveat on Realtime) |
| 4 | **Zonos** | ❌ v0.1 MPS broken/open ([#185](https://github.com/Zyphra/Zonos/issues/185), [PR#190](https://github.com/Zyphra/Zonos/pull/190) open, quality loss); hybrid needs mamba_ssm = hard no-go | ✅ v0.1 CPU ("kind of slow") | ? | ✅ **only for ZONOS2** (`mlx-community/Zyphra-ZONOS2`) | ✅ `Zyphra/ZONOS2-GGUF`, `ZONOS1-GGUF` | **ZONOS2 via MLX or GGUF** | **Apache-2.0** |
| 5 | **Llasa** | ? **未核实** | ? | ? | ⚠️ only `srinivasbilla/llasa-3b-Q4-mlx` (unofficial) | ✅ `NikolayKozloff/Llasa-{1B,3B}-Q8_0-GGUF` | unofficial GGUF only (未核实) | **CC-BY-NC-4.0** |
| 6 | **Step-Audio** | ❌ README: *"NVIDIA GPU with CUDA support is required"*; custom flash-attn | ? | ✅ ONNX on several checkpoints | ⚠️ only Step-Audio-**EditX** via mlx-speech | — | `appautomaton/step-audio-editx-8bit-mlx` | Code **Apache-2.0**; weights per-model; **Step-Audio 2 mini: Apache-2.0** |
| 7 | **Kimi-Audio** | ⚠️ needs flash-attn removal patch ([#15](https://github.com/MoonshotAI/Kimi-Audio/issues/15)) | ✅ | ? | ✅ `mlx-community/kimi-audio-7b` (2026-04-05) | ✅ `vokra/kimi-audio` (gguf) | **`mlx-community/kimi-audio-7b`** | Qwen2.5-derived code **Apache-2.0**, rest **MIT**; HF card `mit` |
| 8a | **Qwen3-TTS** | ⚠️ [#69](https://github.com/QwenLM/Qwen3-TTS/issues/69) hangs 20 min on M4 Pro, closed NOT_PLANNED; MPS PR [#124](https://github.com/QwenLM/Qwen3-TTS/pull/124) **still open**; M1 iMac 16GB *"works, a little slow"* | ✅ | ✅ **CoreML** (`TheStageAI`, `erjigit17` ANE) + ONNX | ✅✅ **best-supported**: mlx-community, aufklarer, odiak; **RTF 0.68→0.60 on M1 Pro 16GB** | ✅ `Serveurperso/Qwen3-TTS-GGUF` (475k dl), `ggml-org/...-GGUF` | **mlx-audio / mlx-community**, or CoreML for ANE | **Apache-2.0** |
| 8b | **Qwen2.5-Omni** | ⚠️ understanding OK, **direct audio output fails** on M2 Max 32GB ([HF disc. #19](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/discussions/19)) | ✅ | ? | ❌ no official MLX | ❌ no official | unverified community Space `Jimmi42/Qwen2.5-Omni-Apple-silicon` | `license:other` on HF (text 未核实) |
| 9 | **MOSS-TTSD / MOSS-TTS** | ❌ upstream CUDA/SGLang ([#22](https://github.com/OpenMOSS/MOSS-TTSD/issues/22): *"CUDA error: device kernel image is invalid"*) | ? | ✅ `MOSS-TTS-Nano-100M-ONNX` | ✅✅ `appautomaton/openmoss-ttsd-mlx`, mlx-community MOSS-TTS/MOSS-TTSD | ✅ CrispASR (moss-* engines) | **`appautomaton/openmoss-ttsd-mlx` via mlx-speech** | **Apache-2.0** |
| 10 | **Chatterbox** | ✅ **officially** `device="mps"`; needed `map_location` fixes ([#85](https://github.com/resemble-ai/chatterbox/issues/85), [#357](https://github.com/resemble-ai/chatterbox/issues/357) open); works on M1 Air as of 2025-12-29 | ✅ Nano 110M "3x realtime on 8 CPU cores" | ✅ ONNX (reference only) | ✅ mlx-community chatterbox v2/v3 + Turbo | ✅✅ **`gianni-cor/chatterbox.cpp`** — Metal M3 Ultra RTF **0.16**; `cstr/chatterbox-GGUF` | **`chatterbox.cpp` (ggml+Metal)** — best Apple numbers | **MIT** |
| 11 | **NeuTTS Air** | ❌ N/A (llama.cpp path) | ✅ **iMac M4 16GB: 111 tok/s** (LLM only, excl. codec) | ✅ `neutts-air-onnx` | ? | ✅✅ native GGUF + `llama-cpp-python` (Accelerate or Metal) | **`neutts-air-q4-gguf` + llama.cpp** | **Apache-2.0** (Nano/2e: NeuTTS Open License 1.0) |
| 12 | **dots.tts** | ⚠️ upstream CUDA; `--optimize` depends on `torch.compile` | ? | ✅ ONNX/CoreML for Qwen3-TTS sibling only | ✅✅ **M5 Max: MLX ~2x (fp32) / ~3.8x (int4) faster than PyTorch-MPS**; 8.2GB→2–3GB | ✅ `cstr/dots-tts-soar-GGUF`, audio.cpp, EvoAwaken | **`sb1992/dots-tts-mlx`** or **`sammcj/mlx-swift-dots-tts`** | **Apache-2.0** (code+checkpoints); `dots.tts.edit` = `other` |
| 13 | **Kani-TTS** | ? | ✅ | ? | ✅✅ **official MLX** (`kani-mlx`); **TTFC 2–3 s on M2 MacBook Air 8GB** | ✅ `mradermacher/kani-tts-*-GGUF` | **`nineninesix-ai/kani-mlx`** | Code **Apache-2.0**; codec **NVIDIA Open Model License**; newer weights `other` |
| 14 | **OuteTTS** | ❌ `aten::col2im` unsupported/CPU-fallback ([#51](https://github.com/edwko/OuteTTS/issues/51)) | ✅ | ✅ `onnx-community/OuteTTS-0.2-500M` | ✅ `mlx-community/OuteTTS-1.0-0.6B-fp16` | ✅✅ README-sanctioned llama.cpp Metal; ⚠️ [#75](https://github.com/edwko/OuteTTS/issues/75) slow + **missing words on M1** | **llama.cpp `-DGGML_METAL=on`** or mlx-audio | 0.1-350M `cc-by-4.0`; 0.2-500M `cc-by-nc-4.0`; 0.3-500M `cc-by-sa-4.0`; 0.3-1B & Llama-1.0-1B `cc-by-nc-sa-4.0` |
| 15 | **Muyan-TTS** | ? | ? | ? | ❌ none found | ✅ GGUF exists (mradermacher/NikolayKozloff) | none confirmed — **未核实** | **Apache-2.0** |
| 16 | **VoiceCraft** | ⚠️ manual MPS/CPU patching + espeak path fix ([#33](https://github.com/jasonppy/VoiceCraft/issues/33), M1 Max 64GB) | ✅ | ? (`zhisheng01/VoiceCraft-X` onnx, 未核实) | ❌ | ❌ (only unrelated Mistral-based gguf) | none official — **未核实** | Code **CC BY-NC-SA 4.0**; weights **Coqui Public Model License 1.0.0** |
| 17 | **Seed-VC** | ✅ README: "Mac M Series (Apple Silicon)" supported; GPU strongly recommended for real-time | ✅ | ? | ❌ | ❌ | none found — **未核实** | **GPL-3.0** (HF) |
| 18 | **OpenVoice v2** | ⚠️ works with `PYTORCH_ENABLE_MPS_FALLBACK=1`; MPS breaks without it; **maintainers: "we do not recommend running on M1"** | ✅ | ⚠️ `seasonstudio/openvoice_tone_clone_onnx` (未核实) | ❌ | ❌ | none robust — **未核实** | **MIT** |

---

## 21. Cross-cutting conclusions

1. **MLX is the answer for 10 of 18 models.** Confirmed real MLX ports exist for: Dia, Higgs Audio v2 (and v3), VibeVoice (ASR), Qwen3-TTS, MOSS-TTS + MOSS-TTSD, Chatterbox v2/v3, Kani-TTS, OuteTTS, ZONOS2, dots.tts, Kimi-Audio. Pin the specific `mlx-community` / `mlx-audio` artefact — `mlx-audio` is MIT and requires **M1/M2/M3/M4** expressly.
2. **Best-documented M-series hard numbers in this batch:** `chatterbox.cpp` Metal on **Mac Studio M3 Ultra (RTF 0.16 Turbo / 0.35 Multilingual)** and **M4 (RTF 1.37/1.65)**; Higgs Audio v2 MLX RTF **0.33–0.60× on M5 Max**; dia MLX **RTF 0.32x on M4 Pro 48GB**; Qwen3-TTS MLX **RTF ~0.60 on M1 Pro 16GB**; dots.tts MLX **~2–3.8× faster than PyTorch-MPS on M5 Max**; NeuTTS Air llama.cpp **111 tok/s on iMac M4 16GB** (LLM only); VibeVoice **59 s audio in 72 s on M1 Max 64GB**; Dia PyTorch-MPS **0.10x on M3 Pro 36GB**.
3. **Do not use PyTorch MPS for:** Zonos v0.1 (❌), OpenVoice (❌ without fallback flag), OuteTTS (❌), Step-Audio (❌), MOSS-TTSD (❌). All four have a non-PyTorch escape hatch.
4. **License traps to flag to the parent:** Higgs Audio v2 is **NOT Apache-2.0** (community licence; mlx-audio's README is wrong); **Llasa = CC-BY-NC-4.0** and **VoiceCraft = CC BY-NC-SA 4.0 + Coqui PML** are non-commercial; Kani-TTS code is Apache-2.0 but its **codec is under the NVIDIA Open Model License** and newer weights flipped to `license:other`; **Qwen2.5-Omni is `license:other`** (terms not fetched); OuteTTS checkpoints span cc-by-4.0 → cc-by-nc-sa-4.0.
5. **Genuinely unverified (未核实) and should not be asserted:** Llasa MPS behaviour and any M-series performance; Muyan-TTS on any Apple backend; VoiceCraft/Seed-VC/OpenVoice non-PyTorch routes; Qwen2.5-Omni direct-audio on Apple; the mlx-audio Qwen3-TTS batch-throughput table's hardware; `CrispStrobe/CrispTTS` claims (repo README 404); the codersera.com blog claims about Kimi-Audio on MLX.
