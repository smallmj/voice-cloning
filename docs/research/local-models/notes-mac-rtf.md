# Raw notes: measured Apple Silicon + Windows numbers (fetched 2026-09-16)

## tts-bench (5uck1ess/tts-bench) — the single best decision artifact
Repo: https://github.com/5uck1ess/tts-bench  | Demos: https://5uck1ess.github.io/tts-bench/
Arena: https://5uck1ess-tts-arena.hf.space/ (blind A/B, human-preference Elo)
74 models tracked. Rigs: Windows 9950X3D CPU, Windows RTX 5090 CUDA, Linux RTX 3090, Apple M4 10C/16GB.
Artifacts: docs/results.md (May 2026), docs/known-issues.md, docs/considered.md, README TLDR (June 2026).

### Apple M4 10C 16 GB — predefined voice (results.md, May 2026)
| Model | device | TTFA cold | TTFA warm | RTFx warm |
| Piper | cpu | 90ms | 62ms | 33.5x |
| Kokoro-82M | mps | 595ms | 193ms | 13.8x |
| Kokoro-82M | cpu | 507ms | 299ms | 9.0x |
| KittenTTS | cpu | 316ms | 331ms | 8.2x |
| Soprano 80M | cpu | 341ms | 324ms | 6.1x |
| Soprano 80M | mps | 750ms | 427ms | 4.7x |
| Supertonic ~99M ONNX | cpu | 1952ms | 681ms | 4.2x |
| StyleTTS 2 | cpu | 2444ms | 2208ms | 3.82x |
| VibeVoice-Realtime-0.5B | mps | 3288ms | 2642ms | 1.1x |
| VibeVoice-Realtime-0.5B | cpu | 6478ms | 6977ms | 0.3x |
| Magpie-TTS Multi 357M | cpu | 5690ms | 6268ms | 0.5x |
| VibeVoice-1.5B | cpu | 75160ms | 70841ms | 0.06x |
README TLDR (June 2026) says Mac M4 fastest = Piper 208ms warm TTFA, 32x RTFx (later rebench; conflicts with May results.md 62ms/33.5x -- cite both + dates).

### Apple M4 16 GB — zero-shot cloning
| Pocket-TTS | cpu | 33ms | 29ms | 8.8x |  (fastest cloner; M4 beats Ryzen 9)
| OpenVoice v2 ~100M | mps | 2074ms | 1012ms | 5.91x |
| OpenVoice v2 | cpu | 10573ms | 2399ms | 2.49x |
| NeuTTS Nano GGUF Q4 | cpu | 684ms | 281ms | 2.8x |
| NeuTTS Nano GGUF Q4 | mps | 1579ms | 490ms | 2.2x | (MPS gives no win; GGUF runs CPU-side via llama-cpp)
| NeuTTS Air GGUF Q4 | cpu | 1337ms | 354ms | 2.1x |
| Coqui XTTS-v2 | mps | 3739ms | 1377ms | 2.0x |
| MOSS-TTS-Nano ~100M | cpu | 3709ms | 3661ms | 1.93x | (48kHz; mps 0.40x = worse)
| NeuTTS Air GGUF Q4 | mps | 2182ms | 545ms | 1.9x |
| Coqui XTTS-v2 | cpu | 2369ms | 2108ms | 1.4x |
| ChatterBox Turbo 744M | mps | 3902ms | 2033ms | 1.1x |
| ChatterBox Turbo | cpu | 3158ms | 2627ms | 0.8x |

### gpu-class on Mac (sub-realtime, auto-skipped) — RTFx warm
ChatterBox base (mps 0.5x / cpu 0.3x) | Sesame CSM-1B 0.2x | Qwen3-TTS Base 0.2x | IndexTTS-2 0.1x |
OmniVoice (default cpu 0.57x / mps 0.79x; cloning cpu 0.18x 5/5, mps ~0.2x 4/5) | F5-TTS <=0.1x |
Fish Speech 1.5 (cpu 0.34x / mps 0.31x) | Zonos v0.1 (cpu 0.15x, no Mac GPU path) |
MOSS-TTS-Nano/mps 0.40x | VibeVoice-1.5B (cpu 0.06x; mps OOM at load) | ZipVoice (cpu ~0.1x 10/11; mps OOM ~20GiB)
Timed out >600s/cell: MARS5, VoxCPM. Install-blocked on Apple Silicon: LuxTTS (no arm64 piper-phonemize wheel).

### Windows RTX 5090 CUDA (results.md) — for the contrast
Kokoro 101x (TTFA warm 69ms, 0.7GB) | OmniVoice default 8.5x / cloning 9.2x (2.0/2.4GB) |
F5-TTS 5.2x / 5.3x (0.8GB) | XTTS-v2 4.2x / 4.1x (2.0/1.9GB) | NeuTTS Nano 2.7x / 2.5x |
ChatterBox 2.0x / 2.0x (3.1/3.5GB) | NeuTTS Air 1.6x / 1.4x | VoxCPM2 1.2x / 1.2x (5.4/5.7GB) |
IndexTTS-2 1.1x / 1.1x (7.2/7.3GB) | Qwen3-TTS Base 0.6x / 0.6x (4.4/4.5GB) | VibeVoice-0.5B 2.1x (2.6GB) | Magpie 357M 1.7x (3.6GB)

### Windows Ryzen 9 9950X3D CPU (results.md)
Piper 47x / TTFA warm 39ms | Kokoro 13x (245ms) | KittenTTS 6.6x | Supertonic 6.1x (509ms) |
Pocket-TTS 2.9-3.0x (97-150ms) | VibeVoice-Realtime-0.5B ~0.5x | NeuTTS Nano 1.3-1.4x | NeuTTS Air 0.88-0.90x |
ChatterBox ~0.30x | F5-TTS ~0.05x

### tts-bench Apple-Silicon-specific bug notes (docs/known-issues.md)
- **Kokoro MLX is SLOWER than PyTorch-MPS**: head-to-head on M4 16GB, same 5 prompts: warm MLX 0-6% slower, never faster
  (RTFx p1 13.74 vs 13.77x; p3 16.23 vs 16.31x; p4 14.94 vs 15.55x; p5-FR 14.63 vs 15.54x). Cold start ~3.5x worse
  (first-call TTFA ~2057ms vs ~595ms, MLX JIT kernel compile). mlx-audio CRASHES on prompt 2:
  `ValueError: [broadcast_shapes] Shapes (1,140400,1) and (1,140700,9) cannot be broadcast`.
  => MPS stays the recommended Kokoro path on Mac; published speed-only.
- **OmniVoice MPS spurious OOM**: "MPS backend out of memory (MPS allocated: 3.38 GiB, other allocations: 16.73 GiB)" on 16GB M4
  but real usage ~1.1GB. Fix: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`. Default voice then ~0.4-0.7x RTFx;
  cloning mps 4/5 prompts, only 30-word long-form OS-killed (SIGKILL) >16GB; cpu 5/5. Both sub-realtime (~0.1-0.3x).
- **VibeVoice 1.5B not viable on 16GB Mac**: cpu ~0.03-0.07x RTFx (67-87s for 2-5s audio), long-form times out at 600s;
  mps OOMs at load `RuntimeError: Invalid buffer size: 10.94 GiB`. Now tagged GPU-class. 0.5B variant fine on cpu+mps.
- **MARS5**: no MPS path (cpu/cuda only), sub-realtime on M4 CPU, times out. "Treat as unusable on Mac."
- **LuxTTS install-blocked on Apple Silicon + Windows**: piper-phonemize ships no arm64-macOS wheel.
- **Fish Speech 1.5 Mac gotchas**: needs `brew install portaudio`; install.sh's cu128 torch reinstall has no macOS-arm64 wheel
  (now Darwin-guarded). Benches on Mac but sub-realtime (cpu 0.34x / mps 0.31x).
- **Zonos**: `--device cpu` crashes on CUDA hosts (import-time DEFAULT_DEVICE); harmless on a Mac (already cpu). No Mac GPU path.
- **Zonos2**: Linux+CUDA only (compiled CUDA kernels) — no CPU/MPS, no Windows path.
- **ZipVoice**: mps OOMs ~20GiB; cpu works 10/11 with ffmpeg installed.
- **Voxtral**: "is a Mac/MLX row in practice"; MLX port cannot clone from a wav (default lens only).
- **dia, qwentts_fast (CUDA-graph)**: CUDA-only; torch reports no available device on Apple Silicon.

### tts-bench per-model licence findings (docs/known-issues.md + cloning table)
MIT: Pocket-TTS, Piper, ChatterBox, F5-TTS(see note), VibeVoice(community fork) | Apache 2.0: NeuTTS, Kokoro, KittenTTS, LuxTTS, dots.tts
CPML 1.0 non-commercial: Coqui XTTS-v2
CC-BY-NC-SA-4.0 (applies to GENERATED AUDIO too): Echo-TTS, Fish Speech 1.5 / S2-Pro
NVIDIA Open Model License (commercial OK w/ terms): Magpie-TTS Multilingual 357M
LTX-2 Community (non-commercial): DramaBox
Boson Higgs Audio v3 Research & Non-Commercial: Higgs Audio v3 TTS
CC-BY-NC weights: OmniVoice (code Apache-2.0) — reclassified 2026-07 after HF discussion #29 / GH issue #211
GPL-3.0 w/ MIT runtime carve-out: sanoTTS
Note DISCREPANCY: tts-bench's cloning table lists IndexTTS-2 as "Apache 2.0" but the primary repo (index-tts/index-tts) states
"released under the bilibili Model Use License Agreement" + commercial use requires contacting indexspeech@bilibili.com. Trust the repo.
Same: tts-bench lists F5-TTS as MIT, but HF card SWivid/F5-TTS declares cc-by-nc-4.0 (code MIT, weights NC).
