# Raw notes: Apple Silicon TTS — Soniqo / MLX / MNN / ONNX (fetched 2026-09-16)

## soniqo/speech-swift + soniqo/speech-core
- speech-core: C++17, Apache-2.0, Linux/Windows/Android, ONNX + LiteRT backends. https://github.com/soniqo/speech-core
- Apple sibling: https://github.com/soniqo/speech-swift (MLX + CoreML). CLI binary `speech`, Swift package `speech-swift`.
- HF org: https://huggingface.co/soniqo and https://huggingface.co/aufklarer
- Benchmarks page: https://soniqo.audio/zh/benchmarks  (M5 Pro 48GB, macOS 25.5, release build + compiled metallib, unless noted)

### TTS round-trip intelligibility (30 English dialogue sentences; synth -> Qwen3-ASR 0.6B re-transcribe; WER%)
| engine | params | size | WER% | RTF |
|---|---|---|---|---|
| CosyVoice3 | 0.5B 4-bit | ~1.9GB | 3.25 | 0.59 |
| Qwen3-TTS | 1.7B 4-bit | ~2.3GB | 3.47 | 0.79 |
| Qwen3-TTS | 1.7B 8-bit | ~3.5GB | 3.66 | 0.85 |
| Kokoro-82M CoreML | 82M | ~170MB | 3.90 | 0.17 |
| Qwen3-TTS | 0.6B 8-bit | ~960MB | 9.74 | 0.76 |
| Qwen3-TTS | 0.6B 4-bit | ~675MB | 15.58 | 0.76 |

### Zero-shot voice cloning (12s reference; M5 Pro 48GB; RTF = wall/audio)
| engine | params | Peak RSS | audio | synth | RTF | ASR roundtrip |
|---|---|---|---|---|---|---|
| Higgs TTS 3 (clone) | 4B bf16 | 8.6 GB | 6.04s | 4.73s | 0.78 | 完全一致 |
| F5-TTS (16 steps) | 336M fp16 | 0.8 GB | 5.09s | 2.91s | 0.57 | 差1词 |
| F5-TTS (32 steps) | 336M fp16 | 0.8 GB | 5.09s | 5.75s | 1.13 | 差1词 |
| IndexTTS2 (clone) | 1.5B-class fp16 | 3.0 GB | 5.80s | 5.8s | 1.0 | 完全一致 |

### Per-model details
- Kokoro-82M: CoreML ANE, RTFx ~0.7 (page) / RTF 0.17 (bench table, inconsistent -> note both), ~170MB (1 decoder bucket), 54 voices, 10 langs incl zh zf_xiaobei/zm_yunjian. 3-stage CoreML. buckets p16/p32/p64/p128; decoder_5s/10s/15s. iOS18/macOS15+.
- CosyVoice3 (Fun-CosyVoice3-0.5B): MLX, 4 variants (4bit ~1.2GB / 8bit ~1.4GB / 8bit-full ~1.6GB / bf16 ~2.1GB) at aufklarer/CosyVoice3-0.5B-MLX-*; CAM++ speaker encoder via CoreML ANE (aufklarer/CamPlusPlus-Speaker-CoreML, ~14MB, 12ms). 9 langs. Alibaba has NOT released 3.5 open weights (only cloud API id cosyvoice-v3.5-plus).
- F5-TTS: MLX fp16, CC-BY-NC-4.0 weights (non-commercial). steps 12 -> RTF 0.43; 16 -> 0.57; 32 -> 1.13. ~0.8GB RSS. EN+ZH, pinyin lexicon bundled (99.2% token-exact vs rjieba/pypinyin).
- Chatterbox Multilingual: MLX fp16 on Apple, LiteRT for edge; MIT. 23 langs. T3 + S3Gen + HiFi-GAN/HiFT.
- Higgs TTS 3: MLX bf16 + fp32 codec; Boson Higgs TTS 3 Research and Non-Commercial License. 4B. RTF 0.78 M5 Pro. ~8.6GB peak RSS. inline tags <|emotion:...|> <|style:whispering|> <|sfx:laughter|>.
- VibeVoice: Realtime-0.5B (~1GB bf16 / 350MB INT4 / 570MB INT8), 1.5B long-form (~3GB bf16 / 1GB INT4). MIT. EN+ZH only. up to 90 min, 4 speakers. 1.5B INT4 RTFx 1.48 on M2 Max.
- Magpie-TTS Multilingual 357M (NVIDIA): MLX INT4 ~247MB / INT8 ~411MB, 9 langs (incl zh), 5 preset speakers, NO zero-shot cloning. M4 Pro batch RTF 0.32 (short) / 0.23 (sentence) / 0.24 (23s sampling); streaming RTF 0.93; first-chunk ~120ms.
- VoxCPM2 (OpenBMB, 2B): MLX bf16 ~5.0GB / int8 ~3.0GB / int4 ~1.9GB. 48kHz! 30 langs. Apache-2.0. Round-trip WER/RTF: bf16 2.04%/1.38, int8 0.00%/1.02, int4 4.08%/0.83. Voice design via natural language.
- Fish Audio S2 Pro: MLX fp16, 44.1kHz, research/non-commercial, CLI pending.
- Pocket TTS 100M (Kyutai): ONNX INT8 120.3MiB, English only, fixed Alba voice, CC BY 4.0. Galaxy S23 Ultra (ONNX RT 1.27, 2 CPU threads): TTFA p50 127.7ms/p95 133.7ms, total p50 467.5ms, median RTF 0.4495, peak RSS 373.7MiB, WER 8.00%/CER 3.47% on 46 phrases. No Apple/CoreML backend in this release.

### Non-TTS but relevant
- Qwen3-ASR 1.7B MLX 5-bit: WER 1.32% LibriSpeech test-clean, RTF 0.027 (36x), 1.92GB
- Speaker embeddings: CAM++ CoreML ANE 12ms; WeSpeaker ResNet34-LM MLX 64ms / CoreML 143ms

## index-tts-2.5 PyPI family (yunfengwang, license: Bilibili-IndexTTS)
Source: IndexTTS-2.5 -> https://modelscope.cn/models/IndexTeam/IndexTTS-2.5
- `index-tts-2.5-mlx` v0.1.1: Apple Silicon MLX, int8 GPT. **RTF ~0.45, ~2.4x faster than official PyTorch MPS backend.** torch-free, MLX only. M1+, macOS13+, py3.10+.
- `index-tts-2.5-mnn` v0.1.1: MNN CPU-only, torch-free, fp16 (~3.6GB) or fp32 (~7GB). **Apple M5 Pro, 4 threads, ~3s audio: MNN fp32 synth 9.20s RTF 3.16; MNN fp16 9.48s RTF 3.25; ONNX Runtime fp32 CPU 9.53s RTF 3.27.** MNN vocoder ~4x faster than ORT CPU (1.12s vs 4.81s). fp32 bit-exact vs PyTorch CPU ref (73/73, 83/83 greedy tokens).
- `index-tts-2.5-onnx` v0.1.0: ONNX Runtime, fp32 ~7GB, `--device auto|cpu|cuda|coreml` (has a coreml option!). Bit-exact fp32. Torch-free.

## speech-swift (soniqo) — Apache-2.0 runtime, fetched 2026-09-16
https://github.com/soniqo/speech-swift  |  Homebrew formula `speech`  |  HF: https://huggingface.co/aufklarer
Extra RTF: Kokoro 0.08 RTF on iPhone 16 Pro; Supertonic-3 (99M, CoreML ANE, 31 langs, 44.1kHz) 0.15 RTF on iPhone 16 Pro.
TTS in speech-swift: Qwen3-TTS (MLX+CoreML, 0.6B/1.7B, 10 langs), CosyVoice3 (MLX 0.5B, 9 langs), VoxCPM2 (MLX 2B, 48kHz, 30 langs),
IndexTTS2 (MLX 1.5B-class fp16, EN/ZH), F5-TTS (MLX 336M fp16, EN/ZH, non-commercial), Higgs TTS 3 (MLX 4B bf16, non-commercial),
Kokoro-82M (CoreML ANE, 10 langs), Supertonic-3 (CoreML ANE 99M, 31 langs), VibeVoice Realtime-0.5B + 1.5B (MLX, EN/ZH),
Magpie-TTS (MLX/CoreML INT8 357M, 9 langs), Chatterbox Multilingual (MLX 0.8B fp16, MIT, 23 langs) + Chatterbox Flash (CoreML, EN),
OmniVoice (MLX, 600+ langs, Apache-2.0), Indic-Mio, Fish Audio S2 Pro (MLX 0.5B-class), CSM-1B (MLX int8/fp16, Apache-2.0).

## MLX TTS ports — verified by HF repo existence + download counts (2026-09-16)
mlx-community: index-tts2-mlx (22,049 dl!) | Kokoro-82M-bf16 (85,637 dl, 66 likes) | Qwen3-TTS-12Hz-1.7B-Base-bf16 (13,355) |
chatterbox-multilingual-v3 (2,205) / chatterbox-turbo-fp16 (1,062, 23 likes) / chatterbox-fp16 (675) |
Spark-TTS-0.5B-bf16 (1,039) | MOSS-TTS-Nano-100M (563) / MOSS-TTS-8B-8bit (131) | pocket-tts (539) |
Voxtral-4B-TTS-2603-mlx-4bit (1,034) | Breeze-TTS-2-mlx (1,031) | csm-1b | Dia-1.6B-fp16 | OuteTTS-1.0-0.6B-fp16
aufklarer: CosyVoice3-0.5B-MLX-bf16 (511) / -4bit (287) / -8bit (165) | Higgs-TTS-3-4B-MLX-bf16 (307) |
Magpie-TTS-Multilingual-357M-MLX-8bit (363) | VoxCPM2-MLX-bf16 (207) / int8 (204) | Chatterbox-Multilingual-MLX-fp16 (219) |
IndexTTS2-MLX-fp16 (199) | VibeVoice-1.5B-MLX-INT4 (139) | Kokoro-82M-CoreML (3,779) / CoreML-INT8 (154) |
Chatterbox-Flash-CoreML (15,081!) | Supertonic-3-CoreML (6,662) | Magpie-TTS-Multilingual-357M-CoreML-8bit (309)
soniqo (ONNX/LiteRT, non-Apple + Apple): Kokoro-82M-ONNX (33), Kokoro-82M-LiteRT (56), Chatterbox-LiteRT (60), VoxCPM2-ONNX (15), Pocket-TTS-100M-ONNX-INT8 (22)

## Licenses confirmed from HF API (2026-09-16)
| repo | license | downloads | likes | lastModified |
|---|---|---|---|---|
| hexgrad/Kokoro-82M | apache-2.0 | 11,594,066 | 6,931 | 2025-04-10 |
| SWivid/F5-TTS | cc-by-nc-4.0 | 946,804 | 1,201 | 2025-03-21 |
| coqui/XTTS-v2 | other (CPML) | 7,300,281 | 3,806 | 2023-12-11 |
| fishaudio/openaudio-s1-mini | cc-by-nc-sa-4.0 | 2,123 | 712 | 2026-02-06 |
| neuphonic/neutts-air-q4-gguf | apache-2.0 | 8,244 | 69 | 2026-08-26 |
| OuteAI/Llama-OuteTTS-1.0-1B | cc-by-nc-sa-4.0 | 3,048 | 245 | 2025-09-08 |
| yl4579/StyleTTS2-LibriTTS | (none declared) | 0 | 56 | 2023-11-21 |
Also exist: fishaudio/fish-speech-1.5, onnx-community/Kokoro-82M-v1.0-ONNX, neuphonic/neutts-air-q4-gguf.

## kokoro-onnx (thewh1teagle) README quote
"Fast performance near real-time on macOS M1"; "Lightweight: ~300MB (quantized: ~80MB)"; ONNX Runtime >=1.20.1. PyPI kokoro-onnx.
Piper: OHF-Voice/piper1-gpl, pip install piper-tts, espeak-ng phonemizer, VITS -> onnxruntime, C/C++ API libpiper, Java piper-jni.

## Benchmarks — TTS Arena V2 API (fetched 2026-09-16)
https://tts-agi-tts-arena-v2.hf.space/api/leaderboard  (JSON, 41 rows)
open-weight flagged rows only: Kokoro v1.0 (elo 1477, 1033 votes, open=true), Fluxions Vui (1392, open), Veena (1363, open).
Rest are closed. Full table captured in arena2.json.

## TTS Arena V1 (LEGACY, read-only, last modified 2025-05-02) — Arena Score
#1 ElevenLabs 1325 | #2 Hume Octave 1281 | #3 Papla P1 1252 | #4 Play.HT 2.0 1244 | #5 Play.HT 3.0 Mini 1244 |
#6 Kokoro v0.19 1243 | #7 Kokoro v1.0 1238 | #8 Fish Speech v1.5 1230 | #9 XTTSv2 1200 | #10 PlayDialog 1187 |
#11 MetaVoice 1180 | #12 StyleTTS 2 1164 | #13 PlayDialog 1.0 1153 | #14 OpenVoice 1142 | #15 MeloTTS 1133 |
#16 Fish Speech v1.4 1133 | #17 GPT-SoVITS 1125 | #18 WhisperSpeech 1110 | #19 CosyVoice 2.0 1109 |
#20 Parler TTS Large 1107 | #21 Parler TTS 1105 | #22 Vokan TTS 1087 | #23 OpenVoice V2 1078 |
#24 VoiceCraft 2.0 1075 | #25 Pheme 962

## Artificial Analysis TTS Arena (provider-voice), fetched 2026-09-16 — 90 entries
Open-weight-only subset (elo | name | appearances | commercialUseAllowed):
1202.2 Breeze TTS 2 1413 false
1120.8 Fish Audio S2 Pro 2334 false
1101.5 Step Audio EditX (Mar 2026) 2071 false
1073.3 Voxtral TTS 2147 false
1061.2 Magpie-Multilingual 357M (Feb 2026) 2041 TRUE
1061.1 Kokoro 82M v1.0 5298 TRUE
1041.7 OpenAudio S1 Mini 1681 false
1040.9 Maya1 3270 TRUE
1038.0 Higgs Audio V3 TTS 2047 false
1019.6 Chatterbox 4660 TRUE
1000.0 Zonos-v0.1 4869 TRUE
 961.5 VibeVoice 1.5B 2143 false
 951.8 OpenVoice v2 2992 TRUE
 915.0 XTTS v2 2553 false
 894.2 StyleTTS 2 2457 TRUE
 837.8 MetaVoice v1 1375 TRUE
Top overall: Cartesia Sonic 3.6 1277.1 / Inworld Realtime TTS-2 1243.5 / Speechify Simba 3.2 1237.5

## seed-tts-eval via SOTA2 aggregator (page updated ~2026-06; numbers as reported in papers)
EN page (WER | SIM): Qwen3-Omni 1.39 | F5-TTS 1.8 (SIM 0.679/0.69) | MiniMax-Speech 1.9 (0.738) | Seed-TTS 2.2 (0.762) |
CosyVoice3-1.5B 2.2 (0.72) | Qwen2.5-Omni 2.3 | CosyVoice 2 2.4 (0.66) | CosyVoice3 2.46 | CosyVoice2 2.57 |
MaskGCT 2.62 | Qwen2.5-Omni 7B 3.1 | Step-Audio2-mini 3.2
ZH page (WER): Qwen3-Omni-A3B-Instruct 1.07 | Qwen2.5-Omni 7B 1.21 | Qwen2.5-Omni 1.4 | UAS-Audio 1.4 |
LongCat-Next 1.9 | MiMo-Audio-In 1.93 | Step-Audio2-mini 2.1 | UniAudio 2.0 2.3 | Baichuan-Audio 2.9 | Kimi-Audio 13.46
Combined (test) page: HoliTok 1.33/0.62 ... Qwen3-TTS 1.46/0.715 ... F5-TTS 2.0/0.67 (2.04/0.671) |
Human 2.14/0.734 | Index-TTS 2 2.18/0.709 | CosyVoice 3 2.21/0.72 | IndexTTS2 2.23/0.71 | Seed-TTS 2.25/0.762 |
CosyVoice 3 0.5B 2.5/0.698 | MaskGCT 2.57/0.713 | CosyVoice 2 2.61/0.659 | MegaTTS3 2.79/0.77 |
VibeVoice 3.04/0.69 | Spark-TTS 3.14/0.57 | Llasa 8B 3.63/0.581
