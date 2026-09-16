# Voicebox (github.com/jamiepine/voicebox) — TTS/STT engine & weights license audit

- **Audit date:** 2026-09-16
- **Subject:** open-source app Voicebox (jamiepine/voicebox), 7 TTS engines + Whisper STT + bundled local LLM
- **Method:** HuggingFace Hub API (`huggingface.co/api/models/...` → `cardData.license`, `tags`, `gated`, `lastModified`), GitHub REST via authenticated `gh api` (`license.spdx_id`), raw file fetch from `raw.githubusercontent.com` / HF `raw/main`, PyPI JSON API (`info.license` + License classifiers), GitHub code search (`search/code?q=repo:jamiepine/voicebox+...`).
- **`web_fetch` not used** (broken in this environment); all facts come from `curl`/`gh` against huggingface.co, raw.githubusercontent.com, api.github.com, pypi.org, plus `web_search` corroboration.
- Every license identifier below is the **actual observed value** plus its source URL. Anything not directly observed is marked **未核实 / unverified**.

---

## 0. Voicebox repo itself

| Item | Observed value | Source |
|---|---|---|
| Repo | `jamiepine/voicebox` (53,982 stars, pushed_at 2026-08-09) | `gh api repos/jamiepine/voicebox` |
| LICENSE | **MIT** — "MIT License / Copyright (c) 2026 Voicebox Contributors" | https://raw.githubusercontent.com/jamiepine/voicebox/main/LICENSE |
| GitHub license.spdx_id | **MIT** | `gh api repos/jamiepine/voicebox --jq .license.spdx_id` |
| NOTICE file | **ABSENT** | see below |
| THIRD-PARTY / third-party-licenses file | **ABSENT** | see below |

Recursive default-branch tree listing (`gh api repos/jamiepine/voicebox/git/trees/main?recursive=1`) filtered on `LICENSE|NOTICE|THIRD|licen|watermark|credit` returned **exactly one path: `LICENSE`**. There is no `NOTICE`, no `NOTICE.md`, no `THIRD-PARTY.md`, no `THIRD_PARTY_LICENSES.md`, no `CREDITS` file anywhere in the repo.

License display inside the app is **runtime-derived from HuggingFace**, not a bundled audit:

```
app/src/components/ServerSettings/ModelManagement.tsx
  // Derive license from HF data
  const license = hfModelInfo?.cardData?.license
    || hfModelInfo?.tags?.find(t => t.startsWith('license:'))?.replace('license:','');
  formatLicense(): {'apache-2.0':'Apache 2.0', mit:'MIT', 'cc-by-4.0':'CC BY 4.0',
                    'cc-by-sa-4.0':..., 'cc-by-nc-4.0':'CC BY-NC 4.0',
                    'openrail++':'OpenRAIL++', openrail:'OpenRAIL'}
```

Note the mapping table already includes `cc-by-nc-4.0` (non-commercial) → the UI is built to *display* NC licenses, not to block them.

Source: https://raw.githubusercontent.com/jamiepine/voicebox/main/app/src/components/ServerSettings/ModelManagement.tsx

---

## 1. Qwen3-TTS (Alibaba Qwen) — base / cloning models

Repos actually used by Voicebox (`backend/backends/pytorch_backend.py` L54-55, `backend/backends/mlx_backend.py` L47-50):

| Repo | Weights license (`cardData.license`) | Gated | lastModified | Source URL |
|---|---|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | **apache-2.0** | false | 2026-01-23 | https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | **apache-2.0** | false | 2026-01-29 | https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | **apache-2.0** | false | 2026-01-29 | https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign |
| `Qwen/Qwen3-TTS-Tokenizer-12Hz` | **apache-2.0** | false | 2026-01-29 | https://huggingface.co/api/models/Qwen/Qwen3-TTS-Tokenizer-12Hz |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | **apache-2.0** | false | 2026-01-25 | https://huggingface.co/api/models/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | **apache-2.0** | false | 2026-01-25 | https://huggingface.co/api/models/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 |

- HF tag lists include literal `license:apache-2.0` on all of the above. Model cards' YAML front-matter says `license: apache-2.0`.
- **Code license:** `QwenLM/Qwen3-TTS` → `license.spdx_id = **Apache-2.0**`; repo file `LICENSE` is the verbatim Apache License 2.0 text ("Apache License / Version 2.0, January 2004"). Source: https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/LICENSE
- PyPI package `qwen-tts` → license **Apache-2.0** (https://pypi.org/pypi/qwen-tts/json).
- **Commercial use: YES.** No commercial restriction observed anywhere.
- README of `Qwen/Qwen3-TTS-12Hz-0.6B-Base` contains **no** disclaimer / ethics / consent / responsible-use section (grep for `disclaim|ethic|consent|responsib|misuse|lawful` over the full 1360-line README → **zero hits**).
- **Answer to "Apache-2.0?": CONFIRMED Apache-2.0, weights *and* code.** Only the standard Apache-2.0 §6 trademark carve-out applies.
- Caveat worth noting: the base checkpoints are *explicit voice-clone* models (3-second zero-shot cloning). The license grant is Apache-2.0, so **the license itself imposes no consent/voice-likeness check**; real-world consent obligations for a commercial product come from local law (e.g. voice-likeness / personality rights), not from this license. 未核实: no operator-facing consent policy is published.

---

## 2. Qwen CustomVoice (preset timbres)

Voicebox repo IDs (`backend/backends/qwen_custom_voice_backend.py` L52-55):

| Repo | Weights license | Gated | lastModified | Source URL |
|---|---|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | **apache-2.0** | false | 2026-01-29 | https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | **apache-2.0** | false | 2026-01-29 | https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice |

- Code license: same `QwenLM/Qwen3-TTS` repo → **Apache-2.0** (see §1).
- **Preset voice count correction:** the task premise said "50+ curated preset voices via Kokoro and Qwen CustomVoice". The README sentence is being read as one number, but they are two engines:
  - README: "50 curated preset voices, tiny 82M model" = **Kokoro**
  - README: "9 curated preset voices with natural-language delivery control" = **Qwen CustomVoice**
  - Voicebox's own doc confirms it: https://raw.githubusercontent.com/jamiepine/voicebox/main/docs/content/docs/overview/preset-voices.mdx → table "**Kokoro 82M** | 50 voices | 9 languages" and "**Qwen CustomVoice** | 9 (premium curated) | 4 languages".
  - The 9 CustomVoice timbres are named in the model card: **Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee** (with "Voice Description" + native-language columns). Source: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/raw/main/README.md
- **Restriction on commercial use of the preset timbres: NONE STATED.** 未核实 / unverified for anything beyond Apache-2.0 — the model card grants the weights under Apache-2.0 and says nothing about the timbres as personalities. There is **no** voice-cloning consent policy, likeness clause, or acceptable-use addendum in the license or the card. The Apache-2.0 file has no field-of-use restriction.
- **Risk framing:** the preset timbres are *baked-in named voices* (some are described by region: "Youthful Beijing male", "Lively Chengdu male"). A commercial product shipping those exact timbres has an **unverified likeness/consent position**, because no licensor statement addresses it. The license does not grant it explicitly and does not forbid it either.

---

## 3. LuxTTS

**Publisher: Yatharth Sharma — GitHub `ysharma3501/LuxTTS`, HuggingFace `YatharthS/LuxTTS`.**
Correction to the task premise: **`YatharthS/LuxTTS` on GitHub does not exist** (`gh api repos/YatharthS/LuxTTS` → **404**). The HF repo's Usage section points to `https://github.com/ysharma3501/LuxTTS.git`.

| Item | Observed value | Source |
|---|---|---|
| Weights | **apache-2.0** (`cardData.license`; tag `license:apache-2.0`), gated=false, lastModified 2026-01-23 | https://huggingface.co/api/models/YatharthS/LuxTTS |
| Model card license statement | "Model and code is released under Apache-2.0 license." | https://huggingface.co/YatharthS/LuxTTS/raw/main/README.md |
| Code | `ysharma3501/LuxTTS` → `license.spdx_id = **Apache-2.0**` | `gh api repos/ysharma3501/LuxTTS` |
| Voicebox reference | `backend/backends/luxtts_backend.py` L28: `LUXTTS_HF_REPO = "YatharthS/LuxTTS"` | raw.githubusercontent.com |
| Upstream base | Based on **ZipVoice**, distilled to 4 steps; `k2-fsa/ZipVoice` = **Apache-2.0** (`gh api repos/k2-fsa/ZipVoice`); 48 kHz **Vocos** vocoder (`vocoder/vocos.bin`) | model card + repo metadata |

- **Commercial use: YES** (Apache-2.0, weights and code).
- Voicebox docs description **confirmed** (Lightweight ~1GB VRAM, 48 kHz output, cloned / zero-shot voice cloning, >150x realtime): `docs/content/docs/overview/preset-voices.mdx` and README engine table; model card says "Fits within 1gb vram", "Clear 48khz speech generation", "speeds exceeding 150x realtime".
- **License gap inside the LuxTTS install path (see §10):** Voicebox installs `linacodec @ git+https://github.com/ysharma3501/LinaCodec.git`. `gh api repos/ysharma3501/LinaCodec --jq .license` → **`{"license":null}`** = **no license file detected → 未核实 / unverified**.

---

## 4. Chatterbox Multilingual + Chatterbox Turbo (Resemble AI)

| Item | Observed value | Source |
|---|---|---|
| `ResembleAI/chatterbox` (this is the **multilingual** 23-language model; there is no separate `ResembleAI/chatterbox-multilingual` repo) | **mit** (`cardData.license`; tag `license:mit`), gated=false, lastModified 2026-06-10 | https://huggingface.co/api/models/ResembleAI/chatterbox |
| `ResembleAI/chatterbox-turbo` (350M, English, paralinguistic tags) | **mit**, gated=false, lastModified 2025-12-15 | https://huggingface.co/api/models/ResembleAI/chatterbox-turbo |
| `ResembleAI/chatterbox-nano` | **mit**, lastModified 2026-07-21 | https://huggingface.co/api/models/ResembleAI/chatterbox-nano |
| Code | `resemble-ai/chatterbox` → `license.spdx_id = **MIT**` (26,442 stars, pushed_at 2026-07-21) | `gh api repos/resemble-ai/chatterbox` |
| PyPI `chatterbox-tts` | **MIT License, Copyright (c) 2025 Resemble AI** | https://pypi.org/pypi/chatterbox-tts/json |
| Watermarker dependency `resemble-perth` | **MIT** (classifier MIT) | https://pypi.org/pypi/resemble-perth/json |
| PerTh tool repo | `resemble-ai/Perth` → **MIT** | `gh api repos/resemble-ai/Perth` |
| Voicebox repo IDs | `ResembleAI/chatterbox`, `ResembleAI/chatterbox-turbo` (`backend/backends/chatterbox_backend.py` L30, `chatterbox_turbo_backend.py` L30) | raw.githubusercontent.com |

### PerTh watermark — CONFIRMED, verbatim from both model cards and the GitHub README

> "## Built-in PerTh Watermarking for Responsible AI
> Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy."

Present identically on:
- https://huggingface.co/ResembleAI/chatterbox/raw/main/README.md (L178-180)
- https://huggingface.co/ResembleAI/chatterbox-turbo/raw/main/README.md (L139-141)
- https://raw.githubusercontent.com/resemble-ai/chatterbox/master/README.md (L173-175)

**Description / mechanics as documented by Resemble AI:**
- Imperceptible **neural** watermark embedded in every generated file.
- Survives MP3 compression, audio editing and "common manipulations".
- Claims "nearly 100% detection accuracy".
- Extraction is a supported public API — the README ships the script:
  ```python
  import perth, librosa
  watermarked_audio, sr = librosa.load(AUDIO_PATH, sr=None)
  watermarker = perth.PerthImplicitWatermarker()
  print(watermarker.get_watermark(watermarked_audio, sample_rate=sr))  # 0.0 or 1.0
  ```
  → a 0/1 presence detector, not a payload/ID extractor (no per-generation identifier documented).
- Chatterbox README also carries: "## Disclaimer — Don't use this model to do bad things. Prompts are sourced from freely available data on the internet."
- Resemble AI markets this on their site: "The only open-source TTS with built-in watermarking… MIT license" (https://www.resemble.ai/learn/models/chatterbox-turbo).

- **Commercial use: YES** — MIT, no commercial restriction, no field-of-use clause anywhere in the license or the cards. The watermark is a technical/provenance constraint, **not** a licensing restriction; there is no license term requiring you to keep, disclose, or refrain from removing it. Note MIT gives you the legal right to modify the library, so the watermark's persistence is a *default behavior* choice, not a license mandate.

---

## 5. TADA by HumeAI — **NOT Apache-2.0**

Repos used by Voicebox (`backend/backends/hume_backend.py` L37-44): `HumeAI/tada-codec`, `HumeAI/tada-1b`, `HumeAI/tada-3b-ml`.

| Repo | Weights license (`cardData.license` / LICENSE file) | Gated | lastModified | Source URL |
|---|---|---|---|---|
| `HumeAI/tada-1b` | **llama3.2** — LICENSE file is the full **"LLAMA 3.2 COMMUNITY LICENSE AGREEMENT"** | false | 2026-03-17 | https://huggingface.co/api/models/HumeAI/tada-1b · https://huggingface.co/HumeAI/tada-1b/raw/main/LICENSE |
| `HumeAI/tada-3b-ml` | **llama3.2** — same Llama 3.2 Community License text | false | 2026-03-17 | https://huggingface.co/api/models/HumeAI/tada-3b-ml · https://huggingface.co/HumeAI/tada-3b-ml/raw/main/LICENSE |
| `HumeAI/tada-codec` | **mit** (only the encoder/codec is permissive) | false | 2026-03-13 | https://huggingface.co/api/models/HumeAI/tada-codec |
| `HumeAI/mlx-tada-1b` | **llama3.2** | false | 2026-03-24 | https://huggingface.co/api/models/HumeAI/mlx-tada-1b |
| `HumeAI/mlx-tada-3b` | **llama3.2** | false | 2026-03-24 | https://huggingface.co/api/models/HumeAI/mlx-tada-3b |
| Base models | `base_model:meta-llama/Llama-3.2-1B` / `Llama-3.2-3B` (per HF tags) | — | — | HF API `tags` |

**Code license ≠ weights license:**
- `HumeAI/tada` GitHub repo has **two** license files. `gh api repos/HumeAI/tada` → `license.spdx_id = **NOASSERTION**` (GitHub could not classify the pair).
- `LICENSE` = **Llama 3.2 Community License Agreement** (weights) — https://raw.githubusercontent.com/HumeAI/tada/main/LICENSE
- `LICENSE_CODE` = **MIT** ("Copyright 2026 Hume AI", verbatim MIT text) — https://raw.githubusercontent.com/HumeAI/tada/main/LICENSE_CODE
- The repo README states this explicitly: *"This repository contains both model weights and code, which are licensed separately: **Model weights** are licensed under the Llama 3.2 Community License Agreement; **Code** in this repository is licensed under the MIT License. You must comply with the terms of the Llama 3.2 license when using the models."* (https://raw.githubusercontent.com/HumeAI/tada/main/README.md, License section)
- PyPI `hume-tada` **metadata is self-contradictory**: `info.license` = "LLAMA 3.2 COMMUNITY LICENSE AGREEMENT" while the classifier says `License :: OSI Approved :: MIT License`. (https://pypi.org/pypi/hume-tada/json) → **flag: misleading metadata**, the classifier is wrong.
- Note the Hub card also advertises an "MIT" badge in the GitHub README HTML (`img.shields.io/badge/License-MIT-green`) even though the LICENSE file is Llama 3.2 — a cosmetic mislabel worth flagging.

### Is it non-commercial / research-only / a "HumeAI Community License"?

**No — none of those three.** It is **Meta's Llama 3.2 Community License**, a *custom, restricted, non-OSI* license that **does** permit commercial use, subject to conditions. There is no separate "HumeAI Community License". Concrete obligations a commercial product inherits (§ numbers as in the LICENSE file):

1. **§1.b.i — Attribution by display.** If you distribute the Llama Materials (or derivatives, or a product/service containing them) you must (A) provide a copy of the Agreement, and (B) **prominently display "Built with Llama"** on a related website, user interface, blogpost, about page, or product documentation. If you use it (or its outputs) to create/train/fine-tune/improve a distributed model, **the model name must begin with "Llama"**.
2. **§1.b.iii — Notice file.** All distributed copies must retain, in a **"Notice" text file**, the notice: *"Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved."*
3. **§1.b.iv — Acceptable Use Policy.** Use must comply with applicable law (incl. trade compliance) and the **Llama 3.2 Acceptable Use Policy** at `https://www.llama.com/llama3_2/use-policy`, incorporated by reference.
4. **§2 — Scale trigger.** If MAU of the licensee's products/services/affiliates exceeded **700 million** in the preceding calendar month, you must request a separate license from Meta and may not exercise the rights otherwise.
5. **§5.a / §5.c** — no trademark license beyond the "Llama" mark as required by 1.b.i; all licenses terminate if you sue Meta over the materials.
6. **§6 / §7** — Meta may terminate on breach (must then delete and cease use); governed by **California law**, exclusive California jurisdiction.
7. **Practical access gate:** `HumeAI/tada` README says *"TADA models are built on Meta Llama 3.2. You must request access to the Llama models before using TADA"* — the configs hardcode gated `meta-llama/Llama-3.2-1B`. Voicebox had to work around this: `docs/content/docs/developer/tts-engines.mdx` L525 documents "TADA hardcodes `meta-llama/Llama-3.2-1B` in both its `AlignerConfig` and `TadaConfig`. This silently fails without HF authentication." and L531 substitutes `UNGATED_TOKENIZER = "unsloth/Llama-3.2-1B"`. **Voicebox ships an un-gating workaround for a gated/license-gated upstream repo.**

**Verdict: commercial use is allowed but conditional and restricted** — it is the *only* engine in the shipped set whose weights are under a non-OSI, attribution+AUP-bound license. Not research-only, not non-commercial. Cite: https://huggingface.co/HumeAI/tada-1b/blob/main/LICENSE, https://github.com/HumeAI/tada/blob/main/LICENSE, https://developer.meta.com/ai/llama3_2/license/

---

## 6. Kokoro (hexgrad)

| Item | Observed value | Source |
|---|---|---|
| `hexgrad/Kokoro-82M` weights | **apache-2.0** (`cardData.license`; tag `license:apache-2.0`), gated=false, lastModified 2025-04-10 | https://huggingface.co/api/models/hexgrad/Kokoro-82M |
| Model card statement | "With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects." | https://huggingface.co/hexgrad/Kokoro-82M/raw/main/README.md |
| Code | `hexgrad/kokoro` → `license.spdx_id = **Apache-2.0**` | `gh api repos/hexgrad/kokoro` |
| G2P code | `hexgrad/misaki` → `license.spdx_id = **Apache-2.0**`; PyPI `misaki` = Apache | `gh api repos/hexgrad/misaki`, https://pypi.org/pypi/misaki/json |
| PyPI `kokoro` | **Apache License** (classifier: OSI Approved :: Apache Software License) | https://pypi.org/pypi/kokoro/json |
| Voicebox reference | `backend/backends/kokoro_backend.py` L35: `KOKORO_HF_REPO = "hexgrad/Kokoro-82M"`; L5 comment "Apache 2.0 license" | raw.githubusercontent.com |

### Voicepacks

- The voicepacks ship **inside the Apache-2.0 repo** as individual files: `voices/af_alloy.pt`, `voices/af_heart.pt`, `voices/am_adam.pt`, `voices/bf_alice.pt`, … (~54 files incl. `af_*`, `am_*`, `bf_*`, `bm_*`, `ef_*`, `em_*`, `ff_*`, `hf_*`, `hm_*`, `if_*`, `im_*`, `jf_*`, `jm_*`, `pf_*`, `pm_*`, `zf_*`, `zm_*`). File list from https://huggingface.co/api/models/hexgrad/Kokoro-82M (`siblings[].rfilename`).
- **No per-voice license column exists.** `VOICES.md` (https://huggingface.co/hexgrad/Kokoro-82M/raw/main/VOICES.md) has columns only: Name | Traits | Target Quality | Training Duration | Overall Grade | SHA256. There is **no** per-voice license field.
- **Per-voice license issue: OPEN AND UNANSWERED.** `hexgrad/kokoro` issue **#219, "voice License using Kokoro native language pack"**, opened **2025-06-06**, state **open**, **0 comments**:
  > "Does the voice generated using the Kokoro native language pack only need to comply with the Apache 2.0 license? Or is there a different license that applies specifically to the voice output?"
  Sources: `gh api repos/hexgrad/kokoro/issues/219` (`state: open`, `comments: 0`), https://github.com/hexgrad/kokoro/issues/219
  → **Output licensing per voicepack = 未核实 / unverified.** The only positive statement available is the maintainer's blanket "Apache-licensed weights" wording on the model card, which does not address output audio.
- **Training-data licenses are disclosed and include attribution obligations:** Kokoro's card says it was trained "exclusively on permissive/non-copyrighted audio data and IPA phoneme labels" (public domain / Apache / MIT / **synthetic from *closed* TTS providers**), plus a "Creative Commons Attribution" table:
  - Koniwa `tnc` — <1h — **CC BY 3.0** — added v0.19 / 22 Nov 2024
  - SIWIS — <11h — **CC BY 4.0** — added v0.19 / 22 Nov 2024
  Sources: https://huggingface.co/hexgrad/Kokoro-82M/raw/main/README.md (Training Details + Creative Commons Attribution sections)
- Voicebox's own doc repeats the license claim: "**Repository:** `hexgrad/Kokoro-82M` · Apache 2.0 licensed" (`docs/content/docs/overview/preset-voices.mdx`).
- **Commercial use: YES** per the model card ("production environments", "deployed in numerous projects and commercial APIs"), **subject to the unresolved per-voice output question above**.

---

## 7. Whisper (STT)

Exact repos Voicebox requests, from `backend/backends/__init__.py` L39-45:

```python
WHISPER_HF_REPOS = {
    "base":  "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium":"openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "turbo": "openai/whisper-large-v3-turbo",
}
```
(consumed in `backend/backends/pytorch_backend.py` L266/L295; the MLX path goes through `mlx-audio`'s own STT loader — `backend/requirements-mlx.txt`)

| Repo | Weights license | lastModified | Source URL |
|---|---|---|---|
| `openai/whisper-base` | **apache-2.0** | 2024-02-29 | https://huggingface.co/api/models/openai/whisper-base |
| `openai/whisper-small` | **apache-2.0** | 2024-02-29 | https://huggingface.co/api/models/openai/whisper-small |
| `openai/whisper-medium` | **apache-2.0** | 2024-02-29 | https://huggingface.co/api/models/openai/whisper-medium |
| `openai/whisper-large-v3` | **apache-2.0** | 2024-08-12 | https://huggingface.co/api/models/openai/whisper-large-v3 |
| `openai/whisper-large-v3-turbo` | **mit** ⚠️ *(differs from the rest)* | 2024-10-04 | https://huggingface.co/api/models/openai/whisper-large-v3-turbo |
| `openai/whisper-tiny` (not used, for reference) | **apache-2.0** | 2024-02-29 | https://huggingface.co/api/models/openai/whisper-tiny |
| **Code**: `openai/whisper` | **MIT** (`license.spdx_id`) | pushed 2026-08-31 | `gh api repos/openai/whisper` |

- So: **code = MIT, weights = Apache-2.0 except turbo = MIT.** Both allow commercial use.
- **MLX conversion layer unverified:** `mlx-community/whisper-large-v3-turbo` has **no `cardData.license` and no LICENSE file** (siblings are only `.gitattributes`, `README.md`, `config.json`, `weights.safetensors`) → **未核实 / unverified** at that layer. By contrast `mlx-community/whisper-large-v3-mlx` declares **mit**. Source: https://huggingface.co/api/models/mlx-community/whisper-large-v3-turbo
- **Commercial use: YES.**

---

## 8. Qwen3 LLM (bundled local personality/refinement LLM)

Voicebox repo IDs (`backend/backends/qwen_llm_backend.py` L26-36): PyTorch `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`; MLX `mlx-community/Qwen3-0.6B-4bit`, `mlx-community/Qwen3-1.7B-4bit`, `mlx-community/Qwen3-4B-4bit`.

| Repo | Weights license | license_link | lastModified | Source URL |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | **apache-2.0** | `.../Qwen3-0.6B/blob/main/LICENSE` | 2025-07-26 | https://huggingface.co/api/models/Qwen/Qwen3-0.6B |
| `Qwen/Qwen3-1.7B` | **apache-2.0** | `.../Qwen3-1.7B/blob/main/LICENSE` | 2025-07-26 | https://huggingface.co/api/models/Qwen/Qwen3-1.7B |
| `Qwen/Qwen3-4B` | **apache-2.0** | `.../Qwen3-4B/blob/main/LICENSE` | 2025-07-26 | https://huggingface.co/api/models/Qwen/Qwen3-4B |
| `mlx-community/Qwen3-0.6B-4bit` | **apache-2.0** | — | 2025-04-28 | https://huggingface.co/api/models/mlx-community/Qwen3-0.6B-4bit |
| `mlx-community/Qwen3-4B-4bit` | **apache-2.0** | — | 2025-04-28 | https://huggingface.co/api/models/mlx-community/Qwen3-4B-4bit |

- **Commercial use: YES.** Apache-2.0, confirmed on both PyTorch and MLX checkpoints.

---

## 9. Does Voicebox itself watermark generated audio?

**No.** Voicebox adds **no watermark of its own**. Grep-level evidence over the default branch via GitHub code search:

| Query | Hits | Paths |
|---|---|---|
| `watermark` | **3** | `CHANGELOG.md`, `backend/build_binary.py`, `docs/content/docs/developer/tts-engines.mdx` |
| `SynthID` | **0** | — |
| `provenance` | **4** | `README.md`, `CHANGELOG.md`, `landing/src/components/TokenStats.tsx`, `app/src/components/History/HistoryTable.tsx` |

- All three `watermark` hits are about **bundling Chatterbox's PerTh model**, not about Voicebox watermarking anything:
  - `backend/build_binary.py` L227-230: `# perth ships pretrained watermark model files (hparams.yaml, .pth.tar) in perth/perth_net/pretrained/ — needed by chatterbox at runtime` → `--collect-all perth`
  - `CHANGELOG.md` L403: "Collect all `perth` files — bundles the pretrained watermark model (`hparams.yaml`, `.pth.tar`) needed by Chatterbox at runtime"
  - `docs/content/docs/developer/tts-engines.mdx` L430: "Chatterbox | `FileNotFoundError` for watermark model | `perth` ships pretrained model files … | `--collect-all perth`"
- All four `provenance` hits refer to the **in-app version-lineage feature** ("Generation Versions … provenance tracking" in README: Original / Effects versions / Takes / Source tracking), i.e. database lineage between generated versions — **not** audio provenance marking.
- **Consequence:** output is **unwatermarked** for Qwen3-TTS (base/MLX), Qwen CustomVoice, LuxTTS, Kokoro, and TADA; output is **PerTh-watermarked** for Chatterbox Multilingual and Chatterbox Turbo, because Voicebox ships `resemble-perth>=1.0.1` (`backend/requirements.txt`) and its `chatterbox_backend.py` / `chatterbox_turbo_backend.py` contain **no `perth`/`watermark` reference at all** — i.e. the library default (watermark always on) is left in place.
- **Disclosure gap:** there is **no user-facing documentation** stating that Chatterbox output is watermarked. The only in-repo mention is buried in a developer/PyInstaller troubleshooting doc and the changelog. Nothing in `docs/content/docs/overview/*` (generating-speech, preset-voices, introduction, etc.), nothing in the README feature list.

---

## 10. Python dependencies — actual files and copyleft / non-commercial review

### The real dependency files

| Path | HTTP | Notes |
|---|---|---|
| `backend/requirements.txt` | **200** | the real list (below) |
| `backend/requirements-mlx.txt` | **200** | `mlx`, `miniaudio`; `mlx-audio==0.4.1` installed `--no-deps` |
| `backend/requirements-rocm.txt` | **200** | `--extra-index-url repo.radeon.com`, `torch==2.9.1+rocm7.2.1`, `torchaudio`, `torchvision` |
| `requirements.txt` (root) | 200 | stale/minimal: `uvicorn fastapi sqlalchemy torch torchvision soundfile librosa python-multipart huggingface_hub` |
| `backend/pyproject.toml` | 200 | **declares NO `[project] dependencies` at all** (only name/version/requires-python + ruff/pytest config) → packaging-metadata gap |
| `backend/requirements.txt` | **200** | the task's suggested path **is correct**. First attempt in this session returned HTTP `000` (transient connection failure); an immediate re-fetch returned `200` with the full file below. |
| `backend/tests/test_rocm_requirements.py` | exists | test asserting ROCm pins |

### `backend/requirements.txt` (verbatim, grouped as in the file)

```
fastapi>=0.109.0 / uvicorn[standard]>=0.27.0 / pydantic>=2.5.0
sqlalchemy>=2.0.0 / alembic>=1.13.0
torch>=2.2.0 / transformers>=4.36.0,<=4.57.6 / accelerate>=0.26.0
huggingface_hub>=0.20.0 / qwen-tts>=0.0.5
--find-links https://k2-fsa.github.io/icefall/piper_phonemize.html
linacodec @ git+https://github.com/ysharma3501/LinaCodec.git
Zipvoice  @ git+https://github.com/ysharma3501/LuxTTS.git
conformer>=0.3.2 / diffusers>=0.29.0 / omegaconf / pykakasi
resemble-perth>=1.0.1 / s3tokenizer / spacy-pkuseg / pyloudnorm
torchaudio
kokoro>=0.9.4 / misaki[en,ja,zh]>=0.9.4
en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
unidic-lite>=1.0.8
audioop-lts>=0.2.1; python_version >= "3.13"
librosa>=0.10.0 / soundfile>=0.12.0 / numpy>=1.24.0,<2.0
numba>=0.60.0,<0.61.0 / pedalboard>=0.9.0
httpx>=0.27.0
fastmcp>=3.0,<4.0 / sse-starlette>=2.0
python-multipart>=0.0.6 / Pillow>=10.0.0
```
Plus the two engines installed `--no-deps` by the setup script (per the file's comments): `chatterbox-tts` (pins numpy<1.26 / torch==2.6, incompatible with py3.12+) and `hume-tada` (pins torch>=2.7,<2.8; `descript-audio-codec` deliberately not installed).

### ⚠️ Copyleft / licensing problems found

| Package | License observed | Where | Impact |
|---|---|---|---|
| **`pedalboard`** (Spotify) | **GPL-3.0** — PyPI classifier `GNU General Public License v3 (GPLv3)`, license text = GNU GPL; GitHub `spotify/pedalboard` → `license.spdx_id = "GPL-3.0"` | https://pypi.org/pypi/pedalboard/json · `gh api repos/spotify/pedalboard` | **Strongest copyleft exposure.** It powers the 8 shipped post-processing effects (pitch/reverb/delay/chorus/compressor/gain/HPF/LPF) and is bundled into the app. GPL-3.0 inside a distributed commercial product is a real compliance problem, not a theoretical one. |
| **`pykakasi`** | **GPL-3.0-or-later** — classifier `GPLv3+`; source moved to **Codeberg** (`https://codeberg.org/miurahr/pykakasi`); `gh api repos/rspeer/pykakasi` and `repos/pykakasi/pykakasi` = **404** | https://pypi.org/pypi/pykakasi/json | Second GPL-3.0 dep, pulled in by the Chatterbox multilingual (Japanese) path. Same copyleft concern. |
| **`espeak-ng` data shipped via `piper-phonemize`** | **`espeak-ng/espeak-ng` → `license.spdx_id = "GPL-3.0"`**; `piper-phonemize` itself is **MIT** (`rhasspy/piper-phonemize`) | `gh api repos/espeak-ng/espeak-ng`, `gh api repos/rhasspy/piper-phonemize` | Voicebox explicitly **bundles** `espeak-ng-data/` for the LuxTTS phonemizer (`docs/.../tts-engines.mdx` L428: "`espeak-ng-data` not found … `--collect-all piper_phonemize` + set `ESPEAK_DATA_PATH`"). GPL-3.0 data/library in the frozen distribution. |
| **`linacodec`** | **no license detected** (`gh api repos/ysharma3501/LinaCodec --jq .license` → `{"license":null}`) | `gh api repos/ysharma3501/LinaCodec` | Git-only dependency of the LuxTTS/ZipVoice path. **未核实 / unverified** — shipping a dependency with no license file means no granted rights at all. |
| `hume-tada` (PyPI) | metadata self-contradiction: `info.license` = "LLAMA 3.2 COMMUNITY LICENSE AGREEMENT" **vs** classifier `OSI Approved :: MIT License` | https://pypi.org/pypi/hume-tada/json | Misleading PyPI metadata; the real weights license is Llama 3.2 (see §5). |

### Verified-permissive dependencies (no copyleft found)

`torch`/`torchaudio`/`torchvision` (BSD), `transformers`/`accelerate`/`huggingface_hub`/`diffusers`/`s3tokenizer` (Apache-2.0), `qwen-tts` **Apache-2.0**, `chatterbox-tts` **MIT**, `resemble-perth` **MIT**, `Perth` repo **MIT**, `conformer` **MIT**, `spacy-pkuseg` **MIT**, `kokoro` **Apache**, `misaki` **Apache**, `unidic-lite` **MIT**, `mlx` **MIT**, `mlx-audio` **MIT**, `omegaconf` **BSD**, `pyloudnorm` **MIT**, `descript-audio-codec` **MIT**, `fastmcp` **Apache-2.0**, `alembic` **MIT**, `librosa` ISC, `soundfile` BSD, `numpy` BSD, `numba` BSD, `Pillow` HPND, `fastapi` MIT, `uvicorn` BSD, `sqlalchemy` MIT, `pydantic` MIT, `ZipVoice` (`k2-fsa/ZipVoice`) **Apache-2.0**, `LuxTTS` (`ysharma3501/LuxTTS`) **Apache-2.0**, `StyleTTS2` (`yl4579/StyleTTS2`) **MIT**.

### Non-commercial engines Voicebox deliberately kept OUT

From `docs/PROJECT_STATUS.md` (candidate tracker), Voicebox's own criterion is *"PyPI + Apache/MIT licensing preferred"*:

- **Fish Speech / Fish Audio S2** — "Candidate — license TBD"; watch list: *"benchmark leader but research/non-commercial license — same blocker as Fish Speech"*
- **XTTS-v2** (Coqui) — "Candidate — **CPML license likely blocker**"
- **NeuTTS Nano** (neuphonic) — "*Nano has a separate NeuTTS Open License — split needs review*" (Air is Apache-2.0)
- **CosyVoice2-0.5B** — **Abandoned** (quality, not license)
- Top candidate **MOSS-TTS-Nano** — "**Top candidate** — Apache 2.0"; also tracked: dots.tts (Apache-2.0 code+checkpoints), FireRedTTS-2 (Apache-2.0), Maya1 (Apache-2.0), Pocket TTS (MIT).

→ **None of the seven shipped engines has non-commercial weights.** The single restricted one is **TADA (Llama 3.2 Community License)** — restricted, but commercial-capable.

---

## 11. Cross-check: what Voicebox itself claims about licenses in-repo

| Claim | Where | Verified? |
|---|---|---|
| "Kokoro 82M by hexgrad. … Apache 2.0 licensed." | `app/src/components/ServerSettings/ModelManagement.tsx` L71 | ✅ matches HF cardData |
| "**Repository:** `hexgrad/Kokoro-82M` · Apache 2.0 licensed" | `docs/content/docs/overview/preset-voices.mdx` | ✅ |
| "TADA 1B — English speech-language model built on Llama 3.2 1B" / "TADA 3B Multilingual — built on Llama 3.2 3B" | `ModelManagement.tsx` L67-69 | ✅ names the base model, but **does not state the Llama 3.2 Community License**, nor the "Built with Llama" / Notice-file / AUP duties |
| "LuxTTS | Lightweight (~1GB VRAM), 48kHz output, 150x realtime on CPU" (README table) | README | ✅ matches model card |
| README feature blurb: "50+ curated preset voices via Kokoro and Qwen CustomVoice" | README | ⚠️ **misleading** — 50 = Kokoro, 9 = Qwen CustomVoice (per the engine table and preset-voices.mdx) |
| Any statement of the Chatterbox PerTh watermark for end users | — | ❌ **absent** outside developer/build docs |
| NOTICE / third-party license inventory | — | ❌ **absent** (repo has only `LICENSE`) |

---

## 12. Riskiest findings (ranked)

1. **TADA weights are Llama 3.2 Community License, NOT Apache-2.0/MIT** (`license: llama3.2`). Commercial use is permitted but conditional: prominent **"Built with Llama"** display, a required **"Notice"** text-file attribution, Llama-prefixed naming for derived models, Meta's **Acceptable Use Policy**, and a 700M-MAU re-licensing trigger. Voicebox ships TADA as a normal engine and its UI only says "built on Llama 3.2".
2. **`pedalboard` is GPL-3.0** and powers the shipped post-processing effects → copyleft in a distributed commercial product.
3. **`espeak-ng` (GPL-3.0) data is explicitly bundled** for LuxTTS via `piper-phonemize` (MIT wrapper, GPL payload), **plus `pykakasi` GPL-3.0-or-later** on the Chatterbox path → two more copyleft components in the frozen builds.
4. **Qwen CustomVoice preset timbres: no license statement anywhere beyond Apache-2.0 weights; no consent/likeness policy.** 未核实 whether a commercial product may ship those 9 named timbres as product voices. The license neither grants nor forbids it.
5. **Kokoro per-voice output licensing is an open, unanswered question** (hexgrad/kokoro #219, open since 2025-06-06, 0 comments). Model card says "Apache-licensed weights"; nothing addresses voicepack output.
6. **Chatterbox (both variants) PerTh-watermarks every output**, and Voicebox leaves it enabled while never telling end users. Not a license violation (MIT), but a product/provenance disclosure gap that matters for a commercial audio pipeline.
7. **No NOTICE / third-party-license inventory in Voicebox at all**, while it bundles GPL-3.0 components and downloads Llama-3.2-licensed weights. Voicebox's per-model license display is fetched live from HF and relies entirely on upstream metadata being correct.
8. **Unverified layers:** `mlx-community/whisper-large-v3-turbo` (no license metadata, no LICENSE file), `ysharma3501/LinaCodec` (no license at all), `YatharthS/LuxTTS` voice-clone output terms, `hume-tada` PyPI metadata contradicting itself.
9. **Minor premised-in-task corrections:** GitHub `YatharthS/LuxTTS` = 404 (real repo is `ysharma3501/LuxTTS`); "50+ preset voices via Kokoro and Qwen CustomVoice" is 50 + 9, not 50+ from each; Hume's TADA is not a bespoke "HumeAI Community License" but verbatim Meta Llama 3.2.
