# 开源 TTS / 声音克隆项目的 Apple Silicon（M1–M5）支持现状

**报告日期：2026-09-16**（所有页面检索日；页面自带日期者单独标注为「信息日期」）
**方法论**：所有结论均来自一手来源（GitHub issue/PR 正文、README、LICENSE 原文、HF model card / HF API、官方文档站）。二手来源（博客）已单独标记。无法从一手来源确认的写 **未核实**。

**RTF 定义**：RTF = 生成耗时 ÷ 音频时长。**越低越快**，< 1.0 为快于实时（与 soniqo 文档所用约定一致）。

---

## 0. 关键横向发现（先看这三条）

1. **`soniqo/speech-swift` 是本批次最重要的非 PyTorch macOS 路线**。它是原生 Swift 包，后端只有 MLX（Metal GPU）与 CoreML（Neural Engine），覆盖本批次中的 **CosyVoice3 / IndexTTS2 / F5-TTS / Kokoro / CSM / Fish Audio S2 Pro**，另有 Chatterbox、VibeVoice、Higgs 等。文档日期为 2026（含 M5 Pro 实测），检索日 2026-09-16。
   - https://github.com/soniqo/speech-swift
   - https://soniqo.audio/architecture
2. **`k2-fsa/sherpa-onnx` 覆盖 Piper(VITS) / Kokoro / Matcha / KittenTTS / Supertonic**，ONNX Runtime + 官方 CoreML EP 可用，是 Pi​per、Kokoro 最稳的 macOS 非 PyTorch 路线。https://k2-fsa.github.io/sherpa/onnx/tts/all/English/index.html
3. **本批次有一批模型在 MPS 上不是「慢」而是「直接崩」**，错因高度集中在同一个 PyTorch MPS 算子限制：
   `NotImplementedError: Output channels > 65536 not supported at the MPS device.`
   已确认命中的模型：**IndexTTS-2、Spark-TTS、MeloTTS（旁证）**；StyleTTS2/XTTS/Fish-Speech 是另外的错因。

---

## 1. IndexTTS-2（index-tts / IndexTeam）

- **仓库**：https://github.com/index-tts/index-tts ｜ 权重 https://huggingface.co/IndexTeam/IndexTTS-2
- **(a) 能否跑 / 后端**：能跑，但 **MPS 上有硬性算子缺失**。
  - 官方 CLI 提供 MPS 支持：PR #78「Add a new Command-Line Interface and support for mps device (Apple Silicon)」**已合并**（信息日期 2025-04-11，merge commit `f07a9032`）https://github.com/index-tts/index-tts/pull/78
  - 但 macOS 上的实际可用后端是 **CPU**（`--device cpu`）或 **MLX**。
- **(b) 崩溃证据（原文引用）** — Issue #193「Fail on MacOS M3, 65536 not supported at the MPS device」（信息日期 2025-06）https://github.com/index-tts/index-tts/issues/193
  - 环境：`Apple M3 Max`，`PhysMem: 97G used`，`indextts 0.1.4`
  - 报错原文：
    ```
    >> start inference...
    NotImplementedError: Output channels > 65536 not supported at the MPS device.
    As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`
    to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
    ```
  - 作者自述的绕行与实测：「add option `--device "cpu"` to indextts will make it work, but very slow」
  - 线程内版本差异（很重要，说明是 PyTorch 版本相关）：一位用户称「Upgrade torch and torchaudio fixed the problem on my Mac M4」（`pip install torch==2.7.0 torchaudio`），但另一位回复「My mac is M1 max, and torch==2.7.0 is still have this problem」。→ **结论：M4 + torch 2.7 可能修复；M1/M3 不一定**。
  - 维护者回复（#81 comment）：「Please upgrade your macOS to 15.1 or higher」——但报告者已在 macOS 15.5 上仍然失败。
  - 另有第三方博客记录了 torchaudio.save 相关的「static/choppy」音频问题（**二手来源，未经我复核**）：https://exocreate.online/blog/indextts2-apple-silicon-tts-fix
- **(c) 实测数字**
  | 硬件 | 后端 | RTF | 来源 |
  |---|---|---|---|
  | Apple M3 Max（97G used） | **CPU**（`--device cpu`） | **3.5343**（generated 2.01s 音频 / total 7.09s） | Issue #193，2025-06 |
  | M3 Max 128GB 统一内存 | MLX fp32 | zh short 1.127 / zh long 1.232 / en short 1.157 / en long 1.193 | HF card vanch007，2026-08-31 |
  | 同上 | MLX fp16 | 1.538 / 1.584 / 1.462 / 1.511 | 同上 |
  | 同上 | MLX 8bit | **0.966 / 1.035 / 0.914 / 0.956**（最快路线） | 同上 |
  | 同上 | 优化过的 PyTorch MPS | 1.446 / 1.699 / 2.192 / 1.783 | 同上 |
  | Apple M5 Max | MLX fp32, batch | zh 端到端 **RTF ≈ 1.9** | HF card mlx-community/IndexTTS-2-MLX，2026-09-02 |
  来源：https://huggingface.co/vanch007/mlx-indextts2-standard-fp16 ｜ https://huggingface.co/mlx-community/IndexTTS-2-MLX
- **(d) 非 PyTorch macOS 路线：有，且是 MLX（本批次里最完整的一个）**
  - `mlx-community/IndexTTS-2-MLX`（MLX safetensors fp32）https://huggingface.co/mlx-community/IndexTTS-2-MLX
  - 运行时 `Jup33Q/mlx-indextts`（该 card 的安装命令用的是 `hf download Jup33QE/IndexTTS-2-MLX`）https://github.com/Jup33Q/mlx-indextts
  - `vanch007/mlx-indextts2-standard-fp16`（fp16，~2.0GB，运行时 `solar2ain/mlx-indextts`）— card 明确动机是「stable Vietnamese and multilingual TTS on an M3 Max Mac **without PyTorch**」和「**MPS memory crashes**」https://huggingface.co/vanch007/mlx-indextts2-standard-fp16 ｜ https://github.com/solar2ain/mlx-indextts
  - **soniqo/speech-swift**：模块 `IndexTTS2TTS`，默认 bundle `aufklarer/IndexTTS2-MLX-fp16`，1.5B-class，MLX，流式不支持。文档自评「Upstream PyTorch numerical parity, subjective listening quality… still need broader checks before treating this port as benchmark-grade」https://github.com/soniqo/speech-swift/blob/main/docs/models/indextts2.md
- **(e) 许可证：⚠️ 不是 Apache-2.0，是 bilibili 自定义许可**
  - 仓库 `LICENSE` 原文首行：**「bilibili Model Use License Agreement」**（`https://raw.githubusercontent.com/index-tts/index-tts/main/LICENSE`）
  - README 也写明：「This project is released under the [bilibili Model Use License Agreement](LICENSE).」并注明商用联系 `indexspeech@bilibili.com`
  - **坑点**：`pip show indextts` 的 metadata 把 License 标成 `Apache-2.0`（见 Issue #193 的 `pip3.11 show indextts` 输出），**与实际 LICENSE 文件不符**。以 LICENSE 文件为准。
  - 关键条款（2.2）：「either (i) … more than 100 million monthly active users in the immediately preceding calendar month, or (ii) … annual revenue … exceeded RMB 1 billion, You must request a separated license from us」
  - 4.4(c)：禁止用该模型/衍生作品去改进其他 AI 模型（除 IndexTTS2 自身、其衍生作品、或非商用 AI 模型）
  - 管辖法：中华人民共和国法律；争议提交上海仲裁委员会
  - MLX 转换权重（`mlx-community/IndexTTS-2-MLX`）card 标注 `license: other`，并写明「IndexTTS-2 weights are released under the upstream IndexTTS license… apply to the converted weights as well」。

---

## 2. CosyVoice 2 / CosyVoice 3（FunAudioLLM / QwenAudio）

- **仓库**：https://github.com/FunAudioLLM/CosyVoice → **已迁移**，PR/issue 现落在 https://github.com/QwenAudio/CosyVoice
- **(a) 能否跑 / 后端**：CPU 可跑；**MPS 属于「能跑但更慢」+ 官方长期不合并**的状态。
  - 维护者原话（Issue #134，信息日期 2024-07-14，open，2025-11-11 仍有更新）：
    > 「There used to be PR using MPS, but it is **slower than cpu on mac**, maybe it is not mature yet」
    https://github.com/FunAudioLLM/CosyVoice/issues/134
  - 同 issue 中社区给了一个 macOS 支持的 fork：https://github.com/lonelygo/CosyVoice （但同 thread 有人报 `ImportError: cannot import name 'cached_download' from 'huggingface_hub'` 跑不起来）
  - MPS PR 历史：
    - PR #1129「feat: Add limited support for MPS devices」— **closed**（信息日期 2025-04-01 创建，2026-04-07 更新）+33 −8 in 3 files https://github.com/QwenAudio/CosyVoice/pull/1129
    - PR #1869「feat: add Apple Silicon (MPS) support for macOS ARM64」— **open**（信息日期 2026-04-05 创建，2026-05-18 更新）+220 −35 in 10 files，merge commit `1cb52d5b` https://github.com/QwenAudio/CosyVoice/pull/1869
  - Issue #1767「mac gpu支持」— 请求 `AutoModel` 加 `device` 参数以支持 `torch.device("mps")`，**因 stale 被自动关闭**（信息日期 2026-01-04 创建，2026-02-18 closed）https://github.com/QwenAudio/CosyVoice/issues/1767
  - 附带报错（原文）：「Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.」
- **(c) 实测数字（M 系列）**
  | 硬件 | 后端 | 数字 | 来源 |
  |---|---|---|---|
  | Apple M2 Max | MLX（soniqo CosyVoice3） | **RTF ≈ 0.5**（快于实时） | https://soniqo.audio/guides/cosyvoice |
  | Apple M3 Pro | CPU（原版 PyTorch） | 无数字，用户报「only CPU is used」 | Issue #134 |
  | iPhone 16 Pro | CoreML | 见 soniqo iPhone 基准表（ASR/Kokoro 等，CosyVoice 未单列） | https://github.com/soniqo/speech-swift/blob/main/docs/benchmarks/ios-coreml.md |
  - MPS vs CPU 的量化提升：**未核实**（Issue #134 里有人直接问「请问，rtf指标有多少提升？相比用CPU跑」，无回答）
- **(d) 非 PyTorch macOS 路线：有，两条，都很实**
  - **FluidInference / FluidAudio**（Swift 包，CoreML）：HF `FluidInference/CosyVoice3-0.5B-coreml`，**Apache-2.0**。四个 stage 分别放在 CPU+ANE（LLM prefill / decode）、CPU+GPU（Flow）、CPU+ANE（HiFT）。文档原文关键点：「Flow-N250-fp16 | CPU + GPU | … (pure CPU overflows fused LayerNorm → **NaN**; ANE refuses to compile; GPU path uses fp32 accumulators internally and is stable)」。磁盘 ~6.6GB。得名 https://huggingface.co/FluidInference/CosyVoice3-0.5B-coreml ｜ https://github.com/FluidInference/FluidAudio
    - 相关 PR：#582「fix(tts): guard direct Float16 reads with `#if arch(arm64)` (CosyVoice3, StyleTTS2)」 https://github.com/FluidInference/FluidAudio/pull/582
  - **soniqo/speech-swift**：模块 `CosyVoiceTTS`，MLX，0.5B，9 语言，流式 + voice cloning + emotion tags；MLX bundle ~1.2GB（4-bit LLM）。有「Batch-doubled CFG — CosyVoice3 DiT halves flow matching passes by batching conditional + unconditional together」等优化。https://github.com/soniqo/speech-swift/blob/main/docs/models/cosyvoice-tts.md ｜ https://github.com/soniqo/speech-swift/blob/main/docs/models/cosyvoice-tts.md
- **(e) 许可证**：**Apache-2.0**（`cosyvoice/LICENSE` 原文：`Apache License Version 2.0, January 2004`）。https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/LICENSE

---

## 3. GPT-SoVITS（RVC-Boss）

- **仓库**：https://github.com/RVC-Boss/GPT-SoVITS
- **(a) 能否跑 / 后端**：**能跑，但官方主动把 macOS 降级到 CPU**。
  - README macOS 段落原文加粗警告：
    > 「**Note: The models trained with GPUs on Macs result in significantly lower quality compared to those trained on other devices, so we are temporarily using CPUs instead.**」
  - 安装脚本仍提供 MPS 选项：`bash install.sh --device <MPS|CPU> --source <HF|HF-Mirror|ModelScope>`
  - 官方兼容矩阵里 Apple silicon 是受支持组合：`Python 3.9 + PyTorch 2.5.1 → Apple silicon`、`Python 3.11 + PyTorch 2.7.0 → Apple silicon`
- **(b) 崩溃/无效证据**
  - PR #183「Add MPS Backend Support for macOS」（作者 Lion-Wu，**merged**，+158 −70，9 commits，信息日期 2024-01-24）https://github.com/RVC-Boss/GPT-SoVITS/pull/183
  - Issue #2612「How to use mps on macOS when inference model?」（**open**，信息日期 2025-09-21 创建，2025-09-27 更新）https://github.com/RVC-Boss/GPT-SoVITS/issues/2612 — 原文关键引用：
    > 「Although I've used mps parameter when I run install.sh, but the log shows the device still is CPU but not mps, then I read the install.sh, find that **the treatment for mps and cpu is same!** It maybe means you don't have support for mps?」
    > 「only 111.21it/s, **My GPU is Apple M4 Max with 40CU**, So it must runs on cpu」
  - → 即：M4 Max 40CU 上 MPS 选项实际不生效，退化为 CPU。**这是「MPS 名义存在但实际不可用」的典型**。
  - 另有社区 CPU 优化 fork：https://github.com/baicai-1145/GPT-SoVITS-CPUFast
- **(c) 实测数字（M 系列）** — README 正文直接给出：
  | 硬件 | 数字 | 来源 |
  |---|---|---|
  | **M4 CPU** | **0.526** | README 速度对比行：`0.028 tested in 4060Ti, 0.014 tested in 4090 (1400words~=4min, inference time is 3.36s), 0.526 in M4 CPU` |
  | M4 Max 40CU（MPS 选项） | 111.21 it/s（作者判定实际跑在 CPU） | Issue #2612 |
  - 注：`0.526` 与 `0.028`/`0.014` 是同口径对比值（README 未明写单位，但按同段语境为 RTF 量级）。
- **(d) 非 PyTorch macOS 路线**：**未核实**（未找到可靠的 ONNX/CoreML/MLX 官方或主流移植）。
- **(e) 许可证**：**MIT**（`LICENSE` 原文：`MIT License Copyright (c) 2024 RVC-Boss`）。https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/LICENSE

---

## 4. F5-TTS（SWivid）

- **仓库**：https://github.com/SWivid/F5-TTS
- **(a) 能否跑 / 后端**：**能跑，PyTorch MPS 可用**，但需要 `PYTORCH_ENABLE_MPS_FALLBACK=1`。
- **(b) 崩溃证据（原文引用）** — Issue #443「The operator 'aten::unfold_backward' is not currently implemented for the MPS device」（作者 @cocktailpeanut，COLLABORATOR，**closed (completed)**，信息日期 2024-11-10）https://github.com/SWivid/F5-TTS/issues/443
  - 环境：`macOS 14.4.1, Python 3.10, torch==2.4.1, torchaudio==2.4.1`
  - 原文：「The latest code throws a `The operator 'aten::unfold_backward' is not currently implemented for the MPS device` error when running **both** the inference and the training UI. I can work around it by setting the `PYTOCH_ENABLE_MPS_FALLBACK` environment variable to `1`」
  - 崩点定位在 vocoder：`vocos/heads.py line 68 audio = self.istft(S)` → `torch.istft` 的 `unfold_backward`
  - 维护者诊断（原文）：「Probably for reason that in the update introducing **bigvgan**, the vocoder decode operation is moved to gpu, while istft is fine with cpu previously.」并建议在 `utils_infer.py` 里对 mps 自动设 fallback
  - 后续仍有用户反馈未彻底解决：「The new PR did not resolve this problem for, but running this in the cmd line before running `$: f5-tts_infer-gradio` resolved it: `export PYTORCH_ENABLE_MPS_FALLBACK=1`」；另一位「This problem is still unresolved」，还有「Anyone figure this out?」
  - 安装期坑：Issue #475（Apple M2 Ultra, python3.12）boto3/botocore 无限循环降级下载，根因「**bitsandbytes is not suport on apple silicon**」，由 #477 / commit `cb8ce33` 修复 https://github.com/SWivid/F5-TTS/issues/475
- **(c) 实测数字（M 系列）**：**未核实**（issue 中无 RTF/吞吐数字）。
- **(d) 非 PyTorch macOS 路线：有，两条**
  - ONNX：`endink/F5-TTS-ONNX-Exporter` —「Running the F5-TTS by ONNX Runtime」https://github.com/endink/F5-TTS-ONNX-Exporter
  - MLX：**soniqo/speech-swift** 模块 `F5TTS`，bundle `aufklarer/F5TTS-v1-Base-MLX-fp16`，源 ckpt `F5TTS_v1_Base/model_1250000.safetensors`，DiT flow-matching + Vocos，24kHz。文档明写「License | `cc-by-nc-4.0`; non-commercial bundle」，并列出待办「Add release benchmarks for memory and RTF on representative Apple Silicon」https://github.com/soniqo/speech-swift/blob/main/docs/models/f5-tts.md
  - 注：soniqo README 的 SPM products 含 `F5TTS`。
- **(e) 许可证（双轨，务必分清）** — README 原文：
  > 「Our code is released under MIT License. The **pre-trained models are licensed under the CC-BY-NC license** due to the training data Emilia, which is an in-the-wild dataset.」
  - 代码 `LICENSE` 原文：`MIT License Copyright (c) 2024 Yushen CHEN` https://raw.githubusercontent.com/SWivid/F5-TTS/main/LICENSE

---

## 5. Fish-Speech / OpenAudio S1（fishaudio）

- **仓库**：https://github.com/fishaudio/fish-speech
- **(a) 能否跑 / 后端**：**官方宣称支持 MPS，但实际在 M 系列上大面积报错**。
  - MPS 支持是合并进去的：PR #259「Apple's MPS backend support」（作者 Aintor，**merged**，信息日期 2024-06-01）https://github.com/fishaudio/fish-speech/pull/259
  - 官方历史 issue #356「[QUESTION] Support on MacOS apple silicon」（closed，2024-07-07 创建 / 2024-07-15 closed）。维护者先答「**Mac is not supported for training/Inferring.**」，随后改口「We've supported the inference for mps devices」，并说「Theoretically any cpu device will work.」https://github.com/fishaudio/fish-speech/issues/356
- **(b) 崩溃证据（原文引用）** — Issue #789「Error when using MPS: 'Expected elements.dtype() == test_elements.dtype() to be true, but got false.'」（信息日期 2024-12-26/27，closed）https://github.com/fishaudio/fish-speech/issues/789
  - 环境：`macOS = 15.2 (Apple M3), python = 3.10, torch=2.4.1`
  - 日志原文（启动即崩）：
    ```
    INFO | tools.server.model_manager:__init__:41 - mps is available, running on mps.
    ...
    INFO | tools.vqgan.inference:load_model:43 - Loaded model: All keys matched successfully
    ERROR: ...
    baize.exceptions.HTTPException: ( , "'Expected elements.dtype() == test_elements.dtype()
    to be true, but got false. (Could this error message be improved? ...)'")
    ERROR: Application startup failed. Exiting.
    ```
  - 线程内跨机型确认（原文）：
    - 「encountered the same issue on my **M1 Pro** mac.」
    - 「Same on **M1 Pro**. I would like to discover how running in MPS can accelerate the process.」
    - 「Same issue happen in **mac mini M2**」
    - 「Same with **Mac M3**, **works fine with CPU but MPS is the trouble.**」
    - 「Same error in #779 when using webui」
  - → **这是「MPS broken，CPU 正常」的明确案例，且覆盖 M1 Pro / M2 / M3 三代。**
- **(c) 实测数字（M 系列）**：**未核实**（未找到 M 系列 RTF / tokens/s）。
- **(d) 非 PyTorch macOS 路线：有（MLX），但仍是 experimental**
  - **soniqo/speech-swift** 模块 `FishAudioTTS`，「Experimental Apple-Silicon MLX port of `aufklarer/Fish-Audio-S2-Pro-MLX-fp16`」；覆盖 text→codebook、Fish DAC encode（原始参考音频）、Fish DAC decode → **44.1 kHz**。架构：Fish Qwen3-Omni dual-AR transformer，text model 36 layers d=2560 32Q/8KV。文档标注「License | Research/non-commercial bundle; do not expose as a commercial engine without a Fish Audio license」https://github.com/soniqo/speech-swift/blob/main/docs/models/fish-audio-s2-pro.md
  - ONNX 导出**不完整**：PR #830「add onnx export code for vqgan model」，TODO 里写「add onnx export code for vqgan **encoder** model and **llama**」——即只有 vqgan decoder 有导出 https://github.com/fishaudio/fish-speech/pull/830
  - 官方 changelog 提到「Added ONNX export support for better deployment options」与「Includes **bug fixes for Apple Silicon (MPS) compatibility**」https://docs.fish.audio/developer-guide/getting-started/changelog
- **(e) 许可证：⚠️ 两个来源不一致，且都是「非商用默认」**
  - **GitHub 仓库 `LICENSE`（一手，curl 原文）**：`# FISH AUDIO RESEARCH LICENSE AGREEMENT`，**`Last Updated: March 7, 2026`**（信息日期 2026-03-07）
    - 「This Agreement is intended to allow research and non-commercial uses of the Materials free of charge. **Any Commercial use of the Materials requires a separate license from Fish Audio.**」
    - 「Commercial Purpose」定义极宽，明确包含「Your business's or organization's **internal operations**」与任何收费/产生收入的产品或服务。
    - 归属要求：必须分发本协议副本 + 在 Notice 文件保留「This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved.」+ 在网站/UI/博客/产品文档**显著展示「Built with Fish Audio」**。
    - 禁止用其输出「create or improve any foundational generative AI model」。
    - 管辖法：美国加利福尼亚州法
    - README 同段：「This codebase and its associated model weights are released under **[FISH AUDIO RESEARCH LICENSE](LICENSE)**」
  - **HuggingFace `fishaudio/openaudio-s1-mini` 卡（HF API 一手查询，2026-09-16 检索）**：`license: cc-by-nc-sa-4.0`
  - → **报告须同时列出**：仓库 LICENSE 已换成 Fish Audio Research License（2026-03-07 版），但 HF 权重卡仍标 CC-BY-NC-SA-4.0。两者都是「非商用默认、商用需另行授权」。

---

## 6. ChatTTS（2noise）

- **仓库**：https://github.com/2noise/ChatTTS
- **(a) 能否跑 / 后端**：能跑，但**官方明确不推荐 MPS**，实际后端 = **CPU**。
- **(b) 崩溃/质量问题证据（原文引用，非常关键）** — Issue #772「How to use MPS instead of CPU on MacOS?」（**open**，信息日期 2024-10-07）https://github.com/2noise/ChatTTS/issues/772
  - 维护者回复原文：
    > 「**MPS is slow and has memory leak now so we're not recommending you to use it.** You can turn it on by passing `experimental=True` to `Chat.load()`」
  - 但打开后立刻报错（原文）：
    ```
    Errors occur after passing experimental=True to Chat.load()
    masked_input_ids: torch.Tensor = input_ids[text_mask_inv].to(device)
    RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
    ```
    另一用户：「I came across the same issue. Hope the maintainers can solve it. 🙏」
  - → **「慢 + 内存泄漏 + 打开后直接崩」，三层问题齐全。**
  - 另一条 Apple Silicon 安装 issue #263「Error installing and running on apple silicon - M1 chip」（信息日期 2024-06-05，因 stale 于 2024-07-30 关闭）https://github.com/2noise/ChatTTS/issues/263 — 报 `ModuleNotFoundError: No module named 'chattts'`；线程里贴了一份中文 macOS 安装步骤（含 `conda install -c conda-forge pynini=2.1.5` + `pip install WeTextProcessing`）作为绕过方案。
- **(c) 实测数字（M 系列）**：**未核实**（维护者只说 "slow"，无数字）。
- **(d) 非 PyTorch macOS 路线**：**未核实**（未找到可靠的官方或主流 ONNX/CoreML/MLX 移植；且权重为 CC BY-NC 4.0，转换发布受限于许可）。
- **(e) 许可证（双轨）** — README 原文：
  > 「The code is published under **`AGPLv3+`** license.」
  > 「The model is published under **`CC BY-NC 4.0`** license. It is intended for educational and research use, and should not be used for any commercial or illegal purposes.」
  - 仓库 `LICENSE` 原文首行确认为 `GNU AFFERO GENERAL PUBLIC LICENSE Version 3` https://raw.githubusercontent.com/2noise/ChatTTS/main/LICENSE
  - 另注：README 说明训练时「added a small amount of high-frequency noise during the training of the 40,000-hour model, and compressed the audio quality as much as possible using MP3 format, to prevent malicious actors from potentially using it for criminal purposes」——即**音质被人为压低，这是模型固有特性，不是 MPS bug**。

---

## 7. Spark-TTS（SparkAudio）

- **仓库**：https://github.com/SparkAudio/Spark-TTS
- **(a) 能否跑 / 后端**：能跑，但 **MPS 直接 abort，只能 CPU**（或走 MLX）。
- **(b) 崩溃证据（原文引用，两种崩法）**
  - Issue #12「Mac mps support?」（信息日期 2025-03-02）https://github.com/SparkAudio/Spark-TTS/issues/12
    - 用户结论原文：「I was able to work around this by replacing `cuda:{device}` with `cpu` in webui.py and inference.py. **Trying to use mps as the device gives the error message** `NotImplementedError: Output channels > 65536 not supported at the MPS device.`」
    - 更硬的崩法（原文）：
      ```
      /AppleInternal/Library/BuildRoots/.../MetalPerformanceShadersGraph/Core/Files/MPSGraphExecutable.mm:3561:
      failed assertion `Error: MLIR pass manager failed'
      infer.sh: line 45: 16457 Abort trap: 6
      ```
      → **不是异常，是进程 SIGABRT。**
  - Issue #95「How to run this successfully on Apple Silicon?」（**open**，信息日期 2025-03-12，2025-05-08 更新）https://github.com/SparkAudio/Spark-TTS/issues/95
    - 原文：「as soon as I set up the UI for cloning, it starts generating but then after some seconds the terminal throws this: `NotImplementedError: Output channels > 65536 not supported at the MPS device…`」
    - **`PYTORCH_ENABLE_MPS_FALLBACK=1` 也不解决**：「I did the temporary fix and I can confirm CUDA running as false so it is running on CPU but I still get the same error again and again.」
    - 可行方案（原文）：直接改源码强绑 CPU
      ```python
      if platform.system() == "Darwin":
          # device = torch.device(f"mps:{device}")
          device = torch.device("cpu")
      ```
      「If you change the source code to operate with the CPU like this, it will work normally, but **this is not a fundamental solution**.」
    - 一位 M4 Pro / 48GB 用户通过替换 requirements.txt 跑起来（`torch==2.6.0 torchaudio==2.6.0 transformers==4.46.2 gradio==5.18.0`），但**音质不佳**：「since I feed it spanish audio and text, the result is silly... so I guess it does not work for spanish yet」
    - 另有建议「You must install mps: `pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu`」，M1 Pro MacBook Pro 用户反馈「thx, this works for me」——但**同一 thread 后面 M1 Pro 用户复现失败**，说明不稳定。
    - 建议升级 macOS 到 15.1+ 的说法被证伪：用户回复「macOS 15.3.1, **not work**」。
- **(c) 实测数字（M 系列）**：**未核实**。README 里的吞吐表（`876.24 ms / 0.1362`、`920.97 ms / 0.0737`、`1611.51 ms / 0.0704`）是 **Triton/TensorRT-LLM 运行时**（`runtime/triton_trtllm`），**与 Apple Silicon 无关**。https://github.com/SparkAudio/Spark-TTS#-inference-performance
- **(d) 非 PyTorch macOS 路线：有（MLX）**
  - `mlx-community/Spark-TTS-0.5B-bf16`，由 `Blaizzy/mlx-audio` 支持（README 模型表：`| **Spark** | SparkTTS model | EN, ZH | mlx-community/Spark-TTS-0.5B-bf16 |`）https://github.com/Blaizzy/mlx-audio ｜ https://huggingface.co/mlx-community/Spark-TTS-0.5B-bf16
- **(e) 许可证**：**Apache-2.0**。仓库 `LICENSE` 原文首行 `Apache License Version 2.0, January 2004`；README badge 也标 `License-Apache%202.0`。https://raw.githubusercontent.com/SparkAudio/Spark-TTS/main/LICENSE

---

## 8. MegaTTS3（bytedance）

- **仓库**：https://github.com/bytedance/MegaTTS3
- **(a) 能否跑 / 后端**：**无官方 macOS 支持**。官方依赖 CUDA（Docker 基础镜像 `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`，**无 arm64 版本**），且强依赖 `pynini`/`WeTextProcessing`/`openfst`（macOS 上编译困难）。**未发现任何 MPS 支持证据。**
- **(b) 崩溃证据（原文引用）**
  - Issue #14「MacOS install error.」（**open**，信息日期 2025-03-31，2025-04-12 更新）https://github.com/bytedance/MegaTTS3/issues/14
    - 环境原文：`Arch: M3 MAX (ARM)` / `MacOS 15.3.2` / `Python 3.9.21`
    - 报错原文：`Building wheel for pynini (pyproject.toml) ... error` / `× Building wheel for pynini (pyproject.toml) did not run successfully.` / `│ exit code: 1`
    - 线程方案：「`brew install openfst` I installed openfst and it worked.」（但报告者回复「As I mentioned in desc, I have already running `brew install openfst`」→ 未解决）
    - 另一个可用方案原文：「If that doesn't work, try running: `conda install -c conda-forge pynini==2.1.5` `pip install WeTextProcessing==1.0.3` Then: `pip install -e .` **Worked for me on macOS 15.3**」
    - 报告者另提议：「🤔 It would be great if just support `docker` for Linux/Win/Mac User」→ 无人实现
  - Issue #47「Apple M3 macOS 14.4.1 docker安装失败」（**open**，信息日期 2025-04-02，2026-01-16 仍更新）https://github.com/bytedance/MegaTTS3/issues/47
    - 原文关键报错：
      ```
      [1/6] FROM docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
      1 warning found: - InvalidBaseImagePlatform: Base image ... was pulled with platform "linux/amd64", expected "linux/arm64"
      docker run -it -p 7929:7929 --gpus all -e CUDA_VISIBLE_DEVICES=0 megatts3:latest
      WARNING: The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8)
      docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
      ```
    - 维护者/社区回复：「基础镜像 `pytorch/pytorch:2.3.0-cuda12.1` 没有 arm64 版本，可以尝试在构建 docker 镜像时添加 `--platform=linux/amd64`」；另一人贴了纯 CPU Dockerfile（`FROM python:3.10-slim` + CPU index-url）。**CPU 容器可构建，但 `--gpus all` 在 Mac 上永远失败（Docker Desktop 无 GPU 直通）。**
  - Issue #44「Docker install for Mac M1」（信息日期 2025-04-02）同一 `linux/amd64` 平台不匹配问题 https://github.com/bytedance/MegaTTS3/issues/44
- **(c) 实测数字**：**Apple Silicon 上 未核实**。官方在 GPU 上给出过口径 — Issue #21「MegaTTS 3推理性能？」（信息日期 2025-03-31）原文：「a800上显存占用大约6-10G，**RTF大约0.3-0.4左右**，但是也得看prompt和target的长度会上下浮动」https://github.com/bytedance/MegaTTS3/issues/21 （**A800，非 Apple**）
- **(d) 非 PyTorch macOS 路线**：**未核实 / 基本不存在**。未找到 MegaTTS3 的 ONNX/CoreML/MLX/GGUF 移植。
- **(e) 许可证**：**Apache-2.0**。`LICENSE` 原文首行 `Apache License Version 2.0`；README：「This project is licensed under the [Apache-2.0 License](LICENSE).」https://raw.githubusercontent.com/bytedance/MegaTTS3/main/LICENSE

---

## 9. MaskGCT（open-mmlab / amphion）

- **代码**：https://github.com/open-mmlab/Amphion 下 `models/tts/maskgct/` ｜ **权重**：https://huggingface.co/amphion/maskgct
- **(a) 能否跑 / 后端**：**可在 macOS 上跑（CPU / ONNX Runtime）**，但需手动修 bug；**未发现 MPS 支持**。G2P 部分走 ONNX Runtime，macOS 上可用 provider 为 `CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider`。
- **(b) 崩溃证据（原文引用）**
  - Issue #415「[Help]: **mac mini m4** can't run it」（信息日期 2025-03-11）https://github.com/open-mmlab/Amphion/issues/415
    - 关键日志（原文，注意 provider 列表）：
      ```
      UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.
      Available providers: 'CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider'
      ...
      RuntimeError: espeak not installed on your system
      ```
    - 修好 espeak 后撞上源码 bug（原文）：
      ```
      device = torch.device("mps" if torch.cuda.is_available() else "CPU")
      RuntimeError: Expected one of cpu, cuda, ..., mps, meta, ... device type at start of device string: CPU
      ```
      社区修法：「replace `"CPU"` by `"cpu"`」/「Please change the else "CPU" part to else "cpu"」
    - 环境变量绕行（原文）：
      ```
      export PHONEMIZER_ESPEAK_LIBRARY="/usr/local/Cellar/espeak-ng/<版本号>/lib/libespeak-ng.dylib"
      export ESPEAK_DATA_PATH="/usr/local/Cellar/espeak-ng/<版本号>/share/espeak-ng-data"
      export DYLD_LIBRARY_PATH="/usr/local/Cellar/pcaudiolib/<版本号>/lib"
      ```
    - 注意这行逻辑本身写错了：`"mps" if torch.cuda.is_available() else "CPU"` —— cuda 不可用时得到字符串 `"CPU"`（大写，非法）。**所以在 Mac 上它永远走不到 mps，且默认就是坏的。**
  - Issue #450「[BUG]: Cannot Run MaskGCT」（信息日期 2025-06-28）https://github.com/open-mmlab/Amphion/issues/450
    - 报错原文：`ImportError: cannot import name 'setLangfilters' from 'LangSegment.LangSegment'`
    - 修法（原文）：「You can fix this problem by replacing `setLangfilters` with `setfilters` in the file `LangSegment/__init__.py`」
    - 另一用户：「I am running into the same exact error and also very frustrated with the install, I tried locally on **Mac CPU** and on a cloud GPU environment, no luck」
  - → 综合判断：**MaskGCT 在 macOS 上「能跑但装起来很痛」，属于 CPU-only，没有 MPS 路径。**
- **(c) 实测数字（M 系列）**：**未核实**。README 无 Apple 数字；仅宣称「Incredibly fast inference」（第三方站点措辞，非一手）。
- **(d) 非 PyTorch macOS 路线**：**部分有** — G2P（poly_bert_model.onnx）走 ONNX Runtime，可用 CoreML EP；但**主模型本体未找到 ONNX/CoreML/MLX 移植**。
- **(e) 许可证：代码与权重不同**
  - 代码：**MIT**（Amphion `LICENSE` 原文：`MIT License Copyright (c) 2023 Amphion`）https://raw.githubusercontent.com/open-mmlab/Amphion/main/LICENSE
  - **权重：`cc-by-nc-4.0`**（HF API 一手查询 `amphion/maskgct` → `license: cc-by-nc-4.0`，2026-09-16 检索）https://huggingface.co/amphion/maskgct
  - MaskGCT 页面 README 本身**没有** license 段落（已 grep 确认），所以只看 README 会误判。

---

## 10. XTTS-v2（coqui-ai）

- **仓库**：https://github.com/coqui-ai/TTS（**公司已停运、仓库归档**）｜ 权重 https://huggingface.co/coqui/XTTS-v2
- **(a) 能否跑 / 后端**：**MPS 官方不支持且被标记 wontfix**；可用后端 = **CPU**（及 ONNX）。
- **(b) 崩溃证据（原文引用）** — Issue #3649「[Bug] **Unable to use xtts_v2 with mps device on Apple Silicon**」（信息日期 2024-03-28 开，**closed + label `wontfix`**）https://github.com/coqui-ai/TTS/issues/3649
  - 正文原文：「I have a **M1 Max with 32 cores and 64 gb** of unified memory. So if MPS is meant to work, it should work quite fast. But currently …」（后接 MPS 报错）
  - 标签原文：`bug` + **`wontfix`** —「This will not be worked on but feel free to help.」
  - → 因 Coqui 停运，**不会有修复**。这是本批次里「MPS 明确死刑」的案例。
  - 二手补充（未复核）：一份 2026 年指南称「Error: MPS device crashes on Apple Silicon — XTTS uses operations that aren't fully covered by Apple's Metal Performance Shaders backend across all PyTorch versions. If you see MPS-related crashes or assertion errors, force CPU: pass `.to("cpu")`」https://www.qwe.edu.pl/ai-tools/install-xtts-v2-coqui-tts-fork
- **(c) 实测数字（M 系列）**：**未核实**。
- **(d) 非 PyTorch macOS 路线：有**
  - ONNX（明确「no PyTorch required」）：`pltobing/XTTSv2-Streaming-ONNX` —「Streaming text-to-speech inference for XTTSv2 using ONNX Runtime — no PyTorch required.」https://huggingface.co/pltobing/XTTSv2-Streaming-ONNX
  - 注意：`k2-fsa/sherpa-onnx` 的 TTS 模型清单里**没有** XTTS（仅有 Piper/VITS、Kokoro、Matcha、KittenTTS、Supertonic、vits-inflect）。已核对 https://k2-fsa.github.io/sherpa/onnx/tts/all/English/index.html
  - 维护现状：需关注社区 fork（如 idiap 系 fork / `coqui-tts` PyPI 包）——**具体 fork 的维护状态本次未逐一举证，标注为需另行核实**。
- **(e) 许可证（双轨，且权重是非商用）**
  - 代码：**MPL-2.0**（`TTS/LICENSE.txt` 原文首行 `Mozilla Public License Version 2.0`）https://raw.githubusercontent.com/coqui-ai/TTS/dev/LICENSE.txt ；README badge 亦为 `License-MPL 2.0`
  - **权重：Coqui Public Model License（CPML）** — HF card 原文（一手）：
    ```yaml
    license: other
    license_name: coqui-public-model-license
    license_link: https://coqui.ai/cpml
    ```
    「This model is licensed under [Coqui Public Model License](https://coqui.ai/cpml).」https://huggingface.co/coqui/XTTS-v2
    → **CPML 为非商用许可**（Coqui 官方定义，非我推断）。

---

## 11. StyleTTS2（yl4579）

- **仓库**：https://github.com/yl4579/StyleTTS2
- **(a) 能否跑 / 后端**：CPU 可跑；**MPS 从 2023 起就没被启用**（代码里显式禁用）。
- **(b) 证据（原文引用）** — Issue #114「Mac (Metal) support?」（**open**，信息日期 2023-11-30 创建，**2026-02-26 仍被更新**）https://github.com/yl4579/StyleTTS2/issues/114
  - 提问原文：「Any chances of running this model on the unified RAM in silicon macs? 16GB GPU/CPU」
  - 回复原文：「**Same issue. It doesn't work for now.** I've tried HuggingFace space running locally and got: `MPS would be available but cannot be used rn` `RuntimeError: espeak not installed on your system`」
  - 运行日志原文（关键行）：
    ```
    SCIPY TORCH STUFF START 177
    MPS would be available but cannot be used rn
    ...
    Traceback (most recent call last):
      File ".../styletts2/app.py", line 105, in <module>
        btn.click(synthesize, inputs=[inp, voice, multispeakersteps], outputs=[audio], concurrency_limit=4)
    TypeError: EventListenerMethod.__call__() got an unexpected keyword argument 'concurrency_limit'
    ```
  - → **「MPS 可用但被主动禁用」（`MPS would be available but cannot be used rn`）+ 另一条 gradio 版本不兼容 bug。**
  - 线程里有人提议用 MLX：「@fakerybakery have you looked into [mlx](https://github.com/ml-explore/mlx)? It's a new framework from Apple.」——**该建议未被合并**。
  - 维护者给的 macOS 前置条件是 espeak-ng（MacPorts/brew）：「Did you successfully install the espeak-ng in with MacPorts? Can you try running: `echo 'this is a test' | espeak-ng -x -q --ipa -v en-us`」→ 用户确认输出 `ðɪs ɪz ɐ tˈɛst`，即 espeak 能装好，**但 MPS 依然不可用**。
- **(c) 实测数字（M 系列）**：**未核实**。
- **(d) 非 PyTorch macOS 路线：有，且是 CoreML**
  - **FluidInference/FluidAudio**（Swift Package）：PR #582「fix(tts): guard direct Float16 reads with `#if arch(arm64)` (**CosyVoice3, StyleTTS2**)」明确含 StyleTTS2 模块 https://github.com/FluidInference/FluidAudio/pull/582
    - 原文说明：「`Float16` is an arm64-only Swift built-in, so any direct `Float16` typing fails to compile in the x86_64 slice of a Universal build. Four sites in CosyVoice3 and StyleTTS2 do raw `Float16` pointer binds…」
    - 包索引页：https://swiftpackageindex.com/FluidInference/FluidAudio
  - `kokoro` 的架构即基于 StyleTTS 2，故 Kokoro 的 CoreML 路线（见第 12 节）可视为 StyleTTS2 架构的成熟化替代。
- **(e) 许可证（代码与权重不同）** — README 原文：
  > 「Code: MIT License」
  > 「Pre-Trained Models: Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.」
  - 代码 `LICENSE` 原文：`MIT License Copyright (c) 2023 Aaron (Yinghao) Li` https://raw.githubusercontent.com/yl4579/StyleTTS2/main/LICENSE
  - 文本/推理侧许可陷阱（README 原文）：「the inference depends on a **GPL-licensed package**, so it is not included directly in this repository. A [GPL-licensed fork](https://github.com/NeuralVox/StyleTTS2) has an importable script… A [fully MIT-licensed package](https://pypi.org/project/styletts2/) that uses gruut (albeit lower quality…) is also available.」

---

## 12. Kokoro（hexgrad）— 本批次 Apple Silicon 体验最好的模型

- **仓库**：https://github.com/hexgrad/kokoro ｜ 权重 https://huggingface.co/hexgrad/Kokoro-82M
- **(a) 能否跑 / 后端**：**全部可行** — PyTorch MPS、ONNX Runtime（含 CoreML EP）、**原生 CoreML/ANE**、MLX。这是本批次唯一「四种后端都有成熟方案」的模型。
  - README 有专门的 macOS 段落（一手原文）：
    > ### MacOS Apple Silicon GPU Acceleration
    > On Mac M1/M2/M3/M4 devices, you can explicitly specify the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to enable GPU acceleration.
    > ```bash
    > PYTORCH_ENABLE_MPS_FALLBACK=1 python run-your-kokoro-script.py
    > ```
    https://raw.githubusercontent.com/hexgrad/kokoro/main/README.md
- **(b) 崩溃/性能问题证据**
  - **PyTorch 路线不是本批次的坑点**；真正的坑在 **sherpa-onnx CPU** 的性能：
    - PR #1713「Export kokoro to sherpa-onnx」（作者 csukuangfj，**Merged**，信息日期 2025-01-15）https://github.com/k2-fsa/sherpa-onnx/pull/1713
      原文引用：
      > 「Even though kokoro TTS claims it has only about 82M parameters, it is by far the **LARGEST** model we have in sherpa-onnx. It is also the **SLOWEST TTS model on CPU** in sherpa-onnx」
  - MLX 路线的小 bug（一手，来自 model card）：「(The pinned MLX version fails 3-second clips with a **broadcast-shape error**; no time to report.)」
  - CoreML/ANE 路线提到系统级风险：「fix(tts/kokoro-ane): warn on **BNNS-crash-prone OS builds** (26.4…)」（来自 soniqo/speech-swift releases）
- **(c) 实测数字（M 系列）— 本报告中最完整的一组**
  - **sherpa-onnx / ONNX Runtime, CPU, 1 thread**（PR #1713 原文，机器为 `fangjuns-MacBook-Pro`，CPU 型号未标注，py38）：
    ```
    Testing 1/11 - af
    Saved to kokoro-v0_19-af.wav
    Elapsed seconds: 12.287
    Audio duration in seconds: 14.250
    RTF: 12.287/14.250 = 0.862
    ```
    → **RTF 0.862（单线程 CPU）**，模型 `kokoro-v0_19.onnx`
  - **thewh1teagle/kokoro-onnx README 原文**：「**Fast performance near real-time on macOS M1**」；「Lightweight: ~300MB (quantized: ~80MB)」https://github.com/thewh1teagle/kokoro-onnx
  - **mattmireles/kokoro-coreml — 原生 CoreML/ANE，五个固定时长桶**（HF card，测量时间 **2026-06**，信息日期 2026-09-01）https://huggingface.co/mattmireles/kokoro-coreml
    | Audio | M2 Studio (64 GB) | M2 Air (24 GB) | M1 Mini (16 GB) |
    |---|---:|---:|---:|
    | 3s | **51 ms** | 148 ms | 236 ms |
    | 10s | **126 ms** | 466 ms | 712 ms |
    | 30s | **379 ms** | 1,405 ms | 1,958 ms |
    - 原文：「That's **12-79x realtime** across the lineup.」/「30 seconds of speech in 379 ms on a Mac Studio. **2x faster than MLX** on the same hardware. Running on the Apple Neural Engine.」
    - M1 mask-aware A/B（Irvine M1 16GB，macOS 15.7.9，PR #6，**作者自标「engineering results, not publication-grade benchmark numbers」**）：3s 293→236ms（1.24x）、7s 627→494（1.27x）、10s 989→712（1.39x）、15s 1287→1007（1.28x）、30s 2646→1958（1.35x）
    - **vs MLX 对比**（同一批机器/语句/voice `af_heart`，比较对象 `Blaizzy/mlx-audio` 0.4.3 @ `862dfbe` + `mlx-community/Kokoro-82M-bf16`）：
      | Audio | M2 Studio | M2 Air | M1 Mini |
      |---|---|---|---|
      | 7s | 96 vs 224 ms（**2.3x**） | 331 vs 686 ms（**2.1x**） | 494 vs 824 ms（**1.7x**） |
      | 10s | 126 vs 289 ms（**2.3x**） | 466 vs 836 ms（**1.8x**） | 712 vs 1,124 ms（**1.6x**） |
      | 30s | 379 vs 763 ms（**2.0x**） | 1,405 vs 2,600 ms（**1.9x**） | 1,958 vs 3,078 ms（**1.6x**） |
    - iPhone 对照（2026-06, iOS 26.5）：iPhone 15 Pro Max 30s → 6,374 ms（1.2x vs MLX `mlalma/kokoro-ios` 1.0.8 的 7,792 ms）；iPhone 12 Pro 30s → 12,301 ms（MLX 侧 **OOM**）
  - **soniqo KokoroTTS（CoreML/ANE）**：iPhone 16 Pro **0.08 RTF**、峰值内存 676 MB（README 原文：「[Kokoro TTS] On-device TTS (82M, CoreML/Neural Engine, 54 voices, iOS-ready, 10 languages) — **0.08 RTF on iPhone 16 Pro**」）；Mac 侧文档写「Inference RTFx | ~0.7 (faster than real-time)」https://github.com/soniqo/speech-swift/blob/main/docs/benchmarks/ios-coreml.md
- **(d) 非 PyTorch macOS 路线（三条，都可直接用）**
  1. **ONNX Runtime**：`thewh1teagle/kokoro-onnx` — 官方 PyPI `pip install -U kokoro-onnx`，模型文件 `kokoro-v1.0.onnx` + `voices-v1.0.bin` https://github.com/thewh1teagle/kokoro-onnx
  2. **CoreML/ANE**：`mattmireles/kokoro-coreml`（HF `.mlpackage`，运行时是 GitHub 上的 `swift-tts` Swift 包，`KokoroTTS` API，要求 iOS 18.0+ / macOS 15.0+）https://huggingface.co/mattmireles/kokoro-coreml ｜ https://github.com/mattmireles/kokoro-coreml
  3. **sherpa-onnx**（ONNX，C++/Python/Swift 皆有）：模型 `kokoro-en-v0_19`、`kokoro-multi-lang-v1_1`（中英，103 speakers）、`kokoro-multi-lang-v1_0`（53 speakers）https://k2-fsa.github.io/sherpa/onnx/tts/all/English/index.html
  4. **MLX**：`mlx-community/Kokoro-82M-bf16 / 8bit / 6bit / 4bit`，经 `Blaizzy/mlx-audio`
  5. **soniqo/speech-swift**：模块 `KokoroTTS`（CoreML），54 个预置音色 / 10 语言
- **(e) 许可证**：**Apache-2.0（代码 + 权重都是）**
  - 仓库 `LICENSE` 原文首行 `Apache License Version 2.0` https://raw.githubusercontent.com/hexgrad/kokoro/main/LICENSE
  - README 原文：「With **Apache-licensed weights**, Kokoro can be deployed anywhere from production environments to personal projects.」
  - `mattmireles/kokoro-coreml` card 原文：「License — Apache 2.0, inherited from Kokoro-82M. **Ship it. Sell it. Fork it.**」
  - `thewh1teagle/kokoro-onnx` README 原文：「kokoro-onnx: **MIT**；kokoro model: **Apache 2.0**」
  - soniqo Kokoro 文档：「License: Apache-2.0」+「Text is converted to phoneme tokens via a three-tier pipeline — **all Apache-2.0 licensed, no GPL dependencies**」

---

## 13. Piper（rhasspy / OHF-Voice）

- **当前维护仓库**：https://github.com/OHF-Voice/piper1-gpl （**GPL-3.0**，创建于 2025-03-28）｜ 旧仓库 `rhasspy/piper` 已归档 ｜ 音色：https://huggingface.co/rhasspy/piper-voices
- **(a) 能否跑 / 后端**：**是，且是本批次对 Apple Silicon 最友好的架构** — 纯 **ONNX Runtime**，**根本不需要 MPS**（推理是 CPU/ONNX，可选 CoreML EP）。
  - `pip install piper-tts` 即可（PyPI `piper-tts v1.6.0`，`Python: >=3.9`，`License: GPL-3.0-or-later`）https://pypi.org/project/piper-tts/
  - 官方 BUILDING 文档说明用 scikit-build-core + cmake + 内嵌 espeak-ng，且用 Python stable ABI「Piper wheels only need to be built once for each platform (**Linux, Mac, Windows**)」→ 有 macOS wheel https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/BUILDING.md
- **(b) 崩溃/质量问题**：**未发现 MPS 相关问题**（因为不走 MPS）。已知社区记录：
  - 有 Anki 插件作者专门写了「🤖PiperTTS: **MacOS M1~M3 support**」排障笔记，称使用者会遇到 voice 生成失败的提示（**第三方插件 issue，非 Piper 本体，二手**）https://github.com/shigeyukey/my_addons/issues/348
  - 另有社区 ARM64 静态二进制：https://github.com/itsabhishekolkha/piper-arm-build
  - **官方仓库本身未找到 macOS 崩溃 issue。**
- **(c) 实测数字（M 系列）**：**未核实**（官方 README 未给 macOS 数字）。
- **(d) 非 PyTorch macOS 路线：这就是 Piper 的本体**
  - **Piper 本体 = ONNX Runtime**，无需 PyTorch。
  - **sherpa-onnx 官方支持 Piper**：「In this section, we describe how to convert piper pre-trained models from https://huggingface.co/rhasspy/piper-voices」，且提供全部转换好的模型包（`tts-models` release tag）https://k2-fsa.github.io/sherpa/onnx/tts/piper.html
    - 英文可用音色极多（`vits-piper-en_US-lessac-high/medium/low`、`en_US-ryan-high/medium/low`、`en_US-amy-low/medium`、`en_GB-alba-medium`、`en_US-libritts-high`、`en_US-glados-high` 等数十个）https://k2-fsa.github.io/sherpa/onnx/tts/all/English/index.html
  - **CoreML EP**：ONNX Runtime 官方 CoreML Execution Provider 要求「iOS 13+ 或 macOS 10.15+」，「recommended to use Apple devices equipped with Apple Neural Engine」；且现在**官方 wheel 已含 macOS arm64 + Core ML 支持** https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- **(e) 许可证：⚠️ 代码 GPL-3.0，音色逐个不同（必须逐条查）**
  - **代码**：`COPYING` 原文首行 `GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007` https://github.com/OHF-Voice/piper1-gpl/blob/main/COPYING ；PyPI 标 `GPL-3.0-or-later`
  - **音色（voice .onnx）许可逐条不同** — 已在 HF 上逐一下载 `MODEL_CARD` 确认（2026-09-16）：
    - `en/en_US/ryan/high/MODEL_CARD` → 数据集 `https://www.kaggle.com/datasets/roholazandie/ryanspeech`，**`License: CC BY-NC-SA 4.0`**
    - `en/en_GB/alba/medium/MODEL_CARD` → **`License: https://creativecommons.org/licenses/by/4.0/`**
    - `en/en_US/amy/medium/MODEL_CARD` → `License: See URL`（即未明示，需另查）
    - `en/en_US/lessac/medium/MODEL_CARD` → 数据集 Blizzard 2013，**`License: https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html`**
    - → **结论：不能把「Piper」当成一个统一许可的资产；必须按具体音色查 MODEL_CARD。**
  - 现状提醒：README 原文「**Looking for Maintainers** — The Open Home Foundation is looking for maintainers for Piper!」

---

## 14. Orpheus-TTS（canopyai）

- **仓库**：https://github.com/canopyai/Orpheus-TTS ｜ 权重 https://huggingface.co/canopylabs/orpheus-3b-0.1-ft
- **(a) 能否跑 / 后端**：**主路径是 vLLM/CUDA，但官方提供了 Apple Silicon 的非 GPU 路径 = llama.cpp + Metal**。
  - 官方 `additional_inference_options/no_gpu/README.md` 原文：
    > 「You can stream audio without a GPU by using `orpheus-cpp`, which is a **llama.cpp-compatible backend** of the Orpheus TTS model.」
    > ```bash
    > pip install orpheus-cpp
    > # MacOS with Apple Silicon
    > pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
    > ```
    https://github.com/canopyai/Orpheus-TTS/blob/main/additional_inference_options/no_gpu/README.md
  - 主 README 也写明：「For No GPU inference using Llama cpp see implementation documentation」
  - **MLX 也可行**：`mlx-community/orpheus-3b-0.1-ft-4bit`，经 `Blaizzy/mlx-audio` 支持
- **(b) 崩溃/质量问题证据**
  - Issue #178「Feature Request: **Support for Apple Silicon**」（**open**，信息日期 2025-04-22）https://github.com/canopyai/Orpheus-TTS/issues/178
    - 原文：「When I tried this with my Mac I got an error when importing OrpheusModel that **torch had been compiled without CUDA**」→ 即官方 PyTorch/orpheus-speech 路径在 Mac 上直接不可用。
  - MLX 路线功能缺口：`Blaizzy/mlx-audio` Issue #74（信息日期 2025-04-02）https://github.com/Blaizzy/mlx-audio/issues/74
    - 原文：「it still saves a wav file, then starts playing after」；维护者 Blaizzy 回答：「No, you are not. It's a missing feature actually. **Orpheus at the moment generates all the tokens then we play.** Will be fixed :)」→ **即 MLX 侧当时无真流式**（对实时语音对话 pipeline 是硬伤）。
    - 运行时长原文：`mx.metal.set_wired_limt is deprecated…`、`Duration: 00:00:01.365`、`Real-time factor: 1.12x`、`Peak memory usage: 1.92GB`
- **(c) 实测数字（M 系列）**
  | 硬件 | 后端 | 数字 | 来源 |
  |---|---|---|---|
  | **M1 Ultra** | MLX（mlx-audio） | 用户原文：「I have an **M1 ultra** that can generate **faster than realtime** with both orpheus and CSM, and I love the results.」 | mlx-audio #74，2025-04-02 |
  | 未标注机器（应为 M 系列，MLX） | `mlx-community/orpheus-3b-0.1-ft-4bit` | **Real-time factor: 1.12x**，Processing time 1.22s，**Peak memory 1.92GB** | mlx-audio #74（贴出的 CLI 输出） |
  | A100（**非 Apple**，旁证口径） | vLLM FP8 | 原文：「For Orpheus TTS, **~91 tokens/sec = ~1 second of audio** (real-time factor 1.0)」，TTFB ~130 ms | 二手：https://simplismart.ai/blog/orpheus-tts-simplismart （信息日期 2026-01-09） |
- **(d) 非 PyTorch macOS 路线（三条）**
  1. **llama.cpp / GGUF**：官方 `orpheus-cpp` + `llama-cpp-python` **metal wheel**（见上）
  2. **GGUF 权重**：`lex-au/Orpheus-3b-FT-Q8_0.gguf`（HF API 查询：`license: apache-2.0`, gated: False）https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf
  3. **MLX**：`mlx-community/orpheus-3b-0.1-ft-4bit` https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-4bit
- **(e) 许可证**
  - 代码：**Apache-2.0**（仓库 `LICENSE` 原文首行 `Apache License Version 2.0`）https://raw.githubusercontent.com/canopyai/Orpheus-TTS/main/LICENSE
  - 权重：HF API 查询 `canopylabs/orpheus-3b-0.1-ft` → **`license: apache-2.0`**、`tags: ['llama', 'license:apache-2.0']`、**`gated: auto`**（需同意条款方可下载）
  - **风险提示（须注明为观察而非许可主张）**：README 原文「built on the **Llama-3b backbone**」。权重卡自标 apache-2.0，但基座为 Meta Llama-3.2-3B。**派生模型对基座许可的继承关系本次未能从一手文本确认 → 标 未核实。**

---

## 15. CSM-1B（Sesame）

- **仓库**：https://github.com/SesameAILabs/csm ｜ 权重 https://huggingface.co/sesame/csm-1b（**gated：需登录同意条款**）
- **(a) 能否跑 / 后端**：**README 官方支持 MPS（代码里有 `mps` 分支）**，但社区实测**官方仓库在 macOS 上跑不通**；可靠的 macOS 后端是 **MLX**。
  - 官方 README 原文（一手）：
    ```python
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    generator = load_csm_1b(device=device)
    ```
    但同时要求段落写着：「* A **CUDA-compatible GPU**」/「The code has been tested on CUDA 12.4 and 12.6」
  - 另一处（PR #52 的 README 改动，被 review）：「`generator = load_csm_1b(model_path, "cuda")` `# Use "mps" for Apple Silicon or "cpu" for Intel MacBooks`」
- **(b) 崩溃证据（原文引用）** — PR #52「Add Huggingface manual + Macbook GPU/CPU Compatibility」（信息日期 2025-03-14）https://github.com/SesameAILabs/csm/pull/52
  - 评论者 **KyleVasulka** 原文（全文引用，信息量最大）：
    > 「I think the repo currently **does not work with macos** (even with the changes mentioned here). Even after installing bitsandbytes and torch for macos apple sillicon, It seems to **expect triton (which I don't believe is available on macos)**. Additionally, it seems there is an issue in `silentcipher/server.py` (used in `encode_wav`) about trying to convert **MPS Tensor to float64 dtype as the MPS framework doesn't support float64**. please use float32 instead. I think it is **not as easy as swapping out the device type for mps**」
  - PR 获得 7 个 ❤️ 反应（社区共识存在）
  - 另有一手线索：HF 论坛「CSM-1B Model and Metal Shader issue」线程提到「There is a different MPS bug:」（信息日期 2026-02-03）https://discuss.huggingface.co/t/csm-1b-model-and-metal-shader-issue/173053
- **(c) 实测数字（M 系列）**
  | 硬件 | 后端 | 数字 | 来源 |
  |---|---|---|---|
  | **M5 Pro** | MLX int8（soniqo `CSM-1B-MLX-8bit`，Release build） | 原文：「it runs **faster than real-time (≈ 0.5 real-time factor — roughly 0.5 s of compute per second of audio)** in a Release build. Debug builds are several times slower because MLX kernels are unoptimized.」 | https://github.com/soniqo/speech-swift/blob/main/docs/models/csm.md |
  | M1 Ultra（用户自述） | MLX（mlx-audio） | 「My M1 ultra that can generate faster than realtime with both orpheus and CSM」 | mlx-audio #74，2025-04-02 |
  | M2 Air | MLX（csm-mlx） | 原文：「`nn.quantize(csm)` — **Speed up nearly real-time on M2 Air, but loses quality.**」 | https://github.com/senstella/csm-mlx |
  | 内存提示（原文） | — | 「Call `MLX.GPU.clearCache()` between repeated generations to keep GPU memory from growing.」 | soniqo csm.md |
- **(d) 非 PyTorch macOS 路线：非常丰富**
  1. **MLX**：`senstella/csm-mlx`（权重 `senstella/csm-1b-mlx`）https://github.com/senstella/csm-mlx ｜ `endlessreform/csm_mlx`（原文：「Port of Sesame's CSM model to MLX for use on Apple Silicon. **The project goal is realtime streaming inference on a MacBook.**」）https://github.com/endlessreform/csm_mlx ｜ `mlx-community/csm-1b`（经 `Blaizzy/mlx-audio` 的 "CSM / MisoTTS" 条目）https://huggingface.co/mlx-community/csm-1b
  2. **soniqo/speech-swift**：模块 `CSM`，默认 `aufklarer/CSM-1B-MLX-8bit`，复用 PersonaPlex 的 Mimi codec/RoPE/KV cache；int8 group size 64；采样 `temperature=0.9, topK=50, maxFrames=1024` https://github.com/soniqo/speech-swift/blob/main/docs/models/csm.md
  3. **llama.cpp**：PR #12648「tts: implement sesame CSM + Mimi decoder」（作者 ngxson，**OPEN + Draft**，信息日期 2025-03-29 创建，**2026-07-29 仍更新**，+3956 −63 in 20 files）。作者自述「Tbh it is more complicated than expected.」 https://github.com/ggml-org/llama.cpp/pull/12648
     - GGUF 侧还有 `johnbenac/sesame-csm-1b-GGUF-encoder`（含 `examples/tts/tts-csm.cpp`）https://huggingface.co/johnbenac/sesame-csm-1b-GGUF-encoder
  4. **HF Transformers 原生**：自 `transformers` 4.52.1（2025-05-20）起内置 `CsmForConditionalGeneration` https://huggingface.co/docs/transformers/main/en/model_doc/csm
- **(e) 许可证**
  - 代码 + 权重：**Apache-2.0**（仓库 `LICENSE` 原文首行 `Apache License Version 2.0, January 2004`）https://raw.githubusercontent.com/SesameAILabs/csm/main/LICENSE
  - 交叉佐证（一手，第三方独立确认）：soniqo csm.md 明写「**License:** Apache-2.0 (Sesame CSM-1B)」
  - ⚠️ **HF 权重卡的 license 字段本次 未核实**：`https://huggingface.co/sesame/csm-1b` 是 **gated** 仓库（页面原文：「You need to agree to share your contact information to access this model」，需登录），`huggingface.co/api/models/sesame/csm-1b` 亦不可匿名读取。故**只能确认 GitHub 仓库为 Apache-2.0**；权重卡上是否另有条款 未核实。
  - 权重卡可见的使用限制原文（已抓到，非许可条款但属使用约束）：「we **explicitly prohibit** the following: Impersonation or Fraud… Misinformation or Deception… Illegal or Harmful Activities」

---

## 汇总表：按模型 × Apple Silicon 后端

| # | 模型 | MPS (PyTorch) | CPU | ONNX / CoreML | MLX | GGUF/llama.cpp | macOS 上「最好走哪条」 |
|---|---|---|---|---|---|---|---|
| 1 | **IndexTTS-2** | ❌ 崩（`Output channels > 65536`）；M4+torch2.7 部分修复 | ✅ 但 RTF 3.53 | 未核实 | ✅ **成熟**（mlx-community / vanch007 / Jup33Q / solar2ain / aufklarer） | 未核实 | **MLX** |
| 2 | **CosyVoice 2/3** | ⚠️「比 CPU 更慢」（维护者原话）；PR #1869 open | ✅ | ✅ **CoreML**（FluidInference/FluidAudio + CosyVoice3-0.5B-coreml, Apache-2.0） | ✅ MLX（soniqo，M2 Max RTF≈0.5） | 未核实 | **CoreML 或 MLX** |
| 3 | **GPT-SoVITS** | ⚠️ 名义支持，实测退化为 CPU（M4 Max） | ✅ **README 给 M4 RTF 0.526** | 未核实 | 未核实 | 未核实 | **CPU** |
| 4 | **F5-TTS** | ⚠️ 能跑，**必须** `PYTORCH_ENABLE_MPS_FALLBACK=1` | ✅ | ✅ ONNX（endink/F5-TTS-ONNX-Exporter） | ✅ MLX（soniqo `F5TTS`） | 未核实 | **ONNX 或 MLX** |
| 5 | **Fish-Speech / OpenAudio S1** | ❌ 崩（dtype assertion），M1 Pro/M2/M3 均复现 | ✅ | ⚠️ 仅 vqgan decoder 有 ONNX | ✅ **experimental**（soniqo `FishAudioTTS`，44.1kHz） | 未核实 | **CPU 或 MLX(实验)** |
| 6 | **ChatTTS** | ❌ **「slow + memory leak，不推荐」**（维护者原话），开启即崩 | ✅ | 未核实 | 未核实 | 未核实 | **CPU** |
| 7 | **Spark-TTS** | ❌ **SIGABRT**（MLIR pass manager failed）+ `Output channels > 65536` | ✅ | 未核实 | ✅ mlx-community/Spark-TTS-0.5B-bf16 | 未核实 | **MLX** |
| 8 | **MegaTTS3** | ❌ 无 MPS 支持，官方仅 CUDA | ⚠️ 装得起来但需手改（pynini/openfst） | 未核实 | 未核实 | 未核实 | **不推荐 macOS** |
| 9 | **MaskGCT** | ❌ 无 MPS 路径（源码 device 分支坏） | ✅ 但需修 2 个 bug | ⚠️ 仅 G2P 走 ONNX（可用 CoreML EP） | 未核实 | 未核实 | **CPU（需打补丁）** |
| 10 | **XTTS-v2** | ❌ **wontfix**（coqui-ai/TTS #3649），公司已停运 | ✅ | ✅ ONNX（pltobing/XTTSv2-Streaming-ONNX，「no PyTorch required」） | 未核实 | 未核实 | **ONNX** |
| 11 | **StyleTTS2** | ❌ 代码显式禁用（`MPS would be available but cannot be used rn`） | ✅ | ✅ **CoreML**（FluidInference/FluidAudio） | 未验证（有建议无实现） | 未核实 | **CoreML** |
| 12 | **Kokoro** | ✅ README 官方支持（`PYTORCH_ENABLE_MPS_FALLBACK=1`） | ✅ RTF 0.862（1-thread MBP） | ✅✅ **ONNX + 原生 CoreML/ANE**（最成熟） | ✅ mlx-community bf16/8/6/4bit | 未核实 | **CoreML（最快）或 ONNX** |
| 13 | **Piper** | N/A（不走 PyTorch） | ✅ | ✅✅ **ONNX Runtime 本体 + CoreML EP** | 未核实 | 未核实 | **ONNX Runtime** |
| 14 | **Orpheus-TTS** | ❌ torch 无 CUDA 编译，导入即错 | 未核实 | 未核实 | ✅ mlx-community 4bit（RTF 1.12x，Peak 1.92GB） | ✅✅ **官方 `orpheus-cpp` + llama-cpp-python metal wheel**；GGUF `lex-au/…Q8_0.gguf` | **llama.cpp/Metal** |
| 15 | **CSM-1B** | ⚠️ README 有 mps 分支，但社区实测**跑不通**（需 triton、float64 不支持） | ✅ | 未核实 | ✅✅ **成熟**（senstella/csm-mlx、endlessreform/csm_mlx、mlx-community/csm-1b、aufklarer/CSM-1B-MLX-8bit RTF≈0.5 on M5 Pro） | ⚠️ llama.cpp PR #12648 **仍是 Draft** | **MLX** |

### 许可证速查表（均来自一手来源，未核实项已标出）

| 模型 | 代码许可 | 权重/模型许可 |
|---|---|---|
| IndexTTS-2 | **bilibili Model Use License Agreement**（自定义；100M MAU / RMB 10亿 触发单独授权；pip metadata 误标 Apache-2.0） | 同左（MLX 转换权重沿用） |
| CosyVoice 2/3 | **Apache-2.0** | Apache-2.0（FluidInference CoreML 转换亦标 Apache-2.0） |
| GPT-SoVITS | **MIT** | 未核实（README 未单列权重许可） |
| F5-TTS | **MIT** | **CC-BY-NC**（非商用） |
| Fish-Speech / OpenAudio S1 | **FISH AUDIO RESEARCH LICENSE**（2026-03-07 版，非商用免费、商用需单独授权） | HF 卡仍标 **cc-by-nc-sa-4.0**（两处不一致） |
| ChatTTS | **AGPLv3+** | **CC BY-NC 4.0** |
| Spark-TTS | **Apache-2.0** | 未核实（沿用 Apache-2.0 标注） |
| MegaTTS3 | **Apache-2.0** | 未核实（README 仅声明项目 Apache-2.0） |
| MaskGCT | **MIT**（Amphion） | **cc-by-nc-4.0**（HF API 确认） |
| XTTS-v2 | **MPL-2.0** | **Coqui Public Model License (CPML)** — 非商用 |
| StyleTTS2 | **MIT** | 自定义条款（须告知听众为合成音；推理链含 GPL 包） |
| Kokoro | **Apache-2.0** | **Apache-2.0** |
| Piper | **GPL-3.0**（COPYING）/ PyPI `GPL-3.0-or-later` | **逐音色不同**（已确认 ryan/high = CC BY-NC-SA 4.0；alba = CC BY 4.0；amy = "See URL"；lessac = Blizzard 2013 许可） |
| Orpheus-TTS | **Apache-2.0** | HF 卡标 **apache-2.0**（gated: auto）；对 Llama-3.2-3B 基座许可的继承关系 **未核实** |
| CSM-1B | **Apache-2.0** | **未核实**（HF 仓库 gated，需登录；GitHub LICENSE 为 Apache-2.0，soniqo 亦记 Apache-2.0） |

### 主要非 PyTorch macOS 项目索引

| 项目 | 类型 | 覆盖本批次模型 | URL |
|---|---|---|---|
| **soniqo/speech-swift** | Swift 包，MLX + CoreML | IndexTTS2、CosyVoice3、F5-TTS、Kokoro、CSM、Fish Audio S2 Pro（另含 Chatterbox/VibeVoice/Higgs/Qwen3-TTS） | https://github.com/soniqo/speech-swift ｜ https://soniqo.audio/architecture |
| **FluidInference/FluidAudio** | Swift 包，CoreML | CosyVoice3、StyleTTS2、Kokoro | https://github.com/FluidInference/FluidAudio ｜ https://swiftpackageindex.com/FluidInference/FluidAudio |
| **k2-fsa/sherpa-onnx** | C++/Python/Swift，ONNX Runtime | Piper(VITS)、Kokoro、Matcha、KittenTTS、Supertonic（**不含** XTTS/ChatTTS/F5/Orpheus/CSM/IndexTTS/CosyVoice/MaskGCT/MegaTTS3/Spark/StyleTTS2） | https://github.com/k2-fsa/sherpa-onnx ｜ https://k2-fsa.github.io/sherpa/onnx/tts/all/English/index.html |
| **Blaizzy/mlx-audio** | Python，MLX | Kokoro、Spark-TTS、CSM、Orpheus（另含 Dia/Qwen3-TTS/Higgs/Voxtral 等） | https://github.com/Blaizzy/mlx-audio |
| **mattmireles/kokoro-coreml** | Swift SDK + HF `.mlpackage` | Kokoro（ANE，实测最完整） | https://huggingface.co/mattmireles/kokoro-coreml |
| **thewh1teagle/kokoro-onnx** | Python，ONNX Runtime | Kokoro | https://github.com/thewh1teagle/kokoro-onnx |
| **endink/F5-TTS-ONNX-Exporter** | ONNX | F5-TTS | https://github.com/endink/F5-TTS-ONNX-Exporter |
| **pltobing/XTTSv2-Streaming-ONNX** | ONNX | XTTS-v2 | https://huggingface.co/pltobing/XTTSv2-Streaming-ONNX |
| **canopyai orpheus-cpp** | llama.cpp 兼容后端 | Orpheus | https://github.com/canopyai/Orpheus-TTS/blob/main/additional_inference_options/no_gpu/README.md |
| **senstella/csm-mlx** ／ **endlessreform/csm_mlx** | Python，MLX | CSM-1B | https://github.com/senstella/csm-mlx ｜ https://github.com/endlessreform/csm_mlx |
| **Jup33Q/mlx-indextts** ／ **solar2ain/mlx-indextts** | Python，MLX | IndexTTS-2 | https://github.com/Jup33Q/mlx-indextts ｜ https://github.com/solar2ain/mlx-indextts |

---

## 附：一手来源 URL 清单（按模型）

**IndexTTS-2** — https://github.com/index-tts/index-tts ｜ https://github.com/index-tts/index-tts/issues/193 ｜ https://github.com/index-tts/index-tts/pull/78 ｜ https://raw.githubusercontent.com/index-tts/index-tts/main/LICENSE ｜ https://huggingface.co/IndexTeam/IndexTTS-2 ｜ https://huggingface.co/mlx-community/IndexTTS-2-MLX ｜ https://huggingface.co/vanch007/mlx-indextts2-standard-fp16 ｜ https://github.com/soniqo/speech-swift/blob/main/docs/models/indextts2.md ｜（二手）https://exocreate.online/blog/indextts2-mac-install-guide ｜（二手）https://exocreate.online/blog/indextts2-apple-silicon-tts-fix

**CosyVoice** — https://github.com/QwenAudio/CosyVoice ｜ https://github.com/QwenAudio/CosyVoice/issues/134 ｜ https://github.com/QwenAudio/CosyVoice/issues/1767 ｜ https://github.com/QwenAudio/CosyVoice/pull/1129 ｜ https://github.com/QwenAudio/CosyVoice/pull/1869 ｜ https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/LICENSE ｜ https://huggingface.co/FluidInference/CosyVoice3-0.5B-coreml ｜ https://github.com/FluidInference/FluidAudio/pull/582 ｜ https://soniqo.audio/guides/cosyvoice ｜ https://github.com/lonelygo/CosyVoice

**GPT-SoVITS** — https://github.com/RVC-Boss/GPT-SoVITS ｜ https://github.com/RVC-Boss/GPT-SoVITS/issues/2612 ｜ https://github.com/RVC-Boss/GPT-SoVITS/pull/183 ｜ https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/LICENSE ｜ https://github.com/baicai-1145/GPT-SoVITS-CPUFast

**F5-TTS** — https://github.com/SWivid/F5-TTS ｜ https://github.com/SWivid/F5-TTS/issues/443 ｜ https://github.com/SWivid/F5-TTS/issues/475 ｜ https://raw.githubusercontent.com/SWivid/F5-TTS/main/LICENSE ｜ https://github.com/endink/F5-TTS-ONNX-Exporter ｜ https://github.com/soniqo/speech-swift/blob/main/docs/models/f5-tts.md

**Fish-Speech** — https://github.com/fishaudio/fish-speech ｜ https://github.com/fishaudio/fish-speech/issues/356 ｜ https://github.com/fishaudio/fish-speech/issues/789 ｜ https://github.com/fishaudio/fish-speech/pull/259 ｜ https://github.com/fishaudio/fish-speech/pull/830 ｜ https://raw.githubusercontent.com/fishaudio/fish-speech/main/LICENSE ｜ https://huggingface.co/fishaudio/openaudio-s1-mini ｜ https://docs.fish.audio/developer-guide/getting-started/changelog ｜ https://github.com/soniqo/speech-swift/blob/main/docs/models/fish-audio-s2-pro.md

**ChatTTS** — https://github.com/2noise/ChatTTS ｜ https://github.com/2noise/ChatTTS/issues/772 ｜ https://github.com/2noise/ChatTTS/issues/263 ｜ https://raw.githubusercontent.com/2noise/ChatTTS/main/LICENSE

**Spark-TTS** — https://github.com/SparkAudio/Spark-TTS ｜ https://github.com/SparkAudio/Spark-TTS/issues/12 ｜ https://github.com/SparkAudio/Spark-TTS/issues/95 ｜ https://raw.githubusercontent.com/SparkAudio/Spark-TTS/main/LICENSE ｜ https://huggingface.co/mlx-community/Spark-TTS-0.5B-bf16

**MegaTTS3** — https://github.com/bytedance/MegaTTS3 ｜ https://github.com/bytedance/MegaTTS3/issues/14 ｜ https://github.com/bytedance/MegaTTS3/issues/47 ｜ https://github.com/bytedance/MegaTTS3/issues/44 ｜ https://github.com/bytedance/MegaTTS3/issues/21 ｜ https://raw.githubusercontent.com/bytedance/MegaTTS3/main/LICENSE ｜ https://arxiv.org/html/2502.18924v4

**MaskGCT** — https://github.com/open-mmlab/Amphion ｜ https://github.com/open-mmlab/Amphion/blob/main/models/tts/maskgct/README.md ｜ https://github.com/open-mmlab/Amphion/issues/415 ｜ https://github.com/open-mmlab/Amphion/issues/450 ｜ https://raw.githubusercontent.com/open-mmlab/Amphion/main/LICENSE ｜ https://huggingface.co/amphion/maskgct

**XTTS-v2** — https://github.com/coqui-ai/TTS ｜ https://github.com/coqui-ai/TTS/issues/3649 ｜ https://raw.githubusercontent.com/coqui-ai/TTS/dev/LICENSE.txt ｜ https://huggingface.co/coqui/XTTS-v2 ｜ https://coqui.ai/cpml ｜ https://huggingface.co/pltobing/XTTSv2-Streaming-ONNX ｜（二手）https://www.qwe.edu.pl/ai-tools/install-xtts-v2-coqui-tts-fork

**StyleTTS2** — https://github.com/yl4579/StyleTTS2 ｜ https://github.com/yl4579/StyleTTS2/issues/114 ｜ https://github.com/yl4579/StyleTTS2/issues/37 ｜ https://raw.githubusercontent.com/yl4579/StyleTTS2/main/LICENSE ｜ https://github.com/FluidInference/FluidAudio/pull/582 ｜ https://github.com/NeuralVox/StyleTTS2 ｜ https://pypi.org/project/styletts2/

**Kokoro** — https://github.com/hexgrad/kokoro ｜ https://raw.githubusercontent.com/hexgrad/kokoro/main/README.md ｜ https://raw.githubusercontent.com/hexgrad/kokoro/main/LICENSE ｜ https://huggingface.co/hexgrad/Kokoro-82M ｜ https://huggingface.co/mattmireles/kokoro-coreml ｜ https://github.com/mattmireles/kokoro-coreml ｜ https://github.com/thewh1teagle/kokoro-onnx ｜ https://github.com/k2-fsa/sherpa-onnx/pull/1713 ｜ https://k2-fsa.github.io/sherpa/onnx/tts/all/English/index.html ｜ https://github.com/soniqo/speech-swift/blob/main/docs/models/kokoro-tts.md ｜ https://github.com/soniqo/speech-swift/blob/main/docs/benchmarks/ios-coreml.md ｜ https://huggingface.co/mlx-community/Kokoro-82M-bf16

**Piper** — https://github.com/OHF-Voice/piper1-gpl ｜ https://github.com/OHF-Voice/piper1-gpl/blob/main/COPYING ｜ https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md ｜ https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/BUILDING.md ｜ https://pypi.org/project/piper-tts/ ｜ https://huggingface.co/rhasspy/piper-voices ｜ https://k2-fsa.github.io/sherpa/onnx/tts/piper.html ｜ https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html ｜ https://github.com/itsabhishekolkha/piper-arm-build ｜（二手）https://github.com/shigeyukey/my_addons/issues/348

**Orpheus-TTS** — https://github.com/canopyai/Orpheus-TTS ｜ https://github.com/canopyai/Orpheus-TTS/issues/178 ｜ https://github.com/canopyai/Orpheus-TTS/blob/main/additional_inference_options/no_gpu/README.md ｜ https://raw.githubusercontent.com/canopyai/Orpheus-TTS/main/LICENSE ｜ https://huggingface.co/canopylabs/orpheus-3b-0.1-ft ｜ https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf ｜ https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-4bit ｜ https://github.com/Blaizzy/mlx-audio/issues/74 ｜（二手）https://simplismart.ai/blog/orpheus-tts-simplismart

**CSM-1B** — https://github.com/SesameAILabs/csm ｜ https://github.com/SesameAILabs/csm/pull/52 ｜ https://raw.githubusercontent.com/SesameAILabs/csm/main/LICENSE ｜ https://huggingface.co/sesame/csm-1b（gated） ｜ https://discuss.huggingface.co/t/csm-1b-model-and-metal-shader-issue/173053 ｜ https://github.com/senstella/csm-mlx ｜ https://github.com/endlessreform/csm_mlx ｜ https://huggingface.co/mlx-community/csm-1b ｜ https://github.com/soniqo/speech-swift/blob/main/docs/models/csm.md ｜ https://github.com/ggml-org/llama.cpp/pull/12648 ｜ https://huggingface.co/johnbenac/sesame-csm-1b-GGUF-encoder ｜ https://huggingface.co/docs/transformers/main/en/model_doc/csm

**通用** — https://github.com/Blaizzy/mlx-audio ｜ https://github.com/soniqo/speech-swift ｜ https://soniqo.audio/architecture ｜ https://swiftpackageindex.com/FluidInference/FluidAudio ｜ https://k2-fsa.github.io/sherpa/onnx/tts/all/ ｜ https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
