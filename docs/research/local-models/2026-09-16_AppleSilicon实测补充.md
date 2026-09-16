# macOS / Apple Silicon 本地 TTS 引擎选型（内嵌运行时 · torch-free 优先）

**调研日期**：2026-09-16（所有 URL 均于该日读取）
**实测机器**：MacBook Pro Mac17,9 / **Apple M5 Pro** (5 Super + 10 Performance cores) / **24 GB 统一内存** / macOS 26（即目标机型本身）
**方法**：一手来源 = PyPI JSON API、Hugging Face API / raw 文件、GitHub REST API、`gh api`、官方 README/LICENSE；**并在本机真机跑通 4 条 torch-free 中文管线**（uv 隔离环境）。本环境 `web_fetch` 不可用，全部用 `curl` / `gh api`。

> **本文与既有报告的关系**：本仓已有 `research/apple-silicon-tts-status.md`（15 个模型的 MPS 崩溃证据）与 `docs/research/notes-mac-rtf.md`。本篇**不重复**那些内容，只补齐两件它们没有的东西：(1) **本机 M5 Pro 真机实测的 RTF / 内存 / ASR 复核数据**；(2) `index-tts-2.5-mlx/mnn/onnx` 三个包的可信度专项审计。
> 既有报告中「Kokoro 中文 = D 级」的说法需要更正，见 §3.3。

---

## 0. 结论先行

**推荐排序（Mac 侧 v1 默认本地引擎）**

| 名次 | 引擎 | 一句话理由 |
|---|---|---|
| **① v1 默认** | **Qwen3-TTS（mlx-audio + `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit`）** | 唯一同时满足「零样本复刻 + 中文第一梯队 + **Apache-2.0 权重** + 由 7.9k★ 项目维护 + 本机实测热 RTF 0.35–0.40 / 峰值 2.2 GB」的组合。商业许可最干净，供应链风险最低。 |
| **② 效果首选 / 备选 A** | **IndexTTS-2.5（MLX）**，但**必须替换掉 `index-tts-2.5-mlx` 这个包** | 中文/数字/多语言混合朗读本机实测 3/3 完全正确、社区有 CV3-Eval 数字（zh WER 4.36 / SS 77.10）。但 ①作者个人包不可作为核心依赖（§4），②实测 RTF 0.83 比作者宣称的 0.45 慢 1.8×。应走 `vanch007/mlx-indextts2`(MIT) 或 `mlx-community/IndexTTS-2.5-fp16` 权重。 |
| **③ 备选 B（速度/交互）** | **Chatterbox Multilingual V3（mlx-audio）** | MIT 代码+权重、热 RTF 0.41、3.0 GB。**但中文数字/年份会读错**（本机 ASR 复核 1/3 严重错），中文产品需先在管线里补数字归一化并复测。 |
| **④ 轻量兜底（无复刻）** | **Kokoro-82M-v1.1-zh（kokoro-onnx）** | RTF **0.156**、内存 **1.27 GB**、Apache-2.0 —— 快得离谱且极其省资源。但**不支持任何声音复刻**，只能做「固定音色 TTS」，与产品核心定义（`CONTEXT.md`：声音复刻）不符，只能当第二功能或离线兜底。 |
| ❌ 排除 | Piper / Spark-TTS(MLX) | Piper：**无复刻能力**，中文只有 3 个发音人且 `huayan` 数据集许可 "Unknown"。Spark-TTS MLX 权重是 **cc-by-nc-sa-4.0（非商用）**。 |

**一句话给决策**：v1 只用 **Qwen3-TTS 0.6B-Base(8bit) MLX** 一个引擎；把 **IndexTTS-2.5 MLX** 作为「效果优先档」在 v1.1 接入（用第三方 MIT 仓库，不用 `index-tts-2.5-mlx` 包）；**不要**把 `index-tts-2.5-mlx/mnn/onnx` 中的任何一个作为核心依赖。

---

## 1. 总表：候选 × 八维

RTF = 合成耗时 ÷ 音频时长（<1 = 快于实时）。**「实测」列全部是本机 M5 Pro 24GB 今天现跑的数字**；参考音频 = macOS `say -v Tingting` 9.47 s 中文（22.05 kHz mono），ASR 复核用 `faster-whisper medium int8`（CPU）。

| 引擎 | 后端 / 是否 torch-free | 中文效果 | 零样本复刻 | 实测 RTF（M5 Pro） | 实测峰值内存 | 权重体积 | 许可（代码 / 权重） | 成熟度 |
|---|---|---|---|---|---|---|---|---|
| **Qwen3-TTS 0.6B-Base 8bit** | MLX，torch-free | 上游 zh WER **3.27**（同表最低）/ SS 73.02；本机 ASR 3/3 正确 | ✅（ref audio + ref_text） | **0.35–0.40**（热，含 1 行冷启动 0.56） | **2.2 GB** | 1.9 GB | mlx-audio **MIT** / 权重 **apache-2.0** | 上游 HF 3.78M 下载、516★；mlx-audio 7,894★ |
| **IndexTTS-2.5（yunfengwang int8）** | MLX，torch-free | 上游 zh WER 4.36 / SS 77.10；本机 ASR **3/3 完全正确**（含「2025 年 / 100 万」） | ✅（仅需 ref audio） | **0.78–0.88**，均值 **0.826** | **4.9 GB** | 4.6 GB | 包内代码无 LICENSE 文件；权重继承 bilibili 协议（可商用，见 §4.4） | ⚠️ **单人 0.1.x，无任何源码仓库**，见 §4 |
| **IndexTTS-2.5（vanch007/mlx-indextts2）** | MLX，torch-free | 同上 | ✅ | 未实测；作者自测 RTF **0.7767**（与我实测 0.826 一致） | 未实测 | 4.6 GB（用 mlx-community 权重） | 代码 **MIT** / 权重同 bilibili 协议 | 4★、2026-08-11 最后提交、单人 |
| **Chatterbox Multilingual V3** | MLX（mlx-audio），torch-free | ✗ 数字/年份读错（本机 ASR「2025 年处理 100 万」→ 乱码）；上游无 zh 分语言 WER | ✅ | **0.39–0.48**，均值 **0.412** | 3.3 GB | 3.0 GB | 上游 **MIT**（代码+权重）；mlx 转换 **mit** | 上游 26,442★ / HF 1.87M 下载 |
| **Kokoro-82M-v1.1-zh** | ONNX Runtime（CPU），torch-free | 100 h 中文数据；本机 ASR 3/3 正确；早期 v1.0 的 zh 音色仅 D 级（见 §3.3） | ❌ **不支持** | **0.156**（最快） | **1.27 GB** | 0.38 GB | kokoro-onnx **MIT** / 权重 **apache-2.0** | 2,727★、PyPI 月下载 **390,669** |
| **CosyVoice3-0.5B CoreML** | CoreML（Swift/FluidAudio），torch-free | 同表 zh **SS 80.01（最高）**、WER 3.84 | ✅（11 个内置零样本音色包） | 未实测（需 Xcode/Swift 构建） | 未实测 | ~6.6 GB | 转换卡 **apache-2.0**；上游 Apache-2.0 | FluidAudio 2,766★、Apache-2.0、活跃 |
| **VoxCPM2** | MLX（mlx-audio），torch-free | zh WER 3.88 / SS 74.99；30 语种 | ✅ + 音色设计 | 未实测 | 未实测 | 未核实 | 权重 **apache-2.0** | 上游 380K 下载、1,617 likes |
| **MOSS-TTS-Nano-100M** | MLX（mlx-audio） | 20 语种含 zh，小模型 | ✅ | 未实测 | 未实测 | 未核实 | **apache-2.0** | mlx 卡 563 下载 |
| **Ming-omni-tts-0.5B** | MLX（mlx-audio） | EN/ZH，5 亿参数 dense | ✅ + 风格控制 | 未实测 | 未实测 | 未核实 | 上游 **apache-2.0** | 上游 17K 下载、38 likes |
| **Spark-TTS-0.5B MLX** | MLX（mlx-audio） | EN/ZH | ✅ | 未实测 | 未实测 | 未核实 | ⚠️ 权重 **cc-by-nc-sa-4.0（非商用）** | 上游 15,236★ |
| **F5-TTS MLX**（lucasnewman） | MLX | 中文一般（社区） | ✅ | 未实测 | 未实测 | 未核实 | 仓库 MIT；**上游权重 CC-BY-NC** | 644★，**2025-03-19 后停更** |
| **Piper** | ONNX Runtime | zh_CN 3 个发音人（huayan/chaowen/xiao_ya），quality=medium | ❌ **无复刻** | 未实测 | 未实测 | ~60 MB/音色 | 代码 **GPL-3.0**；音色许可**逐条不同**（huayan 数据集 "Unknown"） | 5,593★，OHF 维护、活跃 |
| ❌ **Supertonic** | ONNX | **31 语种里没有中文** | 未核实 | — | — | — | — | 仓库已 **archive**，官方停止支持 |

> 「未实测 / 未核实」= 本次没有一手数据，**不做推断**。

### 1.1 mlx-audio 支持的 TTS 模型全表（一手，README `## Supported Models`，2026-09-16 读取）
`https://github.com/Blaizzy/mlx-audio#supported-models`

Kokoro（EN/JA/ZH/FR/ES/IT/PT/HI）、KittenTTS（EN）、Qwen3-TTS（ZH/EN/JA/KO+）、Higgs Audio v3（100 语）、OmniVoice（646+）、CSM/MisoTTS（EN）、Dia（EN）、OuteTTS（EN）、**Spark**（EN/ZH）、**Chatterbox v2/v3**（23 语）、Soprano（EN）、Ming Omni TTS BailingMM 16.8B-A3B（EN/ZH）、Ming Omni Dense 0.5B（EN/ZH）、KugelAudio（欧洲语）、Voxtral TTS（9 语）、rumik-oss 1（Indic）、**VoxCPM2**（30 语）、**LongCat-AudioDiT**（ZH/EN）、MeloTTS（EN）、MOSS-TTS / MOSS-TTS-Nano（31 / 20 语）、Higgs Audio v2（EN/ZH/KO/DE/ES）。

**额外发现（一手）**：`mlx-audio` 源码里**还有一个 README 未列出的 `indextts` 实现**（`mlx_audio/tts/models/indextts/`：`indextts.py` 15 KB + `gpt2.py` / `bigvgan.py` / `conformer.py` / `perceiver.py` / `ecapa_tdnn/`）。但对照 `mlx-community/IndexTTS-2-fp16/config.json`（`gpt.model_dim=1280, layers=24, condition_type=conformer_perceiver`）可确认它对应的是 **IndexTTS-2（v2）**；`mlx-community/IndexTTS-2.5-fp16` 的卡片明确写它由 `vanch007/mlx-indextts2`（Python-MLX，MIT）与 `xocialize/mlx-indextts2-swift` 消费，**不是 mlx-audio**。→ **不要以为装了 mlx-audio 就能跑 IndexTTS-2.5。**

---

## 2. 本机实测：原始数据（可复现）

环境：`uv 0.11.28`，`uvx`/`uv run` 隔离环境，Python 3.12。参考音频 `ref.wav` = 9.47 s / 22.05 kHz / mono（`say -v Tingting -r 180` + `ffmpeg -ar 22050 -ac 1`）。
测试文本：
- T1 `人工智能模型在2025年处理了100万条数据。`
- T2 `欢迎使用这款桌面软件，它可以在本地完成声音复刻与文字转语音。`
- T3 `千里之行，始于足下。`

### 2.1 IndexTTS-2.5（`index-tts-2.5-mlx` 0.1.1，int8）
```
load=1.17s   build_speaker=0.88s
line0: audio=4.86s synth=4.07s rtf=0.838
line1: audio=6.50s synth=5.07s rtf=0.779
line2: audio=4.39s synth=3.88s rtf=0.883
TOTAL audio=15.75s synth=13.02s RTF=0.826     maxrss=4908 MB
```
CLI 冷跑（首次，含权重元数据拉取）：`load 5.92s / clone 9.08s / synth 4.86s / RTF 1.006`。
分阶段（热）：`gpt=0.61s  cfm=1.66s  bigvgan=1.36s`。
**ASR 复核**：T1/T2/T3 全部逐字正确（含「2025 年」「100 万」）。混合语种 `Please note that this is a mixed language test, 都可以。` 也完整正确。

### 2.2 Chatterbox Multilingual V3（mlx-audio 0.5.4 + `mlx-community/chatterbox-multilingual-v3`）
```
load=18.01s
line0: audio=5.54s synth=2.28s rtf=0.412
line1: audio=7.18s synth=2.77s rtf=0.386
line2: audio=2.66s synth=1.28s rtf=0.483
TOTAL audio=15.38s synth=6.34s RTF=0.412     maxrss=3335 MB
```
**ASR 复核：T1 严重错乱**（`人工智能模型在推拋枕T5GIN處理了環粒條數據`）、T3 错（`千里之行十余足下`）、T2 正确。
日志里的直接原因线索：`pkuseg not available - Chinese segmentation will be skipped` —— mlx-audio 的 chatterbox 路径**没有中文分词/数字归一化**。上游 Chatterbox 的 `punc_norm` 未在此转换中被复现。
> ⚠️ 这意味着「Chatterbox 中文效果」不能直接引用英文 demo 的主观印象；在本机、无额外归一化的情况下，**数字与年份是坏的**。

### 2.3 Qwen3-TTS 0.6B-Base 8bit（mlx-audio + `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit`）
```
load=68.41s   （含 12 文件下载；缓存热加载后显著更快）
line0: audio=4.88s synth=4.46s rtf=0.913   ← 含 Metal kernel 首次编译
line1: audio=6.48s synth=2.27s rtf=0.351
line2: audio=2.40s synth=0.97s rtf=0.403
TOTAL 13.76s / 7.70s  RTF=0.559（含冷首行）；排除首行热 RTF ≈ 0.37
maxrss=2164 MB
```
**ASR 复核**：T1 / T2 / T3 全部正确（T2 的「与」被识别为同音「语」，属 ASR 侧假阳，非引擎错）。

### 2.4 Kokoro-82M-v1.1-zh（kokoro-onnx 0.6.1 + ONNX Runtime CPU）
```
load=3.49s（含 jieba 词典构建）
line0: g2p=0.22s audio=4.57s synth=0.72s rtf=0.158
line1: g2p=0.00s audio=5.95s synth=0.89s rtf=0.150
line2: g2p=0.00s audio=2.18s synth=0.36s rtf=0.168
TOTAL audio=12.70s synth=1.98s RTF=0.156     maxrss=1268 MB
```
**ASR 复核**：3/3 正确。
**警告日志**：`Warning: en_callable is None, so English may be removed` → 中英混读会掉英文（IndexTTS / Qwen3-TTS 在这个用例上正常）。

### 2.5 实测数据可信度说明
- 上述 RTF 是**单次运行**，不是多次取中位数；机器当时并非完全空载。视为「量级可信，±15% 内」。
- **没有做主观听感评测（无 MOS / ABX）**。ASR 复核只能证明「可懂」，不能证明「像不像 / 自不自然」。
- 参考音频是 `say` 合成音，不是真人录音；克隆相似度的绝对值不具代表性。**音色相似度结论本报告不给**。

---

## 3. 逐候选详情

### 3.1 IndexTTS-2.5 MLX
- **上游**：`index-tts/index-tts` 23,995★，最后 push 2026-08-18；论文 arXiv:2601.03888；HF `IndexTeam/IndexTTS-2.5` 14,347 下载 / 205 likes，lastModified 2026-08-12。
- **中文效果（一手，上游 README §📊 Evaluation，CV3-Eval）**：IndexTTS2.5 0.8B → **zh WER 4.36 / SS 77.10**；IndexTTS2.5-RL → **zh WER 3.93 / SS 77.92**。同表中 VoxCPM2 3.88/74.99、OmniVoice 3.41/72.99、CosyVoice3-0.5B 3.84/**80.01**、Qwen3-TTS 1.7B **3.27**/73.02、Fish Audio S2 Pro 3.62/67.79、Moss-TTS 1.5 4.02/72.68、FireRedTTS-2 8.22/68.10。（**这是 IndexTTS 团队自己的评测表，属厂商自评**。）
- **上游 RTF（一手，README §⚡ Inference Speed）**：RTX 4090 上 2.5 bf16 overall **0.2065**、fp32 0.2060 —— 与 Mac 无关，仅供参考口径。
- **零样本复刻**：✅，≤15 s 参考音频，`build_speaker` 一次可复用到无数句。
- **集成**：`uvx index-tts-2.5-mlx synth ...` 或 `pip install`；依赖 `mlx, transformers, librosa(+numba/llvmlite), tiktoken, wetext, kaldi-native-fbank, huggingface-hub, soundfile, pyyaml, numpy`，**无 torch**（实测 `torch ABSENT`，`wetext 0.1.8` 是纯 Python 包，不需要 pynini）。wheel 53 KB，源码全在 wheel 里。首次自动从 HF 拉 ~5 GB（实测落盘 4.6 GB）。

### 3.2 IndexTTS-2.5 MNN / ONNX
- **MNN（`index-tts-2.5-mnn` 0.1.1）**：CPU 推理。fp16 权重 ~3.6 GB / fp32 ~7 GB；HF 仓库**两套都放**，全量 11.26 GB。作者自述 Apple Silicon 上 MLX 版「~7× faster than real-time」（即 RTF≈0.14）—— **与 MLX 版 README 自己写的 0.45–0.47 互相矛盾，且与我实测 0.826 差 5–6×**。CPU 版对 Mac 无意义（本机已有 GPU 路径）。
- **ONNX（`index-tts-2.5-onnx` 0.1.0）**：fp32，HF 仓库 11.73 GB。作者自述「ORT CPU 上 int8 更慢更不准，故只发 fp32」。定位是「Linux/Windows/CUDA 可移植」。对 Mac 无意义。
- 两者都**没有 GitHub 源码仓库**（见 §4）。

### 3.3 Kokoro-82M
- **必须区分两个权重**（一手更正）：
  - `hexgrad/Kokoro-82M`（v1.0）：Apache-2.0，VOICES.md 里 **8 个中文音色全部是 Overall Grade D，训练时长 MM minutes** → 「中文很弱」这个口碑来自这里。
  - `hexgrad/Kokoro-82M-v1.1-zh`：Apache-2.0，**>100 h 中文数据、100 个中文发音人**，21,552 下载 / 228 likes。本机实测用它跑中文 3/3 正确。→ **既有报告「Kokoro 中文 = D 级」需要限定为 v1.0；做中文要用 v1.1-zh。**
- **无复刻**：只能选 103 个预置音色（`zf_001`…`zm_010`），这是它对本项目的**决定性缺陷**。
- **集成**：`kokoro-onnx` 0.6.1（thewh1teagle，2,727★，MIT，最后 push 2026-09-01，PyPI 月下载 **390,669**）。模型文件不在 pip 里，需从 GitHub Releases 下 `kokoro-v1.1-zh.onnx`（325 MB）+ `voices-v1.1-zh.bin`（54 MB）+ HF `config.json`。中文 G2P 需 `misaki-fork[zh]`（拉 jieba 等）。
- 另有 mlx-audio 路线（`mlx-community/Kokoro-82M-bf16`，85,637 下载、apache-2.0），但那是 v1.0，中文音色质量按上面的 D 级。

### 3.4 Chatterbox
- 上游 `resemble-ai/chatterbox` 26,442★，MIT；HF `ResembleAI/chatterbox` **1,866,331 下载 / 1,788 likes，license: mit，标签含 `zh`**（23 语）。V3 = 500M，改进说话人相似度、降低幻觉。
- **MLX 路线存在且成熟**：`mlx-community/chatterbox-multilingual-v3`（2,205 下载，license: mit）+ mlx-audio 原生实现。
- **ONNX / CoreML 路线也存在**（一手 HF 搜索）：`ResembleAI/chatterbox-turbo-ONNX`（3,525 下载/85 likes）、`onnx-community/chatterbox-ONNX`（2,892/26）、`onnx-community/chatterbox-multilingual-ONNX`（814/55）、`aufklarer/Chatterbox-Flash-CoreML`（**15,081 下载**，mit，`en` only）。
- **中文效果**：上游**未发布分语种 WER/SS**（→ 未核实）。本机实测暴露数字归一化缺失，见 §2.2。
- **零样本复刻**：✅（需要参考音频）。

### 3.5 Piper
- `rhasspy/piper`（11,284★）已 **archived**；现役是 `OHF-Voice/piper1-gpl`（5,593★，**GPL-3.0**，C++，最后 push 2026-09-15，README 仍在招 maintainer）。PyPI `piper-tts` 1.8.0（2026-09-04）。
- **中文**：`rhasspy/piper-voices` 下只有 3 个 zh_CN 音色（chaowen / huayan / xiao_ya），quality=medium。MODEL_CARD：`huayan` 数据集 = PlayVoice/HuaYan_TTS，**License: Unknown**，且是「从英文 lessac 微调来的」；`chaowen` 数据集 CC0。
- **无零样本复刻** → 与产品核心需求直接冲突，**排除**。
- 音色许可**逐条不同**（既有报告已逐条查过，此处不重复）。

### 3.6 CosyVoice 的 CoreML 路线
- **有，而且很实**：`FluidInference/CosyVoice3-0.5B-coreml`（`apache-2.0`，language 只标 `zh`）。四个 stage 分别冻结：LLM-Prefill / LLM-Decode 走 **CPU+ANE**，Flow-N250 走 **CPU+GPU**（卡片原文：「pure CPU overflows fused LayerNorm → **NaN**; ANE refuses to compile; GPU path uses fp32 accumulators internally and is stable」），HiFT 走 **CPU+ANE**。**磁盘 ~6.6 GB**。内置 **11 个零样本音色包**（含 AISHELL-3 10 个说话人）+ Qwen2 BPE tokenizer。
- 消费方是 **Swift 包 `FluidInference/FluidAudio`**（2,766★，Apache-2.0，最后 push 2026-09-14）—— 需要 Xcode/Swift 集成，**不是 Python 路线**。
- 另有一条 MLX/Swift 路线：`soniqo/speech-swift`（1,178★，Apache-2.0）的 `CosyVoiceTTS` 模块。
- **中文效果是全部候选里最好的**（SS 80.01，CV3-Eval）。**如果未来把 macOS 侧改成原生 Swift 而不是内嵌 Python，这是第一优先。**
- ⚠️ 对「app 自带 Python 内嵌运行时」的架构来说，CoreML/Swift 路线需要额外一层桥接，v1 不建议。

### 3.7 F5-TTS / Spark-TTS 的 MLX 移植
- **F5-TTS MLX**：`lucasnewman/f5-tts-mlx` 644★，**最后 push 2025-03-19（约 18 个月未更新）**。另有 `Glosik` 56★、`f5-tts-swift` 91★。上游 `SWivid/F5-TTS` 代码 MIT 但**权重 CC-BY-NC**（非商用）→ **不能商用**。
- **Spark-TTS MLX**：`mlx-community/Spark-TTS-0.5B-bf16`（1,039 下载），随 mlx-audio 的 `spark` 模块。**HF 权重卡 license = `cc-by-nc-sa-4.0`（非商用）** —— 注意 GitHub `SparkAudio/Spark-TTS` 仓库 LICENSE 是 Apache-2.0，**两者不一致，以权重卡为准**。→ **不能商用**。
- 其他：`huangyuan3h/Spark-TTS-MLX` 0★；`p-diegmann/F5-TTS-MLX` 2★。

### 3.8 其他值得留意的（一手发现，非本任务要求但影响选型）
- `jamiepine/voicebox` **53,981★**（MIT，TypeScript，2026-01-25 创建）—— "The open-source AI voice studio. Clone, dictate, create." 这是本项目最直接的竞品，值得单独做一次输赢分析（本报告不含）。
- `argmaxinc/ttskit-coreml`（HF，11,192 下载，tags 含 `qwen3-tts`）—— **Argmax（WhisperKit 团队）** 在做 CoreML TTS kit；`argmaxinc/argmax-oss-swift` 6,366★。商用级供应商，中文/复刻能力本次未核实。
- `FluidInference/pocket-tts-coreml` 1,495 下载、`FluidInference/qwen3-tts-coreml`（apache-2.0，含 `zh`）。

---

## 4. 专项：`index-tts-2.5-mlx / mnn / onnx` 可信度审计

这是本篇最重要的部分。**结论：不能作为产品的核心依赖。**

### 4.1 作者身份（一手）
- PyPI `Author` 字段 = `yunfengwang`；HF namespace = `yunfengwang`。
- **GitHub 上不存在 `yunfengwang` 用户**（`gh api users/yunfengwang` → 404）。
- 但 HF 上 `yunfengwang` 的大概率是 GitHub **`vra`**：`gh api users/vra` → `{"name":"Yunfeng Wang","created_at":"2013-09-28","followers":211,"public_repos":120,"bio":"Python|Linux|ComputerVision"}`；且 `vra/indextts-onnx`（IndexTTS2 ONNX 包）的 README 指向 HF `vra/indextts-onnx`。
- **判定：真人、有长期开源履历（`vra/flopth` 131★、`vra/talkGPT4All` 152★、`vra/Thinking-with-Visual-Primitives-pytorch` 166★），不是马甲。但也不是机构。**

### 4.2 源码仓库状态（决定性）
| 仓库 | 状态 |
|---|---|
| `vra/index-tts-2.5-mlx` | **不存在（404）** |
| `vra/index-tts-2.5-mnn` | **不存在（404）** |
| `vra/index-tts-2.5-onnx` | **存在但空仓库**（size=0，created 2026-08-15，0★，无 README） |
| `vra/indextts-onnx`（上一代包） | 0★，128 KB，最后 push 2026-06-21 |

→ **三个包都没有可公开审计的源码仓库、没有 issue tracker、没有提交历史、没有 CI、没有 changelog。** 唯一能拿到的源码是 PyPI wheel 里那 17–26 个文件（53 KB / 27 KB）。

### 4.3 版本、下载量、社区信号（一手）
| 包 | 版本 | 首发 | PyPI 日/周/月下载 | HF 仓库下载 / likes |
|---|---|---|---|---|
| `index-tts-2.5-mlx` | **0.1.1**（0.1.0 → 0.1.1 相隔 **2.5 小时**） | 2026-08-14 | 6 / 44 / **292** | 0 / **1** |
| `index-tts-2.5-mnn` | **0.1.1**（同样 2.5 小时内连发两版） | 2026-08-14 | 3 / 26 / **116** | 0 / 0 |
| `index-tts-2.5-onnx` | **0.1.0** | 2026-08-14 | 0 / 14 / **74** | 0 / 0 |
| （对照）`mlx-audio` | 0.5.4 | — | 10,115 / 58,693 / **368,035** | — |
| （对照）`kokoro-onnx` | 0.6.1 | — | 12,744 / 87,644 / **390,669** | — |

- 作者同期的其他 HF 仓库：`supertonic-tts-mnn`（437 下载 / 4 likes）、`MiniCPM5-2B-MNN-4bit`（43）、`MiniCPM5-2B-MNN-8bit`（55），其余 **绝大多数 0 下载**。→ 作者是「批量做 MNN/ONNX 移植」的模式，**每个项目的社区沉淀都很薄**。
- 从发布日（2026-08-14）到我来查的 2026-09-16，**整整 33 天零版本更新、`mlx` 仓库 1 个 like**。
- **自相矛盾的速度宣称**（同一作者、同一模型）：
  - `index-tts-2.5-mlx` README / PyPI：「RTF ≈ 0.45」「~2.4× faster than PyTorch MPS」
  - `index-tts-2.5-mnn` PyPI：「on Apple Silicon the MLX build (GPU) is **~7× faster than real-time**」（= RTF ≈ 0.14）
  - **我的 M5 Pro 实测：0.826**（比 0.45 慢 1.8×，比「7×」慢 5.9×）
  - 独立旁证：`vanch007/mlx-indextts2` 自测 64 行 bench **RTF 0.7767**，与我的 0.826 高度一致 → **作者宣称的数字偏乐观，我用第三方数据交叉验证过。**

### 4.4 许可（一手核对）
- PyPI metadata 的 `License:` 字段写 `Bilibili-IndexTTS`（**非 SPDX 标识符**）。
- **`yunfengwang/IndexTTS-2.5-mlx` 与 `-onnx` HF 仓库里没有 LICENSE 文件**（HF API 查 siblings 确认）；`-mnn` 返回空结果（同样没有）。
- 真正约束来自上游：`IndexTeam/IndexTTS-2.5` 的 `LICENSE` = **bilibili Model Use License Agreement**（我读了全文）。关键条款：
  - **§2.1 授予全球、非独占、免版税的许可**；**§2.2 商用是允许的**，只有当你的产品或关联方 **上月 MAU > 1 亿** 或 **上一年营收 > 10 亿人民币** 时才需另行申请授权。
  - **§1.5 明确「量化」「合并 checkpoint」都算 Derivative Work** → MLX/ONNX/MNN 转换 **受同一协议约束**。
  - **§3.4(b)「必须在每一份副本中保留全部原始版权声明和本协议的一份拷贝」** → **`yunfengwang` 的三个仓库都没做到**（无 LICENSE 文件）。
  - **§4.1(a) 分发衍生作品时必须在分发页面/文档中写明**「Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder…」→ 三个 README **都没写**。
  - **§6.1/6.2 适用中华人民共和国法律，争议提交上海仲裁委员会。**
- **对照更好的上游转换**：`mlx-community/IndexTTS-2.5-fp16` 的卡片**完整写了** license_name / license_link、**仓库里有 LICENSE 文件（10.5 KB 逐字拷贝）**、有 `conversion_report.json` 张量覆盖率校验、有 base_model revision 钉死（`d0aa86e75bb6f3437f3831e95056fa72842d89ef`）、并显式写了 §4.1(a) 要求的免责声明。**同样一份 bilibili 权重，mlx-community 的合规处理明显更规范。**
- **更正一条既有说法**：`xocialize/mlx-indextts2-swift` 的仓库描述写「weights **NonCommercial** INDEX_MODEL_LICENSE」。我读了 `IndexTeam/IndexTTS-2` 的 `LICENSE.txt` 全文 —— **它就是同一份 bilibili Model Use License Agreement，商用是允许的（同样有 1 亿 MAU / 10 亿营收阈值）**。→ **IndexTTS-2 与 2.5 的权重都不是「非商用」**，xocialize 的描述不准确。

### 4.5 供应链风险汇总（三个包共同）
1. **权重只托管在个人 HF namespace**（0 下载、0–1 like）。作者删库 / 改名 / HF 账号异常 → 用户端「一键装」当场失效。**这是最致命的一条**：产品首次安装时会去拉一个单人维护的仓库。
2. **无源码仓库 = 无法审计、无法提 issue、无法 fork 修复**。虽然 wheel 里有源码（53 KB，可读），但升级只能靠作者发新 wheel。
3. **无版本矩阵 / 无 CI**：0.1.0 → 0.1.1 相隔 2.5 小时，看不出任何测试流程。依赖 `mlx>=0.24`、`transformers>=4.45` 均为**无上界**。
4. **作者自报速度与实测差 1.8–5.9×**，且同一作者的两份文档互相矛盾 → 其他自报指标（「cosine ≥ 0.999」「73/73 完全一致」）也应打折看待。
5. **许可合规缺失**（§4.4），分发给最终用户时是我方的责任（§3.4(a)：你要为下游的违约负责）。
6. `librosa` 依赖链会拉 `numba` / `llvmlite` —— 一个以「torch-free 轻量」为卖点的包，依赖树里塞进了 numba，**内嵌分发时体积与 ABI 风险都在**（实测解析出 `librosa 1.0.0 / numba 0.67.0`）。

### 4.6 我的建议做法
- **不要**把 `index-tts-2.5-mlx` 写进产品的 requirements。
- 如果确定要 IndexTTS-2.5 的效果：**用 `vanch007/mlx-indextts2`（MIT，有源码、有 docs、有 validation 记录）或 `mlx-community/IndexTTS-2.5-fp16`（有 license/转换报告/base revision 钉死）的权重**，自己在自己的 venv 里固化版本；把 4.6 GB 权重**随安装包一起分发或放自己的 CDN**，不要依赖运行时去 HF 拉单人名下的仓库。
- 若只是想要一个「今天就能跑通的参考实现」用来做效果对比，`index-tts-2.5-mlx` 可以**当临时 baseline**，但必须记录「不可作为发布依赖」。

---

## 5. 集成难度对比（内嵌运行时视角）

| 引擎 | pip 装得上？ | 依赖重量 | 权重是否另下 | 现成 API/CLI | 内嵌分发友好度 |
|---|---|---|---|---|---|
| Qwen3-TTS（mlx-audio） | ✅ `pip install mlx-audio` | mlx + numpy + transformers + hf_hub + miniaudio + sounddevice + scipy（**无 torch**） | 是，1.9 GB（可从 `mlx-community` 镜像或自建 CDN） | ✅ `mlx_audio.tts.generate` CLI + Python `load_model().generate()` + **自带 OpenAI 兼容 REST server**（`[server]` extra） | **高**：单一上游包、Apache-2.0 权重可随包分发 |
| IndexTTS-2.5（`index-tts-2.5-mlx`） | ✅ `pip install` / `uvx` | mlx + transformers + **librosa/numba/llvmlite** + tiktoken + wetext + kaldi-native-fbank | 是，4.6 GB | ✅ CLI + Python API | **中**：技术上可行，**但供应链与许可不合规** |
| Chatterbox（mlx-audio） | ✅ | 同 mlx-audio | 是，3.0 GB（含 S3TokenizerV2 0.47 GB） | ✅ 同上 | 高（MIT） |
| Kokoro（kokoro-onnx） | ✅ | onnxruntime + numpy + **phonemizer + espeakng-loader**；中文再要 `misaki-fork[zh]`（jieba 系） | 是，0.38 GB（**在 GitHub Releases，不在 pip**） | ✅ Python API（无官方 CLI/server） | 中高：体积最小，但权重源是 GitHub Releases + 需 espeak-ng 运行时数据 |
| CosyVoice3 CoreML | ❌（Swift 包） | Xcode / SwiftPM | 是，6.6 GB | Swift API | 高（**若 app 是原生 Swift**），否则需桥接 |

**注意一个跨候选的共性坑**：本机环境里 `HF_ENDPOINT=https://hf-mirror.com` 是全局设的，那个镜像今天**反复 SSL 握手失败**（`SSL: UNEXPECTED_EOF_WHILE_READING`），我在 Chatterbox 与 Qwen3-TTS 的测试中都撞到过。→ **产品不能依赖在线拉权重，必须自托管权重**。

---

## 6. 风险清单

### 6.1 通用
1. **在线拉权重是单点故障**：HF 被墙/镜像不稳/仓库被删，都会让「一键装」失败。本次实测已亲历镜像 SSL 失败。→ **权重必须随包或走自建 CDN**。
2. **HF 上的非官方仓库可能被删**：`yunfengwang/*`、`mlx-community/*`、`aufklarer/*`、`FluidInference/*` 都是第三方命名空间。`mlx-community` 相对稳（组织、数千仓库），个人 namespace 不稳。
3. **许可与上游不一致**：Spark-TTS（GitHub Apache-2.0 vs HF 权重 cc-by-nc-sa-4.0）、F5-TTS（代码 MIT vs 权重 CC-BY-NC）、IndexTTS-1.5（HF 标 apache-2.0）vs 2/2.5（bilibili 自定义协议）。**必须按权重卡上标的来。**
4. **`librosa` → `numba`/`llvmlite`** 这条链会显著抬高内嵌分发包体积，并带来 Python 版本 ABI 锁死风险。
5. **mlx 版本漂移**：所有 MLX 包的 `mlx` 依赖都是无上界（`>=0.24` / `>=0.31.1`）。MLX 迭代很快，**必须 pin 死版本并自测**。
6. **中文数字归一化不是免费的**：Chatterbox 在本机把「2025 年 / 100 万」读错，Kokoro 会丢英文。**任何引擎上线前都要跑一遍中文数字/日期/金额/中英混读的回归集。**

### 6.2 逐包
| 包 | 坑 |
|---|---|
| `index-tts-2.5-mlx` | ①无源码仓库 ②权重在个人 namespace ③无 LICENSE 文件（违反上游 §3.4(b)）④无 §4.1(a) 免责声明 ⑤自报速度与实测差 1.8–5.9× ⑥同一作者文档自相矛盾 ⑦`librosa/numba` 依赖 ⑧33 天零更新、1 个 like |
| `index-tts-2.5-mnn` | 同上 + HF 仓库全量 **11.26 GB**（fp16+fp32 都放），按 `--quant` 过滤才不浪费；Mac 上有 GPU，CPU 路线本身无意义 |
| `index-tts-2.5-onnx` | 同上 + HF 仓库 **11.73 GB** fp32；Mac 上无收益 |
| `mlx-audio` | ①每次 load 会去 HF 校验（离线需 `HF_HUB_OFFLINE=1`）②模型模块随版本增删（`indextts` 在 README 里都没列）③`transformers>=5.14.0` 这种大版本跳跃会带副作用④Chatterbox 路径中文分词缺失 |
| `kokoro-onnx` | ①模型文件在 **GitHub Releases**（不是 HF、不在 pip）②需要 espeak-ng 运行时数据 ③中文要额外装 `misaki-fork[zh]` ④**无复刻** |
| Piper | ①**无复刻** ②代码 **GPL-3.0**（内嵌分发要评估 GPL 传染）③音色许可逐条不同、`huayan` 数据集 "Unknown" ④旧仓 archived、新仓在招维护者 |
| Chatterbox MLX | ①中文数字/年份会错 ②无中文分语种评测 ③上游 V3 卡片未公开 zh WER |
| CosyVoice3 CoreML | ①**不是 Python 路线**（需 Swift/Xcode）②~6.6 GB ③Flow 阶段**纯 CPU 会 NaN、ANE 拒绝编译**，必须走 GPU 路径 ④模型冻结在固定 shape（LLM 上下文 256 token、Flow N=250、HiFT T=500）—— 长文本必须切片 |
| Spark-TTS MLX | 权重 **cc-by-nc-sa-4.0** → **不可商用** |
| F5-TTS MLX | 权重 **CC-BY-NC** → **不可商用**；且 `lucasnewman/f5-tts-mlx` 18 个月未更新 |
| Supertonic | **31 语种里没有中文**；仓库已 archive |

---

## 7. 推荐落地方案（给产品）

**v1（Mac 只跑一个本地引擎）**
```
引擎：Qwen3-TTS 0.6B-Base-8bit (MLX, mlx-audio)
理由：唯一「零样本复刻 + 中文第一梯队 + Apache-2.0 权重可随包分发 + 上游 3.78M 下载 + 7.9k★ 维护项目 + 实测热 RTF 0.37 / 峰值 2.2 GB」的组合
内嵌：pin mlx-audio 与 mlx 版本；权重随安装包/自建 CDN；不许运行时去 HF 拉
回归集：必须包含中文数字/日期/金额/中英混读 + 复刻相似度
```

**v1.1（效果优先档，用户可选）**
```
引擎：IndexTTS-2.5 MLX
依赖来源：vanch007/mlx-indextts2（MIT 代码） 或 自研/自托管转换
权重来源：mlx-community/IndexTTS-2.5-fp16（带 LICENSE + conversion_report，合规完整）
禁止：把 index-tts-2.5-mlx 作为发布依赖
预期成本：4.6 GB 权重 + 峰值内存 4.9 GB + 热 RTF ~0.83
```

**不要做**
- ❌ 不要用 Piper / Kokoro 作为「声音复刻」引擎（它们没有复刻能力）。
- ❌ 不要用 Spark-TTS / F5-TTS 权重（非商用）。
- ❌ 不要把 `index-tts-2.5-mlx/mnn/onnx` 写进 requirements。
- ❌ v1 不要为 CosyVoice3 CoreML 改造成 Swift 架构（除非产品本来就走原生 Swift）。

**未来（如果 macOS 侧转为原生 Swift）**
- **CosyVoice3-0.5B CoreML + FluidAudio** 是效果上限最高的选项（zh SS 80.01，全表第一，Apache-2.0，2,766★ 活跃）。
- 次选 `soniqo/speech-swift`（1,178★，Apache-2.0）的 MLX/CoreML 多模型封装。

---

## 8. 一手来源（全部 2026-09-16 读取）

**PyPI JSON API**：`https://pypi.org/pypi/{index-tts-2.5-mlx|index-tts-2.5-mnn|index-tts-2.5-onnx|indextts-onnx|mlx-indextts|mlx-audio|kokoro-onnx|piper-tts}/json` ｜ 下载量 `https://pypistats.org/api/packages/{pkg}/recent`

**Hugging Face API / raw**：
`huggingface.co/api/models/yunfengwang/{IndexTTS-2.5-mlx,IndexTTS-2.5-mnn,IndexTTS-2.5-onnx}` ｜ `.../IndexTeam/IndexTTS-2.5`、`IndexTeam/IndexTTS-2`（含 `LICENSE.txt` 全文） ｜ `.../mlx-community/IndexTTS-2.5-fp16`（含 `README.md`、`LICENSE`、`tree?recursive=true`） ｜ `.../mlx-community/{chatterbox-multilingual-v3,Qwen3-TTS-12Hz-0.6B-Base-8bit,Kokoro-82M-bf16,S3TokenizerV2,VoxCPM2-bf16,LongCat-AudioDiT-1B-bf16,MOSS-TTS-Nano-100M,OmniVoice-bf16,higgs-audio-v2-3B-mlx-q8,Spark-TTS-0.5B-bf16}` ｜ `.../hexgrad/{Kokoro-82M,Kokoro-82M-v1.1-zh}`（含 `VOICES.md`、`EVAL.md`） ｜ `.../ResembleAI/chatterbox`、`onnx-community/chatterbox-ONNX`、`aufklarer/Chatterbox-Flash-CoreML` ｜ `.../FluidInference/{CosyVoice3-0.5B-coreml,qwen3-tts-coreml}` ｜ `.../argmaxinc/ttskit-coreml` ｜ `.../rhasspy/piper-voices` ｜ `.../openbmb/VoxCPM2`、`Qwen/Qwen3-TTS-12Hz-1.7B-Base`、`meituan-longcat/LongCat-AudioDiT-1B`、`inclusionAI/Ming-omni-tts-0.5B`

**GitHub API / raw**：
`api.github.com/repos/{vra/index-tts-2.5-onnx, vra/indextts-onnx, users/vra}` ｜ `Blaizzy/mlx-audio`（README、`git/trees?recursive=1`、`mlx_audio/tts/models/*`、`mlx_audio/utils.py`） ｜ `thewh1teagle/kokoro-onnx` ｜ `OHF-Voice/piper1-gpl` + `rhasspy/piper` ｜ `resemble-ai/chatterbox` ｜ `index-tts/index-tts`（README §Evaluation / §Inference Speed） ｜ `vanch007/mlx-indextts2` ｜ `xocialize/mlx-indextts2-swift` ｜ `soniqo/speech-swift` ｜ `FluidInference/FluidAudio` ｜ `k2-fsa/sherpa-onnx` ｜ `jamiepine/voicebox` ｜ `argmaxinc/argmax-oss-swift` ｜ `lucasnewman/f5-tts-mlx` ｜ `supertone-oss-archive/supertonic`

**本机实测脚本**：`/tmp/ttstest/{bench.py,kbench.py,cbbench.py,qwenbench.py,asr.py,cbasr.py,qwasr.py}`（参考音频 `/tmp/ttstest/ref.wav`）


---

> ℹ️ **补充（2026-09-16 实测，含一次自我更正）**：`HF_ENDPOINT=https://hf-mirror.com` **可用**。但 **hf-mirror 对境外出口 IP 会 308 跳回 `huggingface.co`**——在开着代理 / TUN 模式的机器上它会**表现为失效**，且症状是变慢或超时而非明确报错。本机加入直连规则后恢复正常（`/api/resolve-cache/...`，200）。桌面软件无法控制用户 VPN，故下载层不可只依赖它：优先 ModelScope 或自建 CDN。详见 `docs/research/CORRECTIONS.md` 的 C-001。
