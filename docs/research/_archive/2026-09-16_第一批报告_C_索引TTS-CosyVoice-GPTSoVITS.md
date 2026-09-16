# 开源 TTS / 声音复刻模型深度调研报告（第一批）

**调研日期：2026-09-16（UTC）** · 所有数据均通过 curl 直取一手源（GitHub README / LICENSE / HF model card / PyPI / arXiv）核实。标注「未核实」处为确实无法从一手源确认，绝不编造。

**本批覆盖：** IndexTTS-2 / IndexTTS-2.5 · FireRedTTS-2 · CosyVoice 2 & CosyVoice 3 · GPT-SoVITS

---

## 0. 先说三个跨模型的关键结论

1. **许可证风险分层极其明显。** GPT-SoVITS = MIT（最宽松）、CosyVoice 2/3 = Apache-2.0（宽松）、FireRedTTS-2 = Apache-2.0 但 README 附加「仅供学术研究」免责声明（自相矛盾，需法务判断）、**IndexTTS-2.5 = bilibili 自定义许可（非 OSI 开源，含 1 亿 MAU / 10 亿营收门槛与「禁止用于改进其他 AI 模型」条款）**。做商业桌面软件，IndexTTS 必须重点审。
2. **Apple Silicon 不是「能跑 / 不能跑」的二元问题，而是「官方上游 / 社区移植 / 完全无路」三层。** 关键发现：IndexTTS 官方代码**内置 MPS 支持**（自动探测，优先级 cuda→xpu→mps→cpu）；GPT-SoVITS **官方文档把 Apple silicon 列为已测试环境**但训练强制回落 CPU；CosyVoice 官方主干**没有 MPS**，只有未合并的社区 PR；FireRedTTS-2 **完全无 MPS**。同时四条主流模型里已有三条存在成熟的 MLX / ONNX / GGUF / MNN 第三方 Mac 路线（见各节）。
3. **这四个模型全部不在 Artificial Analysis Speech Arena 的开放权重榜上**（2026-09-16 抓取）。该榜 open-weights 头部是 Fish Audio S2 Pro、Step Audio EditX、Voxtral TTS、Magpie、Kokoro、Chatterbox 等。换言之，**用「TTS Arena 排名」为这四个模型背书是不成立的**，只能引用各家的 seed-tts-eval / CV3-Eval 自测与中文社区口碑。

---

## 1. IndexTTS-2 / IndexTTS-2.5（bilibili / B站）

- 仓库：https://github.com/index-tts/index-tts
- 论文 2.5：https://arxiv.org/abs/2601.03888 （v1 提交 2026-01-07，v5 修订 2026-08-11）
- 论文 2：https://arxiv.org/abs/2506.21619
- HF：https://huggingface.co/IndexTeam/IndexTTS-2.5 ｜ https://huggingface.co/IndexTeam/IndexTTS-2
- Demo：https://index-tts.github.io/index-tts2-5.github.io/

### 1.1 名称 / 参数量 / 发布方 / 最新版本与日期

| 项 | 值 |
|---|---|
| 发布方 | bilibili / IndexTeam（联系 indexspeech@bilibili.com） |
| IndexTTS 1.0 | 2025-03-25 |
| IndexTTS-1.5 | 2025-05-14 |
| **IndexTTS-2** | **2025-09-08**（HF 仓库创建 2025-06-18） |
| **IndexTTS-2.5** | **2026-08-10**（HF 仓库创建 2026-08-10，最后修改 2026-08-12） |
| 参数量 | IndexTTS-2：**1.5B 级**（CosyVoice 官方对比表把 Index-TTS2 列为 1.5B）；**IndexTTS-2.5：GPT 主干约 0.8B**（HF model card 自述 "Parameters: ~0.8B (GPT backbone)"） |
| 架构 | 自回归 GPT 主干（T2S）+ flow-matching 语音到 mel 解码器（S2M）+ BigVGAN 声码器。2.5 把语义 codec 帧率 50Hz→25Hz，S2M 主干从 U-DiT 换成 Zipformer（论文摘要原文） |
| 采样率 | 22.05 kHz |
| 仓库热度 | ⭐ **23,995** ｜ fork 2,859 ｜ open issues 407 ｜ 创建 2025-02-06 ｜ **最后 push 2026-08-18**（活跃） |

### 1.2 能力矩阵

| 能力 | 结论 |
|---|---|
| 零样本克隆 | ✅ 单条参考音频即可（`spk_audio_prompt`） |
| 少样本微调 | ⚠️ 仓库内**无官方训练 / 微调代码**（README 与目录树中未见 train 脚本）。**「不支持」为推论，官方未明确声明 → 未核实** |
| 文本描述设计音色 | ⚠️ 部分。没有通用「voice design」，但有**文本情感描述**：`emo_text` / `use_emo_text`（经 QwenEmotion 模块转情感向量）。2.5 用 `use_emo_text=True` 必须先以 `use_qwen_emo=True` 构造，否则抛 RuntimeError（README 明确警告） |
| 中文质量 | ✅ 极强。seed-tts-eval test-zh CER 1.03（IndexTTS-2 1.5B）；社区公认中文情感天花板 |
| 多语言 | IndexTTS-2：**中文、英文**；IndexTTS-2.5：**中文、英文、日文、西班牙文、阿拉伯文**（HF card 明确列出，含跨语种音色迁移）。韩文、法德俄**无** |
| 情感控制 | ✅ 四路输入：情感参考音频 `emo_audio_prompt`、8 维情感向量 `emo_vector` `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`、文本情感 `emo_text`、强度 `emo_alpha` 0–1。**音色与情感解耦** |
| 方言 | ❌ 官方未声明中文方言支持 → 未核实 |
| 韵律 | ✅ 2.5 新增语速控制 `duration_factor` 0.5–2.0（>1 变慢） |
| 流式 | ❌ speech-swift 的 IndexTTS2 模块明确写 "Streaming: Not supported" |
| 长文本 | ⚠️ 有自动分块（VRAM<10GB 时启用 low_vram 模式）；但英文长段落有已知对齐缺陷（见 1.7） |
| 声音转换 VC | ❌ 未暴露 VC 接口（acknowledgements 提到 seed-vc，但未作为功能提供） |

### 1.3 硬件 / 推理速度 / Apple Silicon（重点）

- **官方显存要求**：IndexTTS-2.5 HF model card 原文 —— "Requires Python 3.10–3.11, an NVIDIA GPU, and roughly **6 GB of VRAM** for inference."
- **半精度**：2.5 用 **BF16**；2.x 用 **FP16**。README 提示半精度「推理更快、显存更低、质量损失极小」。**代码内建 low-VRAM 模式**：检测到 <10GB 显存自动开启，长文本切块，并会跳过 QwenEmotion（除非强制 `--qwen_emo`）。
- **RTF（官方，RTX 4090，`kv_cache=True`）**：

| 文本长度 | 2.0 fp16 | 2.0 fp32 | 2.5 bf16 | 2.5 fp32 |
|---|---|---|---|---|
| 7 字 | 0.4004 | 0.3748 | 0.2871 | 0.2547 |
| 28 字 | 0.3257 | 0.3480 | 0.2065 | 0.1927 |
| 200 字 | 0.3244 | 0.3990 | 0.1997 | 0.2144 |
| **overall** | **0.3257** | **0.3748** | **0.2065** | **0.2060** |

  即 2.5 bf16 在 4090 上约 **4.8 倍实时**；论文摘要称 2.5 相对 2 的 RTF 提升 **2.28×**。

- **CPU 推理**：✅ 代码支持（device 回落 "cpu"，会打印 "Be patient, it may take a while to run in CPU mode"）。CPU RTF **未核实**。

#### 🔴 Apple Silicon MPS 支持：**官方支持（YES）—— 这是四个模型里最强的**

一手证据（`git clone` 后直接 grep 源码）：

- `indextts/infer_v2_5.py` 第 95–112 行，设备自动探测顺序：
  ```python
  elif torch.cuda.is_available():      self.device = "cuda:0"
  elif hasattr(torch, "xpu") and torch.xpu.is_available(): self.device = "xpu"
  elif hasattr(torch, "mps") and torch.backends.mps.is_available():
      self.device = "mps"
      self.use_bf16 = False  # Use bfloat16 on MPS is overhead than float32
  else: self.device = "cpu"
  ```
  **MPS 是官方一等公民设备**，且已针对 MPS 关闭 bf16（因为 MPS 上 bf16 反而更慢）。
- `indextts/gpt/model.py:78` 与 `:179` 有 `if torch.backends.mps.is_available(): torch.mps.empty_cache()` —— 专门的 MPS 显存清理。
- `tools/gpu_check.py` 有专门的 mps 分支，品牌名显示 "Apple MPS"，并注明 Apple Silicon 无 `get_device_name()`。
- `indextts/cli_v2.py` 多处 MPS 处理（`_is_mps_available()`）。
- **注意**：`webui.py` 的 `detect_vram_gb()` 在非 CUDA 时直接 `return None`（不报错，只是检测不到显存）。副作用：Mac 上 `LOW_VRAM=False` → `LOAD_QWEN_EMO` 默认为 True，会加载 QwenEmotion 模型。**建议显式传参控制。**

**Mac 专用轻量路线（均为第三方，非官方，2026-08-14 发布）**：

| 包 | 说明 | 一手来源 |
|---|---|---|
| `index-tts-2.5-mlx` v0.1.1 | **Apple Silicon MLX，int8 量化 GPT，去 PyTorch，RTF ≈ 0.45，比官方 PyTorch MPS 后端快约 2.4×**。要求 M1+ / macOS 13+ / Python 3.10+，走统一内存 GPU。`uvx` 一键 | https://pypi.org/project/index-tts-2.5-mlx/ |
| `index-tts-2.5-mnn` v0.1.1 | MNN CPU 推理，torch-free（numpy + pymnn），跨 x86/ARM、Linux/Windows/macOS。BigVGAN 声码器比 ONNX Runtime CPU 快约 4× | https://pypi.org/project/index-tts-2.5-mnn/ |
| `index-tts-2.5-onnx` v0.1.0 | ONNX Runtime fp32**位精确**（贪心解码与 PyTorch CPU 参考完全一致），torch-free，CPU + CUDA，权重约 7GB | https://pypi.org/project/index-tts-2.5-onnx/ |
| speech-swift (`IndexTTS2TTS`) | **原生 Swift/MLX**，macOS/iOS 可用，Apache-2.0 代码，bundle `aufklarer/IndexTTS2-MLX-fp16`（IndexTTS-**2**，1.5B 级，22.05kHz，不支持流式） | https://github.com/soniqo/speech-swift ｜ https://raw.githubusercontent.com/soniqo/speech-swift/main/docs/models/indextts2.md |

> ⚠️ 三个 PyPI 包的 `license` 字段均标为 `Bilibili-IndexTTS` —— 即许可证约束跟随上游，**换成 MLX 运行不等于换成宽松许可**。

- **磁盘占用**（HF `IndexTeam/IndexTTS-2.5` API 实取）：**合计 5.49 GB** —— `gpt.pth` 3.26GB、`qwen0.6bemo4-merge/model.safetensors` 1.19GB、`codec.pth` 0.61GB、`s2mel.pth` 0.41GB。辅助模型（w2v-bert-2.0 / MaskGCT / CAMPPlus / BigVGAN）首次运行下载到 `checkpoints/hf_cache/`，**不计入上述 5.49GB**。

### 1.4 许可证（代码 + 权重，两者都查了）

| 对象 | 许可证 |
|---|---|
| 仓库 `LICENSE` | **bilibili Model Use License Agreement**（自定义，非 OSI 开源）。仓库英文版 `LICENSE` + 中文版 `LICENSE_ZH.txt`，第 9 条明确「中英文冲突以中文版为准」 |
| HF `IndexTeam/IndexTTS-2.5` | `license: other` / `license_name: bilibili-model-license` / `license_link: LICENSE` ✅ 明确 |
| HF `IndexTeam/IndexTTS-2` | ⚠️ **cardData 中完全没有 license 字段**（仅有 `language: [en, zh]`、`pipeline_tag`）。**权重许可状态在 HF 上未标注 → 未核实**，只能依仓库 LICENSE 推断 |
| 附加 `DISCLAIMER` | 中文免责声明，明确禁止「未经授权将合成声音用于商业目的」（注意：这约束的是**合成的声音**，不是模型本身） |

**bilibili 许可的关键条款（逐条摘录自英文 LICENSE）**：

- **2.1** 授予全球、非独占、不可转让、免版税的**有限**许可（可用于模型及衍生品）。
- **2.2 门槛条款（商业风险核心）**：若你或关联方「前一自然月 MAU 超 1 亿」**或**「上一自然年营收超 1 亿人民币」，**必须**另行申请商业许可，且「我方自行决定是否授予」。未获书面授权不得行使本协议任何权利。
- **3.4(a)** 分发衍生品必须在分发页面/文档注明：「本衍生品对原模型所做修改未获原权利人认可、担保或保证，原权利人免除一切相关责任。」
- **3.4(b)** 必须保留全部原始版权声明与本协议副本。
- **3.4(c)** ⚠️ **不得使用 bilibili indextts2 或其衍生品去改进任何 AI 模型**，除非是 indextts2 自身、其衍生品，或**非商业** AI 模型。（即：**用本模型蒸馏/训练其他模型 → 仅限非商业**）
- **4.2 禁止高风险用途**：医疗诊断、自动驾驶、军事、关键基础设施控制、大规模生物特征监控、自动化决策（信贷/招聘）等。
- **5.1** 违约可撤销许可，须立即停止使用并永久删除所有副本。
- **6** 适用**中华人民共和国**法律；争议提交**上海仲裁委员会**，裁决终局。
- README 补充：「For commercial usage and cooperation, please contact indexspeech@bilibili.com」。

**针对「商业桌面软件打包」的结论**：
✅ 允许商用（在 1 亿 MAU / 10 亿营收门槛之下）；
✅ 允许再分发/打包（许可第 2.1 条含 distribute）；
⚠️ **但有实质义务**：必须携带版权声明 + 完整许可副本、必须在分发页声明免责、下游接收方需受同等约束、**不得用于改进其他 AI 模型（非商业除外）**；
🔴 **不是 OSI 开源许可**。若做闭源商业桌面软件，**建议法务过一遍**，特别是「下游接收方须遵守本协议」这条对闭源分发的实际可执行性。

### 1.5 质量：基准 / 榜单 / 口碑

**官方 CV3-Eval 零样本（摘自仓库 README 表 1）**：

| 模型 | 参数 | zh WER↓/SS↑ | en | es | ja | ar | Avg WER/SS |
|---|---|---|---|---|---|---|---|
| CosyVoice3-0.5B | 0.5B | 3.84 / **80.01** | 4.88 / 74.16 | 4.04 / 78.85 | – / 76.36 | – | – |
| FireRedTTS-2 | 1.5B | 8.22 / 68.10 | 14.92 / 56.93 | – | – | – | – |
| **IndexTTS2.5** | 0.8B | 4.36 / 77.10 | 5.12 / 68.06 | 3.75 / 76.39 | 5.66 / 74.62 | 14.88 / 69.74 | 6.75 / 73.18 |
| **IndexTTS2.5-RL** | 0.8B | 3.93 / 77.92 | **3.89** / 67.79 | **3.33** / 76.68 | **5.30** / 75.41 | **13.58** / **70.36** | **6.00** / **73.63** |

**跨语种 CV3-Eval（中文 prompt → 目标语）**：IndexTTS2.5-RL zh→en 3.55/67.47、zh→es 4.86/64.47、zh→ja 6.38/75.82、zh→ar 9.89/73.05，Avg 6.17/**70.20**（表中平均 SS 最高）。

**seed-tts-eval（转引自 CosyVoice 官方 README 对比表）**：Index-TTS2（1.5B）test-zh CER **1.03** / SS 76.5；test-en WER 2.23 / SS 70.6；test-hard CER 7.12 / SS 75.5。

**第三方榜单**：
- **Artificial Analysis Speech Arena 开放权重榜（2026-09-16 抓取）：IndexTTS 未上榜**。
- tts.ai arena：IndexTTS-2 排第 11，官方分 3.91/5，速度标 "medium"，Standard 档。⚠️ 该站把 IndexTTS-2 参数量写成「300M」，**明显有误**，属于低可信度二手聚合站，引用需谨慎。

**中文社区口碑（`duevan07/chinese-ai-tts-real-review` v2026.06，发布 2026-06-26，多源采集公众号/小红书/B站/知乎/Reddit/GitHub Issues）**：
- IndexTTS2 被列为**口碑一致性最高**的模型：「免费配音界天花板，情绪还原找不到对手」。三维：多音字 ★★★★☆、韵律 ★★★★☆、**语气情感 ★★★★★**。
- 唯一短板（原话）：「语速偏快、停顿不能精控」。
- 多音字可控性上，**只有 IndexTTS2（字符+拼音混合）与 CosyVoice3（拼音修补）有正式纠错机制**。
- 建议场景：想要「朋友般亲近感 + 句子有韵律」的内容创作者，**首选 IndexTTS2 克隆自己的声音**。

### 1.6 工程可用性

- **官方 pip 包**：❌ 无。使用 `uv` 管理环境：`uv sync --all-extras`（README 明确 `uv` 是「required for a reliable installation」）。
- **重依赖**：无 fairseq 需求。有 **DeepSpeed 可选 extra**（README 警告 Windows 上难装）；**不强制 flash-attn**（代码里 `if not torch.cuda.is_available() or not use_flash` 才走非 flash 路径）。安装时若见 CUDA 错误需 **CUDA Toolkit ≥ 12.8**。
- **ONNX / GGUF / llama.cpp 路线**：官方无；**社区有四条**（MLX / MNN / ONNX，见 1.3；GGUF 未发现）。
- **API 服务**：Gradio WebUI（`uv run webui.py`，默认 `http://127.0.0.1:7860`）+ Python API（`indextts.infer_v2_5.IndexTTS2`）+ CLI（`indextts/cli.py`、`cli_v2.py`，含 `cli_v2_usage.md`）。**生产部署有官方 vLLM recipe**：https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5 。未提供独立 FastAPI server。
- **Activity**：⭐23,995 / 最后 push **2026-08-18** —— 四个模型里最活跃之一。QQ 群 663272642、1013410623；Discord。

### 1.7 已知坑

1. **英文长段落对齐坍塌**（Issue #775）：`max_text_tokens_per_segment` 默认 120 是按**中文** token 密度标定的，对英文等字母语言「形同虚设」，单段合成时末句对齐崩坏。→ **英文长文本必须手动切段**。
2. **批处理显存泄漏嫌疑**（Issue #364）：约 200+ 次连续 batch 合成后触发 `CUDA device-side assert triggered`。→ 生产环境需常驻进程时**务必做进程隔离或定期重启**。
3. **低端卡过慢**（Issue #585）：RTX 3060 12GB 上约 **228 秒**合成一条，用户体验差。
4. **输出与参考音色不符 / 语速怪异**（Issue #410, #460）：社区反馈「不管用什么参考音，输出的都慢而怪」→ 质量稳定性存在个体差异。
5. **`use_random=True` 会降低克隆保真度**（README 明确 NOTE）。
6. **2.5 的情绪文本引导是破坏性变更**：必须 `use_qwen_emo=True`，否则 RuntimeError；且 low-VRAM 模式下该模块默认被跳过。
7. **MPS 上强制关闭 bf16**，速度不如 CUDA；且 `webui.py` 在 Mac 上检测不到显存，会误加载 QwenEmotion。
8. **参考音频不需要转写文本**（优点，与 FireRedTTS-2 / CosyVoice 零样本模式不同）——但这也意味着长参考音频的语义对齐靠模型自己推断。

---

## 2. FireRedTTS-2（小红书 Xiaohongshu）

- 仓库：https://github.com/FireRedTeam/FireRedTTS2
- 论文：https://arxiv.org/abs/2509.02020
- HF：https://huggingface.co/FireRedTeam/FireRedTTS2
- Demo：https://fireredteam.github.io/demos/firered_tts_2/
- 注意：v1 是另一个仓库 https://github.com/FireRedTeam/FireRedTTS

### 2.1 名称 / 参数量 / 发布方 / 最新版本与日期

| 项 | 值 |
|---|---|
| 发布方 | FireRedTeam（小红书 智创音频技术团队） |
| 版本 | **FireRedTTS-2（v2）** —— 目前唯一公开版本，无 2.5 / v3 |
| 时间线 | 2025/09/02 技术报告 + demo；**2025/09/08 发布预训练权重与推理代码**；2025/09/12 UI；2025/09/28 bf16（显存 14GB→9GB）；2025/10/11 流式对话；**2025/10/26 微调代码与教程**（最后一个动向） |
| 参数量 | **backbone `qwen-1.5b` + decoder `qwen-200m`**（源码 `bin/finetune_example/config_finetune_1.5b_0.2b.json` 实证），合计约 **1.7B**。CosyVoice 官方对比表标注为 1.5B |
| 架构 | 双 Transformer，作用于「文本–语音交织序列」；**12.5Hz 流式语音 tokenizer**；16 个 codebook、2051 音频词表；文本 tokenizer 参考 Qwen2.5-1.5B；声学解码基于 XCodec2 / Vocos |
| 仓库热度 | ⭐ **1,432** ｜ fork 130 ｜ open issues **18** ｜ 创建 2025-09-02 ｜ 🔴 **最后 push 2025-10-26（截至调研日已停滞约 11 个月）** |

### 2.2 能力矩阵

| 能力 | 结论 |
|---|---|
| 零样本克隆 | ✅ 需 `prompt_wav` **+ `prompt_text`**（参考音频的文本转写） |
| 少样本微调 | ✅ 官方支持，2025/10/26 放出 finetune 代码 + LJSpeech 教程（`bin/finetune_example/tutorial.md`） |
| 文本描述设计音色 | ❌ 无。但有 **Random Timbre Generation**（随机音色生成，用于造 ASR/对话数据），不是文本描述式 voice design |
| 中文质量 | ⚠️ 分裂：seed-tts-eval test-zh CER **1.14** / SS 73.2（不错）；但 **CV3-Eval zh WER 8.22**（明显偏弱）。擅长的场景是**多说话人对话**而非单句克隆 |
| 多语言 | **英、中、日、韩、法、德、俄**（7 种，README 明确），支持跨语种与 code-switching 零样本克隆 |
| 情感控制 | ❌ 无显式情感控制接口 |
| 方言 | ❌ 未声明 → 未核实 |
| 韵律 | ✅ 卖点是 "context-aware prosody" + 可靠的说话人切换 |
| 流式 | ✅ **强项**。12.5Hz tokenizer + 双 transformer，句级生成，**L20 GPU 上首包延迟低至 140ms**；流式 chunk 每包 0.08 秒 |
| 长文本 | ✅ **强项**。目前支持 **3 分钟对话 / 4 个说话人**，可通过扩训练语料扩展 |
| 声音转换 VC | ❌ 未提供 |

### 2.3 硬件 / 推理速度 / Apple Silicon（重点）

- **显存**：官方 news 明确 —— **fp32 需 14GB，bf16 降到 9GB**，从而「enabling consumer-grade GPU deployment」。
- **延迟**：L20 上首包 **140ms**。
- **CPU 推理**：❌ 无官方支持声明。代码 `fireredtts2.py` 用 `torch.cuda.is_bf16_supported()`，默认 `device="cuda"`。CPU RTF **未核实**。
- **RTF**：⚠️ **官方未给出 RTF 数字**（只给首包延迟）→ **未核实**。

#### 🔴 Apple Silicon MPS 支持：**无（NO）**

一手证据：
- `grep -rn "mps" src_FireRedTeam_FireRedTTS2/fireredtts2/` → **零命中**。
- `fireredtts2/fireredtts2.py:46` 为 `if torch.cuda.is_bf16_supported():`，`device` 默认 `"cuda"`，bf16 探测逻辑纯 CUDA 路径。
- `fireredtts2/llm/utils.py` 默认 `device: Union[str, torch.device] = "cuda"`。
- 安装文档只给 `pip install torch --index-url .../cu126`（CUDA 12.6）与 Docker `--gpus=all` 两种路径。
- 未发现任何 MLX / CoreML / ONNX / GGUF 的第三方 Mac 移植。

**结论：Apple Silicon 上目前无可行推理路线**（理论上可强行 `device="cpu"`，但无官方支持、无速度数据、且 4.3GB codec + 8.3GB LLM 在 CPU 上不现实）。

- **磁盘占用**（HF API 实取）：**合计 20.85 GB** —— `llm_pretrain.pt` 8.27GB + `llm_posttrain.pt` 8.27GB + `codec.pt` 4.30GB + `Qwen2.5-1.5B/tokenizer.json` 7MB。⚠️ 两个 8.27GB 的 LLM 权重通常只需其一（推理用 posttrain），**实际运行子集约 12.6GB**。

### 2.4 许可证 —— ⚠️ 存在自相矛盾，必须提示

| 对象 | 许可证 |
|---|---|
| 仓库 `LICENSE` | **Apache License 2.0**（文件头实证） |
| HF `FireRedTeam/FireRedTTS2` | `license: apache-2.0` ✅ 一致 |
| README 徽章 | Apache-2.0 |

✅ **商用允许、允许再分发/打包**（Apache-2.0 标准条款，需保留 NOTICE / 版权声明）。

🔴 **但 README 底部有独立的 `⚠️ Usage Disclaimer`，与 Apache-2.0 冲突**：
> "The project incorporates zero-shot voice cloning functionality; Please note that this capability is intended **solely for academic research purposes**."

**解读**：Apache-2.0 是仓库代码/权重的正式许可（法律上优先），但作者在 README 中表达了「零样本克隆仅供学术研究」的意图，且 demo 页对第三方播客主播声音标注「未经授权不能使用」。**这是「宽松许可 + 窄化意图声明」的典型冲突**。若商业桌面软件要用法，建议：(a) 只把该声明当作对**声音克隆用途**的道德/场景限制而非许可限制；(b) 保守做法是发邮件向 FireRedTeam 取得书面确认。**本报告不对该冲突给出法律结论。**

### 2.5 质量：基准 / 榜单 / 口碑

**seed-tts-eval（转引自 CosyVoice 官方 README 对比表）**：FireRedTTS2（1.5B）test-zh CER **1.14** / SS 73.2；test-en WER **1.95** / SS 66.5。

**CV3-Eval 零样本（摘自 IndexTTS 2.5 README 表 1）**：FireRedTTS-2（1.5B）zh WER **8.22** / SS 68.10；en WER **14.92** / SS 56.93。
→ **在同表中是基线水平**（对比 IndexTTS2.5-RL 的中文 3.93 / 英文 3.89，差距显著）。这印证了它的定位：**不是单句克隆的音质冠军，而是多说话人长对话/播客生成的专项方案**。

**跨语种 CV3-Eval**：zh→en 9.34 / 53.19；zh→es 12.25 / 58.31；zh→ja 19.05 / 64.12 —— 跨语种明显弱。

**第三方榜单**：**Artificial Analysis Speech Arena 未上榜**；tts.ai 榜单**未收录**。
**社区口碑**：中文口碑横评（v2026.06）**未把 FireRedTTS-2 列入 12 款对比** → 中文内容创作者圈层认知度低。小红书本家宣传聚焦「AI 播客」场景。

### 2.6 工程可用性

- **官方 pip 包**：❌ 无。`pip install -e .` + `pip install -r requirements.txt`。
- **依赖**（`requirements.txt` 实证）：`torchtune, torchao, transformers, einops, gradio, librosa, optuna, accelerate, tensorboard`；PyTorch **2.7.1**（cu126）。**无 fairseq / DeepSpeed / flash-attn 硬依赖** —— 依赖相对干净。
- **Docker**：✅ 官方 `docker/Dockerfile`，`docker run --gpus=all`。
- **ONNX / GGUF / llama.cpp 路线**：❌ 均未发现。
- **API 服务**：`gradio_demo.py`（Web UI，支持克隆与随机音色）。❌ 无 FastAPI server。
- **Activity**：🔴 **最后 push 2025-10-26**，停滞约 11 个月。Roadmap 中两项未完成：`[ ] Release a base model with enhanced multilingual support`、`[ ] End-to-end text-to-podcast pipeline`。

### 2.7 已知坑

1. **仓库停滞 11 个月**，roadmap 承诺的多语言增强基座与端到端播客管线**至今未交付**。
2. **零样本克隆必须提供参考音频的文本转写**（`prompt_text`），增加使用门槛；且参考音频需单声道（代码里有 shape 检查）。
3. **单句/跨语种克隆指标偏弱**（CV3-Eval en WER 14.92）—— 不要把它当通用克隆模型用。
4. **许可证意图冲突**（Apache-2.0 vs README「仅供学术研究」）。
5. **无 Apple Silicon 路线**，无 CPU 支持声明，无量化/ONNX 路线 → 部署灵活性最差。
6. 生成依赖 `temperature=0.9, topk=30` 类采样参数（README 示例），**稳定性对采样参数敏感，需自行调参**。

---

## 3. CosyVoice 2 & CosyVoice 3（阿里 Alibaba FunAudioLLM）

- 🔴 **仓库已改名迁移**：`github.com/FunAudioLLM/CosyVoice` → **301 永久重定向** → **`github.com/QwenAudio/CosyVoice`**（实测 HTTP 301，location 指向 QwenAudio）。
- 仓库：https://github.com/QwenAudio/CosyVoice
- 首页：https://funaudiollm.github.io/cosyvoice3
- 论文 3：https://arxiv.org/abs/2505.17589 ｜ 论文 2：https://arxiv.org/pdf/2412.10117
- HF 3：https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 ｜ HF 2：https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
- CV3-Eval 测试集：https://github.com/FunAudioLLM/CV3-Eval

### 3.1 名称 / 参数量 / 发布方 / 最新版本与日期

| 项 | 值 |
|---|---|
| 发布方 | 阿里 FunAudioLLM 团队 |
| CosyVoice 1.0 | 300M，2024-07（含 VC 功能，2024/09） |
| **CosyVoice 2.0** | **0.5B（CosyVoice2-0.5B），2024-12** |
| **Fun-CosyVoice 3.0** | **0.5B（Fun-CosyVoice3-0.5B-2512），2025-12**（HF 创建 2025-12-11；ModelScope 创建时间戳 1765334836 = 2025-12-10；README roadmap 2025/12 条目） |
| 🔴 **CosyVoice 3.5** | **不存在开源版本**。`cosyvoice-v3.5-plus` 只是阿里云百炼（Model Studio）的 **API 模型 ID**，仓库不发布 3.5 权重或推理代码（Soniqo 文档明确说明，2026 抓取） |
| 参数量 | CosyVoice-300M（v1）；**CosyVoice2-0.5B、Fun-CosyVoice3-0.5B**。3.0 的 LLM = **Qwen2.5-0.5B**：24 层、896 hidden、14 Q heads / 2 KV heads（GQA）、FSQ 词表 6561、25Hz；DiT flow：22 层、1024 dim、16 heads、10 步 Euler ODE、CFG 0.7；HiFi-GAN（NSF，8 次谐波，480× 上采样，24kHz） |
| 其他规模 | ⚠️ IndexTTS 2.5 README 对比表中出现 **CosyVoice3-1.5B**（zh WER 3.91 / en 4.99 / es 4.47 / ja 7.57，标注 † 引自原论文），但 **HF/ModelScope 上无该开源权重** → 可能为内部版本，**未核实** |
| 仓库热度 | ⭐ **23,632** ｜ fork 2,686 ｜ open issues **698**（四者中最多）｜ 创建 2024-07-03 ｜ ⚠️ **最后 push 2026-05-25（停滞约 4 个月）** |

### 3.2 能力矩阵

| 能力 | 结论 |
|---|---|
| 零样本克隆 | ✅ 需 `ref_audio` **+ 精确 `ref_text`**（不提供 ref_text 则走 cross-lingual 模式；zero_shot 只在显式给 ref_text 时激活）。参考音频 **≤30 秒**，超长直接拒绝（mlx-audio 已对齐 PyTorch 前端行为） |
| 少样本微调 | ✅ 有训练脚本（`examples/libritts`）；2025/08 新增 **CosyVoice2 GRPO 训练支持** |
| 文本描述设计音色 | ⚠️ 部分。`instruct` / `instruct2` 模式支持**指令式风格控制**：语言、方言、情感、语速、音量。**不是**从零描述音色的 voice design |
| 中文质量 | ✅ 顶级。Fun-CosyVoice3-0.5B-2512 test-zh CER **1.21** / SS **78.0**；RL 版 CER **0.81** / SS 77.4。CV3-Eval 中文 SS **80.01**（同表最高） |
| 多语言 | ✅ **9 种**：中、英、日、韩、德、西、法、意、俄。**另支持 18+ 种中文方言/口音**：广东、闽南、四川、东北、山西、陕西、上海、天津、山东、宁夏、甘肃等 |
| 情感控制 | ✅ 通过 instruct 指令控制情感；另有细粒度控制 token（`[breath]`、`[laughter]` 等，见 `cosyvoice/tokenizer/tokenizer.py#L280`） |
| 方言 | ✅ **强项**，18+ 中文方言/口音 |
| 韵律 | ✅ 官方宣称内容一致性、说话人相似度、韵律自然度均 SOTA |
| 流式 | ✅ **强项**：**双流式**（文本流入 + 音频流出），延迟低至 **150ms** |
| 长文本 | ✅ 支持数字/特殊符号/多种文本格式的**文本归一化**，无需传统前端模块（可选安装 ttsfrd，否则用 wetext） |
| 声音转换 VC | ✅ **有**。v1 `inference_vc` 明确示例；CosyVoice3 亦支持 VC 模式（source_audio + ref_audio，mlx-audio / speech-swift 文档均确认） |
| 发音修补 | ✅ **中英双语**：中文拼音 + 英文 CMU 音素（hotfix 用法，`报道[j][ǐ]予好评`） |

### 3.3 硬件 / 推理速度 / Apple Silicon（重点）

- ⚠️ **官方未给出任何显存数字** → **未核实**。仅能依 0.5B 参数量推断消费级 GPU 充裕。
- **RTF**：⚠️ **官方未给 RTF 表** → 未核实。README 只提「延迟低至 150ms」与「kv cache + sdpa 用于 rtf 优化」。
- **CPU 推理**：⚠️ 官方**未声明支持**。requirements 里 `deepspeed` 与 `tensorrt-cu12` 仅 `sys_platform == 'linux'`；`onnxruntime==1.18.0` 在 darwin/win32 上安装（说明代码可在 Mac 装环境），但**无 CPU 性能数据**。第三方 `lianghsun/cosyvoice3-api` 标注 "macOS / CPU inference works but is significantly slower"。→ **CPU 可用但慢，具体 RTF 未核实**。
- **模型大小（磁盘）**：HF `Fun-CosyVoice3-0.5B-2512` API 实取 **合计 9.75 GB**，但含冗余：`llm.pt` 2.02GB + **`llm.rl.pt` 2.02GB**（RL 版单独权重）、`flow.pt` 1.33GB + `flow.decoder.estimator.fp32.onnx` 1.33GB、`speech_tokenizer_v3.onnx` 0.97GB + `...batch.onnx` 0.97GB、`CosyVoice-BlankEN/model.safetensors` 0.99GB、`hift.pt` 83MB、`campplus.onnx` 28MB。**实际运行子集约 4–5GB**。

#### 🔴 Apple Silicon MPS 支持：**官方主干无（NO）／社区 PR 部分支持（partial，未合并）**

一手证据（clone 后 grep）：
- `grep -rn "mps" src_QwenAudio_CosyVoice/` 在 `.py` 中 **零命中**（仅命中 `json.dumps` 等误报）。→ **官方主干无 MPS 代码路径**。
- 社区 PR（**未合并进主干**）：
  - **PR #1129** "feat: Add limited support for MPS devices" —— 描述原文：「Add **limited support** for MPS devices to enable **partial** compatibility with Apple Silicon. Note: **ONNX Runtime and TensorRT are not supported**, but performance has significantly improved. **Tested on M4 Max.**」https://github.com/FunAudioLLM/CosyVoice/pull/1129
  - **PR #1869** "feat: add Apple Silicon (MPS) support for macOS ARM64" —— 引入设备抽象层 `cosyvoice/utils/device.py`，统一 CUDA/MPS/CPU，替换推理管线中所有硬编码 CUDA 路径。https://github.com/FunAudioLLM/CosyVoice/pull/1869

**但 Mac 上有三条成熟的第三方路线（这才是实操推荐）：**

| 路线 | 说明 | 一手来源 |
|---|---|---|
| **mlx-audio**（`Blaizzy/mlx-audio`；本报告核实的 `DePasqualeOrg/mlx-audio-plus`，**MIT**） | 原生 MLX 支持 **CosyVoice3**。模型 `mlx-community/Fun-CosyVoice3-0.5B-2512`。四种模式齐全：cross-lingual / zero-shot / instruct / **voice conversion**。参考音频 >30s 直接拒绝；zero-shot 需显式 ref_text。流式 chunk 25→50→100 token。**OpenAI 兼容 REST API + Web UI** | https://github.com/DePasqualeOrg/mlx-audio-plus ｜ https://raw.githubusercontent.com/DePasqualeOrg/mlx-audio-plus/main/docs/tts/cosyvoice3.md |
| **speech-swift**（`soniqo/speech-swift`，**Apache-2.0**） | 原生 **Swift/MLX**，macOS + iOS。四个量化档：**4bit ~1.2GB（默认）／8bit ~1.4GB／8bit-full ~1.6GB／bf16 ~2.1GB**。含 8 个内置情感/风格 tag（happy/excited, sad, angry, whispers, laughs, calm, surprised, serious）+ 自由文本指令；多说话人对话解析（`[S1] ... [S2] ...`）；CAM++ CoreML 说话人编码器（~14MB，走神经引擎）；流式 25-token chunk，目标首包 ~150ms；长文本自动分段 + 边缘淡入淡出拼接 | https://raw.githubusercontent.com/soniqo/speech-swift/main/docs/models/cosyvoice-tts.md ｜ https://soniqo.audio/guides/cosyvoice |
| **GGUF**（`cstr/cosyvoice3-0.5b-2512-GGUF`，Apache-2.0） | 供 **CrispASR**（`--backend cosyvoice3-tts`）使用。文件：`llm-f16` 1.29GB / `llm-q4_k` 384MB；`llm-rl-f16` 1.29GB / `llm-rl-q4_k` 384MB；`flow-f16` 665MB / `flow-q8_0` 361MB；`s3tok-f16` 484MB / `s3tok-q4_k` 145MB；`hift-f16` 42MB；`campplus-f16` 14MB。**F16 合计约 2.5GB，量化后约 1.0GB** | https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF |

- 另有第三方 **MLX + PyTorch 混合**的性能实测博客：`How I Made CosyVoice3 2.2× Faster on Apple Silicon`（2026-07-25），通过限制 PyTorch MPS 显存增长 + 只把 dispatch-bound 的 LLM 阶段搬到 MLX，最终达到 **RTF 0.757**（实时以下）。https://www.drmhse.com/posts/running-funaudio-on-mac-mlx-pytorch/

### 3.4 许可证 —— 四个模型里最干净

| 对象 | 许可证 |
|---|---|
| 仓库 `LICENSE` | **Apache License 2.0**（文件头实证） |
| GitHub API 元数据 | `spdx_id: Apache-2.0` |
| HF `FunAudioLLM/CosyVoice2-0.5B` | `license: apache-2.0` |
| HF `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | `license: apache-2.0` |
| ModelScope `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | License 字段为空（未填）→ 以 Apache-2.0 为准 |

✅ **商用允许**；✅ **允许再分发 / 打包进商业桌面软件**；✅ 无 MAU / 营收门槛；✅ 无「禁止改进其他模型」条款；✅ 无领域禁用条款。
⚠️ **需注意的例外**：
- README 末尾 `## Disclaimer` 写「The content provided above is for academic purposes only and is intended to demonstrate technical capabilities. Some examples are sourced from the internet.」—— **该免责声明针对的是文档/示例内容，不是模型许可**，不构成商用限制。
- `CosyVoice-ttsfrd` 是独立的**资源包**（可选，用于更好的文本归一化），其自身条款 **未核实**。不装它则默认走 `wetext`。
- 复用的是 FunASR / FunCodec / Matcha-TTS / AcademiCodec / WeNet 的代码，各自许可 **未逐个核实**（通常同为 Apache/MIT）。

**结论：若目标是闭源商业桌面软件且希望零法务摩擦，CosyVoice 2/3 是本批四个中许可最安全的选项。**

### 3.5 质量：基准 / 榜单 / 口碑

**seed-tts-eval（CosyVoice 官方 README 对比表）**：

| 模型 | 参数 | test-zh CER↓ | SS↑ | test-en WER↓ | SS↑ | test-hard CER↓ | SS↑ |
|---|---|---|---|---|---|---|---|
| Human | – | 1.26 | 75.5 | 2.14 | 73.4 | – | – |
| Seed-TTS（闭源） | – | 1.12 | 79.6 | 2.25 | 76.2 | 7.59 | 77.6 |
| MiniMax-Speech（闭源） | – | 0.83 | 78.3 | 1.65 | 69.2 | – | – |
| F5-TTS | 0.3B | 1.52 | 74.1 | 2.00 | 64.7 | 8.67 | 71.3 |
| **CosyVoice2** | 0.5B | 1.45 | 75.7 | 2.57 | 65.9 | 6.83 | 72.4 |
| FireRedTTS2 | 1.5B | 1.14 | 73.2 | 1.95 | 66.5 | – | – |
| Index-TTS2 | 1.5B | 1.03 | 76.5 | 2.23 | 70.6 | 7.12 | 75.5 |
| VoxCPM | 0.5B | 0.93 | 77.2 | 1.85 | 72.9 | 8.87 | 73.0 |
| **Fun-CosyVoice3-0.5B-2512** | 0.5B | **1.21** | **78.0** | 2.24 | 71.8 | 6.71 | 75.8 |
| **Fun-CosyVoice3-0.5B-2512_RL** | 0.5B | **0.81** | 77.4 | **1.68** | 69.5 | **5.44** | 75.0 |

→ **RL 版的 test-zh CER 0.81 已超越闭源 Seed-TTS（1.12）与 MiniMax-Speech（0.83）**，是本批四个模型中最强的中文错误率数据。

**CV3-Eval 零样本**：CosyVoice3-0.5B zh 3.84 / SS **80.01**（同表 SS 最高）、en 4.88 / 74.16、es 4.04 / 78.85、ja – / 76.36。
**跨语种**：zh→en 3.23 / 62.79、zh→es 4.58 / 64.04。

**第三方榜单**：
- **Artificial Analysis Speech Arena：未上榜**（2026-09-16）。
- **tts.ai arena**：CosyVoice 2 排**第 2**（官方分 4.26/5，速度 "medium"，Standard 档）；CosyVoice3 排第 30（速度 "fast"，尚无用户投票）。⚠️ 该站把 CosyVoice 2 参数量写成 300M（实为 0.5B），**数据准确性存疑**。

**中文社区口碑（v2026.06）**：
- CosyVoice 2/3：多音字 ★★★★☆、韵律 ★★★☆☆、情感 ★★★☆☆。定位：「⚠️ 跨源分歧大：demo 自然度高，长内容情感被实战派挑刺」。
- **正面**：多音字可控性上，只有 IndexTTS2 与 **CosyVoice3（拼音修补）有正式纠错机制**。
- **负面**：海外/实战派评价「不如 GPT-SoVITS」，甚至有「**3.0 不如 2.0**」的说法。
- **正向信号**：知乎答主明确建议新手从旧推荐（VoxCPM 等）**转向 Qwen3-TTS / CosyVoice3**。

### 3.6 工程可用性

- **官方 pip 包**：❌ 无。Conda 环境 + `pip install -r requirements.txt`。
- **重依赖（实测 requirements.txt）**：`torch==2.3.1`、`torchaudio==2.3.1`、`transformers==4.51.3`、**`deepspeed==0.15.1`（仅 linux）**、**`tensorrt-cu12==10.13.3.9`（仅 linux）**、`onnxruntime-gpu`（linux）/ `onnxruntime`（darwin, win32）、`x-transformers`、`HyperPyYAML`、`pyworld`、`openai-whisper`、`wetext`、`librosa`、`gradio==5.4.0`、`fastapi==0.115.6`。
  - ⚠️ **`torch==2.3.1` 是硬钉死的**，这正是 garbled bug 的根源（见 3.7）。
  - ⚠️ 需系统依赖 **sox / libsox-dev**（Ubuntu/CentOS 需手动 apt/yum）。
  - **无 fairseq、无 flash-attn 硬依赖**。
  - 需 `git clone --recursive`（有 submodule）。
- **加速路线**：**vLLM 0.11.x+（V1 engine）或 0.9.0（legacy）**，明确不支持 <0.9.0 与未测试 0.10.x；**TensorRT-LLM** 走 `runtime/triton_trtllm`（Docker compose，据称比 HF transformers 快 4×，仅 Linux）。
- **ONNX / GGUF / llama.cpp 路线**：官方有 ONNX 模型文件；社区有 GGUF（CrispASR）与 MLX（见 3.3）。
- **API 服务**：✅ **官方内建 FastAPI server + client**（`runtime/python/fastapi/`）+ **gRPC server + client**（`runtime/python/grpc/`）+ **Docker 部署**（`runtime/python/docker build -t cosyvoice:v1.0`）+ `webui.py`（Gradio，默认 50000 端口）。**服务化能力是四个模型中最完善的**。
- **Activity**：⚠️ **最后 push 2026-05-25**，停滞约 4 个月；open issues **698**（积压最多）。仓库从 FunAudioLLM 改名至 **QwenAudio**（旧链接会 301，但工具链/脚本里硬编码的 URL 可能失效）。

### 3.7 已知坑

1. 🔴 **torch ≥ 2.7 产生乱码**（Issue #1886）：CosyVoice3 在 torch >= 2.7（**RTX 50 系必需**）下输出 garbled speech。而 requirements 钉死 torch==2.3.1 → **Blackwell（50 系）GPU 支持存在硬冲突**。
2. 🔴 **乱码问题多发**（Issue #1677）："Garbled output with CosyVoice3?" —— 不论什么语言，用 main.py 代码即出现。
3. 🔴 **vLLM 转换后出现非文本发音**（Issue #1911，2026-06-18 创建，仍 open）：CosyVoice3 与 CosyVoice2 微调模型转 vLLM 推理后合成文本出现非文本发音（杂音/幻觉音）。→ **vLLM 加速路线有质量风险**。
4. **零样本回归问题**（Issue #1704）："Regression: CosyVoice zeroshot prompt issue"。
5. **PR #1916 暴露的未合并修复**：`sampling_ids()` 只 mask 了 `speech_token_size`（sos），以及**多 GPU 设备支持**缺失。
6. **日文必须先转片假名**（README 明确 NOTE）：`歴史的世界…` 必须先写成 `レキシ テキ セカイ…` 才能合成。**这是硬性使用门槛**。
7. **零样本需精确参考文本且在 30 秒内**，ref_text 与音频不匹配则质量崩坏；不提供 ref_text 会静默走 cross-lingual 模式（非 zero-shot）。
8. **仓库停滞 4 个月 + 698 个未处理 issue**，社区修复（如 MPS PR、stop_token 修复）迟迟未合并。
9. **CosyVoice3-1.5B 在论文/对比表中出现但无开源权重** —— 不要基于论文表格规划 1.5B 部署。
10. **没有 CosyVoice 3.5 开源版** —— 只有云端 API ID，不要误以为能本地跑。
11. 微调坑：第三方体验报告（Instavar, 2026-03-25）称 CosyVoice3 全量 SFT（506M 参数）失败、建议 LoRA。

---

## 4. GPT-SoVITS（v2 / v4 / pro，RVC-Boss）

- 仓库：https://github.com/RVC-Boss/GPT-SoVITS
- 权重：https://huggingface.co/lj1995/GPT-SoVITS
- Windows 整合包：https://huggingface.co/lj1995/GPT-SoVITS-windows-package
- 在线 demo（HF Space，半张 H200）：https://lj1995-gpt-sovits-proplus.hf.space/
- 中文文档：https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e
- CPU 优化分支（第三方）：https://github.com/baicai-1145/GPT-SoVITS-CPUFast

### 4.1 名称 / 参数量 / 发布方 / 最新版本与日期

| 项 | 值 |
|---|---|
| 发布方 | RVC-Boss（作者 lj1995 / 花儿不哭） |
| 仓库热度 | ⭐ **61,794（本批最高，断层第一）** ｜ fork 6,657 ｜ open issues 894 ｜ 创建 2024-01-14 ｜ **最后 push 2026-08-18**（活跃） |
| **最新 release tag** | **`20250606v2pro`（2025-06-06）**。此前：`20250422v4`（2025-04-22）、`20250228v3`（2025-02-28）、`20240821v2`（2024-08-21） |
| 版本谱系 | v1（2024-01）→ v2（2024-08）→ v3（2025-02）→ v4（2025-04）→ **v2Pro / v2ProPlus（2025-06）** |
| 用户可选版本（WebUI 实证） | `["v1", "v2", "v4", "v2Pro", "v2ProPlus"]`（`webui.py:1498`） |
| 参数量 | ⚠️ **无官方总量数字**。从配置实证：GPT（T2S）= `n_layer: 24, hidden_dim: 512, head: 16, vocab_size: 1025, phoneme_vocab_size: 732, linear_units: 2048` → **约 30–40M**；SoVITS（VITS 系）= 32kHz/24kHz/48kHz 声学模型，约 80–100M 级；另加中文 BERT（chinese-roberta-wwm-ext 级，~100M）。tts.ai 站标为「200M」。**官方未给权威参数量 → 总量未核实** |
| 架构要点 | v1/v2：「直接解码」VQ-VAE，32kHz，无独立声码器；v3：改为 **CFM（Conditional Flow Matching）+ BigVGAN v2**，24kHz，支持 LoRA；v4：**HiFiGAN，48kHz 原生输出**，修复 v3 非整数倍上采样导致的金属音；**v2Pro/v2ProPlus**：回到直接解码 + 32kHz + **增强说话人验证层** |

**各版本定位（README + DeepWiki 整理，DeepWiki 最后索引 2026-01-17）**：

| 版本 | 发布时间 | 输出采样率 | 声码器 | 关键特性 | 训练显存 |
|---|---|---|---|---|---|
| v1 | 2024-01 | 32kHz | 直接解码 | 初版 | ~10GB |
| v2 | 2024-08 | 32kHz | 直接解码 | 韩文/粤语、5k 小时预训练 | ~10GB |
| v3 | 2025-02 | 24kHz | BigVGAN | CFM 架构、LoRA 支持 | 8GB(LoRA)/14GB(全量) |
| v4 | 2025-04 | **48kHz** | HiFiGAN | 修复金属音 | 8GB(LoRA)/14GB(全量) |
| v2Pro | 2025-06 | 32kHz | 直接解码 | 说话人验证层 | ~12GB |
| **v2ProPlus** | 2025-06 | 32kHz | 直接解码 | **增强说话人验证，相似度最高** | ~12GB |

官方定位原话：v4「作者认为 v4 是 v3 的直接替代」；v2Pro「比 v2 略高显存，性能超越 v4，但保持 v2 的硬件成本与速度」；「v1/v2/v2Pro 系列特性相近，v3/v4 相近。**对音质平均的训练集，v1/v2/v2Pro 能给出不错结果，但 v3/v4 不行**」——这是非常重要的选型提示。

### 4.2 能力矩阵

| 能力 | 结论 |
|---|---|
| 零样本克隆 | ✅ **5 秒参考音频**即时语音转换（README 明确） |
| 少样本微调 | ✅ **1 分钟训练数据**即可微调（本仓库核心卖点；实际 5 分钟以上相似度更好） |
| 文本描述设计音色 | ❌ 无 |
| 中文质量 | ⚠️ 「会调参是神器」。多音字处理靠 g2pW 多音字消歧；但幻觉/吞句是结构性问题 |
| 多语言 | **中、英、日、韩、粤语（yue）** 5 种。数据集格式语言码：`zh/ja/en/ko/yue`。支持跨语种推理 |
| 情感控制 | ❌ **无显式支持**。README Todo 中「Enhanced TTS emotion control」被**划掉**，改为「Maybe use pretrained finetuned preset GPT models for better emotion」 |
| 方言 | 仅**粤语** |
| 韵律 | ✅ TTS 语速控制（Todo 已勾选）；v3 起 GPT 模型更稳定、重复和遗漏更少、更易生成丰富情感表达 |
| 流式 | ❌ 无官方流式。有 `inference_webui_fast.py`（快速推理 WebUI）+ 第三方 CPUFast 分支 |
| 长文本 | ⚠️ 支持按句切分，但**长文本吞句问题严重**（见 4.7） |
| 声音转换 VC | ✅ **有**。仓库 tagline 即 "A Powerful **Few-shot Voice Conversion** and Text-to-Speech WebUI"；Todo 中「Zero-shot voice conversion (5s) / few-shot voice conversion (1min)」已勾选 |

### 4.3 硬件 / 推理速度 / Apple Silicon（重点）

**RTF（README 官方原文，v2 ProPlus）**：
> **0.028 on 4060Ti；0.014 on 4090（1400 words ≈ 4 分钟音频，推理仅 3.36 秒）；0.526 in M4 CPU**

→ 这是**四个模型中最强的速度数据**：4090 上约 **70 倍实时**，4060Ti 上约 **36 倍实时**。README 甚至专门加了一句「请不要尬黑 GPT-SoVITS 推理速度慢，谢谢！」

**显存**：
- **训练**（DeepWiki 汇总）：v1/v2 ~10GB；v3/v4 **8GB（LoRA）/ 14GB（全量）**；v2Pro/v2ProPlus ~12GB。
- **推理**：⚠️ 官方未给明确数字 → **未核实**。社区普遍反映 4GB 起（开 `is_half`），6GB 宽裕。
- **Linux 可选 ROCm**（`install.sh --device ROCM`）。

#### 🟡 Apple Silicon MPS 支持：**部分支持（partial）—— 推理 YES，训练 NO**

一手证据：

✅ **官方文档层面承认**：README `Tested Environments` 表格明确列出两个 Apple silicon 行：
| Python | PyTorch | Device |
|---|---|---|
| 3.9 | 2.5.1 | **Apple silicon** |
| 3.11 | 2.7.0 | **Apple silicon** |

✅ **安装器有 MPS 选项**：`install.sh --device <CU126|CU128|ROCM|**MPS**|CPU>`，macOS 安装段写 `bash install.sh --device <MPS|CPU> --source <...>`。

🔴 **但 MPS 选项实际安装的是 CPU 版 PyTorch**（`install.sh` 实证）：
```bash
MPS)
    USE_CPU=true
    ;;
CPU)
    USE_CPU=true
    ;;
```
→ 即「MPS 设备」与「CPU 设备」在安装层面**完全等价**。

✅ **推理代码支持 mps 设备**：`GPT_SoVITS/TTS_infer_pack/TTS.py:1535` 有明确分支
```python
elif str(self.configs.device) == "mps":
    torch.mps.empty_cache()
```
；`webui.py:91` 有注释掉的 `os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 当遇到mps不支持的步骤时使用cpu`（说明作者知道 MPS 有算子缺口，需 CPU 回落）。

🔴 **训练代码硬编码 CPU**：`GPT_SoVITS/s2_train.py:50`、`s2_train_v3.py:50`、`s2_train_v3_lora.py:50` 均为
```python
device = "cpu"  # cuda以外的设备，等mps优化后加入
```
→ **MPS 训练明确未实现（"等mps优化后加入"）**。

🔴 **官方对 Mac GPU 训练的明确警告**（README macOS 段）：
> "**Note: The models trained with GPUs on Macs result in significantly lower quality compared to those trained on other devices, so we are temporarily using CPUs instead.**"

🟡 **第三方学术实测证实 MPS 路径脆弱**：Zenodo 论文 *Benchmarking Real-Time Voice Cloning on Consumer Apple Silicon: A Practical Evaluation of GPT-SoVITS on M-Series Hardware*（2026-04-07，MacBook Pro M4 Pro 24GB）摘要原文：
> "We present the first systematic benchmark of GPT-SoVITS ... running entirely on consumer Apple Silicon hardware. We identify and resolve **seven critical platform incompatibilities, including pervasive float16 precision errors.**"
https://zenodo.org/records/19458410

🟡 **另有 Issue #2612**「How to use mps on macOS when inference model?」（2025-09-21）—— 说明 MPS 推理路径使用上仍有社区疑问。

**结论（GPT-SoVITS on Mac）**：
- 推理：可用。官方 RTF 数据点 **M4 CPU 0.526**（即约 1.9 倍实时）。**M4 GPU/MPS 的 RTF 官方未给 → 未核实**（Zenodo 论文称在 M4 Pro 上做通了全流程，但需解决 7 项不兼容 + 大量 fp16 精度问题）。
- 训练：**不支持 MPS，且官方主动劝阻在 Mac GPU 上训练**。要在 Mac 上微调只能走 CPU（慢）或换 NVIDIA 机器。
- **实操建议**：Mac 上做 GPT-SoVITS **推理**可行但需自备 `PYTORCH_ENABLE_MPS_FALLBACK=1` 与 fp32 处理（避开 fp16 精度坑）；**训练请用 CUDA 机器**。

- **磁盘占用**：⚠️ 官方未给总数字 → **未核实**。据 README 需下载：v4/v2Pro 预训练权重（`s2v4.pth`、`vocoder.pth`、`s2Dv2Pro.pth`、`s2Gv2Pro.pth`、`s2Dv2ProPlus.pth`、`s2Gv2ProPlus.pth`、`pretrained_eres2netv2w24s4ep4.ckpt`）、s1 GPT 权重、G2PWModel（中文）、UVR5 权重、以及 ASR 模型（Fun-ASR-Nano / SenseVoice / Paraformer / VAD / 标点 / faster-whisper-large-v3）。**多个 GB 量级**。

### 4.4 许可证 —— 四个里最宽松

| 对象 | 许可证 |
|---|---|
| 仓库 `LICENSE` | **MIT License, Copyright (c) 2024 RVC-Boss**（文件全文实证） |
| HF `lj1995/GPT-SoVITS`（预训练权重） | **`license: mit`**（cardData 实证）✅ **代码与权重同许可** |
| README 徽章 | MIT |
| Python | 3.10–3.12 |

✅ **商用允许**；✅ **允许再分发 / 打包进闭源商业桌面软件**（仅需保留版权声明与许可文本）；✅ 无 MAU/营收门槛；✅ 无领域禁用条款；✅ 无「禁止改进其他模型」条款。

⚠️ **需注意**：
- README 中「⚠️ Speaker voices: hosts 肥杰 and 惠子 ... Use without authorization is forbidden」只针对 **demo 音色**，不构成模型限制。
- 集成的第三方组件（UVR5 权重、GSA/ASR 模型、G2PWModel、faster-whisper）各自许可 **未逐个核实**，商用前建议单独排查。
- **这是本批四个模型里唯一「代码 + 权重都是标准 MIT」的组合，商业打包摩擦最小。**

### 4.5 质量：基准 / 榜单 / 口碑

⚠️ **官方未发布任何 seed-tts-eval / CV3-Eval 数字** → **无可引用的官方客观基准，未核实**。

**第三方榜单**：
- **Artificial Analysis Speech Arena：未上榜**（2026-09-16）。
- **tts.ai arena**：GPT-SoVITS 排第 **19** 位，**官方评分栏为空**（—），速度标 **"slow"**，Standard 档。
- ⚠️ 注意：tts.ai 标 "slow" 与官方 RTF 0.014（4090）/0.028（4060Ti）**严重矛盾**。可能原因是该站默认走 CPU 或未做优化配置。**引用时需注明冲突**。

**中文社区口碑（v2026.06，这是本批最详细的口碑数据）**：
- 三维评分：多音字 ★★☆☆☆、韵律 ★★☆☆☆、语气情感 ★★★☆☆（对比 IndexTTS2 的 ★★★★☆/★★★★☆/★★★★★）。
- 一句话定位原话：「**会调参是神器，幻觉/吞句/'翻译腔' 劝退纯做内容的人**」。
- 避坑提醒原话：「**GPT-SoVITS 别指望'开箱自然'**：幻觉、吞句、'翻译腔'是结构性问题；只想做内容的人不建议入坑。」
- **跨来源分歧（该报告专门标注）**：中文实测党/B站「新手党劝退（幻觉/吞句）」vs 海外/实战派「**海外有声书开发者当生产首选**」。解读：**「会调参是神器，只想出活是天坑」** —— 这是非常准确的画像。

### 4.6 工程可用性

- **官方 pip 包**：❌ 无（Not on PyPI）。Conda + `install.sh` / `install.ps1`。
- **一键包**：✅ **Windows 整合包**（`GPT-SoVITS-v3lora-20250228.7z` 等，双击 `go-webui.bat`）；✅ **Colab notebook**；✅ **AutoDL 云端 Docker**。
- **重依赖（实测）**：
  - 自建 `requirements.txt` + `extra-req.txt`（`pip install -r extra-req.txt --no-deps` 先装）。
  - 无 fairseq、无 DeepSpeed、无 flash-attn 硬需求 —— **架构轻，依赖比 IndexTTS/CosyVoice 更干净**。
  - 但需外部：**ffmpeg**（三平台都要）、**libsox-dev**（Ubuntu）、**Visual Studio 2017 运行库**（Windows）、**brew ffmpeg**（macOS）。
  - 大量可选模型体积：BERT、G2PW、UVR5、ASR。
- **Docker**：✅ `xxxxrt666/gpt-sovits`（Docker Hub），`docker_build.sh --cuda <12.6|12.8> [--lite]`；提供 `GPT-SoVITS-CU126/CU128` 与 `-Lite` 四种服务；**自动拉取 amd64/arm64 镜像**（arm64 存在 → Apple Silicon Docker 可行，但仍是 CPU 推理）。⚠️ README 警告 Windows Docker Desktop 默认 shm 小，需调大 `shm_size`（如 16g）。
- **ONNX 路线**：✅ **官方有 ONNX 导出**（`GPT_SoVITS/onnx_export.py`），DeepWiki 有「Model Export and ONNX」专门章节。**GGUF / llama.cpp 未发现**。
- **API 服务**：✅ **官方 REST API**（`api_v2.py`，DeepWiki「REST API」章节）+ Gradio WebUI（`webui.py`）+ 独立推理 WebUI（`GPT_SoVITS/inference_webui.py`）+ UVR5 WebUI + ASR 命令行工具。**服务化能力同样完善**。
- **命令行**：`python tools/asr/funasr_asr.py -i <in> -o <out> -l zh` 等。
- **Activity**：⭐61,794、**最后 push 2026-08-18（活跃）**。PR 活跃（如 #2810、#2809「字幕功能 + 彻底解决吞字问题」、#2807）。但 **release tag 停留在 2025-06-06**，改动走 main 分支未打 tag。
- ASR 支持已升级：默认 **Fun-ASR-Nano**（中英日韩 + 自动语种检测），粤语仍走经典 FunASR；可选 SenseVoice（快速转写）。

### 4.7 已知坑

1. 🔴 **吞字 / 吞句（最严重、最常被吐槽）**：
   - Issue #2658「prompt_text 携带会出现大量吞字」（2025-10-31）—— 提供参考音频文本反而大量吞字。
   - Issue #2695「吞整句问题」（2025-12-18）—— 约 100 句的长文本中，随机有若干句整句被吞（如从第 3 句直接跳到第 6 句），100 句里可能吞 2–3 句。
   - Issue #1317「一句里面，丢字现象还是很严重，**就像抽卡**」—— 且「重新抽也没啥，但就是很难判断哪句有问题」。→ **长文本必须逐句人工校对**。
   - Issue #1215「百分百复现吞词？」
2. 🔴 **重复/幻觉**：Issue #1313「英文读音出现重复读」（`twenty twenty twenty three`）。
3. 🔴 **「翻译腔」韵律**——社区共识的结构性问题。
4. 🔴 **Apple Silicon 的 fp16 精度错误**（Zenodo 2026-04-07）+ MPS 算子缺口需 CPU 回落 + Mac GPU 训练质量显著降低 → **Mac 训练请勿使用**。
5. **长文本无完美的「一把过」方案**（行业共识，非本模型独有）：需分段跑 + 人工兜韵律。
6. **无情感控制官方方案**（Todo 已划掉）。
7. **推理显存官方无数字**；WebUI 训练显存需求较高（v2Pro 系列约 12GB，v3/v4 全量 14GB）。
8. **v3/v4 对平均音质训练集表现差于 v1/v2/v2Pro**（官方原话）—— 选错版本会显著掉质量。
9. **版本众多（6 个家族）且有 5 个可选 WebUI 版本**，新手易选错；不同版本权重不能混用。
10. **release tag 停滞在 2025-06**，新特性只在 main 分支，**跟随 main 时缺少版本锚点**。

---

## 5. 横向对比速查表

| 维度 | IndexTTS-2.5 | FireRedTTS-2 | CosyVoice 3 | GPT-SoVITS v2ProPlus |
|---|---|---|---|---|
| 发布方 | bilibili | 小红书 | 阿里 FunAudioLLM | RVC-Boss（个人/社区） |
| 最新版本日期 | **2026-08-10** | 2025-09-08（最后动向 2025-10-26） | 2025-12 | 2025-06（tag） |
| 参数量 | 0.8B（GPT 主干） | ~1.7B（1.5B+0.2B） | 0.5B | ~30M GPT + ~100M SoVITS（未核实总量） |
| GitHub ⭐ | 23,995 | 1,432 | 23,632 | **61,794** |
| 最后 push | 2026-08-18 | 🔴 2025-10-26 | ⚠️ 2026-05-25 | 2026-08-18 |
| 零样本克隆 | ✅ 单参考音频 | ✅ 需参考文本 | ✅ 需精确参考文本（≤30s） | ✅ 5 秒 |
| 少样本微调 | ⚠️ 无官方训练码 | ✅ 有 | ✅ 有 | ✅ **1 分钟数据** |
| 多语言数 | 5 | 7 | **9 + 18 方言** | 5 |
| 情感控制 | ✅ **四路输入（最强）** | ❌ | ✅ instruct | ❌ |
| 流式 | ❌ | ✅ 140ms 首包 | ✅ **150ms 双流式** | ❌ |
| VC | ❌ | ❌ | ✅ | ✅ |
| 官方 VRAM | ~6GB | 9GB(bf16)/14GB(fp32) | 未核实 | 推理未核实；训练 12–14GB |
| 官方 RTF | 0.2065（4090 bf16） | 未给 | 未给 | **0.014（4090）/0.028（4060Ti）** |
| **Apple Silicon MPS** | 🟢 **官方支持** | 🔴 **无** | 🟡 **官方无，社区 PR + MLX/Swift/GGUF 三路线** | 🟡 **推理部分支持，训练不支持** |
| Mac 专用路线 | MLX/MNN/ONNX/Swift | 无 | mlx-audio / speech-swift / GGUF | 仅 CPU（M4 CPU RTF 0.526） |
| 许可证 | 🔴 **bilibili 自定义**（非 OSI） | ⚠️ Apache-2.0 + 学术限定声明冲突 | 🟢 **Apache-2.0** | 🟢 **MIT（代码+权重）** |
| 商用 | ✅（1亿MAU/10亿营收门槛下） | ⚠️ 有争议 | ✅ 无门槛 | ✅ 无门槛 |
| 中文 seed-tts CER | — | 1.14 | **0.81（RL）** | 未核实 |
| 中文口碑 | 🥇 情感天花板 | 未入横评 | 分歧大 | 「会调参是神器」 |
| 官方 pip | ❌ | ❌ | ❌ | ❌ |
| 官方 API server | Gradio + vLLM recipe | 仅 Gradio | ✅ **FastAPI + gRPC + Docker** | ✅ REST API + Gradio |

---

## 6. 给「商业桌面软件 + 声音复刻」场景的初步建议

> ⚠️ 以下是基于本次调研的事实性推论，**不构成法律意见**。

1. **许可证是首要决策轴，不是质量。** 若走闭源商业桌面分发：
   - **最安全**：GPT-SoVITS（MIT 代码+权重）→ CosyVoice 2/3（Apache-2.0 代码+权重）。
   - **需法务审查**：IndexTTS-2.5（bilibili 自定义，含下游传递义务 + 禁止改进其他模型 + 上海仲裁管辖）。
   - **需澄清**：FireRedTTS-2（Apache-2.0 与「仅供学术研究」声明冲突，建议邮件确认）。

2. **Apple Silicon（macOS 桌面应用）选型排序**：
   - 🥇 **IndexTTS-2.5**：官方代码内建 MPS 自动探测，且有 **MLX int8 移植（RTF≈0.45，比官方 PyTorch MPS 快 2.4×）** 和 **原生 Swift/MLX（speech-swift，Apache-2.0 代码）** 双保险。**但许可证最紧**。
   - 🥈 **CosyVoice 3**：官方无 MPS，但有三条成熟第三方 Mac 路线（mlx-audio MIT / speech-swift Apache-2.0 / GGUF-CrispASR），量化后权重仅约 1.0–2.1GB，且**许可证 Apache-2.0 无门槛**。综合「Mac 可跑 + 商用宽松」，**这可能是最优组合**。
   - 🥉 **GPT-SoVITS**：许可证最好、速度最快（CUDA 上），但 Mac 上只能 CPU（M4 CPU RTF 0.526，勉强够用）或走脆弱的 MPS 推理路径；**训练必须回 NVIDIA**。
   - ❌ **FireRedTTS-2**：Mac 无路，且仓库停滞 11 个月，许可证有歧义。

3. **若目标场景是「把用户短音频克隆成自然中文配音」**：IndexTTS-2.5（情感最强）或 CosyVoice 3（许可证最安全 + 中文 CER 最低）。
4. **若目标场景是「播客/多说话人对话」**：FireRedTTS-2 是唯一专为长对话设计的（3 分钟 / 4 说话人 / 140ms 流式），但需接受其单句克隆指标偏弱与停滞风险。
5. **若目标场景是「本地轻量、CPU 可跑、微调成本低」**：GPT-SoVITS（1 分钟数据微调 + 官方 CPU 支持 + Windows 一键包 + MIT）。
6. **所有模型的长文本都必须分段 + 人工校对**——这是行业共识，没有例外。

---

## 7. 未能证实的项（明确标注）

| 项 | 状态 |
|---|---|
| IndexTTS 官方是否支持微调/训练 | 未核实（仓库内未见训练代码，但官方未明确声明不支持） |
| IndexTTS-2 在 HF 上的权重许可 | 未核实（HF cardData 无 license 字段） |
| IndexTTS 中文方言支持 | 未核实（未声明） |
| IndexTTS CPU 推理 RTF | 未核实 |
| FireRedTTS-2 的 RTF | 未核实（官方只给首包延迟 140ms） |
| FireRedTTS-2 CPU 推理 | 未核实（无官方支持声明） |
| CosyVoice 2/3 的官方 VRAM 与 RTF | 未核实（官方未发布数字） |
| CosyVoice3-1.5B 的开放权重 | 未核实（论文对比表出现，HF/ModelScope 无对应开源仓库） |
| CosyVoice-ttsfrd 资源包的许可 | 未核实 |
| CosyVoice 集成的 FunASR/FunCodec/Matcha-TTS/AcademiCodec/WeNet 各自许可 | 未核实 |
| GPT-SoVITS 参数量官方总量 | 未核实（仅从配置推得 GPT 组件规模） |
| GPT-SoVITS 推理显存官方数字 | 未核实 |
| GPT-SoVITS 磁盘占用总量 | 未核实 |
| GPT-SoVITS 在 M4 GPU/MPS 上的 RTF | 未核实（官方只给 M4 **CPU** RTF 0.526） |
| GPT-SoVITS 的 seed-tts-eval / CV3-Eval 官方基准 | 未核实（官方未发布） |
| Voice Clone Arena（TTS-AGI）中四模型排名 | 未核实（页面 JS 渲染，无法抓取静态数据） |
| tts.ai arena 的评分方法学 | 未核实，且发现其参数量数据存在错误（IndexTTS-2 标 300M、CosyVoice 2 标 300M），**可信度存疑** |

---

## 8. 一手来源清单

**GitHub 仓库**
- https://github.com/index-tts/index-tts （README / LICENSE / LICENSE_ZH.txt / DISCLAIMER / indextts/infer_v2_5.py / tools/gpu_check.py / webui.py / docs/README_zh.md）
- https://github.com/FireRedTeam/FireRedTTS2 （README / LICENSE / requirements.txt / fireredtts2/fireredtts2.py / bin/finetune_example/config_finetune_1.5b_0.2b.json）
- https://github.com/QwenAudio/CosyVoice （README / LICENSE / requirements.txt / example.py / cosyvoice/ 全量 grep）— 原 https://github.com/FunAudioLLM/CosyVoice 301 重定向至此
- https://github.com/RVC-Boss/GPT-SoVITS （README / LICENSE / install.sh / webui.py / GPT_SoVITS/configs/ / GPT_SoVITS/TTS_infer_pack/TTS.py / GPT_SoVITS/s2_train*.py）

**许可证文件（直接抓取原文）**
- https://raw.githubusercontent.com/index-tts/index-tts/main/LICENSE （bilibili Model Use License Agreement, 10,554 bytes）
- https://raw.githubusercontent.com/FireRedTeam/FireRedTTS2/main/LICENSE （Apache-2.0, 11,357 bytes）
- https://raw.githubusercontent.com/QwenAudio/CosyVoice/main/LICENSE （Apache-2.0, 11,357 bytes）
- https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/LICENSE （MIT, 1,065 bytes）

**HuggingFace model cards**
- https://huggingface.co/IndexTeam/IndexTTS-2.5 ｜ https://huggingface.co/IndexTeam/IndexTTS-2
- https://huggingface.co/FireRedTeam/FireRedTTS2
- https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 ｜ https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
- https://huggingface.co/lj1995/GPT-SoVITS
- https://huggingface.co/cstr/cosyvoice3-0.5b-2512-GGUF

**PyPI（Mac 轻量路线）**
- https://pypi.org/project/index-tts-2.5-mlx/
- https://pypi.org/project/index-tts-2.5-mnn/
- https://pypi.org/project/index-tts-2.5-onnx/

**Apple Silicon 第三方实现**
- https://github.com/soniqo/speech-swift （Apache-2.0；IndexTTS2TTS / CosyVoiceTTS 模块）
- https://github.com/DePasqualeOrg/mlx-audio-plus （MIT；CosyVoice3 指南 docs/tts/cosyvoice3.md）
- https://soniqo.audio/guides/cosyvoice
- https://www.drmhse.com/posts/running-funaudio-on-mac-mlx-pytorch/ （CosyVoice3 Mac RTF 0.757 实测）

**MPS 支持证据**
- https://github.com/index-tts/index-tts/commit/17a3b0857bf9004723ea6a1795c8af4c2f6fd148 （Adds MPS support for Apple Silicon, 2025-04-11）
- https://github.com/index-tts/index-tts/pull/78 （CLI + mps device）
- https://github.com/FunAudioLLM/CosyVoice/pull/1129 （limited MPS support, tested on M4 Max）
- https://github.com/FunAudioLLM/CosyVoice/pull/1869 （Apple Silicon MPS, device abstraction layer）
- https://github.com/RVC-Boss/GPT-SoVITS/issues/2612 （how to use mps on macOS）
- https://zenodo.org/records/19458410 （GPT-SoVITS on Apple Silicon 系统基准，2026-04-07）

**论文**
- https://arxiv.org/abs/2601.03888 （IndexTTS 2.5，v1 2026-01-07 / v5 2026-08-11）
- https://arxiv.org/abs/2506.21619 （IndexTTS 2）
- https://arxiv.org/abs/2509.02020 （FireRedTTS-2）
- https://arxiv.org/abs/2505.17589 （CosyVoice 3）
- https://arxiv.org/abs/2412.10117 （CosyVoice 2）

**榜单与口碑**
- https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights （开放权重榜，2026-09-16 抓取：四模型均未上榜）
- https://offlinetts.com/blog/tts-arena-leaderboard-2026/ （TTS Arena 2026 解读，2026-05-06 / 更新 2026-08-01）
- https://tts.ai/tts-arena/ （二手聚合站，数据存在错误，谨慎引用）
- https://github.com/duevan07/chinese-ai-tts-real-review （中文 AI 配音真实口碑横评 v2026.06，2026-06-26）
- https://liudon.com/posts/voice-cloning-solution-comparison/ （2026-05-18 / 更新 2026-09-09，RTX 4090 实测部署对比）
- https://deepwiki.com/RVC-Boss/GPT-SoVITS/1.2-model-versions-and-evolution （GPT-SoVITS 版本表，最后索引 2026-01-17）

**已知问题（GitHub Issues）**
- IndexTTS：#775（英文对齐坍塌）、#364（CUDA assert/显存）、#585（3060 慢）、#410（输出怪异）、#460（Index Error）
- CosyVoice：#1886（torch≥2.7 乱码）、#1677（CosyVoice3 乱码）、#1911（vLLM 转换后非文本发音）、#1704（零样本回归）、PR #1916（stop_token / 多 GPU 修复）
- GPT-SoVITS：#2658（prompt_text 吞字）、#2695（吞整句）、#1317（丢字像抽卡）、#1215（吞词）、#1313（英文重复读）、#2612（MPS）

---

*报告生成时间：2026-09-16 UTC ｜ 所有 GitHub/HF/PyPI 数据为当日实时抓取 ｜ 标注「未核实」处未做任何推测性填充*
