# NOTICE 与第三方清单

本仓库所有代码以 **GNU Affero General Public License v3.0**（见 [`LICENSE`](LICENSE)）授权。
本文件是许可证义务的落地处：列出从第三方移植的代码位置、运行时依赖，以及**逐权重**核对的模型许可证（不是只看代码仓库的 LICENSE——权重卡上标的才算数）。

## 1. AI 生成内容声明

- 本应用合成的每一份音频产物都会写入 RIFF `LIST`/`INFO` 元数据（`ICMT` 注释 + `ISFT` 软件名），声明其为 AI 生成音频（见 `sidecar/voiceclone_sidecar/aigc.py`）。响度归一化等衍生副本会保留该声明（`loudness.py`）。
- 首次使用前，应用强制展示声音授权确认（知情同意），用户须确认拥有所上传声音的使用授权。
- 每个引擎在界面明示「上传内容是否用于训练」（能力声明 `upload_used_for_training` + 厂商数据用途说明 `data_usage_note`）。

## 2. 从 VoiceStudio 移植的部分

许可证：**AGPL-3.0** · 上游：https://github.com/debpalash/VoiceStudio

依据 ADR-0006，本项目**只移植「单个引擎怎么调」这一层**——各厂商的鉴权方式、计费规则、错误码语义、音色生命周期等踩坑事实与算法；不取其数据模型与界面。移植/参考落在以下文件（云引擎适配层）：

| 本仓库文件 | 移植内容 | 上游对应 |
| --- | --- | --- |
| `sidecar/voiceclone_sidecar/engines/dashscope_base.py` | 阿里云 DashScope 调用骨架、错误分类（`CloudEngineError`） | 其 DashScope/qwen 引擎适配器 |
| `sidecar/voiceclone_sidecar/engines/qwen_tts_cloud.py` | Qwen-TTS 复刻：参考音频注册、音色生命周期、按汉字计费规则 | 同上 |
| `sidecar/voiceclone_sidecar/engines/qwen_tts_vd_cloud.py` | Qwen-TTS 音色设计（voice design）接口与试听样本处理 | 同上 |

移植处保留 AGPL-3.0 义务：本项目整体以 AGPL-3.0 授权并向公众开放源码。

## 3. 模型权重许可证（逐权重核对）

> 核对方式：逐个读取权重所在 Hugging Face / ModelScope **模型卡标注的许可证字段**，而非其代码仓库的 LICENSE。核对日期：2026-09-17（背景研究见 `docs/research/local-models/`，其中含一手引文；后续升级引擎时须重新核对）。

| 权重 | 用途 | 许可证（权重卡标注） | 商用提示 |
| --- | --- | --- | --- |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | macOS 本地引擎（Qwen3-TTS MLX） | Apache-2.0 | 可商用 |
| `IndexTeam/IndexTTS-2.5` | 本地引擎（Windows CUDA + macOS MPS，同一套权重） | `other`：**bilibili Model Use License** | 有限商用：月活 ≤ 1 亿且年营收 ≤ 10 亿人民币；超阈值须另行书面授权（见 `docs/research/local-models/2026-09-16_AppleSilicon实测补充.md` §4.4） |
| ├ `facebook/w2v-bert-2.0` | IndexTTS-2.5 管线的语义编码器 | MIT（HF 卡片标注） | 可商用 |
| ├ `amphion/MaskGCT`（`semantic_codec/model.safetensors`） | IndexTTS-2.5 管线的语义 codec | **CC-BY-NC-4.0** | ⚠️ **非商用**。该权重是 IndexTTS-2.5 自身管线要求的依赖；**商用部署 IndexTTS-2.5 需自行取得 Amphion 授权** |
| ├ `funasr/campplus`（`campplus_cn_common.bin`） | 说话人特征 | Apache-2.0 | 可商用 |
| └ `nvidia/bigvgan_v2_22khz_80band_256x` | 声码器 | MIT | 可商用 |
| `mlx-community/whisper-small` | macOS 参考音频本地转写 | Apache-2.0（HF 卡片标注；OpenAI 仓库 LICENSE 为 MIT） | 可商用 |
| `Systran/faster-whisper-small` | Windows 参考音频本地转写 | MIT | 可商用 |

云端引擎（DashScope Qwen-TTS 等）不涉及权重下载：许可关系由厂商 API 服务条款约束，见各引擎在设置页展示的计费与数据用途说明。

## 4. Python 运行时依赖

见 `sidecar/pyproject.toml`：FastAPI（MIT）、uvicorn（BSD-3-Clause）、python-multipart（Apache-2.0）、httpx（BSD-3-Clause）、keyring（MIT）。开发依赖 pytest（MIT）、websockets（BSD-3-Clause）。本地引擎按需装入隔离运行时（mlx-audio / index-tts 等），其依赖随引擎文档而非本清单展开。

## 5. 前端依赖

见 `app/package.json`：Electron（MIT）、React / React DOM（MIT）、wavesurfer.js（BSD-3-Clause）、Vite（MIT）、esbuild（MIT）、TypeScript（Apache-2.0）、@vitejs/plugin-react（MIT）、cross-env（MIT）。

## 6. 音频处理组件的 GPL 审计（issue #15）

- 后处理范围**限定为响度归一化与格式/采样率转换**，实现为 `sidecar/voiceclone_sidecar/loudness.py`（纯 stdlib，ITU-R BS.1770-4）与 `segmentation.py`（拼接 + 线性重采样），**不引入任何 GPL 音频库**。
- 诊断与参考音频解码对非 WAV 格式会**可选地**调用系统安装的 `ffmpeg`（`diagnostics.py`、`voices.py`）。ffmpeg **不被本项目打包分发、也不被链接**，仅作为用户机器上的外部可执行文件按需调用；用户可安装 LGPL 构建以避免 GPL 义务。这是本仓库对 GPL 音频组件边界的正式记录。
- 未使用 SoX（GPL）、LAME（LGPL）、pydub、librosa 等组件作为运行时依赖。

## 7. 本仓库自有素材

- 设计文档、ADR、研究笔记：随仓库以 AGPL-3.0 授权。
- 图标/占位头像若无另行标注，为仓库自绘或公有领域素材。
