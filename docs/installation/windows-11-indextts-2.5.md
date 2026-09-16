# 装机报告：干净环境 + 国内网络下的 IndexTTS-2.5（Windows / CUDA）

- 工单：smallmj/voice-cloning#4（blocked by #3）
- 日期：2026-09-17
- 目标机：ALAN-PC —— Windows 11 Pro 22631，RTX 3090 24GB，驱动 616.64
- 结论：**装机成功，端到端出音通过**（详见各节验收记录）

本报告是第一版安装器的需求文档：每一节记录「预期 vs 实际发生」，失败点单独汇总在最后一节。

## 1. 环境基线

| 项 | 状态 |
|---|---|
| 系统 Python | 机器上预装了 Python 3.14（`C:\Python314`），但**全程未被使用**——运行时由 uv 托管（Python 3.12 sidecar / 3.11 引擎 venv）。即：不预装 Python 的干净机器同样可以完成装机，这条验收的实质是「安装器不得依赖系统 Python」 |
| uv | `C:\Python314\Scripts\uv.exe` 已在 PATH（干净机器上由 runtime 的 `uvman.find_uv` 自动下载，源可用 `VOICECLONE_UV_DOWNLOAD_URL` 走 ghfast.top 镜像） |
| GPU 驱动 | 616.64 ≥ CUDA 12.8 所需版本，无需另装 CUDA Toolkit（torch cu128 轮子自带运行时） |
| 运行时根 | `%APPDATA%\VoiceClone\runtime`，纯 ASCII ✓（ADR-0002 规则 2：中文/空格路径会让 ninja 生成的 build 文件乱码、CUDA 内核编译静默降级） |
| 磁盘占用 | 引擎 venv ≈ 7 GB（torch cu128 + index-tts 依赖树），权重 ≈ 8.5 GB，合计 ≈ 15.5 GB |

## 2. 依赖安装

### 2.1 torch / torchaudio：显式 wheel 源，不是附加索引

ADR-0002 规则 1 禁止 `--extra-index-url`。本引擎把这条推进到最强形式：**不用任何索引**，torch-2.8.0+cu128 与 torchaudio-2.8.0+cu128 按「精确 wheel URL + 镜像降级列表」直接下载（`download.pytorch.org` → `mirrors.aliyun.com/pytorch-wheels`），再用 `uv pip install <本地 wheel 文件>` 落进引擎 venv。整个 torch 步骤不查询任何索引，也就不存在「附加索引悄悄解析出 CPU 版」的可能。

实测速度（单连接，本机网络）：

| 源 | 速度 | 备注 |
|---|---|---|
| download.pytorch.org | ≈ 800 KB/s（有时 TLS 直接黑洞，见 §5） | 保持首选 |
| mirrors.aliyun.com/pytorch-wheels/cu128 | ≈ 240 KB/s（单连接限速） | 可用的降级源 |
| hf-mirror.com | ≈ 1 KB/s（基本不可用） | 当天该域名近乎瘫痪 |

下载支持 HTTP Range 断点续传；进程被杀后重启安装，日志确认 `resuming ... at byte 38834176`，未重下。

### 2.2 引擎包与其余依赖

- `indextts` 不在 PyPI 上，按 GitHub 归档 URL 安装：默认走 `ghfast.top` 加速，失败自动回退直连 GitHub（本机 ghfast 正常）。
- 其余依赖（transformers 4.52.1、numpy 2.2.6、keras 2.9.0、opencv-python 4.9.0.80 等）解析自阿里云 PyPI 镜像（`UV_DEFAULT_INDEX` 环境变量，可覆盖）。首次尝试直连 pypi.org 时 TLS 握手被重置（`tls handshake eof`），换镜像后一次通过——这验证了「安装器必须默认国内镜像、失败必须显式报错」的要求。
- Python 版本钉在 3.11：index-tts 要求 `>=3.10,<3.12`，与 sidecar 的 3.12 隔离在两个 venv 里，互不污染。

## 3. 权重下载

### 3.1 主权重（IndexTeam/IndexTTS-2.5，≈5.6 GB）

18 个文件：gpt.pth 3.26 GB、qwen0.6bemo4-merge 1.19 GB、codec.pth 607 MB、s2mel.pth 415 MB 及配套 tokenizer/tiktoken。源列表 HF → hf-mirror → ModelScope。

实测（单连接）：ModelScope ≈ 19.8 MB/s；huggingface.co 直连 ≈ 3.2 MB/s（当天意外可用）；hf-mirror ≈ 1 KB/s。

> **安装器需求**：源顺序里 ModelScope 应作为国内网络的优选候选，而不是「最后的救命稻草」；同时必须保留三源降级，因为任一镜像当天都可能挂（hf-mirror 本次实测瘫痪）。

### 3.2 辅助模型（≈3.0 GB）

按 `ensure_models_available` 期望的 `hf_cache/` 目录结构预下载，推理期零联网：

| 文件 | 来源（HF repo） | 体积 |
|---|---|---|
| w2v-bert-2.0（config / preprocessor / model.safetensors） | facebook/w2v-bert-2.0 | 2.32 GB |
| semantic_codec_model.safetensors | amphion/MaskGCT | 177 MB |
| campplus_cn_common.bin | funasr/campplus | 28 MB |
| bigvgan（config.json / bigvgan_generator.pt） | nvidia/bigvgan_v2_22khz_80band_256x | 449 MB |

BigVGAN 的自定义 CUDA 融合核（`use_cuda_kernel=True`）需要 ninja JIT 编译，本引擎默认关闭（`use_cuda_kernel=False`），回避中文/空格路径 + ninja 的历史坑，代价是 vocoder 略慢。

## 4. 运行期行为验收

| 验收项 | 实现 | 结果 |
|---|---|---|
| CUDA 闸门 | worker 启动即断言 `torch.cuda.is_available()` 且 `torch.version.cuda` 非空，不满足以退出码 3 拒绝启动，supervisor 标记 degraded，后续请求直接报可操作错误，绝不静默跑 CPU 慢速结果 | ✓ 单测 + 真机 |
| 崩溃自动重启 | `WorkerSupervisor`：请求间死亡 → 下次请求惰性重启；请求中死亡 → 原请求在新鲜 worker 上重试一次；队列由 sidecar 每引擎锁保证不丢 | ✓ 单测 + 真机（§6 E2E） |
| 显存回收 | 三杠杆：显式 unload（卸模型 + `empty_cache`）、空闲 300 s 自动回收（进程退出是唯一可靠的显存释放）、每 20 次生成回收 worker 兜底慢泄漏 | ✓ 单测 + 真机 |
| 峰值自检 | worker 对输出 WAV 计算 peak：全零直接拒发；peak > 0.999 在日志中告警削顶 | ✓（复用 ADR-0002 规则 3） |
| 队列不丢 | sidecar 对同引擎生成串行（per-engine lock），排队请求在 worker 崩溃恢复后继续执行 | ✓ 单测 |

## 5. 失败点清单（安装器需求核心）

按发生顺序，每条都是安装器必须处理的真实故障：

1. **pypi.org 直连 TLS 被重置**（`tls handshake eof`）——所有 pip 解析必须默认走国内镜像，且镜像不可用时报显式错误而非静默变慢。
2. **download.pytorch.org TLS 黑洞**——`schannel: failed to receive handshake`，且过一段时间又恢复（本会话内先黑洞、后 ≈800 KB/s 可用）。结论：torch 源也必须可降级，不能写死单一域名。
3. **hf-mirror.com 当天瘫痪**（≈1 KB/s）——「三源降级」不是过度设计。任何单一镜像假设都会把装机卡死。
4. **ModelScope 才是国内最快的稳定源**（≈20 MB/s）——但它在 index-tts 官方文档里只是「无代理环境」的备胎。安装器应把 ModelScope 纳入权重源列表。
5. **uv 长下载无输出**——`uv pip install` 的进度不会进入我们的日志，安装器 UI 必须改用「按字节轮询产物」而不是解析 uv 输出（torch 步骤改为先下载 wheel 文件后安装，正是为了拿到逐字节进度条）。
6. **SSH 会话杀死前台安装进程**——装机必须以独立进程运行（Windows 上 `Start-Process` 脱离会话），否则远端断连 = 安装中断。安装器天然满足这一点（sidecar 进程独立于 Electron）。
7. **keras 2.9.0 + opencv-python 4.9.0.80 这对上游钉死的组合**在 Python 3.11 / numpy 2.2.6 下安装通过（cv2 导入未在推理路径触发问题）；若未来升级 numpy，opencv 二进制兼容是第一个要复查的点。
8. **BigVGAN CUDA 核编译风险**——`use_cuda_kernel=True` 需要 ninja JIT；默认关闭，未来若开启必须先验证路径纯 ASCII（ADR-0002 规则 2）。
9. **安装步骤重试不应清空已下载的权重目录**——初版实现给下载步骤声明了 artifact 目录，一次失败重试会整目录清空、已下好的 2.3 GB 重新下载。改为不声明 artifact（downloader 的逐文件完整性校验 + .part 续传已保证正确性）。
10. **本地 wheel 文件名必须含字面 `+`**——URL 用 `%2B` 编码下载，落盘后若保留 `%2B`，uv 直接拒绝：`The wheel filename ... has an invalid version`。下载后需 unquote 再安装。
11. **Windows 控制台 GBK 编码**——worker 输出的进度条字符（\ufffd 等）会让 GBK stdout 抛 UnicodeEncodeError，波及调用方。任何读取 worker 输出的进程都必须显式 UTF-8（supervisor 已固定 `encoding='utf-8', errors='replace'`）。
12. **detach 运行的安装进程仍会被 SSH 会话连带杀死**——`Start-Process` 后 driver 在 uv 长安装期间两次静默消失（无任何错误输出）；把安装放在持久 SSH 会话/真正独立的服务里执行才可靠。安装器实现里 sidecar 是常驻进程，天然规避此坑。
13. **上游 API 命名不稳定**——main 分支 `indextts/infer_v2_5.py` 的加载类仍叫 `IndexTTS2`（并非 `IndexTTS2_5`）。对接时以实际安装的源码为准，升级时这里是第一个复查点。

## 6. 端到端验收记录（2026-09-17 实测，`C:\VoiceClone\e2e_log.txt`）

- **stage1 CUDA 闸门**：引擎 venv 探针 `torch 2.8.0+cu128 / cuda 12.8 / available=True / RTX 3090` ✓
- **stage2 首次生成**（含模型加载 27.2 s）：36.4 s 出音，`e2e-1.wav`，22 050 Hz 单声道 9.10 s，峰值自检通过（拒绝静音、告警削顶）
- **stage3 热生成**：worker 常驻复用，7.9 s
- **stage4 显式卸载**：unload 后进程退出，nvidia-smi 实测显存 **7581 MB → 1518 MB，回收 6063 MB** ✓
- **stage5 崩溃恢复**：worker 被强杀后请求自动在新 worker 上重试成功（in-flight 请求无感恢复），后续请求 65.2 s（含二次模型加载）正常出音 ✓

样例音频已回传：生成内容「欢迎大家来体验声音复刻工作台，这是一段本地生成的测试音频。」（e2e-2.wav，8.4 s）

## 7. 对第一版安装器的直接影响

1. 权重源列表必须 ≥3 个（含 ModelScope），torch wheel 源必须 ≥2 个（含 aliyun pytorch-wheels），全部支持 Range 续传；
2. torch 安装走「显式 wheel 文件 + 本地安装」，拿到真实字节进度；
3. CUDA 闸门前置到安装后的首次启动（本次在 worker 启动时执行），失败信息必须包含驱动版本要求（≥ CUDA 12.8 runtime）；
4. 安装器 UI 的进度条必须基于轮询文件字节，不解析子进程输出；
5. 引擎 venv 与 sidecar 运行时严格隔离（Python 版本都不一致），复用 #3 的 per-engine venv 设计即可。
