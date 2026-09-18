# 0016 — 下载源三轴分离，首选源 + 静默兜底链，且必须显示实际生效源

日期：2026-09-18 · 状态：已采纳 · 关联：plan §8、CORRECTIONS C-001、ADR-0002、ADR-0015

## 背景

国内首次安装失败是 `plan.md` §8 列的头号风险，而"下载源"是它的唯一缓解手段——但**代码里根本没有这一层**：三个轴的来源全是模块级常量，散在四个文件里，只能靠未文档化的环境变量触达（而其中 `VOICECLONE_*` 那几个在应用里**根本不生效**，见 ADR-0015）。

`docs/research/CORRECTIONS.md` C-001 的实测结论决定了本 ADR 的形状：

> **hf-mirror 对境外出口 IP 会 308 跳回 `huggingface.co`**……一个开代理的用户会让这个镜像路径静默退化成"直连真 HF"，在国内即等于不可用，而且**表现为下载很慢或超时，不是明确报错**。

同时 `docs/installation/windows-11-indextts-2.5.md` 已经把要求写死了：`:105`「权重源列表**必须 ≥3 个（含 ModelScope）**……全部支持 Range 续传」、`:50`「同时**必须保留三源降级**，因为任一镜像当天都可能挂（hf-mirror 本次实测瘫痪）」、`:82`「**ModelScope 才是国内最快的稳定源**（≈20 MB/s）」。

而现状有一处覆盖漏洞：**ModelScope 只覆盖到部分文件**——IndexTTS 主权重有、7 个辅助模型里 5 个有，但 `nvidia/bigvgan_v2_22khz_80band_256x` 没有；**qwen3-tts-mlx 完全没有**（源列表只有 HF → hf-mirror）；**本地转写模型完全没有**。

另外 `HF_HUB_DISABLE_XET=1` 全仓**从未设置**，而文档已把它列为硬要求并解释了机制：Xet 流量走 `cas-server.xethub.hf.co`，**`HF_ENDPOINT` 覆盖不到**（huggingface_hub #4475）。它恰好只在最需要的地方缺失——`transcription.py` 是全仓唯一用 `huggingface_hub` 的路径，它设了 `HF_ENDPOINT` 却没设这个开关。

## 决策

1. **三个轴分离，互不合并**：**权重源**（HF 官方 / hf-mirror / ModelScope）、**包索引**（PyPI 及其镜像）、**轮子源**（PyTorch CUDA wheel）。混在一个"镜像"开关里会复现 plan §3 记录的 torch 事故——传递依赖把 CUDA 版 torch 顶成 CPU 版，29 秒音频跑了 700 秒。

2. **权重源界面是三选一，但选中的是"首选源"，其余两个保留为静默兜底。** 这样既满足用户要的简洁 UX，也保住文档要求的三源降级：任一首选当天挂掉或静默退化时，仍有第二个源接手。

3. **必须显示"本次实际由哪个源服务"。** 因为 C-001 的失效模式是**静默**的（表现为慢而不是报错），只让用户选、不告诉他实际生效的是哪个，用户就只能猜。安装日志需要新增这条结构化信息——当前字节进度只是被拼成日志字符串。

4. **补齐 ModelScope 覆盖**：`qwen3-tts-mlx`、bigvgan 的权重、以及本地转写模型都要有 ModelScope 源（调研已确认 Qwen3-TTS 有官方 ModelScope collection；bigvgan 与 whisper-small 的对应仓库**待核**）。

5. **转写路径必须设 `HF_HUB_DISABLE_XET=1`**。这是文档既有要求，且只有这条路径真正需要它。

6. **来源解析集中在 sources 模块**（ADR-0015 决策 2），引擎只声明"我要哪些文件"，不得自己拼 URL；新增引擎默认继承兜底链。

7. **本轮不做自定义源入口**（填 URL / 选本地目录 / `file://`）。plan §6 原措辞「下载源（ModelScope / 自建 CDN）」随之改为「下载源（首选源 + 兜底链）」。若将来要支持离线分发或自建 CDN，另开一条决策。

## 后果

- 「下载源」的界面归属在「引擎」区（ADR-0013），与本地模型管理同处一页；跨引擎通用的其余配置留在「设置」区。
- 三个轴必须各自可切换：只给"权重源"一个开关会让用户无法处理 PyPI 或 CUDA 轮子被墙的情况。
- 待核两项：(a) ModelScope 是否真有 bigvgan 与 `whisper-small` 的对应仓库，没有则该文件只能两源；(b) `HF_HUB_DISABLE_XET` 在 hf-mirror 场景下是否真如文档所述生效（社区有"好像不生效"的报告，来源未核实）。
- 本 ADR 只定策略与归属；**兜底链的顺序在"实际生效源"可见之前不得自动按速度重排**——自动重排会让用户看到的源与配置的源不一致。

## 落地核验（issue #17 实现，2026-09）

原「待核」两项已实测关闭：

- bigvgan：ModelScope 双胞胎为 `nv-community/bigvgan_v2_22khz_80band_256x`，`config.json` 与 `bigvgan_generator.pt` resolve 均 200，已写入 `AUX_WEIGHTS`。
- 本地转写（Windows faster-whisper-small）：`gpustack/faster-whisper-small` 的 `config.json` / `model.bin` / `tokenizer.json` / `vocabulary.txt` 均 200（该镜像无 `preprocessor_config.json`，faster-whisper 加载不需要）；macOS 的 `mlx-community/whisper-small` 仍无双胞胎，维持两源。
- qwen3-tts-mlx：`mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` 在 ModelScope resolve 200，缺口已关闭。
