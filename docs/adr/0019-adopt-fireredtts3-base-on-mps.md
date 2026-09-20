# ADR-0019: 采纳 FireRedTTS3-Base 作为本地引擎候选（MPS，承接上游维护责任）

日期：2026-09-20 · 状态：已接受 · 关联：issue #26、#25 · 证据：`docs/evaluation/2026-09-20_FireRedTTS3_PoC.md`

## 背景

FireRedTTS3（Apache-2.0，代码 + 权重）的中文复刻指标（Seed-TTS-eval avg 3.04/78.8）、21 种中文方言、以及强过 Qwen3-TTS-VD 的文生音色是真实优势，但三条硬伤：`flash_attn` 钉版本且零 win_amd64 wheel、官方 CUDA 硬编码（MPS 依赖未合并社区 PR）、上游仅 1 贡献者 / 15 commit / 0 release。issue #26 规定有界 PoC 过闸后才可采纳，且采纳必须另立 ADR 记录「为什么愿意承担上游维护」。

PoC 结果（M5 Pro / 24 GB / torch 2.8.0 + MPS / fp32 / SDPA）：四类补丁后 3 样音 + 17 项回归集 **20/20 全过**；RTF 中位 2.22；MPS 峰值 11.7 GB。人工听感判定通过（复刻相似度与可懂度达标）。

## 决策

采纳 FireRedTTS3-Base（仅复刻模式）进入引擎接入序列（追加进 #25），但受以下四条硬约束：

1. **MPS 路线先行，bf16 未验证前不发布。** fp32 RTF 2.22 距社区 M4 Max bf16 实测 0.73–0.76 慢约 3 倍；接入后须先实测 `torch.autocast('mps', bf16)`，不达标则该引擎仅标记实验性。
2. **补丁由本仓库维护**（`scripts/fireredtts3_poc/patch_fireredtts3.py`）：设备选择 / SDPA / autocast 守卫 / seed 守卫。补丁锚定上游原文、漂移即报错——上游任何更新都须人工重核补丁面，这是承接维护的直接成本。
3. **平台范围暂限 macOS MPS**；Windows CUDA 因 `flash_attn` 无 win_amd64 wheel 仍需另做依赖剥离，不在本次采纳范围。
4. **合规两项随接入评审**：README「零样本复刻仅限学术研究用途」声明（不在 LICENSE 内，Apache-2.0 §2 权利不被收窄，但须记录评审结论）；RedAE 训练数据含音效/音乐，来源问题许可不涉及，接受为已知残留风险（已登记 NOTICE.md §3）。

## 为什么愿意承担上游维护

- 指标与能力（中文复刻、21 方言、文生音色）没有零成本的等价替代：VoxCPM2 官方 MPS 支持更好但克隆指标弱于 FireRedTTS3，文生音色弱于 Qwen3-TTS-VD。
- 补丁面小且稳定：四类补丁都是「设备/注意力/精度」级别的模板改动，上游大改的信号（commit / release）容易被监控，最坏情形是钉死当前 commit 使用。
- 权重 Apache-2.0 自持：即使上游消失，已下载权重 + 本仓库补丁 fork 仍可长期自用。

## 后果

- #25 追加 FireRedTTS3-MPS 引擎条目（VoxCPM2 / dots.tts 范围不变）；接入走 #32 抽出的共享基类。
- NOTICE.md 已登记权重许可与补丁来源；回归集结果与 wav 证据长期保留在 `docs/evaluation/fireredtts3_poc/`。
- 若上游后续合并官方 MPS / SDPA 支持，补丁集应相应收缩并重跑 PoC 证据。
