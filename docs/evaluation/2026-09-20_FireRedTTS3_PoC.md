# FireRedTTS3 有界 PoC 结论（issue #26）

- 日期：2026-09-20
- 机器：Apple M5 Pro（Mac17,9），24 GB 统一内存，macOS，torch 2.8.0 + **MPS**
- 精度：**fp32**（补丁把 bf16 autocast 在无 CUDA 时置为 no-op）
- 注意力：**SDPA**（flash_attn 未安装）
- 范围：**仅 Base 复刻**（参考音频 + 参考文本），未碰 Instruct
- 参考音频：`sidecar/data/audio/e2e-offline.wav`（Qwen3-TTS MLX 产物，24 kHz，
  转写「今天天气真好，我们一起去公园散步吧。」）
- 证据：本目录 `results.json` + `wav/`（三段样音 sample-1..3 + 17 项回归集音频）
- 运行方式：`scripts/fireredtts3_poc/`（补丁 + 运行器 + README）

## 结果

**补丁后 20/20 全部跑通**（3 样音 + 17 项回归集，零失败）。摘要：

| 指标 | 值 |
| --- | --- |
| 模型加载 | 16.4 s（mps） |
| RTF（音频时长/墙钟） | 中位 **2.22**，区间 1.66–4.45（sample-1 含首次 warmup） |
| MPS 峰值分配内存 | **11.7 GB**（24 GB 统一内存下有余量） |
| 权重 | Base 所需约 12.3 GB（safetensors，fp32 加载后运行时约 11.7 GB） |
| 分段/拼接 | 官方自带切句 + cross-fade，长文本无需我们再分段 |

分类看：数字/日期类最顺（RTF ≈ 1.7），中英混读与多音字略慢（≈ 2.2–2.6）。
是否读对、像不像，**留给人工听感**（`wav/` 20 个文件；对照 `expect` 口语形态
见 `sidecar/voiceclone_sidecar/regression.py` 的 `REGRESSION_ITEMS`）。

## 与调研预期的对照

plan.md §5 引用的社区实测是 **M4 Max bf16 RTF 0.73–0.76**；本机 fp32 实测
中位 2.22 慢约 3 倍，与 fp32/bf16 的算力差相符——**不构成对调研数字的反驳**，
但意味着：若采纳，MPS 路线还须再验证 bf16 autocast（`torch.autocast(
device_type='mps', dtype=torch.bfloat16)`）或量化，否则本地体验不可接受。

## 结论（技术闸门）

- **技术可行性：通过。** 四类补丁（设备选择 / SDPA / autocast 守卫 / seed 守卫）
  足以在 MPS 上完整跑通 Base 复刻；`use_fasttext=False` 路径正常；无
  flash_attn、无 fasttext、无 torchcodec 依赖。
- **采纳与否：待人工听感判定**（issue #26 标 `ready-for-human` 的原因）。
  听感通过后仍须先解决两件事才可接入：
  1. MPS bf16 路线实测（当前 fp32 RTF 2.22 不达标）；
  2. 另立 ADR 记录「为什么愿意为一个 1 贡献者、15 commit、0 release 的上游
     承担维护」——所需平台能力全部停在未合并 PR 上。
- 合规备注（沿用 plan.md §5，不在本 PoC 范围内重判）：README「零样本复刻仅限
  学术研究用途」声明不在 LICENSE 内，采纳前须评审；已登记 `NOTICE.md` §3。
