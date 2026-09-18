# macOS 装机报告：IndexTTS-2.5（MPS 引擎，indextts-25-mps）

> 姊妹篇：`windows-11-indextts-2.5.md`（Windows CUDA 档）。本文记录 2026-09-19
> 在 MacBook Pro M5 Pro 24GB 上把 IndexTTS-2.5 跑通 MPS 推理的真实过程。

## 结论

- **可行**。torch 2.8.0 + MPS + fp32 端到端真实克隆合成成功，输出通过峰值自检。
- 热 RTF ≈ **5.0**（RTX 3090 CUDA 档是 0.21）——只适合短句；长文本走任务队列。
- 冷启动模型加载约 **75s**（fp32 权重 ~10GB 进统一内存）。
- 权重可**零下载复用**本机已有的 checkpoint 树。

## 与 CUDA 引擎的差异（为什么不是同一个引擎）

| 维度 | indextts-25-cuda | indextts-25-mps |
|---|---|---|
| 设备门 | CUDA gate（torch.version.cuda + cuda.is_available） | MPS gate（mps.is_built + mps.is_available） |
| torch | 显式 cu128 wheel URL（download.pytorch.org → 阿里云镜像） | PyPI 钉版本 `torch==2.8.0` / `torchaudio==2.8.0`（macOS arm64 wheel 自带 MPS） |
| 精度 | bf16 | **fp32**（use_bf16=False）：Apple Silicon 上唯一经真实生成验证的精度路径 |
| 权重来源 | 下载（HF → hf-mirror → ModelScope） | **本地复用优先**：`VOICECLONE_INDEXTTS25_LOCAL_WEIGHTS` 指向已有 checkpoint 树，缺失文件才走下载 |
| 显存管理 | torch.cuda.empty_cache + VRAM 采样 | torch.mps.empty_cache + current_allocated_memory |

## 安装

```bash
# 1. 指向本地已有权重（如 ~/index-tts/checkpoints，~10GB 官方 2.5 权重 + hf_cache 辅助模型）
export VOICECLONE_INDEXTTS25_LOCAL_WEIGHTS="$HOME/index-tts/checkpoints"
# 2. 通过界面引擎卡片的「安装」，或直接：
#    POST /engines/indextts-25-mps/install
```

安装器行为（实测）：
- 18 个主权重 + 7 个辅助模型全部从本地树拷贝，**零网络下载**；辅助模型按
  dest 的 `hf_cache/...` 布局与 repo 相对路径双重查找。
- torch 步骤只装钉版本的 PyPI wheel；不在 macOS 上取任何 CUDA wheel。
- 生成期 `HF_HUB_OFFLINE=1`，全部模型走本地目录，不碰网络。

## 本地 checkpoint 树的来源（本机）

`~/index-tts/checkpoints` 是此前 IndexTTS 官方仓库 clone 时留下的完整
2.5 权重（gpt.pth / codec.pth / s2mel.pth / qwen0.6bemo4-merge/ + hf_cache 辅助
模型）。若机器上没有，可让引擎走常规下载源，或先
`huggingface-cli download IndexTeam/IndexTTS-2.5` 到任意目录再指给环境变量。

## 验证记录（2026-09-19）

| 项 | 值 |
|---|---|
| 参考音频 | `~/index-tts/examples/voice_01.wav` |
| 冒烟 1 | 3.99s 音频 / 25.11s 推理（RTF 6.29，含预热），峰值 0.542 |
| 冒烟 2 | 4.42s 音频 / 22.20s 推理（RTF 5.02），峰值 0.569，墙钟 42.9s |
| 峰值自检 | 两条均通过（无削顶、无静音） |
| 测试 | sidecar 全套 325/325 通过（含 test_indextts25_mps.py 8 条） |

## 已知边界

- **fp32 是选择，不是妥协的默认**：ComfyUI T8 节点在 MPS 上同样走 fp32；
  bf16 autocast on MPS 上游未验证，切换前必须重做端到端验证。
- 长文本（>500 字/请求）超出单次上限，交给任务队列分段（issue #13 机制）。
- `use_qwen_emo` 未启用（情绪模型不加载），与 CUDA 版一致。
- 提交：`fbbc02c feat: indextts-25-mps engine — IndexTTS-2.5 on Apple Silicon via MPS`。
