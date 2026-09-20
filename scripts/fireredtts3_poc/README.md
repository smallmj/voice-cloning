# FireRedTTS3 有界 PoC（issue #26）

在 Apple Silicon（MPS）上验证 FireRedTTS3-Base 的零样本复刻是否可用。
上游：https://github.com/FireRedTeam/FireRedTTS3 （Apache-2.0，代码 + 权重）。
**只验 Base 的复刻**（参考音频 + 参考文本），不碰 Instruct。

## 补丁

`patch_fireredtts3.py` 对上游源码做四类最小改写（详见该文件 docstring）：

1. 设备选择：`torch.device('cuda')` 硬编码 → cuda → mps → cpu 回退；
2. SDPA：所有 `attn_implementation='flash_attention_2'` → `'sdpa'`；
3. autocast 守卫：bf16 autocast 在无 CUDA 时 `enabled=False`（MPS 走 fp32）；
4. seed 守卫：`torch.cuda.manual_seed*` 仅在 CUDA 可用时调用。

fasttext 不装也不改代码：`FireRedTTS3(..., use_fasttext=False)`。

每条补丁都锚定上游原文，锚点丢失即报错——上游漂移时必须人工重新核对，
不允许静默打出无效补丁。补丁幂等，可重复执行。

## 环境搭建（一次性）

```bash
ROOT="$HOME/Library/Application Support/VoiceClone/runtime/engines/fireredtts3-poc"
# 运行时根必须是纯 ASCII（ADR-0002），默认路径已满足

uv venv "$ROOT/venv" --python 3.11
uv pip install --python "$ROOT/venv/bin/python" \
  torch==2.8.0 torchaudio==2.8.0 transformers==5.6.2 einops==0.8.2 regex wetext soundfile
# 注意：不装 flash_attn（全 76 个 PyPI 版本零 win_amd64 wheel）、不装 fasttext、
# 不装 torchcodec（2.8 的 torchaudio 走 soundfile 即可）

# 权重（仅 Base 所需，约 12.3 GB）：
#   campp/campplus_voxceleb.bin
#   fireredtts3_base/{config.json,model.safetensors}
#   redae/{config.json,model.safetensors}
#   text_tokenizer/{tokenizer.json,tokenizer_config.json,vocab.json}
# 下载源照 ADR-0016：HF → hf-mirror 回退

# 上游源码 tarball：
curl -L -o frt3.tar.gz https://codeload.github.com/FireRedTeam/FireRedTTS3/tar.gz/refs/heads/main
```

## 运行

```bash
ROOT="$HOME/Library/Application Support/VoiceClone/runtime/engines/fireredtts3-poc"
"$ROOT/venv/bin/python" scripts/fireredtts3_poc/patch_fireredtts3.py \
  "$ROOT/../fireredtts3-poc-upstream"
"$ROOT/venv/bin/python" scripts/fireredtts3_poc/run_poc.py \
  --upstream "$ROOT/../fireredtts3-poc-upstream" \
  --weights "$ROOT/weights" \
  --reference <参考音频.wav> --prompt-text "<参考音频逐字转写>" \
  --out <输出目录>
```

产出：`results.json`（逐条音频时长 / 墙钟 / RTF / MPS 峰值分配内存）+ `wav/` 下
三段样音与 17 项回归集音频。不加自动评分；ASR 可懂度对照与**人工听感判定**
在 PoC 运行之外完成（issue #26 标 `ready-for-human` 的原因）。
