# 本地引擎参数缺口审计（dots.tts / VoxCPM2 / FireRedTTS3 / Qwen3-TTS-MLX）

> 对齐 issue-38 云端审计范式，覆盖四个本地引擎。调研日期：2026-09-21。依据：直接读取本机已安装的上游库源码（`~/Library/Application Support/voiceclone-app/runtime/engines/<id>/` 下 venv site-packages、fireredtts3 的 patched upstream checkout），dots.tts 依据 worker 源码 + CORRECTIONS.md C-003 + `docs/research/local-models` 选型报告（本机未安装，Windows-only）。
>
> 落地原则（ADR-0018）：每个字段要么 exposed（含 applies_to / to_wire），要么 `exposed=False + not_exposed_reason`（六因闭集）——不存在「靠 reviewer 记住的缺口」。

## 0. 已定决策（本轮访谈结论）

1. **范围**：四个本地引擎（dots.tts-cuda、voxcpm2、fireredtts3-mps、qwen3-tts-mlx）。目标是每个引擎的参数面板「信息量不差于 indextts」，以如实声明为准、不凑数量。
2. **暴露标准（访谈 Q2 = b）**：只暴露对「声音复刻」有实际意义的参数（速度、发音、停顿、情感、seed/随机性、复刻质量旋钮）；纯采样类细粒度参数（temperature/top_p/top_k 等）即使上游支持，也不进显眼区——收进引擎专属折叠区或 `exposed=false` + 原因披露。会破坏流水线或纯 no-op 的照 ADR-0018 惯例用 notExposedReason。
3. **验证标准（访谈 Q3）**：**unverified 参数本次就要真机验证**（Windows 3090 跑 CUDA 引擎、Mac 跑 MPS/MLX），验证通过即翻 exposed，不留给后续 issue。
4. **交付**：不先开 issue；本文档更新完成后走 to-spec。

## 1. 两个疑似 bug（优先级最高）

### 1.1 qwen3-tts-mlx：canonical language 是静默 no-op

- worker 传 `language=`，但 mlx-audio 0.5.4 qwen3_tts `generate` 的真参数名是 **`lang_code`**；`language` 落进 `**kwargs` 被静默忽略，实际语言恒为 "auto"。
- 即使改对名字，现有取值 "Chinese"（大写）也与权重 `config.json` 的 `codec_language_id` 键（全小写：chinese/english/french/german/italian/japanese/korean/portuguese/russian/spanish）不匹配。
- **修复方向（已定）**：wire 改传 `lang_code`；canonical language choices 扩成上游 10 语种全量（canonical 层规范大小写，wire 适配器小写化）；"auto" 由不传 lang_code 表达。当前 6 语种 choices 扩为 10。
- 这比已钉为 no-op 数据的 speed 更隐蔽：用户选了语言、界面记录了选择、实际什么都没发生。

### 1.2 voxcpm2：seed 疑似 TypeError

- worker 把 `seed` 传给 `model.generate()`，但钉扎版本 VoxCPM 2.0.3 的 `VoxCPM.generate → _generate` 签名**没有 seed 参数也没有 `**kwargs`**，`VoxCPM2Model._generate_with_prompt_cache` 全文件 grep seed/manual_seed 0 命中——传 seed 应直接 TypeError。
- capability_matrix.json 里 voxcpm2 生成参数条目标 verified 且列了 seed，与上述矛盾，证据来源版本存疑，需复核。
- **处置**：Windows 3090 真机真实合成验证一次。若坏，两条修法任选：(a) worker 侧 `torch.manual_seed()`（dots 引擎 `seed_everything` 有先例）；(b) 降 `exposed=false`。倾向 (a)，保持与其他本地引擎一致的 seed 能力。

## 2. 逐引擎缺口表

图例：(a) = 上游确定支持、可直接 wire；(b) = 上游支持但需真机验证；(c) = 上游不支持。**决策 3 生效后，(b) 项在本轮真机验证，通过即翻 exposed，未通过转 not_exposed + unverified。**

### 2.1 qwen3-tts-mlx（现 1 exposed + 1 no-op；缺口最多、风险最低，建议先做）

| 候选 | 类 | 决策（按 Q2=b 收敛） |
|---|---|---|
| temperature（默认 0.9）/ top_p（1.0）/ top_k（50）/ repetition_penalty（1.05） | (a) | 上游形参直传；按决策 2 进**引擎专属折叠区**，不进显眼区 |
| **language 修复** | (a) bug | §1.1，本轮必修 |
| speed | (c) | 维持 no-op 数据（audit 2026-09-18 §4 实测：接受/打印/不用） |
| instruct / voice | (c) | wrong-mode（base 模型传了 raise / custom_voice 模型专属） |
| seed | (c) | mlx-audio 该模型路径无 seed 钩子 |
| max_tokens（默认 4096/段） | (b) | 需真机定 500 字上限下的合理语义；按验证结果决定暴露或披露 |
| streaming_interval / streaming_context_size / split_pattern | (c) | breaks-pipeline（管线不适用） |

### 2.2 fireredtts3-mps（现仅 seed 1 个 exposed；可挖空间最大）

| 候选 | 类 | 决策 |
|---|---|---|
| inference_cfg（默认 2.0，CFG 引导强度） | (a) | 暴露——复刻质量旋钮（与 dots.guidance_scale / voxcpm.cfg_value 同族） |
| n_timesteps（默认 10，扩散步数） | (a) | 暴露 |
| cross_fade_ms（默认 50，句间淡入淡出） | (a) | 暴露 |
| stop_threshold（默认 0.5） | (b) | 真机验证截断行为后定 |
| token_max_n / token_min_n / merge_len / do_split（切句粒度） | (b) | 本轮不声明（连 not_exposed 行都不加）——真机听过效果再裁决是披露还是暴露 |
| do_clean | (b) | 同上；与 ADR-0008 边界（clean≠TN）需裁决 |
| do_tn / use_wetext | (c) | ADR-0018 决策 4 点名禁暴露：breaks-pipeline（引擎内 TN 与统一文本规范化层冲突） |
| max_text_len | (c) | audit 已定调内部值（正文从不引用），不暴露 |
| instruct | (c) | wrong-mode（Base 权重） |
| language | (c) | 上游仅区分 "Chinese"/"en"，写死 Chinese（其余 unverified 维持） |
| seed | 已有 (a) | — |

### 2.3 voxcpm2-mps / -cuda（现 3 exposed）

| 候选 | 类 | 决策 |
|---|---|---|
| **seed 修复/验证** | (b) bug | §1.2，本轮必修 |
| min_len / max_len（生成长度上下限） | (b) | 真机验证后定 UI 语义（max_len 与 200 字/请求 vendor 上限的关系）；验证不过不声明 |
| retry_badcase / retry_badcase_max_times / retry_badcase_ratio_threshold（坏例重试） | (b) | 真机验证后定（默认值上游 core 层 True / model 层 False 不一致，需实测） |
| denoise | 维持 (b) | 翻 exposed 需引入 ZipEnhancer（`load_denoiser=True` 拉 ModelScope 模型，与离线承诺冲突）——真机验证时一并裁决 |
| normalize | (c) | breaks-pipeline（wetext；{ni3} 音素语法要求 False） |
| denoise_output | (c) | wrong-mode（denoise 只作用于参考音频） |
| LoRA 权重 | (c) | 超出复刻产品范围 |
| language | (c) | 上游无语言参数（30 语种跟随文本） |

### 2.4 dots-tts-cuda（现 4 exposed + 3 防伪 no-op + server-injected；四引擎中最完备，无缺口）

| 候选 | 类 | 决策 |
|---|---|---|
| — | — | 已声明的 `speaker_scale / num_steps / guidance_scale / seed` + `temperature/speed` 防伪 no-op + `ref_audio/ref_text` server-injected 已覆盖全部真实参数面 |
| temperature / top_p / top_k | (c) | 上游不存在（全连续 latent flow-matching，README 明示）——现有 no-op 数据保留 |
| rate control | (c) | 仅 dots.tts.edit 独立模型有速率控制，接入属新引擎/新模型工作，非参数缺口 |
| optimize（torch.compile） | (c) | 安装期决策（冷启动 vs 推理提速），非生成期用户参数 |
| max_generate_length | (b) | vendor 口径 500 ≈ 80 s 上限，与 capabilities 的时长上限口径一致，可在 UI 提示，不需参数 |
| language | (c) | 上游自动检测 |

## 3. 本轮工作分解（供 to-spec）

1. **qwen3-tts-mlx**：修 language→lang_code + 小写 10 语种 wire 适配 + choices 扩量；补 temperature/top_p/top_k/repetition_penalty 四个 spec（引擎专属折叠区）。
2. **voxcpm2**：真机（3090）验证 seed；坏则 `torch.manual_seed` 修复；顺带验证 min_len/max_len 与坏例重试三参数，过则暴露、不过则定披露口径。
3. **fireredtts3-mps**：补 inference_cfg / n_timesteps / cross_fade_ms 三个 spec（真机验证）；stop_threshold 视验证结果；切句参数与 do_clean 不声明。
4. **capability_matrix.json**：复核 voxcpm2 生成参数条的 verified 证据来源版本；为全部新增声明补证据条目。
5. **测试锁**：按 issue-38 范式补 `test_param_specs_cover_the_documented_request_surface` 的本地引擎对等锁，消除云端/本地审计标准不对称。

## 4. 真机验证清单（决策 3）

| # | 机器 | 验证项 | 判定 |
|---|---|---|---|
| V1 | Windows 3090 | voxcpm2 seed 传参是否 TypeError；`torch.manual_seed` 替代是否生效 | bug 确认/修复依据 |
| V2 | Windows 3090 | voxcpm2 min_len/max_len、retry_badcase* 直传行为与默认值 | 暴露或披露 |
| V3 | Mac (MLX) | qwen3 lang_code 10 语种逐一合成，语言确实变化 | bug 修复验收 |
| V4 | Mac (MLX) | qwen3 max_tokens 在 500 字长文本下的行为 | 暴露或披露 |
| V5 | Mac (MPS) | fireredtts3 inference_cfg / n_timesteps / cross_fade_ms 可听差异 | 暴露依据 |
| V6 | Mac (MPS) | fireredtts3 stop_threshold 截断行为 | 暴露或披露 |
| V7 | Windows 3090 | voxcpm2 denoise（需 ZipEnhancer）与离线承诺的兼容性 | denoise 翻 exposed 或维持 |

## 5. 参考

- 上游接口全参数面（VoxCPM 2.0.3 `_generate` 13 形参、FireRedTTS3 `generate` 17 形参、mlx-audio 0.5.4 qwen3_tts `generate` 16 形参）：见调研记录，均从本机 venv/patched checkout 源码直接核对。
- `docs/audit/issue-38-cloud-param-gap-audit.md`（审计范式）
- `docs/adr/0018-parameters-are-declared-by-engines.md`、`docs/adr/0008-text-normalization-layer-before-engines.md`
- audit 2026-09-18 §4（mlx-audio speed no-op 实测、FireRedTTS3 max_text_len 内部值定调）
