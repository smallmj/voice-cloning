# Issue #38：云端引擎参数缺口核对清单

> 逐引擎核对「当前声明参数 ↔ 官方 API 参数面」。依据：`docs/research/2026-09-minimax-api-and-local-tts.md`（2026-09-18 对官方 OpenAPI 逐页核实，含引用）与 `docs/research/cloud-api/2026-09-16_中国大陆云端API.md`（阿里百炼）。本轮（2026-09-21）用 exa 检索再次抽查了 MiniMax t2a_v2 与 Qwen-TTS API 的公开摘要，与上述文档一致，未发现参数面变动。
>
> 落地原则（ADR-0018）：**每个字段要么 exposed（含 applies_to / to_wire），要么以 `exposed=False + not_exposed_reason`（六因闭集）声明为数据**——不存在「靠 reviewer 记住的缺口」。契约由 `ParamSpec.__post_init__` 构造期强制；`test_param_specs_cover_the_documented_request_surface` 等测试锁住本清单。

## 1. MiniMax speech-2.8（`minimax-speech-cloud`，t2a_v2 / t2a_async_v2）

| 厂商字段 | 结论 | 备注 |
|---|---|---|
| `voice_setting.voice_id` | 已覆盖 | 复刻绑定自动注入；`system_voice` 引擎参数可显式覆盖（#27） |
| `voice_setting.speed` [0.5,2] | 已暴露 | #27 |
| `voice_setting.vol` (0,10] | **本次补齐（exposed）** | 开区间用 `min_open` 表达；默认 1 不发送 |
| `voice_setting.pitch` [-12,12] | **本次补齐（exposed）** | 整数半音；默认 0 不发送 |
| `voice_setting.emotion` | 已暴露 | speech-2.8 七情感（whisper/fluent 为 2.6 专属，不入选） |
| `voice_setting.text_normalization` | not_exposed：`unverified` | 与异步顶层 `english_normalization` 语义重叠，同步路径未实测 |
| `voice_setting.latex_read` | **本次补齐（exposed）** | 仅同步路由携带（异步 schema 未列该键，不臆造） |
| `audio_setting.sample_rate` | not_exposed：`server-injected` | 管线固定 24000 Hz WAV |
| `audio_setting.format` | not_exposed：`breaks-pipeline` | 固定 wav（峰值自检，#27） |
| `audio_setting.bitrate` | not_exposed：`breaks-pipeline` | 压缩格式专属概念 |
| `audio_setting.channel` | not_exposed：`breaks-pipeline` | ADR-0018 决策 4 将「声道」列为管线固定参数（24 kHz 单声道 WAV 峰值自检/分段拼接依赖）；code review 首稿误将其暴露，已回退 |
| `audio_setting.force_cbr` | not_exposed：`no-op` | 仅影响 mp3 CBR/VBR |
| `stream` / `stream_options` | not_exposed：`breaks-pipeline` | 管线消费整段 WAV，无流式消费方 |
| `output_format` | not_exposed：`server-injected` | 固定 `url`（默认 hex 会破坏下载路径，#27） |
| `pronunciation_dict.tone[]` | not_exposed：`unverified` | 研究文档只记载 tone[]「原文/替换」形态；代码库既有的内联 `(xing2)` 语法（`pronunciation.to_minimax`）无引用来源——code review 指出后按 ADR-0018 回退为数据声明，`pronunciation_control=False` 保持；真实合成验证 wire 形态后翻 exposed |
| `timbre_weights[]`（legacy，≤4） | not_exposed：`unverified` | 与复刻音色绑定混音的实际效果未实测 |
| `voice_modify.{pitch,intensity,timbre,sound_effects}` | not_exposed：`unverified` | 与 `voice_setting.pitch` 语义重叠且对复刻音色未实测 |
| `subtitle_enable` / `subtitle_type` | not_exposed：`no-op` | 应用只消费音频，字幕无消费方 |
| `language_boost` | 已暴露（#27） | 注意：vendor enum 有 40 项 + auto，当前声明为 10 语种安全子集；全枚举待拿到完整 vendor 页面后扩充（见「遗留」） |
| 文本 `<#x#>` 停顿标记（0.01–99.99s） | 文本级能力，非 API 参数 | 不需参数声明；使用直接写在文本里 |

复刻期（`/v1/voice_clone`）参数 `clone_prompt` / `text`（预览）/ `accuracy` / `need_noise_reduction` / `need_volume_normalization` / `aigc_watermark` 属**绑定期**输入，不属生成期 ParamSpec 面向的请求；当前绑定流程不暴露，维持现状（如需将成为绑定流程的功能工单，不在 #38 范围）。

**核对结论：MiniMax 生成期参数面已全量声明——4 项本次补齐暴露（vol/pitch/latex_read + 既有 6 项复核无缺口），11 项 not_exposed 数据化，无未声明缺口（除 language_boost 全枚举待扩充）。**

## 2. Qwen3-TTS-VC（`qwen3-tts-vc-cloud`，DashScope 多模态生成接口）

合成请求输入旋钮仅三个：`input.text`（必填）/ `input.voice`（复刻音色，server-injected）/ `input.language_type`。

| 厂商字段 | 结论 | 备注 |
|---|---|---|
| `input.language_type` | 已暴露（canonical `language`，#23/#33） | 11 个取值 = 10 语种 + Auto，与厂商枚举一致，无缺口 |
| `input.voice` | not declared | server-injected（复刻绑定），同 MiniMax voice_id 处理 |
| 情感 | **本次数据化（exposed=False，`no-op`）** | 无情感 API 参数：厂商口径是文本内嵌情感标签/情感词；声明为数据后在 UI 披露，不再是无声缺席 |
| 指令控制（音调/语速自然语言指令） | 不适用 | `qwen3-tts-instruct-flash` 专属能力，vc/vd 模型不支持（wrong-model）；无参数可暴露 |

**核对结论：qwen3-tts-vc 无未声明缺口；唯一「隐性」能力（文本内嵌情感）已数据化披露。**

## 3. Qwen3-TTS-VD（`qwen3-tts-vd-cloud`）

| 厂商字段 | 结论 | 备注 |
|---|---|---|
| `input.voice` | server-injected（设计音色绑定） | |
| `input.language_type` | **本次数据化（exposed=False，`unverified`）** | vc 模型支持；vd 同族大概率支持但未实测——验证一次（真实 Key）后翻 exposed 并在 `synthesize` 接线 |
| 其余 | 无 | 请求体只有 text/voice/language_type 三个旋钮 |

**核对结论：vd 无其他缺口；language_type 以 not_exposed 数据声明，验证路径已在 spec help 中写明。**

## 验证

- `sidecar` 全量契约测试（`make test`）通过（564 passed / 2 skipped）；
- 两轴 code review（Standards / Spec）发现并修复 2 处误声明：`channel` 违反 ADR-0018 决策 4、`pronunciation` wire 形态无引用来源——均已按 not_exposed 数据化回退；
- 新增/更新的锁定测试：`test_param_specs_cover_the_documented_request_surface`（MiniMax 全字段面）、vol/pitch/channel 接线与默认值不发送、越界值丢弃、latex_read 仅同步、发音标注改写、qwen-vc emotion 注解、qwen-vd language 注解。

## 遗留（不阻塞 #38 验收）

1. `language_boost` 全量 40 项枚举扩充——vendor 页面在本环境无法直接抓取，待拿到完整枚举后纯数据扩充。
2. qwen3-tts-vd `language_type` 真实 Key 实测后翻 exposed。
3. MiniMax `timbre_weights` / `voice_modify` 真实听感验证后决定暴露。
4. 复刻期绑定参数（accuracy / noise reduction / volume normalization / aigc_watermark）如产品需要，另开工单。
