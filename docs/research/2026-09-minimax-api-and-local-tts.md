# 声音复刻软件 — Web 调研报告

日期：2026-09-18 ｜ 范围：PART A = MiniMax 云 TTS API；PART B = 本地中文 TTS 候选模型
所有结论以 **2026-09-18 当日抓取的官方文档 / GitHub API / HuggingFace API 原文** 为准。无法证实的标注 **未核实**。

> 抓取方式说明：`web_fetch` 工具对 `platform.minimax.io` / `platform.minimaxi.com` 报 “resolves to a non-public IP address”（本机 DNS 走 fake-IP 代理段 198.18.0.0/15），因此改用 `curl` 直取官方 Mintlify 的 `.md` 原生源与 GitHub/HF REST API。所有引用 URL 均可公开访问。

---

# PART A — MiniMax 云 TTS API 现状

## A1. 当前模型名（确认/否证仓库文档）

官方 OpenAPI `model` 字段的完整 enum（2026-09-18 抓取）为 8 个值：

```
speech-2.8-hd, speech-2.8-turbo, speech-2.6-hd, speech-2.6-turbo,
speech-02-hd, speech-02-turbo, speech-01-hd, speech-01-turbo
```

来源：[Text to Speech (T2A) HTTP — OpenAPI T2aV2Req.model](https://platform.minimax.io/docs/api-reference/speech-t2a-http)

**结论：**
- 仓库提到的 4 个（`speech-2.6-turbo` `speech-2.6-hd` `speech-2.8-turbo` `speech-2.8-hd`）**全部存在且有效**。
- **没有任何更新的模型取代 2.8 系列。** 官方 release notes 里 “Speech-2.8” 条目时间为 **Jan. 23, 2026**，其后（Jan 2026 → Jul 31 2026）全部是 Music / 语言 / 视频模型更新，没有新的语音模型条目。来源：[Models — release notes](https://platform.minimax.io/docs/release-notes/models)
- `speech-01-*` / `speech-02-*` 仍可用但属旧代：定价页把它们单列在 “legacy” 分块中，与 2.6 同价（见 A5）。来源：[Pay as You Go](https://platform.minimax.io/docs/guides/pricing-paygo)
- 2.8 相对 2.6 的关键能力差异：**只有 `speech-2.8-hd/turbo` 支持 interjection tags**（`(laughs)` `(sighs)` `(breath)` … 共 19 个）。且 `speech-2.8-*` **不支持** `whisper` 情感；`fluent` / `whisper` 仅 `speech-2.6-hd/turbo` 支持。来源：[T2A HTTP — T2AVoiceSetting.emotion / text 字段](https://platform.minimax.io/docs/api-reference/speech-t2a-http)
- 30 语言 vs 40 语言的差异：T2A 文档写 `language_boost` enum 40 项；语音合成对外宣传 40 languages。注意 `speech-01/02` 系列**不支持** Persian / Filipino / Tamil。
- **⚠️ 值得注意**：`speech-2.8-hd` 在 Voice Clone 的 interjection 列表比 T2A 多出 `(whistles)` `(crying)` `(applause)` —— 两份文档的 tag 列表不一致，**未核实**哪一份是权威（实现时建议按 T2A 的 19 个白名单做校验）。来源对比：[voice-cloning-clone](https://platform.minimax.io/docs/api-reference/voice-cloning-clone) vs [speech-t2a-http](https://platform.minimax.io/docs/api-reference/speech-t2a-http)

## A2. HTTP API 形状（endpoint / 请求 / 响应 / 鉴权）

### 鉴权（全部接口统一）
```
Authorization: Bearer <API_key>
```
- 类型：`http` / `bearer`，`bearerFormat: JWT`。
- Key 位置：[Account Management > API Keys](https://platform.minimax.io/user-center/basic-information/interface-key)；另有 “Subscription Key” 用于 Token Plan 订阅与 Credits。
- **`GroupId` 已不再是必需字段**：在 T2A / Voice Clone / Voice Design / Get Voice / File Upload 全部 6 份 OpenAPI 中搜索 `groupid` / `group_id` **零命中**。鉴权只用 Bearer header。
  来源：[About APIs FAQ](https://platform.minimax.io/docs/faq/about-apis)（只说 API Key）、[openapi.json 各页](https://platform.minimax.io/docs/api-reference/speech-t2a-http)
  - *历史备注*：MiniMax 早期（`api.minimax.chat` 时代）要求 query 参数 `GroupId=<id>`。**该要求在当前文档中已消失**；但旧版 SDK/示例里可能仍带 —— 传了不影响还是报错，**未核实**（本地无可用 key 做实测）。

### (a) 同步 TTS
- **`POST https://api.minimax.io/v1/t2a_v2`**
- 备用地址（降低 TTFA）：`https://api-uw.minimax.io/v1/t2a_v2`
- `Content-Type: application/json`（openapi 中为 required 且 enum 仅 `application/json`）
- 请求体（必填仅 `model` + `text`）：
  | 字段 | 说明 |
  |---|---|
  | `model` | 见 A1 |
  | `text` | < 10,000 字符；> 3,000 建议开流式。支持 `<#x#>` 停顿标记（0.01–99.99 秒，最多两位小数，不可连续） |
  | `stream` | bool，默认 false |
  | `stream_options.exclude_aggregated_audio` | 末块是否含完整音频，默认 false |
  | `voice_setting` | `voice_id`(required) / `speed`[0.5,2] / `vol`(0,10] / `pitch`[-12,12] / `emotion` / `text_normalization` / `latex_read` |
  | `audio_setting` | `sample_rate` / `bitrate` / `format` / `channel` / `force_cbr` |
  | `pronunciation_dict.tone[]` | `"原文/替换"` 发音规则数组，支持拼音带声调、IPA、日语假名 |
  | `timbre_weights[]` | 音色混合，`{voice_id, weight[1,100]}`，**最多 4 个**（legacy 字段） |
  | `language_boost` | 40 项 enum + `auto` |
  | `voice_modify` | `pitch`/`intensity`/`timbre` 各 [-100,100] + `sound_effects`（`spacious_echo` `auditorium_echo` `lofi_telephone` `robotic`，**只能选一个**） |
  | `subtitle_enable` / `subtitle_type` | `sentence` / `word` / `word_streaming` |
  | `output_format` | `hex`（默认）或 `url`（**仅非流式有效；url 有效期 24 小时**）。**流式只支持 `hex`** |
- 响应：`{ data: { audio: <hex>, subtitle_file, status }, trace_id, extra_info:{...}, base_resp:{status_code,status_msg} }`；`data` **可能为 null，必须做 null check**（文档明写）；`status`: `1`=合成中, `2`=完成。
- 流式：`text/event-stream`，返回 JSON 数组 / 分块，**末块含 `extra_info`**。
- `base_resp.status_code` 关键码：`0` 成功、`1002` RPM 超限、`1004` 鉴权失败、`1039` TPM 超限、`1042` 非法字符 > 10%、`2013` 参数非法、`1000` 未知、`1001` 超时。
- 计费口径：`extra_info.usage_characters` = 本次计费字符数；`word_count` 为朗读内容字数（不含标点）。`invisible_character_ratio` > 10% 直接报错。

来源：[speech-t2a-http.md](https://platform.minimax.io/docs/api-reference/speech-t2a-http)

### (b) 异步 / 长文本 TTS（T2A Async，也叫 T2A Large v2）
- **创建：`POST https://api.minimax.io/v1/t2a_async_v2`**
- **查询：`GET https://api.minimax.io/v1/query/t2a_async_query_v2?task_id=<id>`**
  - 查询接口**限速 10 次/秒**（文档明写）。
- 上限：**单请求最多 1,000,000（1M）字符**。来源：[About APIs FAQ](https://platform.minimax.io/docs/faq/about-apis)、[Async Long TTS Guide](https://platform.minimax.io/docs/guides/speech-t2a-async)
- 输入可以是**文本文件**（先走 file upload，`purpose: "t2a_async_input"`，用 `file_id`）或直接 `text`。
- 查询响应：`{ task_id, status: processing|success|failed|expired, file_id, base_resp }`。
  - `status` 枚举值**小写**（`success`/`failed`/`expired`/`processing`），但文档 example 里写的是 `Processing`（首字母大写）—— **大小写不一致，实现时应做 case-insensitive 比较**。
- 完成后用 `file_id` 调 [File Retrieve API](https://platform.minimax.io/docs/api-reference/file-management-retrieve) 下载；**返回的下载 URL 有效期 9 小时（32,400 秒）**。
- 单文件输入产出 3 个文件：音频、句级字幕、额外 JSON。若输入是 zip 压缩包，会为每个文件生成一个文件夹。

来源：[speech-t2a-async-create](https://platform.minimax.io/docs/api-reference/speech-t2a-async-create)、[speech-t2a-async-query](https://platform.minimax.io/docs/api-reference/speech-t2a-async-query)

### (c) 声音复刻（Rapid Voice Cloning）
两步 —— 先传文件，再复刻：

1. **`POST https://api.minimax.io/v1/files/upload`**（`Content-Type: multipart/form-data`）
   - 表单字段：`purpose`（enum: `voice_clone` / `prompt_audio` / `t2a_async_input` / `video_understanding` / `video_generation_input`）+ `file`（binary）
   - 复刻场景用 `purpose=voice_clone`（格式 mp3/m4a/wav）或 `purpose=prompt_audio`（mp3/m4a/wav）
   - 响应：`{ file: {file_id, bytes, created_at, filename, purpose}, base_resp }`
   - 另有独立文档页 [Upload Audio for Voice Cloning](https://platform.minimax.io/docs/api-reference/voice-cloning-uploadcloneaudio) 与 [Upload Prompt Audio](https://platform.minimax.io/docs/api-reference/voice-cloning-uploadprompt)

2. **`POST https://api.minimax.io/v1/voice_clone`**
   - `Content-Type: application/json`
   - 必填：`file_id`(int64) + `voice_id`(string)
   - 可选：`clone_prompt{prompt_audio, prompt_text}`、`text`（预览文本 ≤1000 字符）、`model`（给预览用，`text` 存在时必填）、`language_boost`、`text_validation`(≤200 字符)、`accuracy`[0,1] 默认 0.7、`need_noise_reduction`、`need_volume_normalization`、`aigc_watermark`（在预览音频尾部加 AIGC 水印音）
   - **`voice_id` 命名规则**：长度 [8,256]、**必须以英文字母开头**、只能含字母/数字/`-`/`_`、**不能以 `-` 或 `_` 结尾**、不能与已有 ID 重复
   - 响应：`{ input_sensitive, input_sensitive_type, demo_audio, extra_info, base_resp }`
   - **重要计费口径**：`demo_audio` 仅在同时传了 `text`+`model` 时返回；`extra_info`（含 `usage_characters`）**仅在传了 `text` 时返回**，即预览合成才计费。

来源：[voice-cloning-clone.md](https://platform.minimax.io/docs/api-reference/voice-cloning-clone)、[file-management-upload.md](https://platform.minimax.io/docs/api-reference/file-management-upload)

### (d) 声音设计（Voice Design）—— **存在**
- **`POST https://api.minimax.io/v1/voice_design`**
- 必填：`prompt`（自然语言音色描述）+ `preview_text`（预览文本，≤500 字符）
- 可选：`voice_id`（自定义；不传则自动生成，形如 `ttv-voice-2025060717322425-xxxxxxxx`）
- 响应：`{ voice_id, trial_audio: <hex>, base_resp }`
- **预览计费 $30 / M 字符**（文档明写，比 T2A 贵）。
- 错误码额外含 `1027` 输出内容错误、`1008` 余额不足。

来源：[voice-design-design.md](https://platform.minimax.io/docs/api-reference/voice-design-design)

### (e) 列出 / 查询音色
- **`POST https://api.minimax.io/v1/get_voice`**（注意是 **POST** 且 **不是** GET）
- 必填：`voice_type` ∈ `system` | `voice_cloning` | `voice_generation` | `all`
- 响应：`{ system_voice: [...], voice_cloning: [...], voice_generation: [...], base_resp }`
- **关键坑**：`voice_cloning` 和 `voice_generation` **只有在音色成功用于过一次语音合成之后才能被查到**。原文：
  > “Voice Cloning voices are in an inactive state and need to be used at least once before they can be queried through this API.”
- 删除音色：**`POST /v1/voice_management/delete`**（页面 [Delete Voice](https://platform.minimax.io/docs/api-reference/voice-management-delete)）。*注：该页未逐个抓取，路径以文档索引标题为准，具体 path 字段 **未核实***。

来源：[voice-management-get.md](https://platform.minimax.io/docs/api-reference/voice-management-get)、[llms.txt 索引](https://platform.minimax.io/docs/llms.txt)

### (f) 文件上传
见 (c) 第 1 步。另外异步 TTS 的文本文件也走同一接口（`purpose=t2a_async_input`）。

### 其他语音相关 endpoint（可能有用）
- **WebSocket 同步 TTS**：`wss://api.minimax.io/ws/v1/t2a_v2`（另有双向流式版 [t2a-websocket-bidi](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket-bidi)），同样上限 10,000 字符/请求。来源：[Synchronous TTS Guide (WebSocket)](https://platform.minimax.io/docs/guides/speech-t2a-websocket)
- **ASR**：`asr-1.0`，付费专属，[speech-to-text](https://platform.minimax.io/docs/api-reference/speech-to-text)。TPM 以**音频秒数**为 token 计量。

## A3. 区域端点（对「国内免翻墙」这个差异化卖点最关键）

| 区域 | API base（servers） | 文档站 | 速率限制表 |
|---|---|---|---|
| 国际 | `https://api.minimax.io` | platform.minimax.io | 单一档位 |
| 中国大陆 | `https://api.minimax.cn` | platform.minimaxi.com（302 → platform.minimax.cn） | **分免费/充值两档** |

- 国际备用低延迟地址：`https://api-uw.minimax.io`
- **中国大陆备用地址：`https://api-bj.minimaxi.com/v1/t2a_v2`**（“备用接口地址”，域名是 `.com`，主域是 `.cn` —— 两个域名都在用）
- **确认：国内站 `api.minimax.cn` 与国内文档站 `platform.minimaxi.com` 都是独立且可用的**，且国内 rate-limits 页面明确按「免费用户 / 充值用户」分行 —— 与国际站的单档表**结构不同**。

来源：[CN 同步语音合成 HTTP](https://platform.minimaxi.com/docs/api-reference/speech-t2a-http)、[CN 速率限制](https://platform.minimaxi.com/docs/guides/rate-limits)、[国际 T2A HTTP](https://platform.minimax.io/docs/api-reference/speech-t2a-http)

### API key 能否跨区通用？
**未核实。** 我未找到任何官方文档明说「同一 key 可用于 .io 和 .cn」，也未找到明说不能。两侧都要求实名的独立账号体系（国际站主体是 **Nanonoble Pte. Ltd.（新加坡）**，见 Terms of Service 末尾；国内站要求「个人或企业认证」）。**实现建议：把区域当作账号级配置，不要假设 key 可复用。**

### GroupId 是否仍必需？
**不再必需**（见 A2 鉴权段）。当前 6 份 OpenAPI 中均无 `GroupId`，鉴权仅 Bearer header。这是相对旧版 `api.minimax.chat` 的重要简化。

## A4. 声音复刻生命周期与限制

### 音频要求（**关键数字**）
| 参数 | 值 | 来源 |
|---|---|---|
| 格式 | **mp3, m4a, wav** | [voice-cloning-clone](https://platform.minimax.io/docs/api-reference/voice-cloning-clone) |
| `file_id` 音频时长 | **最短 10 秒，最长 5 分钟** | 同上 |
| `file_id` 文件大小 | **≤ 20 MB** | 同上 |
| `clone_prompt.prompt_audio`（可选增强） | **< 8 秒**，同样 mp3/m4a/wav，**且必须配 `prompt_text` 逐字转写** | 同上 |

### 生效时间 / 可用性
- 复刻是「秒级」的（“produces a high-fidelity replica in seconds”），来源：[pricing-paygo](https://platform.minimax.io/docs/guides/pricing-paygo)
- **但复刻后的音色处于 inactive 状态**，必须先成功合成一次才：
  1. 变为永久有效；
  2. 能在 `/v1/get_voice` 里查到。
  来源：[voice-management-get](https://platform.minimax.io/docs/api-reference/voice-management-get)

### ✅ voice-id 过期规则 —— 仓库记录的「7 天未用即删除」**确认仍然有效**

**英文站原文（两处）：**
> Voice Clone 页头：**“If a cloned voice is not used within 7 days, the system will delete it.”**
> — [platform.minimax.io/docs/api-reference/voice-cloning-clone](https://platform.minimax.io/docs/api-reference/voice-cloning-clone)

> FAQ：**“The system-generated voice_id is initially in an inactive state. If not activated in time, it will automatically expire 7 days after generation. To ensure the long-term validity of your voice_id, we recommend that users synthesize audio within 7 days via the T2A v2 or T2A Large interface. This will let the system permanently save the voice_id.”**
> 并附注：**“Previewing during voice_clone does not activate the voice_id.”**
> — [platform.minimax.io/docs/faq/about-apis](https://platform.minimax.io/docs/faq/about-apis)

**国内站原文：**
> **「复刻得到的音色若 7 天内未正式调用，则系统会删除该音色。」**
> — [platform.minimaxi.com/docs/api-reference/voice-cloning-clone](https://platform.minimaxi.com/docs/api-reference/voice-cloning-clone)

**⚠️ 产品级关键结论**：`voice_clone` 里传 `text` 生成的**预览音频不算“激活”**。所以复刻流程结束后，**必须再真正调一次 T2A（哪怕合成一句话）才能把音色钉住**。如果 app 只在复刻时做预览试听、用户过几天才回来用，音色会被删。**建议在 adapter 里把「复刻成功 → 立即静默合成一句短文本激活」做成默认步骤。**

### 最大复刻音色数
按订阅套餐的 **Voice slots** 限制（不是固定全局上限）：

| 套餐 | 月费 | Audio points/月 | **Voice slots** | RPM |
|---|---|---|---|---|
| Starter | $5 | 100,000 | **10** | 10 |
| Standard | $30 | 300,000 | **100** | 50 |
| Pro | $99 | 1,100,000 | **250** | 200 |
| Scale | $249 | 3,300,000 | **500** | 500 |
| Business | $999 | 20,000,000 | **800** | 800 |
| Customer Pricing | 议价 | 不限 | 更多 | **不限** |

来源：[Audio Subscription](https://platform.minimax.io/docs/guides/pricing-speech)

*注：按量付费（pay-as-you-go）账号的 voice slot 上限，文档未给出独立数字 —— **未核实**。*

### 实名/活体要求
- **国内站**：`voice_clone` 页面顶部明写 **「调用本接口前，请先完成个人或企业认证。」**（→ 有认证门槛）来源：[platform.minimaxi.com voice-cloning-clone](https://platform.minimaxi.com/docs/api-reference/voice-cloning-clone)
- **国际站**：`voice_clone` 页面**没有**这句前置认证提示。但 ToS 账号条款写 “you must complete real-name authentication in compliance with national regulations”。错误码 `2038: No cloning permission, please check account verification status` 说明存在「克隆权限/账号验证状态」这道闸。来源：[voice-cloning-clone](https://platform.minimax.io/docs/api-reference/voice-cloning-clone)、[Terms of Service](https://platform.minimax.io/docs/guides/terms-of-service)
- **活体/朗读验证**：**未核实**是否存在独立活体验证流程。文档里最接近的机制是：
  - `text_validation`（≤200 字符）+ `accuracy`（默认 0.7）：把上传音频送 ASR，转写文本与 `text_validation` 比对，相似度低于阈值直接拒（错误码 `1043: The asr similarity check failed`）。这是**内容一致性校验**，不是活体检测。
  - ToS「User Rights and Obligations」第 1 条要求深度合成服务方「add non-intrusive identifiers … place a prominent mark」—— 这是**输出标识义务**，不是输入活体校验。
- 复刻时支持 `aigc_watermark: true` 在预览音频**尾部加 AIGC 水印音**（默认 false）。

## A5. 限速与配额

### 国际站（单档，无免费/充值区分）
| API | 模型 | **RPM** | CONN | TPM |
|---|---|---|---|---|
| T2A | speech-2.8-turbo/hd, speech-2.6-turbo/hd, speech-02-turbo/hd | **60** | — | — |
| **Voice Cloning** | — | **60** | — | — |
| **Voice Design** | — | **20** | — | — |
| Speech to Text | asr-1.0 | — | 2 | 30,000 |

来源：[国际 Rate Limits](https://platform.minimax.io/docs/guides/rate-limits)

### 中国大陆站（**分免费 / 充值两档**）
| 接口 | T2A v2 | Voice Cloning | Voice Design | Speech to Text |
|---|---|---|---|---|
| 模型 | speech-02-hd/turbo, speech-2.6-hd/turbo, speech-2.8-hd/turbo | —— | —— | asr-1.0 |
| 限制类型 | RPM | RPM | RPM | CONN / TPM |
| **免费用户** | **10** | 60 | 20 | —— |
| **充值用户** | **20** | 60 | 20 | 2 / 30,000 |

来源：[CN 速率限制](https://platform.minimaxi.com/docs/guides/rate-limits)

### ✅ 仓库记录的「T2A 充值用户仅 20 RPM」—— **在国区确认仍然有效**
国内站表格 `T2A v2` 行、`充值用户` 列 = **20**。**注意**：国际站同一接口是 **60 RPM**，国区付费用户只有国际站的 1/3。这对国内主打的 app 是硬约束。

其他要点：
- RPM 是**账号级**的，不是 key 级。文档：“your account can send up to 120 requests per minute”。
- 需要更高限额 → 邮件 `api@minimax.io`（国际）/ 走订阅套餐（Audio Subscription 最高 Business 800 RPM）。
- **并发（CONN）**：语音接口只有 ASR 给了 CONN=2；T2A / Voice Clone / Voice Design **未公布并发上限** —— 只受 RPM 约束。**未核实**是否有隐藏并发限制。
- 错误码 `1002` = RPM 超限、`1039` = TPM 超限（**这两个码要分别处理，退避策略不同**）。
- ASR 免费用户不可用。

### 定价
**按量付费（Pay as You Go），语音部分：**

| API | 模型 | 价格 | 备注 |
|---|---|---|---|
| T2A (Sync) | speech-2.8-hd | **$100 / M characters** | 实时合成，含音量/音调/语速/混合与位率/采样率选项 |
| T2A (Sync) | speech-2.8-turbo | **$60 / M characters** | 能力同 hd，优化速度与成本 |
| T2A Async | speech-2.8-hd | **$100 / M characters** | 单请求 1M 字符 |
| T2A Async | speech-2.8-turbo | **$60 / M characters** | |
| **Rapid Voice Cloning** | All Models | **$1.5 / voice** | **“Charged when the cloned voice is first used for synthesis, not at cloning time.”** 预览字符按所选预览模型费率计 |
| **Voice Design** | All Models | **$3 / voice** | 同样**首次合成时才计费**；API 内预览按 $60/M 字符 |
| T2A (legacy) | speech-2.6-turbo / speech-02-turbo | $60/M characters | |
| T2A (legacy) | speech-2.6-hd / speech-02-hd | $100/M characters | |
| Speech Recognition | Speech-to-Text | **$0.38 / hour** | 流式 + 说话人分离 + srt/vtt 导出 |

来源：[Pay as You Go](https://platform.minimax.io/docs/guides/pricing-paygo)
**⚠️ 实现要点**：复刻本身不收费，**第一次用它合成时才收 $1.5**。所以 A4 那条「复刻后立即合成一句激活」的动作**会触发 $1.5 计费** —— 产品上要把这笔钱算进「创建一个音色」的成本。

订阅制（Audio Subscription）见 A4 的套餐表，$5–$999/月，以 audio points 计量。来源：[Audio Subscription](https://platform.minimax.io/docs/guides/pricing-speech)

*注：历史/下线模型（speech-01 等）的定价另有 [History Model Pricing](https://platform.minimax.io/docs/faq/history-modelinfo) 页，本次未抓取。*

## A6. 输出音频格式与流式

### 格式（`audio_setting.format`）
enum 共 **7 个值**：
```
mp3 | pcm | flac | wav | pcmu_raw | pcmu_wav | opus
```
- `pcmu_raw` / `pcmu_wav` = **G.711 μ-law 编码，8 kHz**（前者无头原始流，后者包在 WAV 容器里）
- `opus` = Ogg/Opus
- 默认 `mp3`
- **注意文档内部矛盾**：`T2aV2Resp.extra_info.audio_format` 的 enum 只列了 `[mp3, pcm, flac]`（3 个），而请求侧的 `format` 有 7 个。**未核实**哪个是准确的——实现时不要依赖响应里的 `audio_format` 做格式判断，应以自己请求的参数为准。

来源：[speech-t2a-http — T2AAudioSetting](https://platform.minimax.io/docs/api-reference/speech-t2a-http)

### 采样率（`sample_rate`）
```
8000 | 16000 | 22050 | 24000 | 32000 | 44100
```
### 位率（`bitrate`，**仅 mp3 生效**）
```
32000 | 64000 | 128000 | 256000
```
### 声道（`channel`）
`1` = mono（默认）, `2` = stereo。
### 其他
- `force_cbr`：恒定码率，**仅在流式 + mp3 时生效**
- **`voice_modify` 音效的格式约束**：非流式支持 `mp3` / `wav` / `flac`；**流式只支持 `mp3`**。要同时用音效和流式，必须锁 mp3。

### 流式支持 —— **支持，三条路**
1. **HTTP SSE**：`stream: true` → `Content-Type: text/event-stream`，响应体为 JSON 数组/分块，每块 `{data:{audio:<hex chunk>, status:1}, trace_id, base_resp}`，末块 `status:2` + 完整 `extra_info`。**流式下 `output_format` 只能是 `hex`**（`url` 无效）。
2. **WebSocket 同步**：`wss://api.minimax.io/ws/v1/t2a_v2`，事件式（`task_start` / `task_started` / …）。来源：[speech-t2a-websocket](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket)
3. **WebSocket 双向**：支持逐字送文本，服务端缓冲成句。来源：[speech-t2a-websocket-bidi](https://platform.minimax.io/docs/api-reference/speech-t2a-websocket-bidi)

*注意：WebSocket asyncapi 规范未抓取细节，事件字段名 **未核实**。*

## A7. 许可 / ToS 限制（对 app 的「数据与训练」提示最有价值）

来源：**[Terms of Service](https://platform.minimax.io/docs/guides/terms-of-service)**（运营主体 **Nanonoble Pte. Ltd.**，注册地新加坡 152 Beach Road, #14-02 Gateway East, Singapore 189721；管辖法 **新加坡法**；争议走 **SIAC 仲裁**）

### 🔴 转售/再分发限制 —— **明确禁止独立转售**
> “You, as well as any affiliates, employees, service providers, and any third parties controlled, managed, supervised by, or otherwise under your control, may not: … **(c) Independently sublicense, resell, or distribute any or all services outside of any integrated applications**; or (d) Access services in a manner that circumvents fees or otherwise evades usage restrictions.”

**解读**：把 MiniMax 当作**集成进 app 的后端引擎**是允许的（“outside of any integrated applications” 是排除条件）；**单纯转卖 API 额度/裸接口是不允许的**。贵 app 的形态（用户界面里用 MiniMax 合成）属于 integrated application，合规。
另外：不得逆向/反编译提取源码；不得绕过计费或用量限制。

### 🔴 数据用于训练 —— **ToS 保留「用于改进算法」的权利**
> “**Our use or disclosure of Confidential Information for the purpose of improving algorithms or enhancing services does not constitute a breach of confidentiality obligations.**”

**解读**：这是 ToS 里最接近「你的数据会不会被拿去训练」的条款，方向是 **MiniMax 可以拿数据改进算法/服务**。**注意**：Confidential Information 的定义包含 “user information”（用户信息），但客户输入的合成文本是否落入该定义、以及是否有独立的「不用于训练」开关，**未核实**。
> 建议 app 的「数据与训练」提示对 MiniMax 一栏写：*官方 ToS 明示 MiniMax 可能将信息用于「改进算法或增强服务」，且该用途不构成保密义务的违反；是否提供关闭选项未在公开文档中说明。*

### 🔴 深度合成标识义务 —— **强制，直接落在 app 上**
> “The services provided by us are based on deep synthesis technology. **If you offer services based on these technologies, you must** … adopt technical measures to **add non-intrusive identifiers**, maintain log information …, and **place a prominent mark in reasonable positions or areas within generated or edited content to inform the public of the use of deep synthesis technology.** You must also fulfill **filing, modification, and deregistration requirements** according to applicable local laws.”

**解读（对产品影响最大的一条）**：因为 app 是基于 MiniMax 深度合成技术向下提供服务，**app 有义务在生成内容上加显著标识**（水印/角标）并满足备案要求。API 侧有 `aigc_watermark` 参数（仅作用于 Voice Clone 的预览音频尾部），但 ToS 要求的是**面向公众的内容标识**，需要在 app UI/导出物上自己实现。

### 其他相关条款
- **AI 输出不保证准确性**：AI Output 可能 “false, inaccurate, incomplete, or out of date”；客户对 AI Output 的全部使用（含终端用户的使用）负全责，须自行审阅验证。
- **实名认证**：账号必须完成实名认证，且作为账号归属判断依据。
- **美国区域额外限制**：若选择美国为服务区域，不得处理/存储受 **ITAR** 管制或受 **HIPAA** 约束的数据。
- **出口管制/制裁**：不得为受制裁国家/地区或名单主体使用（列举：Iran, Cuba, North Korea, Syria, Crimea/Sevastopol, DNR, LNR）。
- **保密例外**：法律强制披露不算违约（但须及时通知）。
- **责任上限**：累计责任不超过你已支付的服务费；服务期超 12 个月时以损害发生前 6 个月已实付费用为上限。
- **服务可随时调整/下线**：「We reserve the right to adjust or terminate certain or all services … at any time based on its operational needs」——对 app 的稳定性是个风险点，建议 adapter 做多引擎降级。
- MiniMax 可在营销材料中使用你的名称与 logo（可书面要求停止）。

---

# PART B — 候选本地中文 TTS 模型现状

*数据来源：GitHub REST API `/repos/*` 与 `/search/repositories`、各仓库 `main` 分支 README / requirements / pyproject 原文、HuggingFace `api/models/*`。抓取日期 2026-09-18。*
*以下 stars / pushed 均为抓取当日实时值。*

## B0. 先说结论：Apple Silicon 的现实

**坏消息**：检视的「MLX 原生」中文 TTS 实现**普遍是低星个人实验项目**，没有生产级成熟选项：

| MLX 端口 | stars | license | 最后 push | 判断 |
|---|---|---|---|---|
| [lucasnewman/f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx) | 644 | MIT | **2025-03-19** | 是唯一有星数的，但**已 18 个月未更新**，实质停更 |
| [sb1992/dots-tts-mlx](https://github.com/sb1992/dots-tts-mlx) | 21 | Apache-2.0 | 2026-06-10 | 个人端口 |
| [masanishi/voxcpm2-mlx](https://github.com/masanishi/voxcpm2-mlx) | 1 | MIT | 2026-04-20 | 玩具级 |
| [vanch007/mlx-voxcpm2](https://github.com/vanch007/mlx-voxcpm2) | 0 | 无 | 2026-07-06 | 玩具级 |
| [xocialize/mlx-voxcpm2-tts-swift](https://github.com/xocialize/mlx-voxcpm2-tts-swift) | 0 | MIT | 2026-09-12 | 玩具级 |
| [huangyuan3h/Spark-TTS-MLX](https://github.com/huangyuan3h/Spark-TTS-MLX) | 0 | MIT | 2026-08-16 | 玩具级 |
| [chenlu-hung/GPT-SoVITS-MLX](https://github.com/chenlu-hung/GPT-SoVITS-MLX) | 0 | 无 | 2026-05-16 | 玩具级 |
| [KaedeTai/sovits-mlx](https://github.com/KaedeTai/sovits-mlx) | 0 | MIT | 2026-05-25 | 玩具级 |
| [zxc524580210/mlx-fish-speech](https://github.com/zxc524580210/mlx-fish-speech) | 3 | Apache-2.0 | 2026-04-03 | 玩具级 |
| [gafiatulin/fish-speech-mlx](https://github.com/gafiatulin/fish-speech-mlx) | 0 | MIT | 2026-03-22 | 玩具级 |
| [brickheadbs/mlx-fish-speech-finetuning](https://github.com/brickheadbs/mlx-fish-speech-finetuning) | 4 | MIT | 2026-06-20 | 微调脚本 |
| [mlalma/kokoro-ios](https://github.com/mlalma/kokoro-ios) | 289 | MIT | 2026-01-10 | Kokoro 的 iOS 端，**Kokoro 中文支持弱** |
| [fikrikarim/parlor](https://github.com/fikrikarim/parlor) | 2068 | Apache-2.0 | 2026-08-03 | Kokoro/其他模型的端侧 runtime，非中文专用 |

**MLX 生态里唯一有规模的框架**：[Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) — **7,908 stars, MIT, pushed 2026-09-15**（活跃）。但它是**多模型 runtime**，不保证包含中文 zero-shot 克隆模型；本次未逐模型核对它支持哪些中文 TTS —— **未核实**。

**CosyVoice 没有任何 MLX 端口**：GitHub 搜索 `cosyvoice+mlx` 只命中一个 0 星仓库 `xwbcl123/agent-voice`，与 CosyVoice 本体无关。

**好消息**：**VoxCPM2 官方原生支持 MPS**，且已有人用 GGUF + Metal 跑通（见 B1）。所以 Apple Silicon 的可行路径是 **MPS / llama.cpp-omni Metal**，而不是 MLX。

**Windows CUDA 侧**：主流的 F5-TTS / IndexTTS / dots.tts 都是 CUDA-first，成熟度远高于 MLX 侧。

## B1. VoxCPM2

| 项 | 事实 | 来源 |
|---|---|---|
| 仓库 | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) | GitHub API |
| Stars / forks | **37,738 / 4,287**（open issues 121） | GitHub API |
| **License** | **Apache-2.0**（代码）；[pyproject.toml](https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/pyproject.toml) 亦写 `license = "Apache-2.0"` | GitHub API + pyproject |
| 权重 license | **Apache-2.0**（HF 模型卡 tag `license:apache-2.0`） | [HF API openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) |
| 创建 / 最后 push | 2025-09-16 / **2026-09-02**（活跃） | GitHub API |
| 权重位置 | **HuggingFace + ModelScope 双份**：`openbmb/VoxCPM2` / `modelscope.cn/models/OpenBMB/VoxCPM2` | [README](https://github.com/OpenBMB/VoxCPM) |
| HF 热度 | downloads **385,459**，likes **1,626**，lastModified 2026-08-18 | [HF API](https://huggingface.co/api/models/openbmb/VoxCPM2) |
| 维护方 | **OpenBMB（面壁智能）机构项目，非个人** | README |

**模型规格（官方对比表）**：
| | VoxCPM2 (🟢 Latest) | VoxCPM1.5 | VoxCPM-0.5B |
|---|---|---|---|
| Backbone 参数 | **2B** | 0.6B | 0.5B |
| 采样率 | **48 kHz** | 44.1 kHz | 16 kHz |
| languages | **30** | 2 (zh,en) | 2 |
| Cloning Mode | **Isolated Reference & Continuation** | Continuation only | Continuation only |
| Voice Design | ✅ | — | — |
| Controllable Voice Cloning | ✅ | — | — |
| SFT/LoRA | ✅ | ✅ | ✅ |
| RTF (RTX 4090) | ~0.30 | ~0.15 | ~0.17 |
| RTF in Nano-VLLM (4090) | ~0.13 | ~0.08 | ~0.10 |
| **VRAM** | **~8 GB** | ~6 GB | ~5 GB |

架构：tokenizer-free 扩散自回归；`LocEnc → TSLM → RALM → LocDiT`，在 AudioVAE V2 潜空间工作。技术报告 arXiv 2606.06928。

### 平台支持
- **CUDA**：官方要求 `Python ≥3.10 (<3.13), PyTorch ≥2.5.0, CUDA ≥12.0`。
- **✅ Apple Silicon / MPS —— 官方一等支持**。README CLI 段：
  > “Supported values are `auto`, `cpu`, `mps`, `cuda`, and `cuda:N`. **On Apple Silicon Macs, `auto` uses MPS when available.**”
- **Metal via GGUF**：官方 README 推荐第三方 [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)（MIT，264 stars，pushed 2026-09-11）做纯 C++ 端侧推理，支持 **CPU / Metal / CUDA / Vulkan**，有 VoxCPM2 GGUF 原生支持。实测数据：
  > **“RTF ~1.76 (Q8_0) on Apple M4 Pro / Metal.”**（即比实时慢 1.76 倍 —— **不能实时流式，但可离线合成**）
  GGUF 权重：[DennisHuang648/VoxCPM2-GGUF](https://huggingface.co/DennisHuang648/VoxCPM2-GGUF)
- **Apple Neural Engine**：第三方 [VoxCPMANE](https://github.com/0seba/VoxCPMANE)（未核实 star/维护度）
- 另有 [VoxCPM.cpp](https://github.com/bluryar/VoxCPM.cpp)（CPU/CUDA/Vulkan）、[audio.cpp](https://github.com/0xShug0/audio.cpp)（2,828 stars，ggml，CPU/CUDA/Vulkan/**Metal**，无 Python，pushed **2026-09-18** — 当天仍在更新）
- MLX 端口存在但都是玩具级（见 B0）。

### 安装依赖 —— **✅ 无 pynini / fairseq / flash-attn / DeepSpeed**
`pyproject.toml` dependencies 实际内容：
```
torch>=2.5.0, torchaudio>=2.5.0, torchcodec, transformers>=4.36.2,
gradio>=6,<7, wetext, modelscope>=1.22.0, datasets>=3,<4, soundfile
```
- 安装命令：`pip install voxcpm`（有正式 PyPI 包）
- **`wetext`** 是纯 Python 的文本归一化库（**不是** 需要 pynini 的 `WeTextProcessing`）→ **无 Windows 编译雷**
- `torchcodec` 是需要 FFmpeg 的原生扩展，**在 Windows/macOS 上偶有装不上的报告** —— *本次未找到 2026-09 的具体 issue 证据，标记 **未核实**，但这是一个值得在 CI 里先验证的点*
- 可选 `pip install "voxcpm[timestamps]"`（stable-ts）做时间戳
- 有 vLLM 加速路径：`uv pip install vllm==0.19.0 --torch-backend=auto` + [nanovllm-voxcpm](https://github.com/a710128/nanovllm-voxcpm)（310 stars，pushed 2026-09-02）

### 是否需要参考音频转写？—— **三种模式，转写可选**
README 列出三种克隆能力：
1. **Voice Design** — 纯自然语言描述生成全新音色，**完全不需要参考音频**：
   格式 `"(your voice description)The text to synthesize."`（描述放 `text` 开头的括号里）
2. **Controllable Cloning** — `reference_wav_path` 参考音频，**不需要转写**，还可附加风格指令（语速/情感）：
   ```python
   wav = model.generate(text="(slightly faster, cheerful tone)…", reference_wav_path="voice.wav")
   ```
3. **Ultimate Cloning** — `prompt_wav_path` + `prompt_text` **需要精确转写**，复现全部声音细节（音色/节奏/情感/风格），同 VoxCPM1.5 行为：
   ```python
   wav = model.generate(text=…, prompt_wav_path="voice.wav",
                        prompt_text="The transcript of the reference audio.",
                        reference_wav_path="voice.wav")  # reference 可选，提升相似度
   ```
   → **提示：想拿最高相似度，把同一段音频同时传给 `reference_wav_path` 和 `prompt_wav_path`。**

### 控制维度
- **发音控制（拼音）**：README 未提及拼音/polyphone 控制。**✅ 但控制指令是自然语言括号前缀**（“slightly faster, cheerful tone”）→ 这是 **emotion / 语速 / 风格控制**（通过 text prompt 而非参数）。**拼音级别的发音控制：未核实**（VoxCPM2 是 tokenizer-free、不用 G2P，可能天然不提供拼音接口）。
- **情感控制**：✅ 通过自然语言描述 / 括号指令实现。
- **方言**：README 提到方言能力（东北话/河南话/陕西话/山东话/天津话/闽南话 出现在 HF 相关描述中），但这是**第三方 HF 页面**的描述，非官方 README 原文 —— **未核实**。

## B2. LongCat-AudioDiT

| 项 | 事实 | 来源 |
|---|---|---|
| 仓库 | [meituan-longcat/LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT) | GitHub API |
| Stars / forks | **576 / 59**（open issues 16） | GitHub API |
| **License** | **MIT —— 代码与权重均是** | GitHub API + README |
| License 原文 | > “This repository, **including both the model weights and the source code**, is released under the **MIT License**. … This license does not grant any rights to use Meituan trademarks or patents.” | [README](https://github.com/meituan-longcat/LongCat-AudioDiT) |
| 创建 / 最后 push | 2026-03-30 / **2026-04-03** | GitHub API |
| ⚠️ 维护信号 | **自 2026-04-03 起 5.5 个月零提交**，open issues 16 无回应 → **实质停滞** | GitHub API |
| 维护方 | 美团 LongCat 团队（机构） | README |
| 权重 | HF **双规格**：[LongCat-AudioDiT-1B](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B)（1,173 downloads, 46 likes）、[LongCat-AudioDiT-3.5B](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B)（1,223 downloads, 77 likes），两者 lastModified 均 **2026-04-03**，tag `license:mit` | HF API |

**规格**：非自回归扩散模型，直接在 **waveform latent space** 操作（跳过 mel-spectrogram）。组件仅 **Wav-VAE + diffusion backbone**。两个变体 **1B / 3.5B**。论文 arXiv 2603.29339。

**官方自报的 SOTA 数字（Seed benchmark，README 表格）**：
- LongCat-AudioDiT-1B：ZH CER 1.18 / **ZH SIM 0.812** / EN WER 1.78 / ZH-Hard SIM 0.787
- LongCat-AudioDiT-3.5B：ZH CER 1.09 / **ZH SIM 0.818** / EN WER 1.50 / ZH-Hard SIM **0.797**
- 对比参照：**MiniMax-Speech**（闭源）ZH CER 0.99 / ZH SIM 0.799；F5-TTS ZH SIM 0.741；CosyVoice2 ZH SIM 0.748
- **⚠️ 这是模型方自报数据，未做第三方复现验证。**

### ✅ 安装依赖 —— **最干净的候选，零雷**
`requirements.txt` 全文仅 8 行：
```
transformers>=5.3.0
torch>=2.0.0
torchaudio>=2.0.0
safetensors>=0.4.0
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0
einops>=0.8.0
```
**✅ 无 pynini、无 fairseq、无 flash-attn、无 DeepSpeed、无 WeTextProcessing、无 torchcodec。** 安装命令就是 `pip install -r requirements.txt`。**这是本次调研中依赖最轻的中文 zero-shot 克隆模型。**

⚠️ **但需注意 `transformers>=5.3.0`** —— 这是一个**非常新的大版本要求**，会与其它依赖旧版 transformers 的 TTS 项目（CosyVoice、fish-speech 等）**冲突**，必须独立 venv。

### 平台支持
- **CUDA**：代码基于 PyTorch + transformers，CUDA 路径是主路径。
- **Apple Silicon / MPS**：README **完全没有提及** macOS / MPS / Metal / MLX。仓库内 **无 MLX 端口**（GitHub 搜索无结果）。**未核实** MPS 是否可跑 —— 理论上纯 PyTorch + transformers 的实现有 MPS 可行性，但**未经验证**。
- **VRAM**：README **未给出显存要求**。按参数规模推算 1B 约 2–4 GB（fp16）、3.5B 约 7–10 GB（fp16）—— **这是推算，未核实官方数字**。
- **无 GGUF / llama.cpp 路径**（对比 VoxCPM2 有多个）。

### 是否需要参考音频转写？—— **✅ 需要**
CLI 克隆必须同时给 `prompt_text` 和 `prompt_audio`：
```bash
python inference.py --text "…" \
    --prompt_text "小偷却一点也不气馁，继续在抽屉里翻找。" \
    --prompt_audio assets/prompt.wav \
    --output_audio output.wav --model_dir meituan-longcat/LongCat-AudioDiT-1B \
    --guidance_method apg
```
Python API 里文本编码是 **`prompt_text` 与 `gen_text` 拼接后送入**：
```python
inputs = tokenizer([f"{prompt_text} {gen_text}"], padding="longest", return_tensors="pt")
```
→ **参考音频转写是硬性输入，不可省。** 且 `--guidance_method apg`（adaptive projection guidance）是推荐参数。

### 控制维度
- **发音控制（拼音）**：README **无任何提及**。**未核实**（大概率不支持）。
- **情感控制**：README **无任何提及**。**未核实**（大概率不支持显式情感参数）。
- `duration` 是**显式 latent frames 参数**（示例 `duration=62`、`duration=138`）→ **时长可以人工控制**，这是它一个明确的控制维度。

### 综合判断
技术上有竞争力（MIT、权重齐、依赖干净、ZH SIM 0.818 超越 MiniMax-Speech 自报值），但 **3 个红旗**：
1. **5.5 个月无提交**，16 个 open issue 无回应 → 维护停滞
2. **576 stars** 对比 VoxCPM 的 37,738 → 社区验证严重不足
3. 无 MPS/Metal 路径，Apple Silicon 支持未知
**建议定位：Windows CUDA 侧的「质量档」候选之一，先做 PoC 验证再决定；不建议作为 Apple Silicon 主引擎。**

## B3. 其他强 zero-shot 中文克隆 TTS（含 MLX 端口排查）

### 🥇 dots.tts（**本次调研最大的新发现，强烈建议纳入**）

| 项 | 事实 |
|---|---|
| 仓库 | [studio-dots-ai/dots.tts](https://github.com/studio-dots-ai/dots.tts)（原 `rednote-hilab/dots.tts`，小红书） |
| Stars / forks | **1,337 / 135**（open issues 30） |
| **License** | **Apache-2.0**（代码 + 权重 + 微调代码） |
| 创建 / 最后 push | 2026-06-04 / **2026-08-17**（活跃） |
| 权重 | HF 多检查点：`pretrained` / `SOAR` / `MeanFlow-distilled`（`dots.tts-mf`、`dots.tts-mf-2steps`、`dots.tts-mf-2steps-stts`） |
| 规模 | **2B** fully continuous AR TTS |

**✅ 基准数据极强（官方自报，Seed-TTS-Eval）**：
- WER **0.94% / 1.30% / 6.60%**，SIM **81.0 / 77.1 / 79.5**（zh / en / zh-hard）
- MiniMax 24 语言多语基准上 **最高平均说话人相似度 83.9**
- 中文 hard 子集 **0.73% WER / 81.6 SIM**（README 表格 — 该行可能是 MF 蒸馏版）
- 与 `gpt-4o-mini-tts` 由 Gemini-2.5-Pro 盲评：SOAR 拿下最高 Syntactic Complexity 65.7%（**高于所有闭源系统**）；Pretrain 版拿下**开源系统中最高 Emotions 72.7%**

**✅ 拼音发音控制 —— 支持（方式特殊，要适配）**
> “**Force a pronunciation with Pinyin for polyphones.** Replace the character in the input text with its **tone-marked** pinyin — e.g. write `我生平不hào此道` to force `好` to be read as `hào`. **Use tone-marked pinyin only (`hǎo`, `hào`, `bā`); numbered forms like `hao4` or `ha4o` are NOT recognized.**”
→ **与 MiniMax 的 `hao4`/括号格式不同**：dots.tts 要**声调符号**，MiniMax 要**声调数字**。适配层需要做格式转换。

**转写要求**：**要求参考音频转写**。CLI `--prompt-text "The exact transcript of the reference audio."`；README 明写 “Voice cloning (**reference audio + transcript required**)”。另有 “Edit speaker guidance defaults to auto” 与 “instructions that derive an empty transcript are rejected”。
- 有一处提到 “transcripts are optional: when omitted or blank, both are derived from the …”（用于 local transcription）—— 那是**自动转写**路径，仍需先本地 ASR。**未核实**具体触发条件。

**情感控制**：✅ 盲评 Emotions 分开源第一 72.7%，且有 speaker guidance 指令机制。具体 API 参数名 **未核实**。

**⚠️ 安装/平台雷点**：
- **实测路径全部是 CUDA**：README 性能表硬件写 **“single H800, torch 2.8 + CUDA 12.8”**，部署走 SGLang-Omni（CUDA-graph backbone decode）。**Apple Silicon / MPS / Metal 完全未提及。**
- `--optimize` 用 `torch.compile`，**冷启动在 H800 上要 ~3 分钟**（走所有 DiT compile bucket）。`warmup_on_optimize=False` 可跳过。
- **显存**：峰值分配 **5.29–6.53 GB**（按 patch 数 <64 / <128），见 README 表。
- ⚠️ **`pynini` 雷（通过 WeTextProcessing）**：dots.tts 的 Windows 打包（第三方 [dots.tts-Pinokio](https://github.com/pierrunoyt/dots.tts-pinokio)）提交说明中明确：
  > “**pynini (a dependency of WeTextProcessing, used for text normalization) has no official Windows wheels**, so the installer uses a prebuilt community wheel from [billwuhao/pynini-windows-wheels](https://github.com/billwuhao/pynini-windows-wheels)”
  → **这正是仓库要规避的 pynini 雷。** 官方 `dots.tts` 仓库的 requirements 是否直接依赖 WeTextProcessing，**未核实**（我只确认了第三方 Windows 打包版遇到此问题）。**若要选它，必须先核对官方 requirements。**

**MLX 端口**：[sb1992/dots-tts-mlx](https://github.com/sb1992/dots-tts-mlx)（**21 stars**，Apache-2.0，pushed 2026-06-10）。README 自述 “A pure-MLX port of `rednote-hilab/dots.tts` — multilingual zero-shot voice-clone text-to-speech, running natively on Apple Silicon.” → **存在但个人项目、21 星、3 个 open issue，成熟度低。**

**许可证注意**：README 有 “Attribution + license” 段落（**未展开核实**具体条款）；Apache-2.0 下需保留声明。

---

### F5-TTS
| 项 | 事实 |
|---|---|
| 仓库 | [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) |
| Stars | **15,241** ｜ License **MIT** |
| 创建 / push | 2024-10-08 / **2026-07-23** |
| 转写要求 | **需要**（zero-shot，prompt audio + prompt text） |
| 发音/情感控制 | 无拼音控制；**未核实**情感参数 |
| **MLX** | ✅ [lucasnewman/f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx) **644 stars, MIT, 但 2025-03-19 后停更**（18 个月） |
| 依赖雷 | F5-TTS 历史上依赖 `pynini`/`WeTextProcessing` 做文本归一化（中文场景）—— *本次未抓 requirements 逐行确认，**未核实*** |
| 基准 | LongCat README 引用：F5-TTS ZH CER 1.56 / ZH SIM 0.741（**明显低于 LongCat/VoxCPM2/dots.tts**） |

### IndexTTS / IndexTTS2
| 项 | 事实 |
|---|---|
| 仓库 | [index-tts/index-tts](https://github.com/index-tts/index-tts)（bilibili） |
| Stars | **24,051** ｜ License **NOASSERTION**（GitHub 未能识别为标准协议，页面显示 “License: Other”） |
| 创建 / push | 2025-02-06 / **2026-08-18**（活跃，894 open issues） |
| 版本 | 最新 **IndexTTS-2.5**；IndexTTS2 主打「情感表达 + 时长控制」的自回归 zero-shot |
| ⚠️ **Windows 安装雷 —— 有实证** | 官方/社区明确承认 Windows 依赖困难。第三方打包项目原文：<br>> “for many Windows users unfamiliar with Python and complex compilation environments, running such a project is not easy. Every step, from environment configuration and installing numerous dependencies to **handling special libraries that are difficult to install directly on Windows**, can become [a problem]”<br>来源：[pyVideoTrans「Building an Out-of-the-Box Windows Package for Index-TTS」](https://en.pyvideotrans.com/blog/create-indext-tts-windows-package) |
| 情感控制 | ✅ 主打卖点（emotionally expressive） |
| 拼音控制 | **未核实** |
| MLX | 未找到 MLX 端口 |
| **License 风险** | `NOASSERTION` = 非标准/自定义协议。**必须人工读 LICENSE 原文**才能确认是否允许商用 —— 对桌面商业 app 是**法务风险点**（对比 VoxCPM/LongCat/dots.tts 的 Apache-2.0/MIT 明确许可） |

### CosyVoice（**注意：仓库已迁移**）
| 项 | 事实 |
|---|---|
| ⚠️ **仓库迁移** | `FunAudioLLM/CosyVoice` **已 301 重定向**到 **[QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice)**（repo id 823430322）。旧路径仍可用但应改。 |
| Stars / forks | **23,672 / 2,690** ｜ License **Apache-2.0** |
| 创建 / push | 2024-07-03 / **2026-05-25**（近 4 个月无提交） |
| 版本 | 模型卡显示 **Fun-CosyVoice3-0.5B**（9 语言 + 18+ 中文方言，~150ms 双流式延迟）。*该 0.5B 描述来自第三方评测页，**部分未核实*** |
| **pynini 雷 —— 确认存在** | README 明确：<br>- 功能清单含 “**WeTextProcessing support when `ttsfrd` is not available**”<br>- 安装说明：“Notice that this step is not necessary. **If you do not install `ttsfrd` package, we will use wetext by default.**”<br>→ `ttsfrd` 是阿里私有 wheel（**非公开 PyPI**），拿不到就退回 `wetext`；而 `WeTextProcessing` 依赖 `pynini`（Windows 无官方 wheel）。**这是仓库点名要规避的模型。** |
| 基准 | LongCat README 引用：CosyVoice ZH SIM 0.723、CosyVoice2 0.748（**低于 LongCat/VoxCPM2/dots.tts**） |
| MLX | ❌ **无 MLX 端口**（GitHub 搜索 `cosyvoice+mlx` 仅命中无关的 0 星仓库） |
| CUDA 部署 | 官方提供 gRPC / FastAPI runtime 与 Docker（`cosyvoice:v1.0`） |

### GPT-SoVITS
| 项 | 事实 |
|---|---|
| 仓库 | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) |
| Stars / forks | **61,877 / 6,660**（open issues **894**）｜ License **MIT** |
| 创建 / push | 2024-01-14 / **2026-08-18**（活跃） |
| 特点 | 少样本**微调**路线（“1 min voice data can also be used to train a good TTS model”），非纯 zero-shot |
| 依赖雷 | 历史上以 WebUI 打包为主；是否含 pynini **未核实** |
| MLX | 仅 0 星端口（见 B0） |
| 判断 | 许可证干净、社区最大，但**是训练/微调范式**，不是 zero-shot 克隆；集成到桌面 app 需考虑是否要做本地训练流程 |

### fish-speech
| 项 | 事实 |
|---|---|
| 仓库 | [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) |
| Stars / forks | **32,729 / 2,823**（open issues 仅 15 —— 维护响应良好） |
| License | ⚠️ **NOASSERTION**（非标准）。**法务需读原文** —— fish-speech 历史上用 CC-BY-NC-SA（**非商用**），若仍如此则**不能用于商业 app**。**未核实当前版本** |
| 创建 / push | 2023-10-10 / **2026-09-16**（非常活跃） |
| MLX | 多个 0–4 星端口（`zxc524580210/mlx-fish-speech` 3 星 Apache-2.0；`gafiatulin/fish-speech-mlx` 0 星；`brickheadbs/mlx-fish-speech-finetuning` 4 星）→ **无生产级 MLX** |

### Kokoro
| 项 | 事实 |
|---|---|
| 仓库 | [hexgrad/kokoro](https://github.com/hexgrad/kokoro) |
| Stars | **8,885** ｜ License **Apache-2.0** |
| 创建 / push | 2025-01-10 / **2025-08-06**（**已 13 个月无更新**） |
| 规模 | **82M**（Kokoro-82M）—— 极小，CPU 可跑 |
| ⚠️ 关键限制 | **不是 zero-shot 克隆模型**，不支持任意音色克隆。中文支持依赖 `Misaki` G2P（[mlalma/MisakiSwift](https://github.com/mlalma/MisakiSwift) 32 星 Apache-2.0） |
| 定位 | 适合做**轻量固定音色**的旁白，不适合「声音复刻」核心卖点 |

### Higgs（boson-ai/higgs-audio）
| 项 | 事实 |
|---|---|
| 仓库 | [boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio) |
| Stars | **8,355** ｜ License **Apache-2.0** |
| 创建 / push | 2025-07-20 / **2026-06-05**（3.5 个月无提交） |
| 定位 | “Text-audio foundation model”，零样本克隆 + 语音理解 |
| MLX / 拼音 / 情感 | **未核实** |
| 判断 | 许可证干净、有一定社区，但**中文专项能力与中文基准数据未找到** → 优先级低于 dots.tts / VoxCPM2 |

### Spark-TTS / MegaTTS3（补充）
- [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS) —— **未核实**（GitHub API 本次未返回该 repo 数据）
- [bytedance/MegaTTS3](https://github.com/bytedance/MegaTTS3) — **6,097 stars，Apache-2.0，pushed 2026-06-15**（3 个月无提交）。字节跳动出品，有 WavVAE + 扩散。中文能力设计上较强，但**未核实**其拼音/情感控制与安装依赖细节（MegaTTS3 历史上需要 `ffmpeg` 且模型较大）。
- 基准参照（LongCat README 表）：SparkTTS ZH SIM **0.672**（较差）、IndexTTS2 ZH SIM 0.765、CosyVoice3-1.5B 0.781、CosyVoice3.5 0.797、MOSS-TTS 0.788、Qwen3-TTS 0.770

### 实用工具（非模型，但相关）
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — **14,833 stars, Apache-2.0, pushed 2026-09-17**（当天仍在更新）。ONNX 端侧 ASR/TTS/声纹/VAD 全家桶。**若需在 Apple Silicon / Windows 上做无 Python 依赖的本地推理，这是最成熟的工程底座**；支持哪些中文 TTS 模型需进一步核对 —— **未核实**。

---

# 汇总：可执行建议

## PART A
1. **端点**：国区用 `https://api.minimax.cn`（备用 `api-bj.minimaxi.com`），国际用 `https://api.minimax.io`（备用低延迟 `api-uw.minimax.io`）。**把区域做成账号级配置，不假设 key 跨区通用（未核实）。**
2. **鉴权**：只发 `Authorization: Bearer <key>`，**不需要 GroupId**。
3. **模型**：用 `speech-2.8-hd`（质量）/ `speech-2.8-turbo`（成本）。旧代 `speech-02-*`/`speech-01-*` 同价但能力弱，无理由选用。
4. **🔴 复刻必做动作**：`/v1/files/upload(purpose=voice_clone)` → `/v1/voice_clone` → **立即调一次 T2A 真正合成**（否则 7 天后音色被删且 `get_voice` 查不到）。这一步会触发 **$1.5 的 first-use 计费**，要计入产品成本。
5. **音频校验**：mp3/m4a/wav，**10 秒 – 5 分钟，≤20 MB**；`voice_id` **必须字母开头、8–256 字符、不能以 `-`/`_` 结尾**。这些要在 UI 层做前置校验，避免把 400 抛给用户。
6. **限速**：国区付费 T2A **仅 20 RPM**（国际 60）—— 这是国内产品的硬瓶颈，adapter 必须实现 RPM 令牌桶 + `1002`/`1039` 分开退避。需要更高并发就引导用户买 Audio Subscription（Standard 50 RPM / Pro 200 RPM）。
7. **格式**：默认 mp3/32kHz；要流式就用 `stream:true` + `format:mp3` + `output_format:hex`（流式下 `url` 和音效非 mp3 都不可用）。
8. **「数据与训练」提示文案**：ToS 明示 MiniMax 可将信息用于「improving algorithms or enhancing services」且不构成保密违约 → 提示应如实写明「可能用于改进模型/服务，官方未公开关闭选项」。
9. **🔴 合规必做**：ToS 强制要求深度合成内容加**显著标识**（面向公众）+ 非侵入式标识 + 日志留存 + 备案。这是 app 侧的责任，**不是**调个 API 参数就能满足的。另外**不得独立转售 API**（集成进 app 允许）。

## PART B
| 模型 | License | Apple Silicon | Win CUDA | 依赖雷 | 需转写 | 拼音 | 情感 | 维护 | 建议 |
|---|---|---|---|---|---|---|---|---|---|
| **VoxCPM2** | **Apache-2.0**（码+权重） | ✅ **MPS 官方支持**；Metal 经 llama.cpp-omni（M4 Pro RTF 1.76） | ✅ CUDA≥12 | ✅ 干净（wetext，非 pynini） | 可选（3 档模式） | 未核实 | ✅ 自然语言指令 | **活跃**（2026-09-02，37.7k★） | **🥇 首选，双平台唯一都有路** |
| **dots.tts** | **Apache-2.0** | ⚠️ 未提及；MLX 端口仅 21★ | ✅ H800 实测 | ⚠️ 第三方打包遇到 pynini | **需要** | ✅ 声调符号 | ✅ 盲评开源第一 | 活跃（2026-08-17，1.3k★） | **🥈 质量最优候选**（基准最强），但要先验依赖+Apple 支持 |
| **LongCat-AudioDiT** | **MIT**（码+权重） | ❌ 未提及 MPS | ✅ | ✅ **最干净（8 行依赖）** | **需要** | 未核实(大概率无) | 未核实(大概率无) | ⚠️ **停滞 5.5 月**（576★） | 🥉 Win CUDA 质量档，先 PoC |
| IndexTTS2/2.5 | ⚠️ **NOASSERTION** | ❌ | ⚠️ Windows 装配难（有实证） | 存在难装库 | 需核实 | 未核实 | ✅ 主打 | 活跃（24k★） | 法务先读 LICENSE |
| CosyVoice(3) | Apache-2.0 | ❌ **无 MLX** | ✅ | 🔴 **pynini/WeTextProcessing + 私有 ttsfrd** | 需要 | 未核实 | 未核实 | 4 月无提交（23.7k★） | **按仓库要求排除** |
| F5-TTS | MIT | ⚠️ MLX 端口已停更 18 月 | ✅ | ⚠️ 疑 pynini（未核实） | **需要** | 无 | 未核实 | 活跃（15.2k★） | 基准偏低，作后备 |
| GPT-SoVITS | MIT | ❌ | ✅ | 未核实 | 微调范式 | 未核实 | 未核实 | 活跃（61.9k★） | 非 zero-shot，另议 |
| fish-speech | ⚠️ NOASSERTION | ❌ | ✅ | 未核实 | 是 | 未核实 | 未核实 | 活跃（32.7k★） | **许可或为 NC，法务风险** |
| Kokoro | Apache-2.0 | ✅ mlx 生态成熟 | ✅ | 轻 | ❌ 非克隆 | ❌ | ❌ | **停更 13 月** | 仅适合轻量固定音色 |
| Higgs | Apache-2.0 | 未核实 | 未核实 | 未核实 | 未核实 | 未核实 | 未核实 | 3.5 月无提交 | 中文数据缺失，低优先 |

**本地引擎选型结论**：
- **Apple Silicon 的可行路径是 MPS / GGUF-Metal，不是 MLX。** MLX 中文 TTS 生态目前只有低星个人实验（唯一有星数的 f5-tts-mlx 已停更 18 个月）。
- **VoxCPM2 是唯一「官方明确支持 Apple Silicon（MPS）+ CUDA + 依赖干净 + Apache-2.0 + 活跃维护」四项全中的候选**，且它的三种克隆模式（Voice Design / Controllable / Ultimate）天然匹配 app 的能力分级 —— **且 Voice Design 模式不需要参考音频**，与 A2(d) 的 MiniMax Voice Design 可对齐 UI。
- **dots.tts 基准数据最强、且原生支持拼音发音控制**，但要让工程先验证两件事：(1) 官方 requirements 是否引入 pynini；(2) 是否存在可用的 Apple Silicon 路径。MLX 端口只有 21 星，不能算生产可用。
- **LongCat-AudioDiT 依赖最干净（8 行、零雷）但已停滞 5.5 个月**，且无 Apple Silicon 路径 → 只建议在 Windows CUDA 侧做质量档 PoC。

---

## 未核实清单（需后续人工确认）
1. 同一个 MiniMax API key 能否跨 `.io` / `.cn` 两区使用。
2. `Delete Voice` API 的确切 path（本次只确认页面存在，未读 OpenAPI）。
3. WebSocket（asyncapi）的事件字段名细节。
4. `api.minimax.io` 是否为「香港/新加坡」节点、`.cn` 是否为「北京」节点 —— 只有 `api-uw`（国际）与 `api-bj`（国内）的备用地址暗示了地理分布。
5. T2A 响应 `extra_info.audio_format` 只列 3 格式 vs 请求侧 7 格式，哪个准确。
6. `speech-2.8-hd` 的 interjection tag 白名单到底是 T2A 的 19 个还是 Voice Clone 的 22 个。
7. MiniMax 是否提供「不将客户数据用于训练」的关闭开关。
8. 按量付费账号的 voice slot 上限。
9. T2A / Voice Clone / Voice Design 的隐藏并发（CONN）上限。
10. LongCat-AudioDiT 的官方显存要求（README 未给）与 MPS 可行性。
11. dots.tts 官方 requirements 是否含 WeTextProcessing/pynini（只确认了第三方 Windows 打包版遇到）。
12. dots.tts 的「Attribution + license」附加条款具体内容。
13. index-tts 的 `NOASSERTION` 许可证原文（是否允许商用）。
14. fish-speech 当前许可证是否仍为 CC-BY-NC-SA（非商用）。
15. VoxCPM2 的方言支持与拼音发音控制能力。
16. `torchcodec` 在 Windows / macOS 上的实际安装成功率。
17. `Blaizzy/mlx-audio` 具体支持哪些中文 zero-shot 克隆模型。
18. `sherpa-onnx` 支持哪些中文 TTS 模型（含克隆）。
19. MegaTTS3 的拼音/情感控制与安装依赖。
20. Higgs-audio 的中文能力与 MLX 支持。
