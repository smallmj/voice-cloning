# 合规护栏：同意闸门、AIGC 元数据标记与逐权重许可清单

issue #15 要求三类护栏：声音授权的知情确认、生成产物的 AI 生成标记、以及真实的第三方清单。本 ADR 记录它们的落点与取舍。

## 决定

### 1. 首次使用的声音授权确认，持久化在 sidecar 而不是渲染层

同意状态存 `<data>/app_state.json`（`GET/PUT /consent`），而不是 localStorage：数据根目录是整库备份/恢复（issue #14）的搬运单位，授权确认属于「这台机器的库的属性」，随库走；localStorage 会在恢复备份后被静默丢掉。同意记录带版本号（`voice-consent-v1`），未来文案实质变更时升版本号即可让老用户重新确认。

### 2. AIGC 标记写进音频文件本身（RIFF LIST/INFO），而不是只写在历史记录里

生成记录里加字段很容易，但产物一旦被下载、转发、二次剪辑，记录就与文件脱钩。因此每一份合成产物在落盘后由 sidecar 原位写入 RIFF `LIST`/`INFO` chunk：`ICMT`（「AI 生成音频 / AI-generated audio」+ 音色/引擎/模型/时间血缘）与 `ISFT`（软件名）。实现为纯 stdlib 的 `aigc.py`——sidecar 惯例是无重依赖。

衍生副本不得丢标记：响度归一化（对比试听、规范化副本）会重写 WAV，`normalize_wav_lufs` 因此把源文件的 INFO chunk 原样带过去。分段任务的拼接产物（`job-*.wav`）同样打标。

**范围限定**：音色设计（issue #11）的试听样本不打标——按 ADR-0010 它就是该音色的参考音频，不是面向传播的产物。

### 3. 「上传内容是否用于训练」读引擎声明，不做全局开关

能力位 `upload_used_for_training` 与 `data_usage_note` 早已随 issue #9 落地，本工单只补齐展示面：生成面板在选择引擎后明示「是/否」并原样展示厂商数据用途说明；参考音频上传区提示授权确认。真值来源是各引擎声明（`fake`、`indextts25`、`qwen3_tts` 均声明 False，云端引擎的说明文字里含「训练」字段），界面不发明事实。

### 4. 许可清单逐权重核对，NOTICE.md 为唯一落点

`NOTICE.md` 扩写为完整第三方清单：VoiceStudio（AGPL-3.0）移植文件对照表（ADR-0006 的义务）、运行时依赖、以及**逐权重**的许可证表——逐个读取 Hugging Face/ModelScope 模型卡的许可字段，而非引擎代码仓库的 LICENSE（IndexTTS-2.5 的 HF 卡标 `bilibili-model-license`，与其早期仓库 README 的差异正是「逐权重」规则要抓的坑）。核对发现：IndexTTS-2.5 管线依赖的 `amphion/MaskGCT` semantic codec 为 **CC-BY-NC-4.0（非商用）**，作为已知风险如实记录而非粉饰。

### 5. GPL 音频组件边界

后处理限定响度归一化与格式/采样率转换，均为 stdlib 实现，不引入 GPL 音频库。ffmpeg 仅作为**用户机器上的可选外部可执行文件**按需调用（非 WAV 解码），不打包、不分发、不链接；此边界在 NOTICE.md §6 正式记录。

## Testing Decisions

- `tests/test_aigc.py`：INFO chunk 的嵌入/读取/替换、非 WAV 拒绝、`wave` 模块仍可解析打标文件；归一化后标记保留与否（含未打标源不产生假标记）。
- `tests/test_compliance.py`（走真实 sidecar 进程）：consent 契约（默认未确认、PUT 校验 `acknowledged`、持久化进数据根目录的 `app_state.json`）；完整生成后产物记录 `aigc_marked` 且文件内 ICMT 含 AIGC 声明与引擎血缘。

## Out of Scope

- 音频水印（inaudiowatermark 等盲水印）：标记是元数据，可被有意剥离；盲水印是独立的大工程，不在本工单。
- 逐分段（segment）级别打标：分段任务只对最终拼接产物打标，分段音频仅存于本地历史。

## 修正（2026-09-18）

决策 1 与 Testing Decisions 中「整库备份/恢复的搬运单位是数据根目录」的表述需要收紧：索引层改用 SQLite（**ADR-0017**）之后，在 WAL 模式下直接拷贝数据目录会拿到不一致状态。备份改为**先 `VACUUM INTO` 生成单文件快照再打包**，恢复路径需能识别并接受该快照格式。同意状态的存放位置随 ADR-0017 一并迁入数据库，其「随库走、不随 localStorage」的结论不变。
