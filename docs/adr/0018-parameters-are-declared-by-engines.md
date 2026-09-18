# 0018 — 参数由引擎声明，规范层与引擎专属层分离，且「不支持」必须是数据

日期：2026-09-18 · 状态：已采纳 · 关联：ADR-0008、plan §1②、§2 #15

## 背景

界面只渲染引擎声明的 `param_specs()`（`capabilities.py` 明写），机制是对的——但**几乎没人用**：六个引擎里只有两个声明了任何参数（阿里云端一个 `language_type`、假引擎一个 `fake_mode`）。两个 IndexTTS 引擎声明了 `pronunciation_control=True`，而**模型确实支持**发音控制（`<汉字|PINYIN>` 尖括号标注、声调数字大写，源码 `apply_pronunciation_annotations()`），**却没有任何 `param_specs()`** —— 于是用户没有任何入口去用它。

八引擎参数调研（2026-09-18）给出两个决定性结论：

**(a) 30+ 个参数是条件性或无条件失效的。** 典型两例：mlx-audio 的 `speed` 被接受、被打印（`Speed: {speed}x`）、**代码里从不使用**，而**我们正在用它**；FireRedTTS3 的 `max_text_len` 声明了但正文从不引用。

**(b) 共享控制集而不做逐引擎适配器，产出的是静默错误的音频，而不是报错。**

| 概念 | 不可等价的具体表现 |
|---|---|
| 语速 | **IndexTTS 的 `duration_factor` 是时长倍率，与所有引擎的 speed 相反**——"更快"会变慢 |
| 重复惩罚 | IndexTTS 默认 **10.0**，mlx-audio 默认 **1.05** |
| 音高 | **四种互不兼容的单位**：半音 [-12,12]、[-100,100]、**比值** [0.5,2.0]、`shift the pitch by N step(s)`；**MiniMax 一个请求体里就有两个不同单位的 `pitch`** |
| 情感 | MiniMax 9 值（缺 `melancholic`、多 `fluent`/`whisper`）、dots.tts 8 值带 level、IndexTTS 是 8 维向量 + alpha + 独立音频/文本提示 |
| 语种 | 六套方案，且有一个引擎根本没有这个参数 |

## 决策

1. **两层结构，位置固定**：**规范层**（跨引擎语义一致的概念，供盲听对比与偏好画像取得可比性）+ **引擎专属层**（位置固定、**默认折叠**的一个区）。规范层**只收录值域语义真正一致**的概念，且**每个规范参数必须配逐引擎适配器**（含单位换算与反向映射，例如 `duration_factor = 1/speed`）。引擎专属参数不发明规范槽位——凭空造槽位只会产出上面那类静默错误。

2. **`ParamSpec` 扩展为可表达真实参数空间**，至少包含：`bool` 控件（否则 20+ 个开关只能被迫渲染成假的 `select:["on","off"]`）；带 min/max/step 且区分整数与浮点的 `number`；**开区间**（MiniMax `vol` 是 `(0,10]`）；**动态上界**（IndexTTS 的 `max_mel_tokens` 上界来自检查点）；`textarea` 与长度上限；**嵌套分组与线上路径**（`voice_setting`/`audio_setting`/`voice_modify`）；**对象数组**（`timbre_weights[{voice_id,weight}]` ≤4、`pronunciation_dict.tone[]`、`emo_vector[8]` 且**顺序敏感**）；**用户值 → 线上载荷的映射**（FireRedTTS3 要发 `adjust the speed to 0.8x`、dots.tts 要发 `<rate, factor=1.19>`、VoxCPM2 要发 `(…)text`——"用户看到的 ≠ 引擎收到的"必须可表达）；`kind:"output"`（只读展示，如 FireRedTTS3 的 CoT `gen_text`）；以及 `unit`（`x`/`s`/`ms`/`%`/`Hz`/`kbps`/`semitone`——没有 unit，MiniMax 那两个 `pitch` 在界面上无法区分）。

3. **「不支持」必须是数据，不是评审习惯。** 每个参数声明 `appliesTo {engine, model, mode}` 与 `ignoredWhen [predicate]`；凡不暴露者必须带 `exposed: false` + `notExposedReason`，取值限定为 `no-op | server-injected | wrong-mode | paid-tier | breaks-pipeline | unverified`。界面据此**隐藏或禁用并说明原因**，而不是静默丢弃。上面 (a) 类的 30+ 个参数就此变成一条数据，而不是每次靠人记得。

4. **三类参数在任何方案下都不暴露**：
   - 会破坏本应用管线的——输出格式 / 采样率 / 声道（峰值自检与分段拼接依赖 WAV，拼接层规定格式不一致是硬错误）；
   - 服务端注入的——`ref_audio`、`voice_id`、`target_model`、`action`；
   - 与 ADR-0008 冲突的——FireRedTTS3 的 `do_tn` / `use_wetext` 自带文本归一化，暴露即等于允许用户双重归一化或**绕过必经归一化层**。

5. **能力矩阵必须覆盖被暴露的参数。** 一个参数出现在界面上，其能力矩阵条目不得缺失——pronunciation 就是三者不一致的现成反例（能力声明为真、矩阵无证据、界面无入口）。

6. **生成参数按引擎记忆上次使用的值**并持久化在设置存储（随整库备份走，与 ADR-0012 对同意状态、ADR-0014 对主题的处理一致）。

## 后果

- 「引擎不支持就不显示」从此是数据驱动，不再依赖人工评审记得。
- 新增引擎的成本里多了一项硬义务：逐参数声明 `appliesTo`/`ignoredWhen`/`notExposedReason`，否则界面会撒谎。
- **规范层会很薄**（语速、语种、种子等少数几项），绝大多数参数落在引擎专属层。这是预期结果，不是失败——调研已经证明强行拉宽规范层会产出静默错误的音频。
- 发音控制是这套机制的第一个真实用例，且三种语法互不兼容（IndexTTS `<行|XING2>` 大写声调数字 / MiniMax `(chu3)(li3)` 小写声调数字 / dots.tts `hào` 声调符号），正好验证"规范输入 + 逐引擎适配器"是唯一可行形态。
