# 0015 — 引擎配置走注入式接缝，源解析集中，云端抽厂商中立基类

日期：2026-09-18 · 状态：已采纳 · 关联：plan §3、§8、ADR-0002、ADR-0006

## 背景

接第 2 个云端厂商（MiniMax）与后续本地引擎之前，三处**已核实**的问题必须先解决：

1. **引擎配置接缝是断的。** `registry.default_registry(output_dir=None, key_store=None)` **没有 `env` 参数**，所有引擎只按 `output_dir=` 构造，于是落回各自模块里的字面量字典，**从不与 `os.environ` 合并**。后果：
   - `docs/installation/macos-indextts-2.5-mps.md` 教的 `export VOICECLONE_INDEXTTS25_LOCAL_WEIGHTS=…` 与 plan §5 宣传的「本地已有权重可零下载复用」**在跑起来的应用里都没有效果**；
   - `VOICECLONE_RUNTIME_ROOT` 同样无效（`runtime_root()` 收到的是那个字面量字典而非 `None`）；
   - 用户自己设的 `UV_DEFAULT_INDEX` 被 `_runtime_env()` 的阿里云值**反向覆盖**（`uvman._run` 是 `os.environ.copy()` 后 `update(env)`，字面量字典赢）。
   旁证：`installer.InstallContext` 有 `env` 字段并注释 `# mirror overrides etc.`，但 `installer.py` **从不读它**——这条线本来就打算做，做了一半停了。

2. **三个下载轴全是模块级常量**，散在四个文件里：HF 兼容权重源（`indextts25.py` 三源、`qwen3_tts.py` 两源、`transcription.py` 用子进程注入 `HF_ENDPOINT`）、PyPI 索引（`indextts25.py` 阿里云；`qwen3_tts` 与 `transcription` 没有）、PyTorch CUDA wheel 源（`indextts25.py` 用精确 wheel URL，不走索引）。每个引擎还各写一份 `_runtime_env()`，内容还不一致。

3. **云端基类是"名字厂商专属、内容大半通用"。** `dashscope_base.py` 里的 `_key()` / `_http()`（可注入 httpx client，供 `MockTransport` 契约测试）/ `_vendor_error()` / `_probe_sample_rate()` 都与厂商无关；真正 DashScope 专属的只有 `BASE_URL` 与文案。更麻烦的是**共享代码里有两处厂商文案硬编码**（`main.py` 的 key-missing 分支、`dashscope_base._key()`）：一个 MiniMax 引擎走那条路径会给用户显示「需要阿里百炼 API Key」。另外，判定"云端音色已死、需重建绑定"的分类器 `_voice_missing_error`（issue #12 最关键的行为）住在**克隆**适配器里，被**音色设计**适配器跨模块导入。

## 决策

1. **引擎构造改为接收注入式 config 对象，而不是读进程环境变量。** 环境变量只作为兜底默认值的来源之一。配置由设置存储读入、经 `default_registry()` 注入每个引擎——因此设置改动可以即时生效而不必重启进程，且能做引擎级覆盖。

2. **三个轴的解析集中到一个 sources 模块**（权重源 / PyPI 索引 / CUDA 轮子源）。各引擎只声明"我要哪些文件"，不再自己拼 URL。ADR-0002 的两条硬规范（CUDA 索引用显式 index、**绝不** `--extra-index-url`）由该模块统一保证，引擎不得绕过。

3. **抽一个厂商中立的 cloud base**：key 取用、可注入的 httpx client、厂商错误格式化、采样率探测，以及**厂商标签抽象**（错误文案由此生成而不是写死）。DashScope 与 MiniMax 各只保留自己的 base URL、模型名、payload 构造与错误分类。

4. **`main.py` 中硬编码的厂商文案改为从引擎取**；`_voice_missing_error` 与 `billed_chars` 从克隆适配器搬进 cloud base，成为可复用的分类器。

5. **下载源的界面形态与源列表策略另见后续 ADR**（依赖产品决策）。本 ADR 只定接缝与解析归属。

## 后果

- 这是"第三次之前动手"的时机：plan §5 还留着第 3 个云端厂商（Fish Audio）。拖到它之后再改，就是三份拷贝加三处文案。
- 本地引擎侧的 `_runtime_env()` 副本一并消失。`VOICECLONE_*` 系列要么**真正生效**、要么从文档里删除——不允许"文档承诺了但代码不兑现"继续存在。
- 新引擎的契约测试因此可以走生产构造路径（`default_registry()`），修掉"测试证明了产品里不存在的路径"这类**假绿**。
- 三轴必须分离暴露：把 PyPI 索引与 CUDA 轮子源混进一个"镜像"开关，会复现 plan §3 记录的 torch 事故——传递依赖把 CUDA 版 torch 顶成 CPU 版，29 秒音频跑了 700 秒（VoxCPM #60）。
