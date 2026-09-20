# 调研更正记录

本目录下的报告是不同 agent 在 2026-09-16 并行产出的。后续核实推翻或修正的结论集中记在这里。

---

## C-001 · `hf-mirror.com` 在代理 / TUN 模式下会失效（2026-09-16 实测，含一次自我更正）

**起因**：多份报告建议把 `HF_ENDPOINT` 指向 `hf-mirror.com` 以加速国内下载。一台开启 TUN 模式 VPN 的机器上实测该镜像"已死"，据此曾判定该建议失效——**该判定是错的，已撤回**。

**两次实测对照（同一台机器，改动仅为在 VPN 中为 hf-mirror.com 加入直连规则）**：

| | 加入直连规则前 | 加入直连规则后 |
|---|---|---|
| `/api/models` | 200（内容一致） | 200，final 仍为 hf-mirror.com |
| `/resolve/main/config.json` | **308 → `location: https://huggingface.co/...`** | **307 → `https://hf-mirror.com/api/resolve-cache/...`** |
| `/` | — | 200，14062 bytes |

**真实机制**：**hf-mirror 对境外出口 IP 会 308 跳回 `huggingface.co`**。因此当用户的代理 / TUN 模式把 hf-mirror 的出口变成境外时，镜像会**表现为失效**——不是挂了，是被重定向回真站。同机 DNS 解析到 `198.18.0.48`（TUN fake-IP 段），即为此现象的来源。

**对本项目的实际含义（这才是重点）**：
1. `HF_ENDPOINT=https://hf-mirror.com` 本身可用，原建议成立。
2. **但桌面软件无法控制终端用户的 VPN 状态**——一个开代理的用户会让这个镜像路径静默退化成"直连真 HF"，在国内即等于不可用，而且**表现为下载很慢或超时，不是明确报错**。
3. 因此下载层**不应只依赖 hf-mirror**：优先 **ModelScope**（实测 `https://modelscope.cn/api/v1/models/...` → 200，0.24s，且不做地域重定向），或自建 CDN / 随发布产物分发权重。
4. 排查建议：遇到"镜像失效"时先确认出口 IP 是否境外，再判断镜像本身。

**受影响文件**（均已在其尾部加入更正指针）：`desktop/2026-09-16_本地Python运行时管理.md`、`desktop/2026-09-16_技术栈与竞品.md`、`local-models/2026-09-16_中文社区共识.md`、`local-models/2026-09-16_TTS与VC模型.md`、`local-models/2026-09-16_AppleSilicon实测补充.md`。


---

## C-002 · Tauri「无法播放 audio blob」不是现存的阻塞问题（2026-09-16 用 `gh api` 核实）

**原结论**（`desktop/2026-09-16_技术栈与竞品.md` §B）：Tauri issue **#9573「Can't play blob audio」** 是对音频应用的"直接威胁"，而"生成 → 试听"正是 TTS 应用的核心路径。

**核实结果**：**该 issue 已于 2024-04-26 关闭**（`state=closed`，`closed_at=2024-04-26T15:17:56Z`，7 条评论）——开票后两天即关闭。它不是未决阻塞项。

**仍然存在的相关 issue**：**#9514「macOS audio bug with .webm」→ `state=open`**（3 条评论）。但这是 **webm 格式专属**的问题，不是通用音频播放能力缺失。

**订正后的技术栈图景**（同批核实）：
| | Tauri | Electron |
|---|---|---|
| 仓库 star | 111,100 | 123,087 |
| open issues | 1,465 | 774 |
| 最新版本 | **tauri-v3.0.0-alpha.1（2026-09-15）**；稳定线 v2.11.5（2026-07-01） | **v44.4.1（2026-09-16）** |

**结论**：技术栈选择**不应**再建立在"Tauri 播不了音频"这个已被关闭的 issue 上。真实的比较维度是：音频波形生态（wavesurfer.js / peaks.js）、参考实现与社区知识存量、打包/签名/自动更新的成熟度，以及两个已知的 Tauri + Python 先例各自的痛点来源（voicebox 的崩溃源于 **PyInstaller 冻结**而非 Tauri；VoiceStudio 正在从 Tauri 迁往 Electron）。

---

## C-003 · dots.tts 依赖 pynini 的「疑云」核实为真（issue #25，2026-09-20，一手证据）

**原结论**（`docs/plan.md` §5 决策 1，2026-09-16）：dots.tts 与 MegaTTS3 依赖 `pynini`（PyPI 上 win_amd64 wheel = 0），`uv pip install` 解析期即失败，故 Windows 侧重型引擎排除 dots.tts。但该判断当时基于社区打包的二手信息，issue #25 的验收要求在接入前用一手来源核实「官方依赖是否引入本仓库明令拒绝的文本归一化族」。

**核实结果（2026-09-20，官方源码一手证据）**：

1. `pyproject.toml`（github.com/studio-dots-ai/dots.tts @ main，tag v0.3.1 一致）`[project].dependencies` **含 `"WeTextProcessing"`，且为无版本钉扎的硬依赖**（不是 extra）：https://raw.githubusercontent.com/rednote-hilab/dots.tts/main/pyproject.toml
2. `src/dots_tts/utils/text.py` **模块顶层**直接 `from tn.chinese.normalizer import Normalizer as ZhNormalizer` / `from tn.english.normalizer import Normalizer as EnNormalizer` —— 不装 WeTextProcessing 连 `import dots_tts` 都会失败，即使推理默认 `normalize_text=False`（TN 完全关着）。
3. WeTextProcessing 拉入 **pynini + OpenFst**；pynini 至今**无官方 win_amd64 wheel**（上游 issue wenet-e2e/WeTextProcessing#287 承认 "windows is not fully supported"）。plan §5 的排除理由**被官方依赖证实，二手疑云升级为一手事实**。
4. 除 WeTextProcessing 外依赖面干净：无 pynini 直依赖、无 nemo/fun_text_processing/num2words；**无 flash-attn**（stock SDPA attention）；官方 OS 声明仅 Linux + macOS（runtime 设备自动选择为 cuda→cpu，无 MPS 路径）。

**对本仓库的实际影响（接入决策）**：本仓库的架构是**文本归一化在进引擎之前由自建归一化层完成**（ADR-0008），引擎内的 TN 是重复层。因此接入 dots.tts 时不安装 pynini/WeTextProcessing，而是沿用 FireRedTTS3（issue #26）的补丁先例：以锚点校验的最小补丁把 `text.py` 顶层的 `tn.*` 导入改为惰性可选，`normalize_text` 保持关闭并声明为不暴露（`breaks-pipeline`——引擎内 TN 与 ADR-0008 层职责重复）。补丁与证据记录于 `NOTICE.md` §3 与能力矩阵。**若未来上游把 TN 导入改为可选依赖，本补丁应收缩。**
