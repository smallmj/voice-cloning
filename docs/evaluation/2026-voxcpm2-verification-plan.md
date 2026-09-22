# VoxCPM2 新能力真机验证计划（issue #54 / 工单 #59）

状态：**草案**——待 #56/#58 合并后按最终参数面修订执行。

执行环境：
- MPS：本机 macOS（voxcpm2-mps，权重 openbmb/VoxCPM2 已由 issue #25 安装）
- CUDA：`ssh home_windows`（3090，voxcpm2-cuda；安装形态以引擎安装步骤为准）

每项验证结果回填 `sidecar/voiceclone_sidecar/data/capability_matrix.json` 对应条目的 evidence/verification，并在本文追加实测记录一节。

## 清单

| # | 项目 | 平台 | 方法 | 通过判据 |
|---|------|------|------|----------|
| V1 | control_instruction 拼接（Voice Design 语法） | MPS+CUDA | 带控制指令合成一段，日志确认 `(指令)正文` 到达引擎；无指令时不拼前缀 | 听感符合指令；无指令行为与旧版一致 |
| V2 | control_instruction 归一化安全 | MPS | 指令含数字（如 "a 30-year-old voice"），正文含数字日期 | 指令原样、正文归一化正常 |
| V3 | CMUDict 英文音素 | MPS | `world|W ER1 L D` 标注合成 | 花括号音素到达引擎、读音受控 |
| V4 | 非语言标签 | MPS | 正文含 `[laughing]`、`[sigh]` | 标签原样透传、非语言声音出现 |
| V5 | design_voice 端到端 | MPS+CUDA | /voices/design 用 voxcpm2 设计音色 → 预览样本固化为参考 → 用该音色合成 | 音色入档、合成成功、origin=designed |
| V6 | 语种抽查 | MPS+CUDA | 中文、英文、粤语、四川话各合成一段 | 声音为对应语种/方言，无崩坏例 |
| V7 | 回归：既有 8 参数 | MPS | cfg_value/inference_timesteps/min/max_len/retry_badcase/seed 各跑缺省+设定 | 与 #25 基线行为一致 |
| V8 | CUDA 全量 | CUDA | V1–V7 在 3090 重跑 | 矩阵 CUDA 条目 unverified → verified（逐格） |

## 证据口径

- 每格 evidence 附：命令、时间、听感结论（或输出文件 hash）、机器标识。
- 30 语种其余语种标「声明性支持（官方口径），未逐语种验证」。
- 不通过项：矩阵保持/降级诚实标记，并在 #59 评论说明。
