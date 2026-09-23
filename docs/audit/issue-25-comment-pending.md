# issue #25 关闭评论草稿（2026-09-20 生成；api.github.com 当时不可达，待网络恢复后 `gh issue comment 25` 发布）

实现完成（commits 734e4fe → c88e3f2），验收逐条对照：

- ✅ VoxCPM2 三种克隆模式：reference-only / prompt_wav+prompt_text 成对（Hi-Fi）/ 两者同给（默认路径）；本机 MPS 实测两种模式均出声。
- ✅ dots.tts 归一化族核实：官方硬依赖 WeTextProcessing（pynini 族），plan §5 疑云核实为真 → docs/research/CORRECTIONS.md C-003；接入不装 pynini，锚点校验补丁改惰性导入，normalize_text 声明 breaks-pipeline（ADR-0008）。
- ✅ 注册 + 平台门控：voxcpm2-mps / fireredtts3-mps（darwin+arm64），voxcpm2-cuda / dots-tts-cuda（win32）。
- ✅ 能力矩阵 4 个 engine_id 逐格带验证等级；集合断言更新。
- ✅ 参数如实：appliesTo 全覆盖；breaks-pipeline / unverified / wrong-mode / no-op 均以数据声明；新增 voxcpm {ni3} 发音语法。
- ✅ macOS 17 项回归：voxcpm2-mps 17/17（RTF≈0.75）、fireredtts3-mps 17/17，全管线（POST /regression/run）；docs/evaluation/2026-09-20_issue25_接入验证记录.md。
- ✅ FireRedTTS3 补丁表唯一真源入引擎包；fp32 实验性；学术研究声明评审在 ADR-0019/NOTICE。

测试：36 条新引擎测试 + 安装器重试修复回归；全量 504 passed / 2 skipped。

遗留：Windows 3090 真机装机与回归；FireRedTTS3 bf16 实测；dots.tts 听感判定。
