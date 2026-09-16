# 主技术栈为 Electron + React + Python sidecar

桌面外壳用 Electron + React，推理跑在独立的 Python sidecar 进程里（由应用自带的 uv 运行时管理，见 ADR-0002），两者通过本机 HTTP + WebSocket 通信。

**为什么不是 Tauri**：Tauri 的头号卖点是体积，但在一个要分发数 GB 模型的应用里这个优势不成立——离线分发需要打 WebView2 fixedVersion，安装包要加约 180 MB，与 Electron 的差距缩到约 1.7 倍。曾被引为"直接威胁"的 Tauri 音频播放 issue #9573（无法播放 audio blob）**实际已于 2024-04-26 关闭**（见 `docs/research/CORRECTIONS.md` C-002），因此该论据不成立。真正的决定性理由是**音频波形生态**（wavesurfer.js / peaks.js）——波形、片段选择、A/B 对比试听是本产品的核心功能面，而 Electron 在这里的生态优势碾压 PySide6，Tauri 的 WebView 分裂（旧 macOS 拿到旧 WebKit，Web Audio 成熟度随之变化）则引入额外不确定性。附带理由：它是 AI 桌面应用的事实标准（可查证的 9 个中 6 个），且 ComfyUI Desktop 用 TypeScript 实现了与 ADR-0002 完全同构的 uv/venv 管理器，可直接参考。

**Considered Options**：
- **Tauri 2**：体积小、Rust 后端、已有两个 TTS 应用先例（voicebox、VoiceStudio）。但两者各有痛点——voicebox 的崩溃源于 **PyInstaller 冻结**而非 Tauri；VoiceStudio 正从 Tauri 迁往 Electron。生态知识存量差一个量级。
- **PySide6**：单一语言、无 IPC、同进程 `import torch`，架构最简单，且因本项目是 AGPL-3.0，与 PySide6 的 GPL 模式许可兼容。但 UI 精美度与波形生态明显较弱，而"精美的面板"是明确的产品要求。
