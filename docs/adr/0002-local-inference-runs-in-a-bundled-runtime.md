# 本地推理走内嵌运行时，而不是打包单文件或依赖用户环境

本地模型一律跑在应用自带的运行时里：随包分发 `uv` 二进制 + 由它托管的 Python（python-build-standalone）+ 每个引擎实例独立的 venv + 运行时安装依赖 + 一个独立的推理服务进程。用户不需要预先安装 Python、conda，也不需要自己搭服务。

**为什么不是显而易见的做法**：任何打包器都无法把 PyTorch 干净地冻成单文件——PyInstaller 打 torch 约 4GB 且跨机器报 DLL 错误，pex 实测失败，Nuitka 报错。所以「分发运行时」不是偷懒，是唯一可行路线。黄金参考实现是 ComfyUI Desktop 的 `src/virtualEnvironment.ts`。

**已知代价（必须正面解决，不是忽略）**：这套路线在国内首次安装时是全行业最大的失败源（ComfyUI 官方 discussion #390 直接劝国内用户改用整合包；半成品 `.venv` 会让重试永久失败）。因此国内镜像（`UV_PYTHON_INSTALL_MIRROR`、ModelScope、`HF_HUB_DISABLE_XET=1`）属于架构的一部分，不是后期优化。

**平台不对称（已核实）**：PyTorch/MPS 路线对 TTS 基本不可用（`nn.Conv1d` 的 65536 输出通道上限，IndexTTS 在 M3 Max 上复现）。因此 macOS 侧与 Windows 侧使用不同运行时——macOS 走 torch-free 路线（MLX / MNN / ONNX），Windows 走 PyTorch + CUDA——但两者对上层暴露同一套协议。

**运行时管理器的三条硬规范**（否则环境会被静默破坏）：

1. **CUDA 版 torch 必须钉死，不能让它被传递依赖顶掉。** `funasr` / `modelscope` 这类依赖会从默认 PyPI 索引取 CPU-only 的 `torch`，把 GPU 版本覆盖掉。已实测的后果：装好后 `torch` 变成 `2.8.0+cpu`、CUDA 不可用、**29 秒音频跑了 700 秒**，且反复重装无效。
   → `torch` / `torchaudio` 必须是 project **直接依赖**；CUDA 索引用 `[[tool.uv.index]]` + `explicit = true`，**不用 `--extra-index-url`**；**安装后断言 `torch.cuda.is_available()` 且 `torch.version.cuda` 非空，不满足则拒绝启动**。
2. **安装路径必须纯 ASCII。** 含中文或空格会让 ninja 生成乱码 `build.ninja`，导致 CUDA 内核编译失败并**静默降级**（只变慢，不报错）。
3. **每个引擎的 venv 完全隔离并钉死 torch 版本。** TorchAudio 2.9+ 起 `save()` 改走 TorchCodec，会**静默削顶饱和** WAV 文件——不报错、不告警。因此需要逐引擎钉版本，并加**输出峰值自检**。

**显存不在本决定范围内，但同一处置**：实测确认 IndexTTS 连续生成后显存从 12GB 爬到上限且不释放，`unload()` 不足够，需要「空闲超时释放 + 达到阈值重启 worker 进程」的进程级回收（见 `docs/plan.md` §3）。
