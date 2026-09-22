# VoiceClone 声音复刻软件

一款 macOS / Windows 桌面应用（Electron + React 前端 + FastAPI Python sidecar）：用统一界面编排**云端 API** 与**本地模型**，完成声音复刻、音色设计与文字转语音。领域词汇见 [CONTEXT.md](CONTEXT.md)。

## 功能特性（v1.1.0）

- **声音复刻**：上传参考音频克隆音色；参考音频自动诊断 + 自动转写（本地或云端转写提供方）。
- **音色设计**：不依赖参考音频，用文字描述直接生成音色。
- **文字转语音**：长文本自动分段、任务队列 / 限流 / 取消，生成历史批量管理。
- **本地引擎**（按平台自动注册，能力矩阵见 docs）：
  - Qwen3-TTS（MLX，macOS Apple Silicon）
  - IndexTTS-2.5（CUDA，Windows NVIDIA；MPS，macOS）
  - VoxCPM2 / dots.tts / FireRedTTS3（CUDA / MPS 变体）
- **云端引擎**：多厂商接入（BYOK，厂商级 API Key），音色健康检查与自动重建，按能力矩阵声明式呈现参数。
- **对比与调优**：跨引擎盲听对比（LUFS 归一化）、偏好画像；文本归一化层与归一化预览。
- **数据管理**：音色包导出导入、整库备份恢复；存储层 SQLite + 一次性迁移 + 快照备份。
- **应用在线更新**（v1.1.0）：启动自动检测新版本并弹窗提醒（可跳过此版本）；设置页「关于 / 更新」内下载并引导安装；**更新渠道**三选一（自动 / GitHub 官方 / 国内镜像，镜像优先 + 官方静默兜底），下载按 sha512 清单校验。

## 下载安装（Release）

从 [GitHub Releases](https://github.com/smallmj/voice-cloning/releases/latest) 下载：

| 平台 | 文件 |
| --- | --- |
| macOS（Apple Silicon） | `VoiceClone-<版本>-arm64.dmg` 或 `.zip` |
| Windows（x64） | `VoiceClone Setup <版本>.exe` |

安装包内嵌钉版 [uv](https://docs.astral.sh/uv/)；首次启动时由 uv 在用户数据目录自动创建 Python 运行时（无需预装 Python）。发版资产与命名约定见 [docs/release/update-publishing.md](docs/release/update-publishing.md)。

安装后无需手动盯 Releases：应用会在启动时自动检测新版本并在设置页提供应用内下载；版本变更明细见 [CHANGELOG.md](CHANGELOG.md)。

### macOS：应用未签名，无法打开？

当前版本未做 Developer ID 签名与公证，首次打开会被 Gatekeeper 拦截。**在终端里运行**（打开一次即可，只需执行一次）：

```bash
xattr -rd com.apple.quarantine /Applications/VoiceClone.app
```

该命令移除下载标记（quarantine 属性），之后可正常双击打开。如果应用还没拖进「应用程序」，把命令里的路径换成 `.app` 所在位置即可。

备选方式：在 Finder 中**右键点击**应用 → 「打开」→ 再点「打开」（此方式对部分 macOS 版本可能不再生效，推荐用上面的终端命令）。也可以到「系统设置 → 隐私与安全性」底部点击「仍要打开」。

### Windows：SmartScreen 提示？

安装包未做 Authenticode 签名，SmartScreen 可能显示「Windows 已保护你的电脑」——点击「更多信息」→「仍要运行」即可。

## 目录结构

```
app/       Electron 壳 + React 界面（TypeScript, vite, vitest）
sidecar/   Python 本地服务：引擎注册表、生成契约、任务队列（uv 管理）
docs/      ADR、装机文档、能力矩阵、审计记录
```

## 环境要求

- Python ≥ 3.12 与 [uv](https://docs.astral.sh/uv/)（sidecar 及所有本地引擎都由 uv 托管）
- Node.js ≥ 20 与 npm

## 快速开始

```bash
# 1. 准备 sidecar 依赖（一次性）
cd sidecar && uv sync && cd ..

# 2. 前端依赖（一次性）
cd app && npm ci && cd ..

# 3. 开发模式：Electron 会自动从 ../sidecar 拉起 sidecar 进程
cd app && npm run dev:main
# 可选：另开终端跑 vite 热更新，dev:main 已默认指向 http://localhost:5173
# cd app && npm run dev:renderer
```

生产构建与打包：

```bash
cd app
npm run build        # vite build + esbuild main
npm run dist:mac     # macOS 安装包（自动下载钉版的 uv）
npm run dist:win     # Windows 安装包
```

## 命令集（也可用根目录 Makefile）

| 命令 | 在哪跑 | 作用 |
| --- | --- | --- |
| `make test` / `cd sidecar && uv run pytest` | repo 根 | Python 测试套件（~520 个用例，pytest-timeout 60s 兜底） |
| `make test-frontend` / `cd app && npm test` | repo 根 | 前端 vitest 测试 |
| `make typecheck` / `cd app && npm run typecheck` | repo 根 | `tsc --noEmit` |
| `make lint` | repo 根 | ruff（sidecar）+ eslint（app） |
| `make format` | repo 根 | ruff check --fix + ruff format + prettier --write |
| `cd app && npm run lint` | app | 仅 eslint |
| `cd sidecar && uv run ruff check .` | sidecar | 仅 ruff |

## CI

`.github/workflows/ci.yml` 在每次 push（main）与 PR 上运行：

1. **sidecar**（macOS runner——默认引擎注册表按平台注册，套件断言 darwin/arm64 行为）：`ruff check` + `pytest`（timeout 护栏强制开启）。
2. **frontend**（ubuntu）：`eslint` + `tsc --noEmit` + `vitest run`。

CI 失败即红；不要合入让 CI 变红的改动。

## 代码质量约定

- **ruff**（`sidecar/pyproject.toml`）：启用 BLE（blind-except）、RUF100（未使用的 noqa）等——源码里每一条 `# noqa: XXX - 原因` 都是被 linter 校验的「真压制」：规则真的触发、原因真实存在。不允许无 linter 配置的裸压制。
- **eslint**（`app/eslint.config.mjs`）：typescript-eslint recommended + react-hooks（`exhaustive-deps` 开启；目前两处显式 disable 均为有意设计并被此配置校验）。
- **prettier**（`app/.prettierrc.json`）：提供 `npm run format`；存量代码尚未整体格式化，新增/改动文件请顺手格式化。
- **pytest-timeout**：`[tool.pytest.ini_options]` 中声明 `timeout = 60`——任何测试挂死都会在 60 秒内失败，而不是卡住整个套件。

## License

见 [LICENSE](LICENSE) 与第三方清单 [NOTICE.md](NOTICE.md)。
