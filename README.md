# VoiceClone 声音复刻软件

一款 macOS / Windows 桌面应用（Electron + React 前端 + FastAPI Python sidecar）：用统一界面编排**云端 API** 与**本地模型**，完成声音复刻、音色设计与文字转语音。领域词汇见 [CONTEXT.md](CONTEXT.md)。

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
| `make format` | repo 根 | ruff format --fix + prettier --write |
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
