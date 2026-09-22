# Release 发布约定（在线更新配套）

对应 issue #67；规范见 #62 与 ADR-0020（docs/adr/0020-online-update-semi-auto-with-dual-channel.md）。
应用内更新检测以 GitHub Releases 的 `/releases/latest` 为唯一数据源，因此每次正式发版必须满足以下约定，否则更新器无法解析。

## 必须附带的资产

每个 Release **必须**同时附带以下四类文件（`latest*.yml` 由 electron-builder 自动生成，内含 sha512 校验与文件大小）：

| 资产 | 来源 | 用途 |
| --- | --- | --- |
| Windows NSIS 安装包 `*.exe` | `dist:win` 构建 | Windows 端下载安装 |
| macOS dmg `*.dmg` | `dist:mac` 构建 | macOS 端下载安装 |
| `latest.yml` | electron-builder 自动产出 | Windows 更新清单（指向 exe，含 sha512） |
| `latest-mac.yml` | electron-builder 自动产出 | macOS 更新清单（指向 dmg，含 sha512） |

electron-builder 已在 `app/electron-builder.json5` 配置（`productName: "VoiceClone"`，输出目录 `app/release/`），无需改动。

## 资产命名必须稳定

更新器的平台匹配规则为：Windows → `*.exe`，macOS → `*.dmg`，文件 URL 取自 `latest*.yml` 中的 `path`/`url` 字段。因此：

- **不要**在发布前手工重命名 exe/dmg，否则清单中的 sha512 与 URL 全部失配。
- electron-builder 默认产物名（`version` 取自 `app/package.json`）：
  - macOS：`VoiceClone-<版本>-arm64.dmg`（另有 `VoiceClone-<版本>-arm64-mac.zip`，可选附带）
  - Windows：`VoiceClone Setup <版本>.exe`
- 发布资产时直接上传构建目录 `app/release/` 下的原始文件名。

## 正式版与预发布版

- 只有**正式（非 prerelease）**Release 会触发应用内更新提醒：`/releases/latest` 接口只返回最新正式版，prerelease 不会成为 `latest`，更新器自然忽略。
- 想先小范围验证安装包，可先发 prerelease，但不要指望它触发用户端更新；验证通过后再发正式版。

## 无 Release 时的静默路径

当前仓库尚无任何 Release 时，`/releases/latest` 返回 404，更新器走"检测不可用 → 静默跳过"路径（无报错、无弹窗），属预期行为，不是故障。首次发版后该路径自动消失。

## 发版清单（checklist）

1. **同步版本号**：按既有 version-sync 约定，`app/package.json` 的 `version` 与 `sidecar/voiceclone_sidecar/_version.py` 保持一致，一并更新。
2. **构建**：在 `app/` 下运行 `npm run dist:mac`（macOS, Apple Silicon）与 `npm run dist:win`（Windows x64，可在 CI 或对应平台执行）。
3. **核对产物**：`app/release/` 下确认 `.exe`、`.dmg`、`latest.yml`、`latest-mac.yml` 均存在，且 `latest*.yml` 中的版本号/文件名与本次 tag 一致。
4. **上传**：创建 GitHub Release（tag = 版本号，如 `v1.0.1`），上传上述全部产物，保持原始文件名。
5. **发布**：确认无误后把 Release 发布为正式版（勾掉 prerelease），更新提醒即对该版本生效。
