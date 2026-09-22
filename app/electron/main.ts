// Electron main process: spawn the Python sidecar, read its JSON handshake
// line from stdout, and expose connection info to the renderer via IPC.
//
// Issue #29:
// - uv is resolved to an absolute path (bundled under Resources/uv/<platform>
//   in packaged builds); we never rely on a Finder-inherited minimal PATH.
// - Single-instance lock: a second launch focuses the existing window instead
//   of starting a second sidecar (per-user dataDir + full-memory JSON stores
//   mean two sidecars silently lose data).
import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import { spawn, spawnSync, ChildProcess } from "child_process";
import * as path from "path";
import * as fs from "fs";
import * as crypto from "crypto";
import type { ReadableStream as NodeWebReadableStream } from "node:stream/web";
import {
  checkForUpdate,
  DEFAULT_MIRROR_PREFIX,
  GITHUB_REPO,
  resolveInstallerFileName,
  sanitizeMirrorPrefix,
  verifySha512Hex,
  type UpdatePlatform,
  type UpdaterResult,
} from "./updater";
import type { UpdateEvent, UpdateSettings } from "../src/updater-types";

// Release-page fallback opened in the browser when the in-app download or the
// sha512 verification fails (spec #62 user story 16).
const RELEASE_PAGE_URL = `https://github.com/${GITHUB_REPO}/releases/latest`;

let sidecar: ChildProcess | null = null;

interface SidecarInfo {
  baseUrl: string;
  token: string;
}

let sidecarInfo: SidecarInfo | null = null;
let pendingWaiters: ((info: SidecarInfo | null) => void)[] = [];

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  // Another instance owns the sidecar; quitting lets its focus handler run.
  app.quit();
} else {
  app.on("second-instance", () => {
    const win = BrowserWindow.getAllWindows()[0];
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });
}

function isPackaged(): boolean {
  return app.isPackaged;
}

function resourcesDir(): string {
  return isPackaged() ? process.resourcesPath! : path.resolve(app.getAppPath());
}

function sidecarDir(): string {
  // Packaged: the sidecar Python project is copied into Resources/sidecar by
  // electron-builder extraResources (ADR-0002 "内嵌运行时", issue #29).
  if (isPackaged()) {
    return path.join(process.resourcesPath!, "sidecar");
  }
  // Dev: ../sidecar next to the app.
  return path.resolve(app.getAppPath(), "..", "sidecar");
}

function dataDir(): string {
  const dir = path.join(app.getPath("userData"), "data", "audio");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function runtimeDir(): string {
  // Mutable runtime (engine venvs, uv-hosted pythons) must live in userData:
  // inside a packaged .app Resources is code-signed content, and per-user
  // state must survive updates.
  const dir = path.join(app.getPath("userData"), "runtime");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function bundledUvPath(): string | null {
  // Absolute path to the pinned uv binary fetched by scripts/fetch-uv.mjs
  // (same UV_VERSION pin as sidecar/voiceclone_sidecar/runtime/uvman.py).
  const exe = process.platform === "win32" ? "uv.exe" : "uv";
  // Layout produced by scripts/fetch-uv.mjs and wired via
  // electron-builder extraResources: Resources/uv/<platform>-<arch>/uv.
  const candidate = path.join(resourcesDir(), "uv", `${process.platform}-${process.arch}`, exe);
  try {
    if (fs.existsSync(candidate)) return candidate;
  } catch {
    // unreachable in practice; fall through
  }
  return null;
}

function resolveUv(): string {
  const override = process.env.VOICECLONE_UV_PATH;
  if (override && fs.existsSync(override)) return override;
  const bundled = bundledUvPath();
  if (bundled) return bundled;
  // Dev fallback only: in a packaged build this would repeat the minimal
  // launchd-PATH failure, so warn loudly.
  if (isPackaged()) {
    console.warn("[main] bundled uv not found; falling back to PATH 'uv'");
  }
  return "uv";
}

function sidecarPidFile(): string {
  return path.join(runtimeDir(), "sidecar.pid");
}

// Terminate the whole sidecar process tree, not just the uv wrapper: uv does
// not reliably forward signals to the python grandchild, which used to leave
// orphan sidecars behind after app quit (issue #29).
function killTree(pid: number): void {
  try {
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(pid), "/T", "/F"], { stdio: "ignore" });
    } else {
      // Negative pid signals the whole process group (requires detached).
      process.kill(-pid, "SIGTERM");
    }
  } catch (err) {
    console.warn(`[main] could not signal sidecar tree ${pid}:`, (err as Error).message);
  }
}

// Idempotent cleanup: if a previous run died without killing its sidecar,
// the pid it left behind lets us terminate the stale tree before spawning a
// new one. Only acts when the recorded pid is still alive.
function killStaleSidecar(): void {
  try {
    const raw = fs.readFileSync(sidecarPidFile(), "utf8").trim();
    const pid = Number.parseInt(raw, 10);
    if (!Number.isInteger(pid) || pid <= 1) return;
    if (process.platform === "win32") {
      // Can't signal-check on Windows; taskkill /F is a no-op on a dead pid.
      spawn("taskkill", ["/pid", String(pid), "/T", "/F"], { stdio: "ignore" });
    } else {
      process.kill(-pid, 0); // throws ESRCH if the group is gone
      killTree(pid);
      console.warn(`[main] killed stale sidecar process group ${pid} from a previous run`);
    }
  } catch {
    // No pid file or the process is gone — nothing to clean up.
  } finally {
    try {
      fs.unlinkSync(sidecarPidFile());
    } catch {
      // Already absent.
    }
  }
}

function startSidecar(): void {
  const token = crypto.randomBytes(24).toString("base64url");
  const uv = resolveUv();
  const args = [
    "run",
    "--project",
    sidecarDir(),
    "python",
    "-m",
    "voiceclone_sidecar",
    "--port",
    "0",
    "--audio-dir",
    dataDir(),
  ];

  console.log("[main] starting sidecar:", uv, args.join(" "));
  killStaleSidecar();
  // Token goes through the environment, not argv (argv is readable via ps).
  // The sidecar reuses the same bundled uv for engine venvs, and keeps all
  // mutable runtime state under userData (never inside the .app bundle).
  // detached: true gives the uv→python tree its own process group so the
  // whole tree can be terminated at quit (killing uv alone does not
  // necessarily terminate the python grandchild — observed orphans, #29).
  sidecar = spawn(uv, args, {
    stdio: ["ignore", "pipe", "pipe"],
    detached: process.platform !== "win32",
    env: {
      ...process.env,
      SIDECAR_TOKEN: token,
      VOICECLONE_UV_PATH: uv === "uv" ? process.env.VOICECLONE_UV_PATH ?? "" : uv,
      VOICECLONE_RUNTIME_ROOT: runtimeDir(),
      UV_PROJECT_ENVIRONMENT: path.join(runtimeDir(), "sidecar-venv"),
    },
  });

  if (sidecar.pid) {
    try {
      fs.writeFileSync(sidecarPidFile(), String(sidecar.pid));
    } catch {
      // Best-effort bookkeeping for stale-sidecar cleanup on next launch.
    }
  }

  let buffer = "";
  sidecar.stdout!.on("data", (chunk: Buffer) => {
    buffer += chunk.toString();
    let idx: number;
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line);
        if (msg.event === "ready") {
          sidecarInfo = { baseUrl: `http://127.0.0.1:${msg.port}`, token };
          console.log(`[main] sidecar ready on port ${msg.port}`);
          const waiters = pendingWaiters;
          pendingWaiters = [];
          waiters.forEach((w) => w(sidecarInfo));
        }
      } catch {
        console.log("[sidecar stdout]", line);
      }
    }
  });
  sidecar.stderr!.on("data", (chunk: Buffer) => console.error("[sidecar stderr]", chunk.toString()));
  sidecar.on("error", (err) => {
    // e.g. `uv` not found — must not crash the main process.
    console.error("[main] failed to start sidecar:", err.message);
    const waiters = pendingWaiters;
    pendingWaiters = [];
    waiters.forEach((w) => w(null));
    notifyRenderer("sidecar-error", `sidecar 启动失败：${err.message}`);
  });
  sidecar.on("exit", (code) => {
    console.log(`[main] sidecar exited with code ${code}`);
    sidecarInfo = null;
    notifyRenderer("sidecar-error", `sidecar 已退出（code ${code}）`);
  });
}

function notifyRenderer(channel: string, payload: unknown): void {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send(channel, payload);
  }
}

function waitForSidecar(timeoutMs = 30000): Promise<SidecarInfo | null> {
  if (sidecarInfo) return Promise.resolve(sidecarInfo);
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      pendingWaiters = pendingWaiters.filter((w) => w !== resolve);
      resolve(null);
    }, timeoutMs);
    pendingWaiters.push((info) => {
      clearTimeout(timer);
      resolve(info);
    });
  });
}

function createWindow(): void {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
    },
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    win.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
}

// ---------------------------------------------------------------------------
// Online update (issue #65, spec #62 / ADR-0020) — the side-effect shell.
// ALL logic (channel resolution, semver, asset matching, sha512) lives in
// electron/updater.ts; this block only wires IPC, persistence, download
// progress events, installer launch / dmg mount, and the browser fallback.
// ---------------------------------------------------------------------------

const UPDATER_EVENT_CHANNEL = "updater:event";
const UPDATER_NEW_VERSION_CHANNEL = "updater:new-version";

let lastUpdateCheck: UpdaterResult | null = null;
let updateSettings: UpdateSettings = {
  channelMode: "auto",
  mirrorPrefix: DEFAULT_MIRROR_PREFIX,
  skippedTag: null,
};
let activeDownload: { controller: AbortController } | null = null;

function updatePlatform(): UpdatePlatform | null {
  return process.platform === "win32" || process.platform === "darwin" ? process.platform : null;
}

/** Map the sidecar's flat settings keys onto the renderer-facing shape. */
function sanitizeUpdateSettings(raw: Record<string, unknown>): UpdateSettings {
  const mode = raw["updater_channel_mode"];
  const prefix = raw["updater_mirror_prefix"];
  const tag = raw["updater_skipped_tag"];
  return {
    channelMode: mode === "official" || mode === "mirror" ? mode : "auto",
    // Review fix (PR #63 finding 2): the prefix is spliced in front of GitHub
    // URLs, so persistence-time sanitization rejects anything that is not
    // https://host[/path] (file://, http://, whitespace, …) and falls back to
    // the default. The sidecar validates the same rule at PUT time.
    mirrorPrefix: sanitizeMirrorPrefix(prefix),
    skippedTag: typeof tag === "string" && tag !== "" ? tag : null,
  };
}

/**
 * Updater preferences persist in the sidecar's existing settings store
 * (GET/PUT /settings/ui, the same SQLite key/value block issue #44 uses) —
 * one source of truth shared with the renderer, never a second store.
 */
async function sidecarSettingsRequest(method: "GET" | "PUT", body?: unknown): Promise<Record<string, unknown> | null> {
  const info = await waitForSidecar(10_000);
  if (!info) return null;
  try {
    const res = await fetch(`${info.baseUrl}/settings/ui`, {
      method,
      headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!res.ok) return null;
    return (await res.json()) as Record<string, unknown>;
  } catch {
    return null;
  }
}

async function loadUpdateSettings(): Promise<void> {
  const raw = await sidecarSettingsRequest("GET");
  if (raw) updateSettings = sanitizeUpdateSettings(raw);
}

async function saveUpdateSettings(next: UpdateSettings): Promise<void> {
  updateSettings = next;
  await sidecarSettingsRequest("PUT", {
    updater_channel_mode: next.channelMode,
    updater_mirror_prefix: next.mirrorPrefix,
    updater_skipped_tag: next.skippedTag,
  });
}

/** Run one check. `manual` checks ignore 跳过此版本 (spec user story 9). */
async function runUpdateCheck(manual: boolean): Promise<UpdaterResult> {
  const platform = updatePlatform();
  if (!platform) {
    // No installer asset exists for this platform (e.g. linux dev builds).
    return { status: "unavailable", reason: "no-platform-asset", effectiveChannel: null, attempts: [] };
  }
  await loadUpdateSettings();
  const result = await checkForUpdate({
    currentVersion: app.getVersion(),
    channelMode: updateSettings.channelMode,
    mirrorPrefix: updateSettings.mirrorPrefix,
    // Startup checks honor the skipped tag; manual checks never do.
    skippedTag: manual ? null : updateSettings.skippedTag,
    fetch,
    platform,
  });
  lastUpdateCheck = result;
  // Only a genuinely-available result opens the dialog state; a skipped
  // update resolves as status "skipped" and stays silent (spec story 8).
  // Startup checks only: a manual 检查更新 reports its result in the card and
  // must NOT trigger the startup 弹窗 (spec stories 7/8/10, review fix #63
  // finding 4).
  if (!manual && result.status === "available") {
    notifyRenderer(UPDATER_NEW_VERSION_CHANNEL, result);
  }
  return result;
}

function updateEvent(event: UpdateEvent): void {
  notifyRenderer(UPDATER_EVENT_CHANNEL, event);
}

async function runInstaller(filePath: string): Promise<void> {
  if (process.platform === "win32") {
    // NSIS installer: detached so it outlives the app we are about to quit.
    spawn(filePath, [], { detached: true, stdio: "ignore" }).unref();
    return;
  }
  // macOS: mount the dmg and open the mounted volume in Finder, then explain
  // the drag-to-Applications step (半自动引导, ADR-0020).
  const attach = spawnSync("hdiutil", ["attach", filePath, "-nobrowse"], { encoding: "utf8" });
  const mountMatch = attach.stdout?.match(/\/Volumes\/.+$/m);
  const mountPath = mountMatch ? mountMatch[0].trim() : null;
  if (mountPath) {
    await shell.openPath(mountPath);
  }
  await dialog.showMessageBox({
    type: "info",
    title: "安装更新",
    message: "请退出应用后完成安装",
    detail:
      "安装镜像已打开。请先退出本应用，然后将应用拖入「应用程序」文件夹完成更新。" +
      (mountPath ? "" : "\n\n（未能自动打开镜像，请手动双击已下载的 .dmg 文件。）"),
  });
}

// Review fix (PR #63 finding 10): a generous overall installer-download
// timeout — a stalled mirror connection must end in failure (browser
// fallback + failed event) instead of hanging forever.
const DOWNLOAD_TIMEOUT_MS = 10 * 60 * 1000;

async function startUpdateDownload(): Promise<void> {
  if (activeDownload) return; // one download at a time; the UI serializes this
  const controller = new AbortController();
  // Occupy the download slot BEFORE the pre-download recheck (review fix PR
  // #63 finding 9): a 取消下载 clicked while the recheck runs must abort here
  // instead of being a no-op that lets the transfer start anyway.
  activeDownload = { controller };
  let filePath: string | null = null;
  try {
    let available = lastUpdateCheck?.status === "available" ? lastUpdateCheck : null;
    if (!available) {
      const result = await runUpdateCheck(true);
      if (result.status === "available") available = result;
    }
    if (controller.signal.aborted) return; // cancel arrived during the recheck
    if (!available) {
      updateEvent({ type: "failed", message: "当前没有可下载的更新" });
      return;
    }
    // Review fix (PR #63 finding 1): the installer name comes from the
    // release JSON and is joined under temp/ — basename-validate it and
    // require the platform installer extension before it can reach spawn;
    // anything else is treated as a failed download (browser fallback).
    const platform = updatePlatform();
    const fileName = platform ? resolveInstallerFileName(available.installerName, platform) : null;
    if (!fileName) throw new Error("安装包文件名非法，已拒绝下载");
    // The installer downloads through the channel that served the check
    // (review fix PR #63 finding 6: reuse the prefix captured with the check
    // result, not the possibly-changed current settings value).
    const url = `${available.resolvedPrefix}${available.installerUrl}`;
    filePath = path.join(app.getPath("temp"), fileName);
    // Overall stall timeout (review fix PR #63 finding 10).
    const timer = setTimeout(() => controller.abort(), DOWNLOAD_TIMEOUT_MS);
    let digest: string;
    try {
      const res = await fetch(url, { signal: controller.signal });
      if (!res.ok || !res.body) throw new Error(`下载失败（HTTP ${res.status}）`);
      const total = Number(res.headers.get("content-length") ?? 0);
      const out = fs.createWriteStream(filePath);
      const hash = crypto.createHash("sha512"); // incremental (review fix finding 8)
      let received = 0;
      let lastPercent = -1;
      try {
        for await (const chunk of res.body as unknown as NodeWebReadableStream<Uint8Array>) {
          received += chunk.byteLength;
          hash.update(chunk);
          // Honor write backpressure (review fix PR #63 finding 7).
          if (!out.write(chunk)) {
            await new Promise<void>((resolve, reject) => {
              out.once("drain", resolve);
              out.once("error", reject);
            });
          }
          const percent = total > 0 ? Math.min(100, Math.floor((received / total) * 100)) : 0;
          if (percent !== lastPercent) {
            lastPercent = percent;
            updateEvent({ type: "progress", percent, received, total });
          }
        }
        await new Promise<void>((resolve, reject) => out.end((err?: Error | null) => (err ? reject(err) : resolve())));
      } finally {
        out.close();
      }
      digest = hash.digest("hex");
    } finally {
      clearTimeout(timer);
    }
    // sha512 verification against the manifest value captured at check time.
    // A missing manifest is a warning, not a failure (spec: 无清单则警告但
    // 允许继续), so the installer still runs but the renderer sees the flag.
    const verdict = verifySha512Hex(digest, available.manifestSha512);
    if (!verdict.ok && !verdict.warning) {
      throw new Error("文件校验失败（sha512 不匹配），安装包可能被截断或篡改");
    }
    await runInstaller(filePath);
    updateEvent({ type: "done", ...(verdict.warning ? { warning: verdict.warning } : {}) });
  } catch (err) {
    const aborted = err instanceof Error && err.name === "AbortError";
    // Review fix (PR #63 finding 7): never leave a partial installer behind
    // on failure or cancel (on success the file is the installer we just
    // launched / mounted, so it is kept).
    if (filePath) {
      try {
        fs.rmSync(filePath, { force: true });
      } catch {
        // Best-effort cleanup only.
      }
    }
    if (!aborted) {
      // Fallback: let the user grab the installer from the release page.
      try {
        await shell.openExternal(RELEASE_PAGE_URL);
      } catch {
        // Even the browser fallback failing must not crash the app.
      }
      updateEvent({ type: "failed", message: err instanceof Error ? err.message : String(err) });
    }
  } finally {
    activeDownload = null;
  }
}

ipcMain.handle("sidecar:info", async () => waitForSidecar());

ipcMain.handle("updater:status", async () => {
  await loadUpdateSettings();
  return { currentVersion: app.getVersion(), lastCheck: lastUpdateCheck, settings: updateSettings };
});

ipcMain.handle("updater:check", async (_event, manual: boolean) => runUpdateCheck(manual === true));

ipcMain.handle("updater:settings:get", async () => {
  await loadUpdateSettings();
  return updateSettings;
});

ipcMain.handle("updater:settings:set", async (_event, settings: UpdateSettings) => {
  await saveUpdateSettings(sanitizeUpdateSettings({ ...settings }));
});

ipcMain.handle("updater:skip", async (_event, tag: string) => {
  if (typeof tag !== "string" || tag === "") return;
  await saveUpdateSettings({ ...updateSettings, skippedTag: tag });
  // The current reminder state flips to "skipped" immediately so a status
  // read after skipping does not re-offer the same version.
  if (lastUpdateCheck?.status === "available" && lastUpdateCheck.latestTag === tag) {
    lastUpdateCheck = {
      status: "skipped",
      latestTag: lastUpdateCheck.latestTag,
      effectiveChannel: lastUpdateCheck.effectiveChannel,
    };
  }
});

ipcMain.handle("updater:download", async () => {
  // Resolves immediately; progress/completion arrives via updater:event.
  void startUpdateDownload();
});

ipcMain.handle("updater:cancel-download", async () => {
  activeDownload?.controller.abort();
});

app.whenReady().then(() => {
  if (gotLock) {
    startSidecar();
    createWindow();
    // Startup update check (spec #62): async, the 5s timeout lives inside the
    // updater module, and every failure path resolves silently — the window
    // must never wait on it. The result is pushed via updater:new-version.
    void runUpdateCheck(false).catch((err) => console.warn("[main] update check failed:", (err as Error).message));
  } else {
    // Another instance owns the sidecar; quit without flashing a window.
    app.quit();
  }
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (sidecar && sidecar.exitCode === null && sidecar.pid) {
    killTree(sidecar.pid);
    sidecar.kill();
  }
});
