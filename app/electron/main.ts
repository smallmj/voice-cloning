// Electron main process: spawn the Python sidecar, read its JSON handshake
// line from stdout, and expose connection info to the renderer via IPC.
//
// Issue #29:
// - uv is resolved to an absolute path (bundled under Resources/uv/<platform>
//   in packaged builds); we never rely on a Finder-inherited minimal PATH.
// - Single-instance lock: a second launch focuses the existing window instead
//   of starting a second sidecar (per-user dataDir + full-memory JSON stores
//   mean two sidecars silently lose data).
import { app, BrowserWindow, ipcMain } from "electron";
import { spawn, ChildProcess } from "child_process";
import * as path from "path";
import * as fs from "fs";
import * as crypto from "crypto";

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

function notifyRenderer(channel: string, message: string): void {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send(channel, message);
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

ipcMain.handle("sidecar:info", async () => waitForSidecar());

app.whenReady().then(() => {
  if (gotLock) {
    startSidecar();
    createWindow();
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
