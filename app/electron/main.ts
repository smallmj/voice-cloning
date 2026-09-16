// Electron main process: spawn the Python sidecar, read its JSON handshake
// line from stdout, and expose connection info to the renderer via IPC.
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

function sidecarDir(): string {
  // Dev: ../sidecar next to the app. Packaged builds will bundle the runtime
  // per ADR-0002; that lands in a later ticket.
  return path.resolve(app.getAppPath(), "..", "sidecar");
}

function dataDir(): string {
  const dir = path.join(app.getPath("userData"), "data", "audio");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function startSidecar(): void {
  const token = crypto.randomBytes(24).toString("base64url");
  const args = [
    "run",
    "--project",
    sidecarDir(),
    "python",
    "-m",
    "voiceclone_sidecar",
    "--port",
    "0",
    "--token",
    token,
    "--audio-dir",
    dataDir(),
  ];

  console.log("[main] starting sidecar: uv", args.join(" "));
  sidecar = spawn("uv", args, { stdio: ["ignore", "pipe", "pipe"] });

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
  sidecar.on("exit", (code) => {
    console.log(`[main] sidecar exited with code ${code}`);
    sidecarInfo = null;
  });
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
  startSidecar();
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (sidecar && sidecar.exitCode === null) {
    sidecar.kill();
  }
});
