import { contextBridge, ipcRenderer } from "electron";
import type { IpcRendererEvent } from "electron";
import type { UpdateBridge, UpdateEvent, UpdateSettings, UpdaterResult } from "../src/updater-types";

// Issue #65: the updater bridge. Every member is a thin ipcRenderer wrapper —
// ALL update logic lives in electron/updater.ts via main.ts; the renderer
// (ticket #66) only sees these typed calls and events.
const updaterBridge: UpdateBridge = {
  getUpdateStatus: () => ipcRenderer.invoke("updater:status"),
  checkForUpdate: (manual: boolean) => ipcRenderer.invoke("updater:check", manual),
  getUpdateSettings: () => ipcRenderer.invoke("updater:settings:get"),
  saveUpdateSettings: (settings: UpdateSettings) => ipcRenderer.invoke("updater:settings:set", settings),
  skipVersion: (tag: string) => ipcRenderer.invoke("updater:skip", tag),
  startUpdateDownload: () => ipcRenderer.invoke("updater:download"),
  cancelUpdateDownload: () => ipcRenderer.invoke("updater:cancel-download"),
  onUpdateEvent: (cb: (event: UpdateEvent) => void) => {
    const handler = (_event: IpcRendererEvent, payload: UpdateEvent) => cb(payload);
    ipcRenderer.on("updater:event", handler);
    // Review fix (PR #63 finding 3): expose removal so renderer components
    // can unsubscribe on unmount (listener stacking on remount otherwise).
    return () => {
      ipcRenderer.removeListener("updater:event", handler);
    };
  },
  onNewVersionAvailable: (cb: (result: Extract<UpdaterResult, { status: "available" }>) => void) => {
    const handler = (_event: IpcRendererEvent, payload: Extract<UpdaterResult, { status: "available" }>) => cb(payload);
    ipcRenderer.on("updater:new-version", handler);
    return () => {
      ipcRenderer.removeListener("updater:new-version", handler);
    };
  },
};

contextBridge.exposeInMainWorld("voiceclone", {
  getSidecarInfo: () => ipcRenderer.invoke("sidecar:info"),
  onSidecarError: (cb: (message: string) => void) => {
    ipcRenderer.on("sidecar-error", (_event, message: string) => cb(message));
  },
  ...updaterBridge,
});
