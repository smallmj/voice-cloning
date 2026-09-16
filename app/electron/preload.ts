import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("voiceclone", {
  getSidecarInfo: () => ipcRenderer.invoke("sidecar:info"),
});
