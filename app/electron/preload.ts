import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("voiceclone", {
  getSidecarInfo: () => ipcRenderer.invoke("sidecar:info"),
  onSidecarError: (cb: (message: string) => void) => {
    ipcRenderer.on("sidecar-error", (_event, message: string) => cb(message));
  },
});
