// Renderer-facing contract for the online-update feature (issue #65, spec #62
// / ADR-0020). Type-only file: the logic lives in electron/updater.ts and the
// side effects in electron/main.ts; this module only pins the shapes that the
// preload bridge (electron/preload.ts) exposes and the settings page
// (ticket #66) consumes. Re-exporting the updater types keeps one source of
// truth — import from here in renderer code, never from electron directly.
import type { SidecarInfo } from "./api";
import type { ChannelMode, UpdaterResult } from "../electron/updater";

export type { ChannelMode, UpdaterResult };

/** Update preferences persisted with the sidecar settings store (#44 shape). */
export interface UpdateSettings {
  /** 更新渠道: auto (default) | official | mirror — NOT 下载源 (ADR-0016). */
  channelMode: ChannelMode;
  /** Editable 镜像前缀, e.g. the gh-proxy default. */
  mirrorPrefix: string;
  /** 跳过此版本: the persisted tag, or null when nothing is skipped. */
  skippedTag: string | null;
}

/** One event pushed from the main process during/after a download. */
export type UpdateEvent =
  | { type: "progress"; percent: number; received: number; total: number }
  | { type: "done"; warning?: "manifest-missing" }
  | { type: "failed"; message: string };

/** Snapshot returned by `updater:status`. */
export interface UpdateStatus {
  currentVersion: string;
  lastCheck: UpdaterResult | null;
  settings: UpdateSettings;
}

/** The `window.voiceclone` additions for updates (merged in src/api.ts). */
export interface UpdateBridge {
  getUpdateStatus(): Promise<UpdateStatus>;
  /**
   * Re-run the check. When `manual` is false (startup path) the persisted
   * 跳过此版本 tag suppresses an otherwise-available update; manual checks
   * ignore the skipped tag and always report the real latest version.
   */
  checkForUpdate(manual: boolean): Promise<UpdaterResult>;
  getUpdateSettings(): Promise<UpdateSettings>;
  saveUpdateSettings(settings: UpdateSettings): Promise<void>;
  /** Persist the tag as 跳过此版本 so the startup check stops alerting. */
  skipVersion(tag: string): Promise<void>;
  /** Start the in-app download; resolves immediately, progress via event. */
  startUpdateDownload(): Promise<void>;
  cancelUpdateDownload(): Promise<void>;
  onUpdateEvent(cb: (event: UpdateEvent) => void): void;
  onNewVersionAvailable(cb: (result: Extract<UpdaterResult, { status: "available" }>) => void): void;
}

declare global {
  interface Window {
    voiceclone: {
      getSidecarInfo: () => Promise<SidecarInfo | null>;
      onSidecarError: (cb: (message: string) => void) => void;
    } & UpdateBridge;
  }
}
