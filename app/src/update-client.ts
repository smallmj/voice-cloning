// Issue #66: renderer-side client for the online-update contract (spec #62,
// ADR-0020). Every `window.voiceclone` updater call is isolated in this one
// module so the exact preload export names can be reconciled here if they
// drift when #64/main lands. All calls are guarded with typeof checks: in
// `vite dev` (no preload bridge yet) and in unit tests the UI must render
// gracefully instead of crashing.

export const DEFAULT_MIRROR_PREFIX = "https://gh-proxy.com/";
export const RELEASES_PAGE_URL = "https://github.com/smallmj/voice-cloning/releases/latest";

// Merger note (PR #63): types now come from ./updater-types — the single
// source of truth that mirrors the real preload bridge (issue #65). The
// UpdaterResult discriminant is `status` (events keep `type`).

import type {
  UpdateBridge,
  UpdateEvent,
  UpdateSettings,
  UpdateStatus,
  UpdaterResult,
} from "./updater-types";

export type { UpdateEvent, UpdateSettings, UpdateStatus, UpdaterResult };
export type UpdateChannelMode = UpdateSettings["channelMode"];
export type NewVersionInfo = Extract<UpdaterResult, { status: "available" }>;

// --- guarded preload bridge -------------------------------------------

type VoicecloneUpdaterApi = Partial<UpdateBridge>;

function updaterApi(): VoicecloneUpdaterApi | null {
  if (typeof window === "undefined") return null;
  const vc = (window as { voiceclone?: unknown }).voiceclone;
  return vc && typeof vc === "object" ? (vc as VoicecloneUpdaterApi) : null;
}

export async function getUpdateStatus(): Promise<UpdateStatus | null> {
  const fn = updaterApi()?.getUpdateStatus;
  if (typeof fn !== "function") return null;
  return fn();
}

export async function checkForUpdate(manual: boolean): Promise<UpdaterResult | null> {
  const fn = updaterApi()?.checkForUpdate;
  if (typeof fn !== "function") return null;
  return fn(manual);
}

export async function saveUpdateSettings(s: UpdateSettings): Promise<void> {
  const fn = updaterApi()?.saveUpdateSettings;
  if (typeof fn !== "function") return;
  await fn(s);
}

export async function skipVersion(tag: string): Promise<void> {
  const fn = updaterApi()?.skipVersion;
  if (typeof fn !== "function") return;
  await fn(tag);
}

export async function startUpdateDownload(): Promise<void> {
  const fn = updaterApi()?.startUpdateDownload;
  if (typeof fn !== "function") return;
  await fn();
}

export async function cancelUpdateDownload(): Promise<void> {
  const fn = updaterApi()?.cancelUpdateDownload;
  if (typeof fn !== "function") return;
  await fn();
}

/** Subscribe to download events. Returns an unsubscribe no-op when the
 * preload bridge is absent (dev / tests). */
export function onUpdateEvent(cb: (e: UpdateEvent) => void): () => void {
  const fn = updaterApi()?.onUpdateEvent;
  if (typeof fn !== "function") return () => {};
  fn(cb);
  return () => {};
}

export function onNewVersionAvailable(cb: (r: NewVersionInfo) => void): () => void {
  const fn = updaterApi()?.onNewVersionAvailable;
  if (typeof fn !== "function") return () => {};
  fn(cb);
  return () => {};
}

// --- labels + pure helpers (unit-testable, no DOM) ---------------------

/** 更新渠道 three-way selector labels (CONTEXT.md terms; NOT 下载源 —
 * that is ADR-0016 model-weights vocabulary). */
export const CHANNEL_OPTIONS: { value: UpdateChannelMode; label: string }[] = [
  { value: "auto", label: "自动（默认）" },
  { value: "official", label: "GitHub 官方" },
  { value: "mirror", label: "国内镜像" },
];

/** 实际生效渠道 display: 镜像 / GitHub 官方 / 自动. */
export function effectiveChannelLabel(channel: string): string {
  if (channel === "mirror") return "镜像";
  if (channel === "official") return "GitHub 官方";
  if (channel === "auto") return "自动";
  return channel;
}

/** Release body → plain-text paragraphs (no HTML rendering: React escapes
 * text nodes, which is the point). Splits on blank lines, trims each. */
export function releaseNotesParagraphs(body: string): string[] {
  return body
    .split(/\r?\n\s*\r?\n|\r?\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);
}

/** Pure settings derivation for a channel-selector change: the component
 * persists exactly this object via saveUpdateSettings. */
export function settingsForChannelChange(
  prev: UpdateSettings,
  channelMode: UpdateChannelMode,
): UpdateSettings {
  return { ...prev, channelMode };
}

/** Pure settings derivation for a mirror-prefix edit (persisted on blur).
 * An emptied prefix falls back to the default, matching the spec's "镜像不
 * 可编辑为空" rule. */
export function settingsForMirrorChange(
  prev: UpdateSettings,
  mirrorPrefix: string,
): UpdateSettings {
  const trimmed = mirrorPrefix.trim();
  return {
    ...prev,
    mirrorPrefix: trimmed.length > 0 ? trimmed : DEFAULT_MIRROR_PREFIX,
  };
}

// --- UI state machine ---------------------------------------------------

export interface UpdateUiState {
  phase:
    | "idle"
    | "checking"
    | "available"
    | "up-to-date"
    | "skipped"
    | "unavailable"
    | "downloading"
    | "done"
    | "failed";
  result: UpdaterResult | null;
  progress: { percent: number; received: number; total: number } | null;
  /** Non-null after "done" when the manifest was missing (sha512 skipped). */
  doneWarning: string | null;
  failMessage: string | null;
}

export const initialUpdateState: UpdateUiState = {
  phase: "idle",
  result: null,
  progress: null,
  doneWarning: null,
  failMessage: null,
};

export type UpdateUiAction =
  | { type: "check-start" }
  | { type: "check-result"; result: UpdaterResult }
  | { type: "download-start" }
  | { type: "progress"; percent: number; received: number; total: number }
  | { type: "done"; warning?: "manifest-missing" }
  | { type: "failed"; message: string };

export function updateReducer(state: UpdateUiState, action: UpdateUiAction): UpdateUiState {
  switch (action.type) {
    case "check-start":
      return { ...state, phase: "checking", failMessage: null };
    case "check-result":
      return {
        ...initialUpdateState,
        phase: action.result.status,
        result: action.result,
      };
    case "download-start":
      return { ...state, phase: "downloading", progress: null };
    case "progress":
      return {
        ...state,
        phase: "downloading",
        progress: {
          percent: action.percent,
          received: action.received,
          total: action.total,
        },
      };
    case "done":
      return {
        ...state,
        phase: "done",
        doneWarning: action.warning ?? null,
      };
    case "failed":
      return { ...state, phase: "failed", failMessage: action.message };
  }
}
