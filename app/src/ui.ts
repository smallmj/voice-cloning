// Pure helpers behind the issue #21 refactor. Kept free of React/DOM so the
// vitest net can test the cross-section behaviours and API-shape parsing
// without a renderer.

import type { EngineInfo, GenerationRecord, JobStatus, LogEvent, LogLevel } from "./api";
import type { MatrixEngine } from "./capability-matrix";
import { CAP_LABELS } from "./labels";

// --- five sections + left navigation (ADR-0013) ----------------------------

export const SECTION_IDS = ["generate", "engines", "voices", "history", "settings"] as const;
export type SectionId = (typeof SECTION_IDS)[number];

export const SECTION_LABELS: Record<SectionId, string> = {
  generate: "生成",
  engines: "引擎",
  voices: "音色库",
  history: "历史",
  settings: "设置",
};

export function isSectionId(value: string | null | undefined): value is SectionId {
  return !!value && (SECTION_IDS as readonly string[]).includes(value);
}

// --- generate section sub-tabs (issue #43, ADR-0013 2026 修订) --------------

/** 「单条生成」「对比盲听」是两个独立功能（非同一流程两段）：各自拥有自己的
 * 布局与状态，切换不回填文本/音色/参数。 */
export const GENERATE_TAB_IDS = ["single", "compare"] as const;
export type GenerateTabId = (typeof GENERATE_TAB_IDS)[number];

export const GENERATE_TAB_LABELS: Record<GenerateTabId, string> = {
  single: "单条生成",
  compare: "对比盲听",
};

/** Persisted/unknown values fall back to the primary tab instead of a blank
 * pane — same contract as isSectionId. */
export function isGenerateTabId(value: string | null | undefined): value is GenerateTabId {
  return !!value && (GENERATE_TAB_IDS as readonly string[]).includes(value);
}

// --- theme (ADR-0014) -------------------------------------------------------

export type ThemePref = "system" | "dark" | "light";

export function isThemePref(value: unknown): value is ThemePref {
  return value === "system" || value === "dark" || value === "light";
}

/** Which token set the document should use: explicit choices win, `system`
 * follows the OS. */
export function resolveTheme(pref: ThemePref, systemPrefersDark: boolean): "dark" | "light" {
  if (pref === "dark") return "dark";
  if (pref === "light") return "light";
  return systemPrefersDark ? "dark" : "light";
}

// --- cross-section: history → generate rerun (ADR-0013 §5) ------------------

export interface RerunState {
  engineId: string;
  text: string;
  voiceId: string | null;
  /** Everything the record stored minus server-side paths the sidecar
   * re-injects itself (ref_audio). */
  params: Record<string, unknown> | null;
  hint: string | null;
}

export function buildRerunState(
  record: GenerationRecord,
  voices: { id: string }[],
): RerunState {
  const params = { ...(record.params ?? {}) };
  delete params.ref_audio;
  const state: RerunState = {
    engineId: record.engine_id,
    text: record.text,
    voiceId: record.voice_id && voices.some((v) => v.id === record.voice_id)
      ? record.voice_id
      : null,
    params: Object.keys(params).length > 0 ? params : null,
    hint: null,
  };
  if (record.voice_id && !state.voiceId) {
    state.hint = "原音色已删除，已预填文本与引擎；请重新选择音色后再生成。";
  }
  return state;
}

// --- shared formatting --------------------------------------------------------

export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  return seconds < 1 ? `${Math.round(seconds * 1000)}ms` : `${seconds.toFixed(2)}s`;
}

export function fmtCost(cost: number | null | undefined): string {
  if (cost == null) return "—";
  return cost === 0 ? "本地（免费）" : `¥${cost.toFixed(4)}`;
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return iso.replace("T", " ").replace("Z", " UTC");
}

// --- cross-section: job settle → history refresh (ADR-0013 §5) ---------------

/** A long-text job counts as "settled" (its outcome should refresh the
 * history list) when it leaves queued/running for a terminal state. */
export function jobSettled(prev: JobStatus | undefined, next: JobStatus): boolean {
  return (
    (prev === "queued" || prev === "running") &&
    next !== prev &&
    next !== "queued"
  );
}

// --- log stream --------------------------------------------------------------

export const LOG_BUFFER_LIMIT = 500;

export function appendLog(prev: LogEvent[], event: LogEvent): LogEvent[] {
  return [...prev.slice(-(LOG_BUFFER_LIMIT - 1)), event];
}

/** The log WebSocket delivers untrusted text; a malformed frame must be
 * dropped, never crash the stream. A missing/unknown level reads as "info"
 * (frames from a pre-#36 sidecar carry none). */
export function parseLogEvent(raw: string): LogEvent | null {
  try {
    const j = JSON.parse(raw) as Partial<LogEvent> | null;
    if (j && j.type === "log" && typeof j.generation_id === "string" && typeof j.message === "string") {
      return {
        type: "log",
        generation_id: j.generation_id,
        message: j.message,
        ts: typeof j.ts === "number" ? j.ts : 0,
        level: isLogLevel(j.level) ? j.level : "info",
      };
    }
    return null;
  } catch {
    return null;
  }
}

export function isLogLevel(value: unknown): value is LogLevel {
  return value === "info" || value === "warn" || value === "error";
}

export const LOG_LEVELS: readonly LogLevel[] = ["info", "warn", "error"];

export const LOG_LEVEL_LABELS: Record<LogLevel, string> = {
  info: "信息",
  warn: "警告",
  error: "错误",
};

export interface LogFilter {
  /** Empty set = every level passes. */
  levels: ReadonlySet<LogLevel>;
  /** Case-insensitive substring match on the message. */
  keyword: string;
}

/** Issue #36: pure log filtering for the sidebar's log pane — level toggle
 * plus keyword substring, both applied client-side over the buffered stream. */
export function filterLogs(logs: LogEvent[], filter: LogFilter): LogEvent[] {
  const kw = filter.keyword.trim().toLowerCase();
  return logs.filter((l) => {
    if (filter.levels.size > 0 && !filter.levels.has(l.level)) return false;
    if (kw !== "" && !l.message.toLowerCase().includes(kw)) return false;
    return true;
  });
}

// --- issue #36: shared right sidebar geometry ---------------------------------

export const SIDEBAR_MIN_WIDTH = 260;
export const SIDEBAR_MAX_WIDTH = 640;
export const SIDEBAR_DEFAULT_WIDTH = 360;

/** Drag clamp — mirrors the server-side clamp on /settings/ui so a stored
 * corrupt value can never pin the divider off-screen. */
export function clampSidebarWidth(width: number): number {
  if (Number.isNaN(width)) return SIDEBAR_DEFAULT_WIDTH;
  return Math.round(
    Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, width)),
  );
}

// --- issue #23 / ADR-0018: two-layer parameter surface -----------------------

import type { ParamSpecInfo } from "./api";

export interface ParamLayers {
  canonical: ParamSpecInfo[];
  engine: ParamSpecInfo[];
  hidden: ParamSpecInfo[];
}

/** Split an engine's declared specs into the canonical layer (primary
 * surface), the engine-specific layer (fixed position, collapsed), and the
 * not-exposed set — "不支持"是数据，渲染时必须连同原因一起可见。 */
export function splitParamLayers(params: ParamSpecInfo[] | undefined | null): ParamLayers {
  const all = params ?? [];
  return {
    canonical: all.filter((p) => p.exposed && p.layer === "canonical"),
    engine: all.filter((p) => p.exposed && p.layer === "engine"),
    hidden: all.filter((p) => !p.exposed),
  };
}

/** Issue #37: evaluate one `ignored_when` predicate, e.g. `emo_mode!=情感向量`
 * or `emo_mode==same-as-reference`. The compared value is the CURRENT param
 * value, falling back to the spec default — exactly what the engine adapter
 * will see. */
function predicateHolds(
  predicate: string,
  specs: ParamSpecInfo[],
  values: Record<string, string>,
): boolean {
  const m = /^([\w.]+)(==|!=)(.+)$/.exec(predicate.trim());
  if (!m) return false; // unknown predicate: never hide on it
  const [, name, op, expected] = m;
  const spec = specs.find((p) => p.name === name);
  const current = values[name] ?? (spec ? String(spec.default ?? "") : "");
  return op === "==" ? current === expected : current !== expected;
}

/** Issue #37: the specs whose ignored_when conditions are NOT all satisfied —
 * i.e. what the engine will actually honour right now. The UI hides specs
 * whose predicates ALL hold (same rule the engine enforces again in
 * prepare_synthesis); unknown predicates never hide anything. */
export function visibleParamSpecs(
  specs: ParamSpecInfo[],
  values: Record<string, string>,
): ParamSpecInfo[] {
  return specs.filter(
    (p) => !p.ignored_when.length || !p.ignored_when.every((pred) => predicateHolds(pred, specs, values)),
  );
}

/** Only exposed, still-declared parameters may ride a generate request —
 * remembered values (ADR-0018 decision 6) or rerun state must never leak a
 * parameter the engine no longer exposes. */
export function sendableParams(
  params: ParamSpecInfo[] | undefined | null,
  values: Record<string, string>,
): Record<string, string> {
  const exposed = new Set((params ?? []).filter((p) => p.exposed).map((p) => p.name));
  return Object.fromEntries(Object.entries(values).filter(([k]) => exposed.has(k)));
}

/** User-facing wording for the closed not-exposed vocabulary (ADR-0018). */
export function reasonLabel(reason: string | null): string {
  const labels: Record<string, string> = {
    "no-op": "引擎接受但不使用",
    "server-injected": "由应用自动注入",
    "wrong-mode": "当前模式不适用",
    "paid-tier": "需要付费档位",
    "breaks-pipeline": "会破坏本应用管线",
    unverified: "未经验证",
  };
  return labels[reason ?? ""] ?? "未暴露";
}

// ---------------------------------------------------------------------------
// Issue #40: engines-page card rework helpers (pure, unit-tested).
// ---------------------------------------------------------------------------

/** Voice design dropdown only offers engines that declare the capability. */
export function voiceDesignEngines(engines: EngineInfo[]): EngineInfo[] {
  return engines.filter((e) => e.capabilities.voice_design);
}

/**
 * 「详情」 copy for one engine card (issue #40): capability declarations plus
 * the curated capability-matrix entry, in stable paragraph order. Missing
 * pieces are simply skipped so the card never shows empty placeholders.
 */
export function engineDetailParagraphs(
  engine: EngineInfo,
  matrix: MatrixEngine | null,
): string[] {
  const c = engine.capabilities;
  const caps = [
    `${CAP_LABELS.voice_cloning} ${c.voice_cloning ? "✓" : "✗"}`,
    `${CAP_LABELS.voice_design} ${c.voice_design ? "✓" : "✗"}`,
    `${CAP_LABELS.pronunciation_control} ${c.pronunciation_control ? "✓" : "✗"}`,
    `${CAP_LABELS.emotion} ${c.emotion ? "✓" : "✗"}`,
    `${CAP_LABELS.languages}：${c.languages.join("/") || "—"}`,
  ];
  const paragraphs: string[] = [];
  if (matrix?.role) paragraphs.push(matrix.role);
  paragraphs.push(`能力声明：${caps.join(" · ")}`);
  if (engine.billing_note) paragraphs.push(`计费口径：${engine.billing_note}`);
  if (engine.data_usage_note) paragraphs.push(`数据与训练：${engine.data_usage_note}`);
  if (matrix?.notes) paragraphs.push(matrix.notes);
  return paragraphs;
}

