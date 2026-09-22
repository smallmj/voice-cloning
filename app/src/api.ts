export interface SidecarInfo {
  baseUrl: string;
  token: string;
}

// Issue #65: the `window.voiceclone` global declaration (sidecar members +
// update-bridge members) moved to ./updater-types.ts so both renderer code and
// the Electron preload share one declaration; keeping a second copy here would
// be a conflicting interface merge.

export interface Capabilities {
  languages: string[];
  voice_cloning: boolean;
  voice_design: boolean;
  pronunciation_control: boolean;
  emotion: boolean;
  commercial_license: boolean;
  cross_device_use: boolean;
  upload_used_for_training: boolean;
  api_closed_loop: boolean;
  requires_reference_text?: boolean;
  // Issue #13: over-length text is auto-segmented at this many characters
  // per request; null/absent means the engine declares no limit.
  max_chars_per_request?: number | null;
}

export interface ParamSpecInfo {
  name: string;
  label: string;
  // Issue #23 / ADR-0018: canonical | engine layers plus the expanded
  // control surface (bool, textarea, ranges, units, not-exposed data).
  kind: "select" | "text" | "textarea" | "number" | "bool" | "array" | "audio" | "output";
  layer: "canonical" | "engine";
  group: string;
  wire_path: string;
  default: string | number | boolean | null;
  choices: string[];
  help: string;
  unit: string;
  min: number | null;
  max: number | null;
  step: number | null;
  integer: boolean;
  min_open: boolean;
  max_open: boolean;
  max_from: string | null;
  max_length: number | null;
  max_items: number | null;
  items: {
    name: string;
    label: string;
    kind: string;
    min: number | null;
    max: number | null;
    choices: string[];
  }[];
  exposed: boolean;
  not_exposed_reason: string | null;
  applies_to: {
    engine: string | null;
    model: string | null;
    mode: string | null;
  } | null;
  ignored_when: string[];
  wire_map: boolean;
}

export interface EngineInfo {
  id: string;
  display_name: string;
  capabilities: Capabilities;
  installed?: boolean;
  // BYOK cloud engines (issue #9): key management + billing/data-usage
  // disclosure, rendered verbatim on the settings page.
  requires_key?: boolean;
  key_configured?: boolean | null;
  // Vendor identity (ADR-0015): engines of one vendor share one API key
  // entry — the engines page groups its key inputs by this label.
  vendor_label?: string | null;
  billing_note?: string | null;
  data_usage_note?: string | null;
  params?: ParamSpecInfo[];
}

export interface KeyStatus {
  engine_id: string;
  configured: boolean;
}

export interface InstallStepState {
  status: "pending" | "running" | "completed" | "failed";
  error?: string | null;
}

/** Byte-level download progress of a running install (issue #35). */
export interface InstallProgress {
  file: string;
  done_bytes: number;
  total_bytes: number | null;
}

export interface EngineInstallStatus {
  id: string;
  installed: boolean;
  installing: boolean;
  steps: Record<string, InstallStepState>;
  progress: InstallProgress | null;
}

export interface GenerationLogLine {
  ts: string;
  message: string;
}

export interface GenerationRecord {
  id: string;
  engine_id: string;
  model_version?: string | null;
  text: string;
  normalized_text?: string;
  params?: Record<string, unknown>;
  voice_id?: string | null;
  voice_name?: string | null;
  status: "running" | "succeeded" | "failed";
  audio_url?: string | null;
  audio_file?: string | null;
  sample_rate?: number | null;
  error?: string | null;
  logs: GenerationLogLine[];
  duration_seconds?: number | null;
  cost?: number | null;
  created_at: string;
  finished_at?: string | null;
}

export interface GenerationListResult {
  records: GenerationRecord[];
  total: number;
}

export interface GenerationBatchDeleteResult {
  deleted: string[];
  missing: string[];
  count: number;
}

export interface NormalizeResult {
  normalized: string;
  changed: boolean;
  /** US19 (issue #54): present only when the request carried engine context
   * (engine_id + params). True when the engine adapter changed the
   * normalized text on top of it — e.g. a control-instruction prefix — so
   * `normalized` here is the COMPLETE text the engine will synthesize. */
  engine_adapter_applied?: boolean;
}

export type LogLevel = "info" | "warn" | "error";

export interface LogEvent {
  type: "log";
  generation_id: string;
  message: string;
  ts: number;
  /** Issue #36: classified at the publish site; frames without a valid
   * level parse as "info". */
  level: LogLevel;
}

export interface VoiceReference {
  filename: string;
  format: string;
  duration_seconds: number;
  size_bytes: number;
  sha256: string;
  transcript?: string | null;
  // Issue #18: the stored transcript is the old fake engine's fixed test
  // text — the UI shows an actionable re-transcribe hint instead of letting
  // the placeholder silently ride along as ref_text.
  transcript_placeholder?: boolean;
}

export interface VoiceBinding {
  status: string;
  reference_sha256: string;
  created_at: string;
  // Issue #12: when the vendor recycled the cloud voice and the automatic
  // rebuild failed, the binding is marked "unavailable" and `error` holds
  // the user-facing reason.
  // Issue #27: engines that must ACTIVATE a freshly cloned cloud voice
  // (MiniMax) record `activated_at` when the first real synthesis ran, and
  // mark the binding "unactivated" (with `activation_error`) when the
  // activation synthesis failed.
  voice_id?: string;
  error?: string | null;
  activated_at?: string | null;
  activation_error?: string | null;
}

export interface VoiceAvatar {
  filename: string;
  format: string;
  size_bytes: number;
}

// A voice is an engine-independent identity (ADR-0001); bindings are a
// rebuildable cache, never the source of truth. Issue #11 adds a second
// origin: "designed" voices are created from a text description and only
// later receive the design engine's preview sample as their reference.
export interface VoiceDesignInfo {
  engine_id: string;
  voice_prompt: string;
  preview_text: string;
}

export interface Voice {
  id: string;
  name: string;
  description: string;
  created_at: string;
  origin: "cloned" | "designed";
  design?: VoiceDesignInfo | null;
  reference: VoiceReference | null;
  avatar: VoiceAvatar | null;
  bindings: Record<string, VoiceBinding>;
}

// --- diagnostics + transcription (issue #8) ---

export interface DiagnosticItem {
  id: "snr" | "speaker" | "clipping" | "silence" | string;
  status: "good" | "warn" | "bad";
  message: string;
  advice: string;
  value?: number | null;
  segments?: [number, number][];
}

export interface ReferenceAnalysis {
  duration_seconds: number;
  sample_rate: number;
  snr_db: number;
  silence_segments: [number, number][];
  clipped_sample_ratio: number;
  diagnostics: DiagnosticItem[];
}

export interface TranscriptionEngineInfo {
  id: string;
  display_name: string;
  requires_key?: boolean;
  key_configured?: boolean | null;
}

export interface TranscriptionProviders {
  provider: string;
  default: string;
  local: {
    supported: boolean;
    installed: boolean;
    label: string | null;
    status?: { installed: boolean; steps: Record<string, unknown> } | null;
    reason?: string;
  };
  engines: TranscriptionEngineInfo[];
}

// --- blind comparison + preference profile (issue #10) ---

export interface CompareEntry {
  label: string;
  // Engine identity only appears once the session is revealed (or scored).
  engine_id?: string;
  generation_id?: string;
  normalized_audio_url: string;
  normalized_file?: string;
  original_lufs?: number | null;
  gain_db?: number;
  achieved_lufs?: number | null;
  peak_limited?: boolean;
  score?: number;
}

export interface CompareFailedLeg {
  engine_id: string;
  error: string;
}

export interface CompareSession {
  id: string;
  created_at: string;
  voice_id: string;
  text: string;
  text_type: string;
  language: string;
  target_lufs: number;
  entries: CompareEntry[];
  failed: CompareFailedLeg[];
  failed_count?: number;
  scored: boolean;
}

export interface PreferenceEngineRow {
  engine_id: string;
  average_score: number;
  score_count: number;
  wins: number;
}

export interface PreferenceCell {
  language: string;
  text_type: string;
  engines: PreferenceEngineRow[];
}

export interface PreferenceProfile {
  cells: PreferenceCell[];
}

// --- compliance guardrails: first-use voice consent (issue #15) ---

export interface ConsentInfo {
  acknowledged: boolean;
  confirmed_at: string | null;
  version: string;
}

// --- long-text jobs: queue / segmentation / cancel (issue #13) ---

export interface JobSegment {
  index: number;
  text: string;
  status: "pending" | "running" | "succeeded" | "failed" | "skipped";
  generation_id: string | null;
  error: string | null;
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface GenerationJob {
  id: string;
  engine_id: string;
  voice_id: string | null;
  voice_name: string | null;
  text: string;
  status: JobStatus;
  segment_count: number;
  segments: JobSegment[];
  audio_url: string | null;
  audio_file: string | null;
  sample_rate: number | null;
  error: string | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface MediaToken {
  media_token: string;
  expires_in_seconds: number;
  expires_at: number;
}

/** Issue #19: fetch the short-TTL, media-scoped token used by <img> /
 * <audio> elements and the log WebSocket. Bearer-only route — never paste
 * the raw sidecar token into a URL. */
export async function fetchMediaToken(baseUrl: string, token: string): Promise<MediaToken> {
  const res = await fetch(`${baseUrl}/media-token`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error(`media-token HTTP ${res.status}`);
  return (await res.json()) as MediaToken;
}

// --- download sources + local model management (issue #17, ADR-0016) ---

export type SourceAxis = "weights" | "pypi" | "cuda";

export interface SourcePrefs {
  weights: string;
  pypi: string;
  cuda: string;
}

export interface SourceChoice {
  id: string;
  label: string;
}

export interface SourceChoiceCatalog {
  weights: SourceChoice[];
  pypi: SourceChoice[];
  cuda: SourceChoice[];
}

export interface SourcesInfo {
  sources: SourcePrefs;
  choices: SourceChoiceCatalog;
}

export interface LocalModelInfo {
  id: string;
  display_name: string;
  installed: boolean;
  model_dir: string;
  weights_dir: string | null;
  disk_usage_bytes: number;
  installing: boolean;
}

export interface LocalModelsInfo {
  models: LocalModelInfo[];
  runtime_root: string;
}
