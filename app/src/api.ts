export interface SidecarInfo {
  baseUrl: string;
  token: string;
}

declare global {
  interface Window {
    voiceclone: {
      getSidecarInfo: () => Promise<SidecarInfo | null>;
      onSidecarError: (cb: (message: string) => void) => void;
    };
  }
}

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
  kind: "select" | "text" | "number";
  default: string | number | null;
  choices: string[];
  help: string;
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

export interface EngineInstallStatus {
  id: string;
  installed: boolean;
  installing: boolean;
  steps: Record<string, InstallStepState>;
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

export interface NormalizeResult {
  normalized: string;
  changed: boolean;
}

export interface LogEvent {
  type: "log";
  generation_id: string;
  message: string;
  ts: number;
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
  voice_id?: string;
  error?: string | null;
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
  engines: { id: string; display_name: string }[];
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
