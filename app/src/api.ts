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
}

export interface EngineInfo {
  id: string;
  display_name: string;
  capabilities: Capabilities;
  installed?: boolean;
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

export interface GenerationRecord {
  id: string;
  engine_id: string;
  text: string;
  normalized_text?: string;
  status: "running" | "succeeded" | "failed";
  audio_url?: string;
  sample_rate?: number;
  error?: string;
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
}

export interface VoiceBinding {
  status: string;
  reference_sha256: string;
  created_at: string;
}

export interface VoiceAvatar {
  filename: string;
  format: string;
  size_bytes: number;
}

// A voice is an engine-independent identity (ADR-0001); bindings are a
// rebuildable cache, never the source of truth.
export interface Voice {
  id: string;
  name: string;
  description: string;
  created_at: string;
  reference: VoiceReference;
  avatar: VoiceAvatar | null;
  bindings: Record<string, VoiceBinding>;
}
