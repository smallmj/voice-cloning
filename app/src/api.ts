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
