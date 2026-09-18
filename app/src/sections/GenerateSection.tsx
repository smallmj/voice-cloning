import type {
  EngineInfo,
  EngineInstallStatus,
  GenerationRecord,
  LogEvent,
  NormalizeResult,
  Voice,
} from "../api";
import { CompareSection } from "../components/CompareSection";
import { LogDrawer } from "../components/LogDrawer";
import { Waveform } from "../components/Waveform";
import { JumpLink } from "../components/bits";
import type { SectionId } from "../ui";

export function GenerateSection({
  baseUrl,
  token,
  mediaToken,
  engines,
  selectedEngine,
  selectedEngineInfo,
  installStatus,
  voices,
  selectedVoice,
  onSelectVoice,
  text,
  onTextChange,
  engineParams,
  onEngineParamChange,
  normalized,
  record,
  generating,
  generateError,
  rerunHint,
  jobHint,
  rerunParams,
  onClearRerunParams,
  onGenerate,
  logs,
  logsOpen,
  onToggleLogs,
  onNavigate,
}: {
  baseUrl: string;
  token: string;
  mediaToken: string;
  engines: EngineInfo[];
  selectedEngine: string | null;
  selectedEngineInfo: EngineInfo | null;
  installStatus: Record<string, EngineInstallStatus>;
  voices: Voice[];
  selectedVoice: string | null;
  onSelectVoice: (id: string | null) => void;
  text: string;
  onTextChange: (text: string) => void;
  engineParams: Record<string, string>;
  onEngineParamChange: (name: string, value: string) => void;
  normalized: NormalizeResult | null;
  record: GenerationRecord | null;
  generating: boolean;
  generateError: string | null;
  rerunHint: string | null;
  jobHint: string | null;
  rerunParams: Record<string, unknown> | null;
  onClearRerunParams: () => void;
  onGenerate: () => void;
  logs: LogEvent[];
  logsOpen: boolean;
  onToggleLogs: () => void;
  onNavigate: (section: SectionId) => void;
}) {
  const notInstalled =
    selectedEngine != null &&
    installStatus[selectedEngine] != null &&
    !installStatus[selectedEngine].installed;

  return (
    <section>
      <h2>生成</h2>
      <div className="voice-picker">
        <label>
          使用音色：
          <select
            value={selectedVoice ?? ""}
            disabled={
              !!selectedEngineInfo && selectedEngineInfo.capabilities.voice_cloning === false
            }
            onChange={(e) => onSelectVoice(e.target.value || null)}
          >
            <option value="">（不用音色，直接合成）</option>
            {voices.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
        </label>
        {selectedVoice &&
          selectedEngine &&
          engines.find((e) => e.id === selectedEngine)?.capabilities.voice_cloning ===
            false && (
            <span className="hint">
              该引擎不支持声音复刻，音色选择已禁用（无参考音频可交给该引擎）。
            </span>
          )}
        {selectedVoice &&
          selectedEngine &&
          engines.find((e) => e.id === selectedEngine)?.capabilities
            .requires_reference_text && (
            <span className="hint">
              该引擎需要参考文本：将自动转写所选音色的参考音频，无需手动输入。
            </span>
          )}
        {selectedEngineInfo && (
          <span className="hint training-note">
            上传内容是否用于训练：
            {selectedEngineInfo.capabilities.upload_used_for_training
              ? "是（该引擎厂商声明会上传内容用于训练）"
              : "否（该引擎厂商声明不上传内容用于训练）"}
            {selectedEngineInfo.data_usage_note
              ? `　—　${selectedEngineInfo.data_usage_note}`
              : ""}
          </span>
        )}
      </div>
      <div className="voice-picker">
        <label>
          当前引擎：
          <strong>{selectedEngineInfo?.display_name ?? "（未选择）"}</strong>
        </label>
        <JumpLink target="engines" label="更换引擎" onNavigate={onNavigate} />
      </div>
      {(selectedEngineInfo?.params ?? []).map((p) => (
        <div className="param-row" key={p.name}>
          <label>
            {p.label}：
            {p.kind === "select" ? (
              <select
                value={engineParams[p.name] ?? String(p.default ?? "")}
                onChange={(e) => onEngineParamChange(p.name, e.target.value)}
              >
                {p.choices.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type={p.kind === "number" ? "number" : "text"}
                value={engineParams[p.name] ?? String(p.default ?? "")}
                onChange={(e) => onEngineParamChange(p.name, e.target.value)}
              />
            )}
          </label>
          {p.help && <span className="hint">{p.help}</span>}
        </div>
      ))}

      <textarea value={text} onChange={(e) => onTextChange(e.target.value)} rows={4} />

      {normalized && normalized.changed && (
        <div className="normalize-preview">
          <div className="normalize-title">归一化预览（生成前引擎将收到以下文本）</div>
          <div className="normalize-text">{normalized.normalized}</div>
        </div>
      )}
      {selectedEngineInfo?.requires_key && !selectedEngineInfo.key_configured && (
        <div className="hint">
          云端引擎需要你自己的 API Key（BYOK）：请先在设置中保存，生成已禁用。
          <JumpLink target="settings" label="打开设置" onNavigate={onNavigate} />
        </div>
      )}
      {selectedEngineInfo?.requires_key && selectedEngineInfo.key_configured && !selectedVoice && (
        <div className="hint">
          云端引擎必须使用音色：请先选择一个音色，首次生成时会自动把参考音频注册为云端音色。
        </div>
      )}
      <button className="primary" onClick={onGenerate} disabled={generating || !selectedEngine || notInstalled}>
        {generating ? "生成中…" : "生成"}
      </button>
      {selectedEngine && notInstalled && (
        <div className="hint">
          该引擎尚未安装，请先安装。
          <JumpLink target="engines" label="前往引擎区安装" onNavigate={onNavigate} />
        </div>
      )}

      {record && record.status === "succeeded" && record.audio_url && (
        <div className="result">
          <Waveform url={`${baseUrl}${record.audio_url}`} token={token} />
        </div>
      )}
      {record && record.status === "failed" && (
        <div className="error">生成失败：{record.error}</div>
      )}
      {generateError && <div className="error">{generateError}</div>}
      {rerunHint && <div className="hint">{rerunHint}</div>}
      {jobHint && <div className="hint">{jobHint}</div>}
      {rerunParams && (
        <div className="rerun-params">
          重跑参数已载入：
          <code>{Object.keys(rerunParams).join("、")}</code>
          <button onClick={onClearRerunParams}>清除</button>
        </div>
      )}

      <CompareSection
        baseUrl={baseUrl}
        token={token}
        mediaToken={mediaToken}
        voices={voices}
        engines={engines}
      />

      {/* ADR-0013 §4: 生成时用户要盯着日志 —— 日志抽屉留在生成区内。 */}
      <LogDrawer logs={logs} open={logsOpen} onToggle={onToggleLogs} />
    </section>
  );
}
