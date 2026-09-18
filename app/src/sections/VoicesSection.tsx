import type { EngineInfo, Voice } from "../api";
import { VoiceCard } from "../components/VoiceCard";
import type { SectionId } from "../ui";
import { JumpLink } from "../components/bits";

export function VoicesSection({
  baseUrl,
  token,
  mediaToken,
  engines,
  voices,
  selectedVoice,
  onNavigate,
  onSelectVoice,
  onTranscribed,
  createVoice,
  createDesignedVoice,
  importVoicePackage,
  deleteVoice,
  creatingVoice,
  importingVoice,
  newVoice,
  setNewVoice,
  design,
  setDesign,
  voiceError,
  designError,
}: {
  baseUrl: string;
  token: string;
  mediaToken: string;
  engines: EngineInfo[];
  voices: Voice[];
  selectedVoice: string | null;
  onNavigate: (section: SectionId) => void;
  onSelectVoice: (id: string | null) => void;
  onTranscribed: (voice: Voice) => void;
  createVoice: () => void;
  createDesignedVoice: () => void;
  importVoicePackage: (file: File) => void;
  deleteVoice: (id: string) => void;
  creatingVoice: boolean;
  importingVoice: boolean;
  newVoice: {
    name: string;
    desc: string;
    file: File | null;
    avatar: File | null;
  };
  setNewVoice: (patch: Partial<{ name: string; desc: string; file: File | null; avatar: File | null }>) => void;
  design: {
    engineId: string;
    name: string;
    prompt: string;
    previewText: string;
    busy: boolean;
  };
  setDesign: (patch: Partial<{ engineId: string; name: string; prompt: string; previewText: string }>) => void;
  voiceError: string | null;
  designError: string | null;
}) {
  const selectedDesignEngine = engines.find((e) => e.id === design.engineId) ?? null;
  return (
    <section>
      <h2>音色库</h2>
      <div className="voice-create-row">
        <label>
          导入音色包（.zip）：
          <input
            type="file"
            accept=".zip"
            disabled={importingVoice}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) importVoicePackage(f);
              e.target.value = "";
            }}
          />
        </label>
        <span className="hint">
          导入后引擎绑定不会带来——需要时对导入的音色重新绑定引擎。
        </span>
      </div>
      <div className="voice-create">
        <div className="voice-create-row">
          <input
            placeholder="音色名称（必填）"
            value={newVoice.name}
            onChange={(e) => setNewVoice({ name: e.target.value })}
          />
          <input
            placeholder="文字说明（可选）"
            value={newVoice.desc}
            onChange={(e) => setNewVoice({ desc: e.target.value })}
          />
        </div>
        <div className="voice-create-row">
          <span className="hint">
            上传参考音频前请确认你拥有该声音的使用授权（首次使用时已确认）。
          </span>
          <label>
            参考音频（WAV/MP3/FLAC/M4A/OGG，3–120s，≤20MB）：
            <input
              type="file"
              accept=".wav,.mp3,.flac,.m4a,.ogg"
              onChange={(e) => setNewVoice({ file: e.target.files?.[0] ?? null })}
            />
          </label>
          <label>
            头像（可选）：
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.webp,.gif"
              onChange={(e) => setNewVoice({ avatar: e.target.files?.[0] ?? null })}
            />
          </label>
          <button
            onClick={createVoice}
            disabled={!newVoice.name.trim() || !newVoice.file || creatingVoice}
          >
            {creatingVoice ? "创建中…" : "创建音色"}
          </button>
        </div>
        {voiceError && <div className="error">{voiceError}</div>}
      </div>
      <div className="voice-design">
        <div className="voice-design-title">用文字描述创造音色（无需参考音频）</div>
        <div className="voice-create-row">
          <label>
            设计引擎：
            <select
              value={design.engineId}
              onChange={(e) => {
                setDesign({ engineId: e.target.value });
              }}
            >
              <option value="">（选择引擎）</option>
              {engines.map((e) => {
                const supported = e.capabilities.voice_design;
                const reason = !supported
                  ? "不支持：引擎未声明音色设计能力"
                  : e.requires_key && !e.key_configured
                    ? "需先配置 API Key"
                    : "";
                return (
                  <option key={e.id} value={e.id} disabled={!supported}>
                    {e.display_name}
                    {reason ? ` — ${reason}` : ""}
                  </option>
                );
              })}
            </select>
          </label>
        </div>
        {selectedDesignEngine && !selectedDesignEngine.capabilities.voice_design && (
          <div className="hint">
            所选引擎不支持音色设计（能力未声明），请改用支持设计的引擎。
          </div>
        )}
        <div className="voice-create-row">
          <input
            placeholder="音色名称（必填）"
            value={design.name}
            onChange={(e) => setDesign({ name: e.target.value })}
          />
          <input
            placeholder="声音描述，如：低沉缓慢的男声，适合纪录片旁白（必填）"
            value={design.prompt}
            onChange={(e) => setDesign({ prompt: e.target.value })}
          />
          <input
            placeholder="试听文本（设计完成后用它生成预览样本）"
            value={design.previewText}
            onChange={(e) => setDesign({ previewText: e.target.value })}
          />
          <button
            onClick={createDesignedVoice}
            disabled={
              design.busy ||
              !design.name.trim() ||
              !design.prompt.trim() ||
              !design.previewText.trim() ||
              !design.engineId ||
              !selectedDesignEngine?.capabilities.voice_design ||
              (!!selectedDesignEngine?.requires_key && !selectedDesignEngine.key_configured)
            }
          >
            {design.busy ? "设计中…" : "设计音色"}
          </button>
        </div>
        {selectedDesignEngine?.requires_key && !selectedDesignEngine.key_configured && (
          <div className="hint">
            该设计引擎需要 API Key（BYOK）：请先在设置中保存后再设计。
            <JumpLink target="settings" label="打开设置" onNavigate={onNavigate} />
          </div>
        )}
        {designError && <div className="error">{designError}</div>}
      </div>
      {voices.length === 0 ? (
        <div className="history-empty">还没有音色。上传一段参考音频创建第一个音色。</div>
      ) : (
        <div className="voice-list">
          {voices.map((v) => (
            <VoiceCard
              key={v.id}
              voice={v}
              baseUrl={baseUrl}
              token={token}
              mediaToken={mediaToken}
              selected={v.id === selectedVoice}
              onSelect={() => onSelectVoice(v.id === selectedVoice ? null : v.id)}
              onDelete={() => deleteVoice(v.id)}
              onTranscribed={onTranscribed}
              onNavigate={onNavigate}
            />
          ))}
        </div>
      )}
    </section>
  );
}
