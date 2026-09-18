import { useCallback, useEffect, useState } from "react";
import type { LocalModelsInfo, SourcesInfo, SourceAxis } from "../api";
import { apiJson } from "../client";

// ---------------------------------------------------------------------------
// 下载源（三轴，issue #17 / ADR-0016）：权重源 / 包索引 / CUDA 轮子源各自
// 可切换，选中的是「首选源」，其余保留为静默兜底。与本地模型管理（权重
// 目录、磁盘占用、卸载）同处「引擎」区一页（ADR-0016 后果条款）。
// ---------------------------------------------------------------------------

const AXIS_LABELS: Record<SourceAxis, string> = {
  weights: "权重源",
  pypi: "包索引（PyPI）",
  cuda: "CUDA 轮子源",
};

const AXIS_HINTS: Record<SourceAxis, string> = {
  weights: "HuggingFace 权重的首选下载站；切换后下次安装生效。",
  pypi: "引擎 Python 包的安装索引；CUDA torch 轮子不走此索引（ADR-0002）。",
  cuda: "PyTorch CUDA 轮子的下载站；与权重源、包索引互相独立。",
};

function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

export function DownloadSourcesSection({
  baseUrl,
  token,
}: {
  baseUrl: string;
  token: string;
}) {
  const [info, setInfo] = useState<SourcesInfo | null>(null);
  const [models, setModels] = useState<LocalModelsInfo | null>(null);
  const [draft, setDraft] = useState<SourcesInfo["sources"] | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    const sources = await apiJson<SourcesInfo>(baseUrl, token, "/settings/sources");
    setInfo(sources);
    setDraft(sources.sources);
    try {
      setModels(
        await apiJson<LocalModelsInfo>(baseUrl, token, "/engines/models"),
      );
    } catch {
      setModels(null); // 模型列表失败不阻塞下载源选择
    }
  }, [baseUrl, token]);

  useEffect(() => {
    void load();
  }, [load]);

  if (!info || !draft) return null;

  const dirty =
    draft.weights !== info.sources.weights ||
    draft.pypi !== info.sources.pypi ||
    draft.cuda !== info.sources.cuda;

  const save = async () => {
    setBusy(true);
    setMessage(null);
    try {
      await apiJson(baseUrl, token, "/settings/sources", {
        method: "PUT",
        body: JSON.stringify({ sources: draft }),
      });
      await load();
      setMessage("已保存；下次安装时生效。");
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "保存失败");
    } finally {
      setBusy(false);
    }
  };

  const uninstall = async (id: string, name: string) => {
    if (
      !window.confirm(
        `卸载「${name}」将删除其权重与运行环境（约 ${formatBytes(
          models?.models.find((m) => m.id === id)?.disk_usage_bytes ?? 0,
        )}），确定吗？`,
      )
    )
      return;
    setBusy(true);
    setMessage(null);
    try {
      await apiJson(baseUrl, token, `/engines/${id}/model`, { method: "DELETE" });
      await load();
      setMessage(`已卸载「${name}」。`);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "卸载失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card" id="download-sources">
      <h3>下载源与本地模型</h3>
      <div className="hint">
        每一轴单独选择首选源；首选失败时自动落到其余源（静默兜底）。安装日志会显示
        「本次实际由哪个源服务」——若日志与所选不符，说明首选源发生了静默退化。
      </div>
      {(["weights", "pypi", "cuda"] as SourceAxis[]).map((axis) => (
        <div key={axis} className="field">
          <label>{AXIS_LABELS[axis]}</label>
          <select
            value={draft[axis]}
            disabled={busy}
            onChange={(e) =>
              setDraft({ ...draft, [axis]: e.target.value })
            }
          >
            {info.choices[axis].map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
          <div className="hint">{AXIS_HINTS[axis]}</div>
        </div>
      ))}
      <button className="install-btn" disabled={busy || !dirty} onClick={() => void save()}>
        保存下载源
      </button>
      {models && models.models.length > 0 && (
        <div className="model-list">
          <h4>本地模型（{models.runtime_root}）</h4>
          {models.models.map((m) => (
            <div key={m.id} className="model-row">
              <span>{m.display_name}</span>
              <span className="hint">
                {m.installed ? `已安装 · ${formatBytes(m.disk_usage_bytes)}` : "未安装"}
              </span>
              <button
                className="install-btn"
                disabled={busy || !m.installed || m.installing}
                onClick={() => void uninstall(m.id, m.display_name)}
              >
                卸载
              </button>
            </div>
          ))}
        </div>
      )}
      {message && <div className="hint">{message}</div>}
    </div>
  );
}
