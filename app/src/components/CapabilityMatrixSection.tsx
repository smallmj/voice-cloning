import { useEffect, useState } from "react";
import type { EngineInfo } from "../api";
import { apiJson } from "../client";
import type { SectionId } from "../ui";
import { JumpLink } from "./bits";

// ---------------------------------------------------------------------------
// 能力矩阵（issue #16）：固化的矩阵数据 + 中文回归集结果，界面直接驱动。
// ---------------------------------------------------------------------------

interface MatrixEvidence {
  field: string;
  value: string;
  verification: "measured" | "verified" | "vendor" | "unverified" | "unknown";
  source: string;
  date?: string;
}

interface MatrixPerf {
  env: string;
  rtf: number | null;
  peak_vram_bytes: number | null;
  verification: MatrixEvidence["verification"];
  source?: string;
  note?: string;
}

interface MatrixEngine {
  engine_id: string;
  display_name: string;
  kind: "builtin" | "local" | "cloud";
  role?: string;
  evidence?: MatrixEvidence[];
  performance?: MatrixPerf[];
  notes?: string;
  runtime?: { capabilities: Record<string, unknown> } | null;
  regression_summary?: {
    ok: number;
    failed: number;
    rtf_mean: number | null;
  } | null;
}

interface CapabilityMatrix {
  updated_at: string;
  categories: { id: string; label: string }[];
  engines: MatrixEngine[];
}

const VERIFY_LABELS: Record<MatrixEvidence["verification"], string> = {
  measured: "实测",
  verified: "已核实",
  vendor: "厂商口径",
  unverified: "未核实",
  unknown: "未知",
};

export function CapabilityMatrixSection({
  baseUrl,
  token,
  engines,
  onNavigate,
}: {
  baseUrl: string;
  token: string;
  engines: EngineInfo[];
  onNavigate: (section: SectionId) => void;
}) {
  const [matrix, setMatrix] = useState<CapabilityMatrix | null>(null);
  const [regression, setRegression] = useState<{
    id: string;
    status: string;
  } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiJson<CapabilityMatrix>(baseUrl, token, "/capability-matrix")
      .then(setMatrix)
      .catch(() => {
        /* sidecar not ready yet */
      });
  }, [baseUrl, token]);

  // 回归在后台线程里跑：启动后轮询直到完成。
  useEffect(() => {
    if (!regression || regression.status !== "running") return;
    const t = setTimeout(async () => {
      try {
        setRegression(await apiJson<{ id: string; status: string }>(
          baseUrl,
          token,
          `/regression/${regression.id}`,
        ));
      } catch {
        /* keep polling */
      }
    }, 1500);
    return () => clearTimeout(t);
  }, [regression, baseUrl, token]);

  async function runRegression() {
    setBusy(true);
    setError(null);
    try {
      setRegression(
        await apiJson<{ id: string; status: string }>(baseUrl, token, "/regression/run", {
          method: "POST",
          body: JSON.stringify({}),
        }),
      );
    } catch (err) {
      setError(`启动回归失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <h2>能力矩阵</h2>
      <div className="hint">
        本矩阵由中文回归集实测与一手调研固化（issue #16）。每一格都标注验证等级：
        <b>实测</b>（本项目真机跑出来的）、<b>已核实</b>（一手来源）、
        <b>厂商口径</b>、<b>未核实</b>（仍未知）、<b>未知</b>。任何公开榜单都给不出这台机器上的数字。
      </div>
      <div className="key-row">
        <button disabled={busy} onClick={() => void runRegression()}>
          {busy ? "启动中…" : "运行中文回归集（数字 / 日期 / 金额 / 中英混读 / 多音字）"}
        </button>
        {regression && (
          <span className="hint">
            {regression.status === "running"
              ? "回归运行中…（引擎逐条生成，完成后自动刷新）"
              : "最近一次回归已完成，结果已合入下方矩阵。"}
          </span>
        )}
      </div>
      {error && <div className="error">{error}</div>}
      {matrix && (
        <div className="settings-list">
          {matrix.engines.map((e) => {
            const installedEngine = engines.find((x) => x.id === e.engine_id);
            return (
              <div className="settings-engine" key={e.engine_id}>
                <div className="settings-engine-head">
                  <strong>{e.display_name}</strong>
                  <span className="badge badge-on">
                    {e.kind === "cloud" ? "云端" : e.kind === "local" ? "本地" : "内置"}
                    {installedEngine && !installedEngine.installed ? " · 未安装" : ""}
                  </span>
                  {e.regression_summary && (
                    <span className="badge badge-on">
                      回归 {e.regression_summary.ok} 成功 / {e.regression_summary.failed} 失败
                      {e.regression_summary.rtf_mean != null
                        ? ` · RTF 均值 ${e.regression_summary.rtf_mean}`
                        : ""}
                    </span>
                  )}
                </div>
                {e.role && <div className="hint">{e.role}</div>}
                <table className="pref-table">
                  <thead>
                    <tr>
                      <th>维度</th>
                      <th>结论</th>
                      <th>验证</th>
                      <th>来源</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(e.evidence ?? []).map((f, idx) => (
                      <tr key={idx}>
                        <td>{f.field}</td>
                        <td>{f.value}</td>
                        <td>{VERIFY_LABELS[f.verification]}</td>
                        <td>
                          {f.source}
                          {f.date ? `（${f.date}）` : ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {(e.performance ?? []).length > 0 && (
                  <table className="pref-table">
                    <thead>
                      <tr>
                        <th>环境</th>
                        <th>RTF</th>
                        <th>峰值显存</th>
                        <th>验证</th>
                      </tr>
                    </thead>
                    <tbody>
                      {e.performance!.map((p, idx) => (
                        <tr key={idx}>
                          <td>{p.env}</td>
                          <td>{p.rtf != null ? p.rtf : "—"}</td>
                          <td>
                            {p.peak_vram_bytes != null
                              ? `${(p.peak_vram_bytes / 1024 ** 3).toFixed(1)} GB`
                              : "—"}
                          </td>
                          <td>{VERIFY_LABELS[p.verification]}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {e.notes && <div className="hint">{e.notes}</div>}
              </div>
            );
          })}
        </div>
      )}
      <div className="hint">
        主观听感（盲听 MOS）请用「盲听对比」：同音色同文本、LUFS 归一化、盲标签评分；
        结论记入 docs/evaluation/ 的听感记录。
        <JumpLink target="generate" label="前往生成区" onNavigate={onNavigate} />
      </div>
    </section>
  );
}
