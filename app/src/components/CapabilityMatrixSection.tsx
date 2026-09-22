import type { EngineInfo } from "../api";
import { VERIFY_LABELS, useCapabilityMatrix } from "../capability-matrix";
import type { SectionId } from "../ui";
import { JumpLink } from "./bits";

// ---------------------------------------------------------------------------
// 能力矩阵（issue #16）：固化的矩阵数据，界面直接驱动。
// Issue #40: the 「运行中文回归集」 button moved into the engines-page
// advanced area; this block now only displays the固化 matrix + regression
// results.
// ---------------------------------------------------------------------------

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
  const matrix = useCapabilityMatrix(baseUrl, token);
  return (
    <section>
      <h2>能力矩阵</h2>
      <div className="hint">
        本矩阵由中文回归集实测与一手调研固化（issue #16）。每一格都标注验证等级：
        <b>实测</b>（本项目真机跑出来的）、<b>已核实</b>（一手来源）、
        <b>厂商口径</b>、<b>未核实</b>（仍未知）、<b>未知</b>。任何公开榜单都给不出这台机器上的数字。
      </div>
      {matrix && (
        <div className="settings-list">
          {matrix.engines.map((e) => {
            const installedEngine = engines.find((x) => x.id === e.engine_id);
            // Issue #53: each engine's evidence collapses behind a native
            // disclosure triangle so the matrix stops being a wall of rows.
            return (
              <details className="settings-engine matrix-group" key={e.engine_id}>
                <summary className="settings-engine-head">
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
                </summary>
                <div className="matrix-group-body">
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
              </details>
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
