"""能力矩阵固化：中文回归集、回归运行记录与矩阵聚合（issue #16）。

回归集是本产品的核心资产之一：五个引擎各跑同一组中文难例
（数字 / 日期 / 金额 / 中英混读 / 多音字），逐条记录成功与否、
RTF（音频时长 / 端到端墙钟时间）与本地 CUDA 引擎的峰值显存。
结果写入本地回归索引（<data>/regression.json），聚合后与仓库内
固化的能力矩阵数据（data/capability_matrix.json）合并，经
GET /capability-matrix 供界面直接驱动——不跑任何东西也能读。

听感（主观 MOS / 盲听）不走本模块：盲听会话复用 issue #10 的
/compare 管线，结论落在 docs/evaluation/ 的人工记录里（ADR-0009：
偏好只给人看，不参与路由）。
"""

from __future__ import annotations

import json
import re
import threading
import wave
from contextlib import contextmanager
from pathlib import Path

from .storage import write_json_atomic

# 固定五类中文难例。类别是封闭集合——自由标签会碎成无法聚合的单例。
REGRESSION_CATEGORIES = ("numbers", "dates", "amounts", "mixed", "polyphones")

CATEGORY_LABELS = {
    "numbers": "数字",
    "dates": "日期",
    "amounts": "金额",
    "mixed": "中英混读",
    "polyphones": "多音字",
}

# 每条：id / category / text（原始待合成文本，进引擎前照常过归一化层）/
# expect（应被听到的口语形态，供 ASR 可懂度对照）。expect 是人工校对
# 的目标读音文本，而不是自动产物——多音字条目必须由人确认读法。
REGRESSION_ITEMS: list[dict] = [
    # -- 数字 ----------------------------------------------------------------
    {"id": "num-01", "category": "numbers", "text": "本次会议共有 1024 人报名参加。", "expect": "本次会议共有一千零二十四人报名参加。"},
    {"id": "num-02", "category": "numbers", "text": "他的手机尾号是 110 和 8080。", "expect": "他的手机尾号是幺幺零和八零八零。"},
    {"id": "num-03", "category": "numbers", "text": "第 3 列的数值是 0.05。", "expect": "第三列的数值是零点零五。"},
    {"id": "num-04", "category": "numbers", "text": "温度达到了 36.6 度。", "expect": "温度达到了三十六点六度。"},
    # -- 日期 ----------------------------------------------------------------
    {"id": "date-01", "category": "dates", "text": "项目于 2024 年 10 月 1 日正式启动。", "expect": "项目于二零二四年十月一日正式启动。"},
    {"id": "date-02", "category": "dates", "text": "航班预计 8:30 起飞，21:05 降落。", "expect": "航班预计八点三十分起飞，二十一点零五分降落。"},
    {"id": "date-03", "category": "dates", "text": "合同有效期至 2026-12-31。", "expect": "合同有效期至二零二六年十二月三十一日。"},
    # -- 金额 ----------------------------------------------------------------
    {"id": "amt-01", "category": "amounts", "text": "这台设备售价 ¥12,999。", "expect": "这台设备售价一万两千九百九十九元。"},
    {"id": "amt-02", "category": "amounts", "text": "账户余额还剩 0.37 元。", "expect": "账户余额还剩三毛七分。"},
    {"id": "amt-03", "category": "amounts", "text": "预算总额为 1.2 亿元。", "expect": "预算总额为一亿两千万元。"},
    # -- 中英混读 --------------------------------------------------------------
    {"id": "mix-01", "category": "mixed", "text": "请把这份 PPT 发到我的邮箱。", "expect": "请把这份 P P T 发到我的邮箱。"},
    {"id": "mix-02", "category": "mixed", "text": "他用 Python 写了一个爬虫。", "expect": "他用 Python 写了一个爬虫。"},
    {"id": "mix-03", "category": "mixed", "text": "新版支持 Wi-Fi 6 和蓝牙 5.3。", "expect": "新版支持 Wi Fi 六和蓝牙五点三。"},
    # -- 多音字 ----------------------------------------------------------------
    {"id": "poly-01", "category": "polyphones", "text": "他重新考虑了这件事的重量。", "expect": "他重新考虑了这件事的重量。"},
    {"id": "poly-02", "category": "polyphones", "text": "银行门口有一条长长的人行通道。", "expect": "银行门口有一条长长的人行通道。"},
    {"id": "poly-03", "category": "polyphones", "text": "大会还讨论了存款利率下调的问题。", "expect": "大会还讨论了存款利率下调的问题。"},
    {"id": "poly-04", "category": "polyphones", "text": "这种乐器叫做大乐,演奏起来很快乐。", "expect": "这种乐器叫做大乐，演奏起来很快乐。"},
]


class RegressionError(ValueError):
    """Raised on invalid regression input; message is user-facing."""


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


def char_error_rate(reference: str | None, hypothesis: str | None) -> float | None:
    """字符错误率（编辑距离 / 参考长度）。任一侧缺失或参考为空 → None。

    参考非空而假设为空是完全听不出内容 → 1.0；反向则无从打分 → None。
    """
    if not reference:
        return None
    ref = re.sub(r"\s+", "", reference)
    hyp = re.sub(r"\s+", "", hypothesis or "")
    if not ref:
        return None
    # 经典 Wagner-Fischer 编辑距离；回归集句子很短，O(n²) 足够。
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
            )
        prev = cur
    return prev[-1] / len(ref)


def wav_duration_seconds(path: Path | str) -> float | None:
    """WAV 文件的音频时长（秒）；文件缺失或不可解析 → None。"""
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
    except (OSError, wave.Error):
        return None
    if rate <= 0:
        return None
    return frames / rate


@contextmanager
def sample_peak_vram(interval: float = 0.2):
    """采样本地 GPU 峰值显存（字节）。

    无 nvidia-smi（Mac / 纯 CPU）时安静地放弃——返回 None 表示
    "此环境不适用"，绝不编造 0。CUDA 机器上以基线为参照返回峰值增量。
    """
    import shutil
    import subprocess

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        yield {"peak_vram_bytes": None, "sampling": False}
        return

    def _used() -> int | None:
        try:
            out = subprocess.run(
                [nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().splitlines()
            return int(out[0]) * 1024 * 1024 if out else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    baseline = _used()
    # The poller mutates this holder in place while the with-body runs; the
    # caller reads it AFTER the body (the with-block's own duration).
    holder = {"peak_vram_bytes": None, "sampling": baseline is not None}
    if baseline is None:
        yield holder
        return

    stop = threading.Event()

    def _poll():
        while not stop.wait(interval):
            v = _used()
            if v is not None and (holder["peak_vram_bytes"] is None or v > holder["peak_vram_bytes"]):
                holder["peak_vram_bytes"] = v

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    try:
        yield holder
    finally:
        stop.set()
        thread.join(timeout=1.0)
        if holder["peak_vram_bytes"] is not None:
            holder["peak_vram_bytes"] = max(0, holder["peak_vram_bytes"] - baseline)


# ---------------------------------------------------------------------------
# 回归会话存储
# ---------------------------------------------------------------------------

_ITEM_KEYS = (
    "item_id", "category", "engine_id", "status", "generation_id",
    "audio_url", "audio_seconds", "wall_seconds", "rtf",
    "peak_vram_bytes", "asr_text", "cer", "error",
)


class RegressionStore:
    """Persistence for regression sessions; items live inside each session."""

    def __init__(self, data_dir: Path) -> None:
        self.index_path = Path(data_dir) / "regression.json"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for session in raw.get("sessions", []):
            if isinstance(session, dict) and session.get("id"):
                self._sessions[session["id"]] = session

    def _save(self) -> None:
        ordered = sorted(
            self._sessions.values(), key=lambda s: s.get("created_at", ""), reverse=True
        )
        write_json_atomic(self.index_path, {"sessions": ordered})

    def create(self, session: dict) -> dict:
        with self._lock:
            self._sessions[session["id"]] = session
            self._save()
        return dict(session)

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return dict(session) if session else None

    def list(self, limit: int = 50) -> list[dict]:
        with self._lock:
            sessions = sorted(
                self._sessions.values(),
                key=lambda s: s.get("created_at", ""),
                reverse=True,
            )
        return [dict(s) for s in sessions[:limit]]

    def update(self, session: dict) -> dict:
        with self._lock:
            if session["id"] not in self._sessions:
                raise KeyError(session["id"])
            self._sessions[session["id"]] = session
            self._save()
        return dict(session)

    def delete(self, session_id: str) -> dict | None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                self._save()
        return session


def summarize_items(items: list[dict]) -> dict:
    """One engine's aggregate across a set of regression items."""
    ok = sum(1 for i in items if i.get("status") == "ok")
    failed = sum(1 for i in items if i.get("status") == "failed")
    rtfs = [i["rtf"] for i in items if i.get("rtf") is not None]
    return {
        "ok": ok,
        "failed": failed,
        "rtf_mean": round(sum(rtfs) / len(rtfs), 3) if rtfs else None,
    }


def aggregate_session(session: dict) -> dict:
    """Per-engine, per-category success rate + mean RTF for one session."""
    engines: list[str] = list(session.get("engines", []))
    items: list[dict] = list(session.get("items", []))
    out: dict[str, dict] = {}
    for engine_id in engines:
        mine = [i for i in items if i.get("engine_id") == engine_id]
        by_category: dict[str, dict] = {}
        for category in REGRESSION_CATEGORIES:
            by_category[category] = summarize_items(
                [i for i in mine if i.get("category") == category]
            )
        summary = summarize_items(mine)
        summary["by_category"] = by_category
        out[engine_id] = summary
    return {"engines": out}


# ---------------------------------------------------------------------------
# 固化的能力矩阵数据（仓库内随包分发）
# ---------------------------------------------------------------------------

_MATRIX_PATH = Path(__file__).parent / "data" / "capability_matrix.json"

# 矩阵里每个事实的可验证等级：
#   measured   本项目在真机上的实测（谁、何时、什么环境）
#   verified   一手来源（官方文档 / 源码 / 许可证原文）核实过
#   vendor     仅厂商宣传口径，未独立验证
#   unverified 调研时未核实；本工单结论为「仍未知」
#   unknown    根本无从核实
VERIFICATION_LEVELS = ("measured", "verified", "vendor", "unverified", "unknown")


def load_matrix_data() -> dict:
    """Read the committed capability-matrix data file."""
    return json.loads(_MATRIX_PATH.read_text(encoding="utf-8"))


def build_capability_matrix(
    registry,
    matrix_data: dict,
    regression_store: RegressionStore | None = None,
) -> dict:
    """Merge curated matrix data with runtime capability declarations and the
    latest regression evidence, producing the payload the UI renders."""
    latest: dict[str, dict] = {}
    if regression_store is not None:
        for session in regression_store.list(limit=5):
            if session.get("status") != "completed":
                continue
            for engine_id, summary in aggregate_session(session)["engines"].items():
                latest.setdefault(engine_id, summary)

    declared: dict[str, dict] = {}
    for engine in registry.list():
        declared[engine.engine_id] = {
            "display_name": engine.display_name,
            "capabilities": engine.capabilities().to_dict(),
            "requires_key": getattr(engine, "requires_key", False),
        }

    engines_out = []
    for entry in matrix_data["engines"]:
        engine_id = entry["engine_id"]
        merged = dict(entry)
        merged["runtime"] = declared.get(engine_id)
        merged["regression_summary"] = latest.get(engine_id)
        engines_out.append(merged)

    return {
        "version": matrix_data.get("version", 1),
        "updated_at": matrix_data.get("updated_at", ""),
        "verification_levels": list(VERIFICATION_LEVELS),
        "categories": [
            {"id": c, "label": CATEGORY_LABELS[c]} for c in REGRESSION_CATEGORIES
        ],
        "regression_items": REGRESSION_ITEMS,
        "engines": engines_out,
    }
