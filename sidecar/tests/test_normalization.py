"""Normalization layer regression tests.

The regression set is a DATA TABLE: each row is (input, expected) and every
row is exercised through the public ``POST /normalize`` endpoint, which is the
same interface the UI preview uses. Normalization must also be idempotent and
must leave already-canonical text untouched.
"""

from __future__ import annotations

import pytest

# (input, expected) — 中文数字 / 日期 / 金额 / 单位 / 中英混读 回归集。
CASES: list[tuple[str, str]] = [
    # --- 中文数字 ---
    ("我买了3个苹果", "我买了三个苹果"),
    ("一共123元", "一共一百二十三元"),
    ("这条视频时长5分20秒", "这条视频时长五分钟二十秒"),
    ("编号是13800138000", "编号是一三八零零一三八零零零"),
    ("圆周率约3.14159", "圆周率约三点一四一五九"),
    ("到了第12层", "到了第十二层"),
    ("1,234本书", "一千二百三十四本书"),
    # --- 日期 / 时间 ---
    ("2024年3月5日发布了", "二零二四年三月五日发布了"),
    ("2024年发布的", "二零二四年发布的"),
    ("3月5日是晴天", "三月五日是晴天"),
    ("会议在9:30开始", "会议在九点三十分开始"),
    ("比赛是14:05:08", "比赛是十四点零五分零八秒"),
    ("23:00收播", "二十三点零零分收播"),
    # --- 金额 ---
    ("这件衣服¥199", "这件衣服一百九十九元"),
    ("票价12.5元", "票价十二元五角"),
    ("捐款￥5000", "捐款五千元"),
    ("存了1000.05元", "存了一千元零五分"),
    ("花了$9.99", "花了九美元九十九美分"),
    # --- 百分比 / 温度 ---
    ("增长了50%", "增长了百分之五十"),
    ("准确率达到99.9%", "准确率达到百分之九十九点九"),
    ("今天25°C", "今天二十五摄氏度"),
    ("气温-5℃", "气温零下五摄氏度"),
    # --- 单位 ---
    ("全程跑了5km", "全程跑了五公里"),
    ("体重60kg", "体重六十千克"),
    ("身高175cm", "身高一百七十五厘米"),
    ("线长10mm", "线长十毫米"),
    ("水深3m", "水深三米"),
    ("重500g", "重五百克"),
    ("装了2L水", "装了二升水"),
    ("等待30s", "等待三十秒"),
    ("睡了8h", "睡了八小时"),
    ("等15min", "等十五分钟"),
    ("室温22 ℃ 偏低", "室温二十二摄氏度 偏低"),
    # --- 中英混读（英文原样保留，不产生副作用）---
    ("Hello world 你好", "Hello world 你好"),
    ("用GPT生成TTS音频", "用GPT生成TTS音频"),
    ("OpenAI released a model", "OpenAI released a model"),
    # --- 对抗用例（评审补充）：分数不是分钟、零头金额 ---
    ("他考了95分", "他考了九十五分"),
    ("半场得到50分", "半场得到五十分"),
    ("这条视频时长2分30秒", "这条视频时长二分钟三十秒"),
    ("$0.99", "零美元九十九美分"),
    # --- issue #56 ①非语言标签：方括号英文标签是正文一部分，原样透传 ---
    ("[laughing]", "[laughing]"),
    ("他讲到一半[laughing]停了一下", "他讲到一半[laughing]停了一下"),
    ("他叹了口气[sigh]然后说3个苹果", "他叹了口气[sigh]然后说三个苹果"),
    ("今天25°C[sigh]别提了", "今天二十五摄氏度[sigh]别提了"),
    # --- 已是规范写法：不产生副作用 ---
    ("今天是二零二四年三月五日", "今天是二零二四年三月五日"),
    ("三个苹果", "三个苹果"),
    ("百分之五十", "百分之五十"),
    ("二十五摄氏度", "二十五摄氏度"),
    ("", ""),
    ("   ", "   "),
]


@pytest.mark.parametrize(("text", "expected"), CASES, ids=[c[0] for c in CASES])
def test_normalize_data_table(client, text, expected):
    r = client.post("/normalize", json={"text": text})
    assert r.status_code == 200
    assert r.json()["normalized"] == expected


def test_normalize_reports_changed_flag(client):
    r = client.post("/normalize", json={"text": "3个苹果"})
    body = r.json()
    assert body["changed"] is True
    r2 = client.post("/normalize", json={"text": "三个苹果"})
    assert r2.json()["changed"] is False


@pytest.mark.parametrize(("text", "expected"), CASES, ids=[c[0] + " (idempotent)" for c in CASES])
def test_normalize_is_idempotent(client, text, expected):
    """normalize(normalize(x)) == normalize(x) for every regression row."""
    r = client.post("/normalize", json={"text": expected})
    assert r.status_code == 200
    assert r.json()["normalized"] == expected


def test_normalize_rejects_missing_text(client):
    r = client.post("/normalize", json={})
    assert r.status_code == 422


def test_generations_use_normalized_text(client, sidecar):
    """Every generation path is normalized before the engine sees the text —
    the record exposes it and the engine contract cannot bypass the layer."""
    r = client.post(
        "/generations",
        json={"engine_id": "fake", "text": "今天是2024年3月5日，共3条"},
    )
    assert r.status_code == 200
    record = r.json()
    assert record["normalized_text"] == "今天是二零二四年三月五日，共三条"


def test_english_description_with_digits_is_protected_by_fixed_ordering(client):
    """Issue #56 ②：含数字的英文描述（如 "a 30-year-old voice"）不被归一化
    改成中文。

    这个保证**不来自归一化层本身**（ADR-0008 层的职责是中文正文的数字/日期/
    单位，纯英文串里的数字仍会被转读——下方第一段断言锁定该现状，防止有人
    误以为本层提供了跨语言保护），而是来自固定的拼接时序（spec #54 /
    CONTEXT.md「控制指令」词条）：先归一化正文、后在引擎适配层拼接控制指令，
    控制指令永不经归一化。引擎侧时序锁定见
    test_voxcpm2.py::test_control_instruction_is_appended_after_normalization。
    """
    r = client.post("/normalize", json={"text": "a 30-year-old voice"})
    assert r.json()["normalized"] == "a 三十-year-old voice"  # 本层现状：会转读
    # ③快速路径：无数字文本（含纯标签文本）走 no-op 快速路径，changed=False。
    r2 = client.post("/normalize", json={"text": "[laughing] [sigh]"})
    assert r2.json() == {"normalized": "[laughing] [sigh]", "changed": False}
