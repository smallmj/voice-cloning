"""Unified text normalization layer (数字 / 日期 / 时间 / 金额 / 单位 / 混读).

Every generation path MUST route its text through :func:`normalize_text`
before handing it to any engine adapter — the layer is applied centrally in
``main.create_generation`` so adapters cannot bypass it. Pure function, no
side effects: text already in canonical form comes back byte-identical.

Polyphone disambiguation (拼音标注) is explicitly OUT of scope here; engines
declare that capability separately.
"""

from __future__ import annotations

import re

_DIGITS = "零一二三四五六七八九"
_UNITS_INT = ["", "十", "百", "千"]
_BIG_UNITS = ["", "万", "亿", "万亿"]

_UNIT_WORDS = {
    "km": "公里",
    "kg": "千克",
    "m": "米",
    "cm": "厘米",
    "mm": "毫米",
    "g": "克",
    "t": "吨",
    "l": "升",
    "ml": "毫升",
    "s": "秒",
    "min": "分钟",
    "h": "小时",
}


def _digit_chinese(d: str) -> str:
    return "".join(_DIGITS[int(ch)] for ch in d)


def _int_chinese(n: int) -> str:
    """Read a non-negative integer in Chinese (万/亿 scale groups)."""
    if n == 0:
        return "零"
    if n < 10:
        return _DIGITS[n]

    # Split into 4-digit groups from the lowest.
    groups: list[int] = []
    while n > 0:
        groups.append(n % 10000)
        n //= 10000

    parts: list[str] = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        parts.append(_four_digit_chinese(g) + _BIG_UNITS[i])
    return _join_groups(parts)


def _four_digit_chinese(g: int) -> str:
    """Read a 0 < g < 10000 group.「一千零五」 style zero filling."""
    thousands, g = divmod(g, 1000)
    hundreds, g = divmod(g, 100)
    tens, ones = divmod(g, 10)
    out = ""
    if thousands:
        out += _DIGITS[thousands] + "千"
    if hundreds:
        out += _DIGITS[hundreds] + "百"
    elif out and (tens or ones):
        out += "零"
    if tens:
        if thousands == 0 and hundreds == 0 and tens == 1:
            # 十五, not 一十五, for bare 10-19.
            out += "十"
        else:
            out += _DIGITS[tens] + "十"
    elif out and ones:
        out += "零"
    if ones:
        out += _DIGITS[ones]
    return out


def _join_groups(parts: list[str]) -> str:
    # Group boundaries never need a zero: 一万二千 not 一万零二千.
    return "".join(parts)


def _decimal_chinese(int_part: str, frac_part: str) -> str:
    return _int_chinese(int(int_part)) + "点" + _digit_chinese(frac_part)


def _money_chinese(int_part: str, frac_part: str, major: str, cents: tuple[str, str] | None) -> str:
    """major: major-unit word. cents: (first-2-digit unit, second unit) or None."""
    out = _int_chinese(int(int_part)) + major
    if not frac_part:
        return out
    if cents is None:
        return out + _decimal_chinese("0", frac_part)[1:]  # drop leading 零点
    jiao = int(frac_part[0])
    fen = int(frac_part[1]) if len(frac_part) > 1 else 0
    if jiao:
        out += _DIGITS[jiao] + cents[0]
    elif fen:
        out += "零"
    if fen:
        out += _DIGITS[fen] + cents[1]
    return out


# --- rule applications, ordered most-specific first -------------------------

_RE_DATE_FULL = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_RE_YEAR = re.compile(r"(\d{4})\s*年")
_RE_MONTH_DAY = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_RE_TIME = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")
_RE_MONEY_YEN = re.compile(r"[¥￥]\s*(\d+)(?:\.(\d+))?")
_RE_MONEY_USD = re.compile(r"\$\s*(\d+)(?:\.(\d+))?")
_RE_MONEY_CNY = re.compile(r"(\d+)(?:\.(\d+))?\s*元")
_RE_PERCENT = re.compile(r"(\d+)(?:\.(\d+))?\s*[%％]")
_RE_TEMP_NEG = re.compile(r"(-?\s?\d+)(?:\.(\d+))?\s*(?:°\s*C|℃|摄氏度)")
# Longest first so e.g. "mm" is never eaten by "m".
_UNIT_PATTERN = "|".join(sorted(_UNIT_WORDS, key=len, reverse=True))
_RE_UNIT = re.compile(r"(\d+)(?:\.(\d+))?\s*(" + _UNIT_PATTERN + r")(?![a-z])", re.IGNORECASE)
_RE_COMMA_NUM = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_RE_DECIMAL = re.compile(r"(\d+)\.(\d+)")
_RE_INT = re.compile(r"\d+")
_RE_CJK_UNIT = re.compile(r"(\d+)\s*(个|条|本|只|层|次|遍|位|名|款|件|岁)")
# Bare "N分" only reads as minutes inside an explicit duration ("5分20秒");
# standalone "95分" is a score and must stay plain.
_RE_MINUTE = re.compile(r"(\d+)\s*分(?=\s*\d+\s*秒)")

# --- normalization exemptions (spec #68, user stories 8/9) ------------------
# Engine marker syntaxes that CONTAIN digits must survive normalization
# byte-for-byte:
#   - MiniMax pause marker  `<#x#>`  (x seconds, 0.01–99.99, two decimals)
#   - IndexTTS-2.5 pinyin annotation  `<汉字|PINYIN2>`  (tone digits)
# They are stripped before the rule pass and restored after (placeholder
# bytes carry no digits, so no rule can touch them); every OTHER number in
# the text is still rewritten as usual.
_RE_PAUSE_MARKER = re.compile(r"<#\d+(?:\.\d+)?#>")
_RE_PINYIN_ANNO = re.compile(r"<[^<>#\n]{1,16}\|[^\s<>]{1,32}>")
# Placeholder: PUA chars only — never a digit (a digit index would itself be
# rewritten by the number rules) and never matched by any rule below.
_PLACEHOLDER = "\ue000{}\ue001"


def _placeholder(index: int) -> str:
    return _PLACEHOLDER.format(chr(0xE100 + index))


def _protect_markers(text: str) -> tuple[str, list[str]]:
    """Strip protected marker spans, returning placeholder text + the spans
    in order of appearance."""
    spans: list[str] = []

    def keep(m: re.Match) -> str:
        spans.append(m.group(0))
        return _placeholder(len(spans) - 1)

    text = _RE_PAUSE_MARKER.sub(keep, text)
    text = _RE_PINYIN_ANNO.sub(keep, text)
    return text, spans


def _restore_markers(text: str, spans: list[str]) -> str:
    for i, span in enumerate(spans):
        text = text.replace(_placeholder(i), span)
    return text


def _apply_date_full(text: str) -> str:
    def sub(m: re.Match) -> str:
        return (
            _digit_chinese(m.group(1))
            + "年"
            + _int_chinese(int(m.group(2)))
            + "月"
            + _int_chinese(int(m.group(3)))
            + "日"
        )

    return _RE_DATE_FULL.sub(sub, text)


def _apply_year(text: str) -> str:
    return _RE_YEAR.sub(lambda m: _digit_chinese(m.group(1)) + "年", text)


def _apply_month_day(text: str) -> str:
    def sub(m: re.Match) -> str:
        return _int_chinese(int(m.group(1))) + "月" + _int_chinese(int(m.group(2))) + "日"

    return _RE_MONTH_DAY.sub(sub, text)


def _minsec_chinese(digits: str) -> str:
    """Time component reading: 30 -> 三十, 05 -> 零五, 00 -> 零零."""
    return _digit_chinese(digits) if int(digits) < 10 else _int_chinese(int(digits))


def _apply_time(text: str) -> str:
    def sub(m: re.Match) -> str:
        out = _int_chinese(int(m.group(1))) + "点"
        out += _minsec_chinese(m.group(2)) + "分"
        if m.group(3):
            out += _minsec_chinese(m.group(3)) + "秒"
        return out

    return _RE_TIME.sub(sub, text)


def _apply_money(text: str) -> str:
    text = _RE_MONEY_YEN.sub(lambda m: _money_chinese(m.group(1), m.group(2) or "", "元", ("角", "分")), text)
    text = _RE_MONEY_USD.sub(lambda m: _money_chinese(m.group(1), "", "美元", None) + _usd_cents(m.group(2) or ""), text)
    return _RE_MONEY_CNY.sub(lambda m: _money_chinese(m.group(1), m.group(2) or "", "元", ("角", "分")), text)


def _usd_cents(frac: str) -> str:
    if not frac:
        return ""
    cents = int((frac + "00")[:2])
    return _int_chinese(cents) + "美分"


def _apply_percent(text: str) -> str:
    def sub(m: re.Match) -> str:
        out = _int_chinese(int(m.group(1)))
        if m.group(2):
            out += "点" + _digit_chinese(m.group(2))
        return "百分之" + out

    return _RE_PERCENT.sub(sub, text)


def _apply_temperature(text: str) -> str:
    def sub(m: re.Match) -> str:
        val = m.group(1).replace(" ", "")
        neg = val.startswith("-")
        out = _int_chinese(abs(int(val)))
        if m.group(2):
            out += "点" + _digit_chinese(m.group(2))
        return ("零下" if neg else "") + out + "摄氏度"

    return _RE_TEMP_NEG.sub(sub, text)


def _apply_units(text: str) -> str:
    def sub(m: re.Match) -> str:
        out = _int_chinese(int(m.group(1)))
        if m.group(2):
            out += "点" + _digit_chinese(m.group(2))
        return out + _UNIT_WORDS[m.group(3).lower()]

    return _RE_UNIT.sub(sub, text)


def _apply_cjk_units(text: str) -> str:
    # Small counts before common Chinese measure words read as plain numbers.
    return _RE_CJK_UNIT.sub(lambda m: _int_chinese(int(m.group(1))) + m.group(2), text)


def _apply_remaining_numbers(text: str) -> str:
    text = _RE_DECIMAL.sub(lambda m: _decimal_chinese(m.group(1), m.group(2)), text)

    def sub_int(m: re.Match) -> str:
        digits = m.group(0)
        # Long digit runs are read digit-by-digit: exactly 11 digits is
        # treated as a mobile phone number. Other lengths read as quantities.
        if len(digits) == 11:
            return _digit_chinese(digits)
        return _int_chinese(int(digits))

    return _RE_INT.sub(sub_int, text)


def normalize_text(text: str) -> str:
    """Normalize mixed Chinese text for TTS engines. Idempotent, side-effect
    free on canonical input.

    Spec #68: marker syntaxes that carry digits (`<#2.5#>` MiniMax pauses,
    `<汉字|XING2>` IndexTTS pinyin annotations) are exempt — they pass
    through byte-identical while every other number is still rewritten."""
    text, protected = _protect_markers(text)
    if not any(ch.isdigit() for ch in text):
        return text if not protected else _restore_markers(text, protected)

    text = _RE_COMMA_NUM.sub("", text)  # 1,234 -> 1234
    text = _apply_temperature(text)  # before units: ℃/°C special-cased
    text = _apply_date_full(text)
    text = _apply_time(text)  # before generic numbers (colon pattern)
    text = _apply_money(text)
    text = _apply_percent(text)
    text = _apply_month_day(text)
    text = _apply_year(text)  # bare year after full date
    text = _apply_units(text)
    text = _RE_MINUTE.sub(lambda m: _int_chinese(int(m.group(1))) + "分钟", text)
    text = _apply_cjk_units(text)
    text = _apply_remaining_numbers(text)
    return _restore_markers(text, protected)


def normalize_with_flag(text: str) -> dict:
    """Shared result shape used by both the preview endpoint and generation."""
    normalized = normalize_text(text)
    return {"normalized": normalized, "changed": normalized != text}
