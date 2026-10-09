# -*- coding: utf-8 -*-
"""
措辞层 —— 把 facts 变成「人话」。
==================================

🔴 为什么判断（`rules.py`）和措辞（本模块）要分开：

  `rules.py` 只回答「**什么时候说、事实是什么**」，本模块只回答「**怎么说**」。
  分开之后，R2.2 的云润色可以**原样插在这一层的位置**：

      模板（本模块） → 云润色 → 数字白名单 → 失败回落模板
                                          ↑
                        R2.2 只需要替换「模板」这一步，rules 一行不改

  这也是本项目三条原则里「**判断在本地、措辞在云上**」的代码形态。

🔴 本模块的三条硬约束：

 1. **不产出数字**：句子里的每个数字都必须来自 facts。
    靠 `fact_allow()` + `tests/test_phrases.py::test_no_invented_numbers` 守着。
    R2.2 的云润色白名单会**复用同一个 `fact_allow()`** —— 本地与云两条路
    用同一个口径，否则「云编数字」和「模板编数字」就成了两套漏法。
 2. **不做算术**：本层只允许"念出 facts 里的数"，**一个数都不许自己算**。
    差、和、比例、四舍五入全部由 `rules.py` 算好放进 facts。
    理由不是洁癖：R2.2 要用同一份 `fact_allow()` 拦 LLM 编数字，
    而"本地偷偷算一个不在 facts 里的差值"会让那把尺当场失效 ——
    实测就是 `projected - best_actual = 0.65` 被判成"编数字"，
    于是差值被移回 rules（`vs_best_s`）。
 3. **不推断失误模式**：「刹车早一点」「入弯太快」这类处方需要驾驶模型，
    本地给不出。本地只给「哪个弯、亏多少、连续几圈」；
    处方留给云（R2.2）—— 那是它擅长、也是唯一该做的事。
 4. **有长度预算**：见 `MAX_CHARS`。超预算要在**写文案时**砍，**不能运行时截断** ——
    截断会把 "0.37" 切一半成 "0.3"，那正是编数字。测试守着每条一句。
"""

from __future__ import annotations

import re
from typing import Any

from .contract import P_HIGH, P_LOW, P_NORMAL

# 数字白名单用的正则：整数或小数（不做千分位/科学计数 —— 我们不产出那种写法）
NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# 句子长度预算 —— **由 ttl 容量反推**，不另立一套数字。
#
# 🔴 曾经把预算与 `ttl_s` 写成两个独立常数，于是必然分叉：
#    预算允许 24 字（≈5.33 s）而 `ttl_s = 5.0` —— 预算允许的"最坏句子"
#    比 ttl 还长，一旦补上过期检查就会被丢掉。这不是理论问题：
#    `brake_warn` 已经因为同样的分叉在 t_go ≤1.0 的整个窗口静默。
#    现在唯一的"源"是下面的 `TTL_CAP_S`，预算 = 容量 × 语速 × 余量。
#
# 依据是**说话时长**，不是别的：中文 TTS 自然语速约 4.5 字/秒，
# 一句 20 字 ≈ 4.4 秒 —— 这是「副驾说完这句、你还没忘掉上一个弯」的上限。
# 超了这个量，玩家要么听漏后半句（信息白给），要么被迫加速播报（像在赶时间）。
#
# A 档（P_CRITICAL / P_HIGH）**不设预算**：它本来就是 ≤9 字的短句
# （"刹车晚了 12"/"换挡"），弯中禁言还会再压成 ≤4 字的 `short`。
CHAR_PER_S = 4.5

# 每个 B 档 key 的 ttl 容量（秒）—— 预算表的**唯一来源**，也是 `ttl_for` 的输入。
# 值本身是"这条信息在车上还有用多久"，与 `rules.py` 里各规则的语义一致。
TTL_CAP_S: dict[str, float] = {
    # R2.4 圈后**综合建议**：把成绩/最慢段/续航/习惯压成一句。它是最长的一句
    # （可能要带 2~3 条事实），所以容量单独给，比单条 lap_summary 大。
    "lap_advice": 7.0,
    "lap_summary": 5.0,     # 圈后成绩，撑到下一个 delta 播报
    "sector_loss": 5.0,     # 圈后分段，同上
    "next_focus": 6.0,      # 习惯类，最长（"连续 N 圈"值得听完）
    "fuel_range": 6.0,      # 续航类，最长
    "projected_lap": 3.0,   # 预测圈速，下一段就刷新
    "tyre_temp": 4.0,       # 胎温变化慢
    "delta": 2.0,           # 秒级刷新，过期极快
    "apex_slow": 2.0,       # 弯心后 60 m 内有效
    "throttle_late": 2.0,   # 同上
}
# 预算 = 容量 × 语速 × 0.9（留 10% 余量：是"说完"，不是"说到最后一个字就过期"）
BUDGET_HEADROOM = 0.9


def _budget_from_cap(key: str) -> int:
    return int(TTL_CAP_S[key] * CHAR_PER_S * BUDGET_HEADROOM)


# 单句预算（B 档全部键都在表里 —— 这样每个键的预算对应它**自己的** ttl，
# 而不是被同类优先级里最长的那条拖着走）。
MAX_CHARS_OVERRIDE: dict[str, int] = {k: _budget_from_cap(k) for k in TTL_CAP_S}
# 兜底：表外的 B 档 key 走这里，取保守值（= 最短那个容量）。
MAX_CHARS_DEFAULT: dict[int, int] = {
    P_LOW: _budget_from_cap("delta"),
    P_NORMAL: _budget_from_cap("apex_slow"),
}


def fmt_lap_time(seconds: float | None) -> str:
    """92.412 → "1:32.412"。"""
    if not seconds or seconds <= 0:
        return "-"
    m = int(seconds // 60)
    return f"{m}:{seconds - m * 60:06.3f}"


# ===========================================================================
# 数字逐位读音（无线电风格）—— 54 → 五四，去掉 十/百/千 等位值字
# ===========================================================================
#
# 🔴 为什么单独做一层：真实赛道/战机无线电报数都是**逐位读**的——
#    速度「54」念「五四」而不是「五十四」，圈速「1:32.412」念
#    「一分三二秒四一二」。带位值字（十/百/千）的读法在高速口播里
#    反而含糊、还慢半拍。这一层只服务于**语音**（TTS），
#    屏幕上的 `text` 仍保留 54 / 1:32.412 等原样数字便于扫读。
#
# 设计约束（与项目其它硬门一致）：
#   · 只改**独立数字**：前面是空白/标点/中文/句首，或被 +/- 修饰的数字；
#     S2 / T3 这类「带字母的编号标签」里的数字**绝不**动（否则 "S2" 变 "S二"）。
#   · 不改变字符数：五四=2 字、54=2 字；零点三七=4 字、0.37=4 字。
#     所以 ttl 预算、长度审计全部按原 `text` 算就够，这层是「等价替换」。
#   · 圈速单独处理（含冒号），且优先于普通数字，避免冒号被误拆。

_DIGIT_CN = "零一二三四五六七八九"

# 独立数字：前面不能是字母或数字（排除 S2/T3 等标签），可带正负号。
_NUM_RE = re.compile(r"(?<![A-Za-z0-9])([+\-]?)(\d+(?:\.\d+)?)")
# 圈速 M:SS.mmm：同样要求前面不是字母/数字。
_LAPTIME_RE = re.compile(r"(?<![A-Za-z0-9])(\d+):(\d{1,2})\.(\d{1,3})")


def _spell_num(sign: str, num: str) -> str:
    """把一个普通数字（可带符号）转成逐位中文。"""
    out = "正" if sign == "+" else ("负" if sign == "-" else "")
    if "." in num:
        intp, frac = num.split(".")
        out += "".join(_DIGIT_CN[int(c)] for c in intp)
        out += "点" + "".join(_DIGIT_CN[int(c)] for c in frac)
    else:
        out += "".join(_DIGIT_CN[int(c)] for c in num)
    return out


def _spell_lap(minutes: str, secs: str, millis: str) -> str:
    """把圈速 M:SS.mmm 转成「M分SS秒mmm」（逐位，无十位值字）。"""
    m = "".join(_DIGIT_CN[int(c)] for c in minutes)
    s = "".join(_DIGIT_CN[int(c)] for c in secs)
    ms = "".join(_DIGIT_CN[int(c)] for c in millis)
    return f"{m}分{s}秒{ms}"


def spell_digits(text: str) -> str:
    """把播报文本里的阿拉伯数字改成逐位中文（无线电风格）。

    54 → 五四；0.37 → 零点三七；1:32.412 → 一分三二秒四一二；
    +0.37 → 正零点三七；-0.37 → 负零点三七。
    S2 / T3 这类带字母的编号原样保留。

    幂等：对已经没有阿拉伯数字的中文串调用返回原串。
    """
    # 先处理圈速（含冒号），再处理普通带符号数字。
    text = _LAPTIME_RE.sub(
        lambda m: _spell_lap(m.group(1), m.group(2), m.group(3)), text)
    text = _NUM_RE.sub(
        lambda m: _spell_num(m.group(1), m.group(2)), text)
    return text


# ===========================================================================
# 数字白名单 —— 本地模板与云润色**共用**的唯一口径
# ===========================================================================

def numbers_in(text: str) -> list[str]:
    """句子里出现的所有数字（字符串形态）。"""
    return NUM_RE.findall(text)


def fact_allow(facts: Any) -> set[str]:
    """facts 里所有数字的「可能写法」集合 —— 白名单本体。

    为什么要多种写法：同一个值在句子里可能被写成 `0.4` / `0.40` / `0`（取整）；
    圈速 `92.412` 写成 `1:32.412` 之后还会碎出 `1` 与 `32.412` 两个 token。
    白名单宁可宽一点 —— 它要拦的是**凭空出现的数字**（"刹车早 15 米"里的 15），
    不是"同一个值的另一种写法"。
    """
    out: set[str] = set()

    def add(v: Any) -> None:
        if v is None or isinstance(v, bool):
            return                       # bool 是 int 的子类，但 True 不是数字
        if isinstance(v, (int, float)):
            out.add(str(v))
            for p in (0, 1, 2, 3):
                out.add(f"{abs(v):.{p}f}")
            # 圈速/时间类（≥60 s）额外允许 m:ss.mmm 及其碎片
            if 60 <= v < 3600:
                m = int(v // 60)
                rest = v - m * 60
                out.add(f"{m}:{rest:06.3f}")
                out.add(str(m))
                out.add(f"{rest:.3f}")
                # 🔴 `1:09.700` 里的 `09.700` 是**零填充**形态，与 `9.700`
                #    不是同一个 token（`NUM_RE` 会原样抓出前导零）。秒数 <10 的
                #    圈速（1:09.700 / 1:00.000…）会因此被误判成"编数字"——
                #    真车圈速大量落在这一档（真实圈速 60~119.999 s）。
                out.add(f"{rest:06.3f}")
                # 秒数常被写成**两位小数**（`1:23.45` 而不是 `1:23.450`）——
                # 同一个值的另一种精度写法，不是"编数字"。不加这一条，
                # 一句完全正确的云句会因为少打一个 0 被整句丢掉。
                out.add(f"{rest:.2f}")
        elif isinstance(v, str):
            out.update(NUM_RE.findall(v))   # "T3" / "S2" 这类带编号的标签
        # 其余类型（dict/list 在 walk 里递归，其它原样忽略）

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        else:
            add(x)

    walk(facts)
    return out


def invented_numbers(text: str, facts: Any) -> list[str]:
    """句子里**不在** facts 白名单中的数字 —— 非空即"编了数字"。"""
    allow = fact_allow(facts)
    return [n for n in numbers_in(text) if n not in allow]


# ===========================================================================
# 第二、三道闸 —— 白名单查不出的两种毛病
# ===========================================================================
#
# 🔴 为什么白名单不够。真 key 冒烟（2026-10-09）抓到两个缺口，都不是"编数字"：
#
#   ① **丢了主体**。`invented_numbers` 问的是"句中出现但 facts 里没有的数字"，
#      它对**少说了一个数**完全无感。而 `_KEY_HINTS["lap_advice"]` 明确让模型
#      "只挑最要紧的两三件说"，实测两次都把圈速主体（`1:23.450`）丢了 ——
#      本地模板却在 `lap_advice` 里把它标成「① 主体，永远保留」。
#      丢掉的恰恰是圈后车手最想听的那一个数，而且丢得**毫无痕迹**：
#      不记违规、不回落模板、句子读起来还很顺。
#
#   ② **编了建议**。模型输出「注意补油」，而 facts 里 `laps_left=2.3`。
#      `phrases.fuel_range` 的文档写着「🔴 只有**真的不够**（≤1 圈）才给行动
#      建议：不然每圈都在喊"进站"」—— 模型违反了这条，却**一个数字都没编**，
#      白名单照样放行。这条比 ① 危险：它是**指令**，车手可能真去提前进站。
#
# 两条对策的共同思路：把本地模板里已经存在的隐含规则（哪些必须留、哪些不许说）
# 显式化成**代码层可判定**的检查 —— prompt 只是第一道且不可靠的约束。

# 「油见底」阈值。🔴 这里定义、`fuel_range` 与 `lap_advice` 两处模板都引用，
#    下面的处方闸也引用 —— 三处必须同一个数，否则"模板允许但闸门拒绝"。
FUEL_CRIT_LAPS = 1.0      # 真的不够：此时才允许给行动建议（"这圈进站"）
FUEL_LOW_LAPS = 3.0       # 进入观察区：只陈述"油够 N 圈"，不含任何指令

# facts 里有就**必须**在句子里出现的字段。
# 不写 key 白名单，而用「facts 里有没有它」自动判定 —— 将来新增带圈速的 key
# 不会漏。实测只有 `lap_summary` / `lap_advice` 会带上 `lap_time_s`
# （`projected_lap` 用的是 `projected_s`，不在此列）。
MANDATORY_WHEN_PRESENT = ("lap_time_s",)


def lap_time_tokens(lap_time_s: float) -> set[str]:
    """圈速「说出来就算数」的数字写法 —— 与 `fact_allow` 同口径。

    🔴 必须与白名单一致：这边认、白名单不认 → 白名单先把它当"编数字"杀掉，
       这条检查根本没机会跑；反过来则会放过一句白名单拒绝的句子。

    🔴 但白名单里有一个**必须剔除**的成员：裸分位数（`1`）。
       若把它也算成"说了圈速"，那么句子里任何一个 `1`（哪怕来自 `T1`）
       都会让"丢了主体"的句子蒙混过关。真机云句
       「这圈83.45，T1段慢了0.4秒」就同时含 `83.45` 与 `T1` 里的 `1` ——
       有 `T1` 在，"分位在不在"这个判据等于永远为真。

    🔴 真机实测的第二点：模型常把圈速写成**原始秒数**（`83.45`）而不是
       `1:23.450`。白名单认它（那正是 facts 里的值），这里也**必须**认 ——
       否则每一句都被判"丢了主体"，`fallback_ratio` 恒为 1.0，云在空烧钱。
       实测就这样：改正之前 6 圈 6 句全部被丢。提示词可以让它偏好 M:SS.mmm
       （见 prompts），但**闸门只管"在不在"，不管"写成什么形式"**。
    """
    allowed = fact_allow({"lap_time_s": lap_time_s})
    m = int(lap_time_s // 60)
    if m:                                  # 剔除裸分位（见上）
        allowed.discard(str(m))
    return allowed


def missing_mandatory(text: str, facts: Any) -> list[str]:
    """facts 里**必须在场**的内容，句子里却找不到 —— 非空即"丢了主体"。"""
    if not isinstance(facts, dict):
        return []
    toks = set(numbers_in(text))
    missing: list[str] = []
    for name in MANDATORY_WHEN_PRESENT:
        v = facts.get(name)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            continue                      # facts 自己就没有 → 无从要求
        # 目前表里只有 `lap_time_s`（圈速专用读法）。将来加别的字段时，
        # 要按字段选对应的读法 —— 宁可在这里显式分支，也不要拿"圈速的
        # 读法"去判一个速度值。
        allowed = lap_time_tokens(v) if name == "lap_time_s" else {str(v)}
        if not (toks & allowed):
            missing.append(name)
    return missing


# 「处方性」词表：这些是**动作指令**，只有 facts 授权才算合法。
# 每项 = (词, 授权字段, 阈值)：字段缺省或超过阈值 → 说了就是"编建议"。
# 授权条件刻意与本地模板的 `FUEL_CRIT_LAPS` 对齐 —— 模板不敢说的话，
# 云也不许说。
#
# ⚠️ 已知取舍：
#   · 「省油 / 滑行」是这份表里最可能误伤的两项（2.3 圈时说"省着点油"，
#     在人类教练看来不算离谱）。保留是因为它们与「进站」同属**燃油管理
#     指令**，前置条件一样；真嫌误伤多，删掉这两个词即可 —— 判据全在这张表。
#   · 「补油」在中英文赛车语境里有歧义（补油降档 / 补充燃油）。这里按
#     燃油解（facts 里带 `unit=油`）。即便误判，代价也只是回落到本地模板。
_ADVICE_RULES: list[tuple[tuple[str, ...], str, float]] = [
    (("进站", "回站", "维修区", "换胎", "换新胎", "加燃料", "补给",
      "补油", "加油", "省油", "滑行"),
     "laps_left", FUEL_CRIT_LAPS),
]


def invented_advice(text: str, facts: Any) -> list[str]:
    """句子里出现、但 facts **没有授权**的处方性指令 —— 非空即"编了建议"。

    与 `invented_numbers` 的分工：那个管**数字**，这个管**动作**。
    「注意补油」一个数字都没有，白名单永远拦不住它。
    """
    if not isinstance(facts, dict):
        return []
    hits: list[str] = []
    for words, field, limit in _ADVICE_RULES:
        found = [w for w in words if w in text]
        if not found:
            continue
        v = facts.get(field)
        allowed = (isinstance(v, (int, float)) and not isinstance(v, bool)
                   and float(v) <= limit)
        if not allowed:
            hits.extend(found)
    return hits


def speech_s(text: str) -> float:
    """这句念出来要几秒 —— 与 `MAX_CHARS_*` 用同一个语速，所以两者永远不会打架。

    🔴 它的搭档是 `Utterance.ttl_s`：句子比 ttl 长 = 那句还没念完就被判过期。
       契约里写着"过期作废"，而 `ttl_s` 到现在**还没有任何消费者**
       （只有声明与转递）。在这条链路补上过期检查之前，
       至少让 `ttl_s >= speech_s(text)` 成为一条可断言的量 ——
       免得将来加了过期检查，第一个栽的就是被 R2.1 加长过的这几句。
       见 tests/test_rules.py::TestTtlFitsSpeech。
    """
    return len(text) / CHAR_PER_S


def over_budget(text: str, priority: int, key: str = "") -> bool:
    """超出该句的长度预算？A 档（无预算）恒 False。

    🔴 **这是测试用的审计口，不是运行时的截断器。**
       超预算要在**写文案时**把句子改短，绝不能在运行时截断 ——
       截断会把 "0.37" 切成 "0.3"，那本身就是"编数字"。

       ⚠️ 别再在这里复制一份预算查找逻辑 —— `char_budget` 负责别名
       （`tyre_cold`→`tyre_temp`）等归一化，这里只管"超没超"。
       曾经两处各写一套，`tyre_cold`（9 字）对不上 `tyre_temp`（16 字）的
       预算而被误报 —— 真车回放抓出来的。
    """
    lim = char_budget(priority, key)
    return lim is not None and len(text) > lim


def char_budget(priority: int, key: str = "") -> int | None:
    """该句的长度预算（A 档无预算 → None）。与 `over_budget` 同一口径。

    🔴 key 匹配是"前缀/别名"式的：`tyre_cold` / `tyre_hot` 两个播报 key
    共享 `tyre_temp` 的预算（都是胎温规则）。硬记"必须叫同一个名字"
    会在 `over_budget('轮胎太凉，抓地不够', P_NORMAL, 'tyre_cold')` 上
    错用 P_NORMAL 兜底（8 字）而误报超预算 —— 真车回放抓出来的。
    """
    if key:
        if key in MAX_CHARS_OVERRIDE:
            return MAX_CHARS_OVERRIDE[key]
        # 别名映射：胎温规则的两个 key 都算 tyre_temp
        if key.startswith("tyre_") and "tyre_temp" in MAX_CHARS_OVERRIDE:
            return MAX_CHARS_OVERRIDE["tyre_temp"]
    return MAX_CHARS_DEFAULT.get(priority)


def ttl_for(text: str, priority: int, key: str = "", *,
            floor_s: float = 0.0, margin_s: float = 0.2) -> float:
    """给一句话算 `ttl_s`，保证**这句话一定念得完**。

    🔴 为什么需要它：`ttl_s` 与长度预算本来是两套独立写的数字，于是必然分叉 ——
     `MAX_CHARS_OVERRIDE["lap_summary"] = 24` 字 ≈ 5.33 s，而 `ttl_s = 5.0`：
       预算允许的那个"最坏句子"比 ttl **还长**，一旦补上过期检查就会被丢掉。
       这不是理论问题：`brake_warn` 已经因为同样的分叉在 `t_go ≤1.0`
       的整个窗口静默（见 `rules._brake_warn`）。

    规则：`ttl = max(floor_s, speech_s(text) + margin_s, 预算对应的时长)`。
      - `floor_s` 给"信息有效期"（如刹车预告要活到进刹车区）；
      - `margin_s` 留一点播完的缓冲（0.2 s ≈ 1 个字）；
      - 预算那一项保证**哪怕文案将来加长到上限**也还装得下 ——
        这样"改文案"不再需要同步改 ttl，分叉从源头消失。
    """
    need = speech_s(text) + margin_s
    lim = char_budget(priority, key)
    if lim is not None:
        need = max(need, lim / CHAR_PER_S)
    return round(max(floor_s, need), 2)


# ===========================================================================
# 逐句措辞
# ===========================================================================
#
# 每个函数都是「facts dict → str」的纯函数：不读全局、不碰状态、不看时间。
# 这样它们才能被 ① 本地直接调用 ② 测试逐条喂 facts ③ R2.2 在云失败时回落。
#
# 🔴 输入 schema 就是**契约**（R2.2 的 prompt 也吃同一份），改动必须同步：
#
#   lap_advice     {lap_time_s?, vs_ref_s?, unit?, laps_left?, sector?,
#                   loss_s?, focus_label?, focus_laps?, focus_loss_s?}
#   lap_summary    {lap_time_s, vs_ref_s?}
#   sector_loss    {sector, loss_s, gain_s?}
#   projected_lap  {projected_s}
#   fuel_range     {unit, laps_left}
#   next_focus     {label, laps, median_loss_s}
#
# 注意：这些 key（facts 的字段名）与 `Utterance.evidence` 里的一致 ——
# 规则把 evidence 原样传进来，**不另造一套中间结构**（少一层就少一处会分叉）。


def lap_summary(f: dict[str, Any]) -> str:
    """圈后成绩：「1:32.412，慢 0.37」。

    🔴 **故意不合并「最亏在 S2」**（虽然那样信息更全）：
       圈速 + delta + 段损失 = 26 字，超 P_LOW 预算（24）会被听漏；
       而两条 15~17 字的短句在嘈杂环境里更容易听全，gate 也有每圈句数上限。
       「把圈速/段/油量/习惯合成一句综合建议」是 **R2.4** 的活 ——
       那里由云做措辞压缩，不占本地模板的长度预算。
    """
    txt = fmt_lap_time(f.get("lap_time_s"))
    d = f.get("vs_ref_s")
    if d is not None and abs(d) >= 0.05:
        txt += f"，{'慢' if d > 0 else '快'} {abs(d):.2f}"
    return txt


def sector_loss(f: dict[str, Any]) -> str:
    """圈后分段：「S2 慢 0.31，还差 0.80」。

    「还差 0.80」= 各段最好值拼起来的理论最快圈比你的实际最快圈还快多少
    （原文案是「潜在 0.80」—— 工程词，开车的人不会这么说）。
    """
    txt = f"S{f['sector']} 慢 {f['loss_s']:.2f}"
    g = f.get("gain_s")
    if g is not None and g >= 0.15:
        txt += f"，还差 {g:.2f}"
    return txt


def projected_lap(f: dict[str, Any]) -> str:
    """预测圈速：「预计 1:22.800」。

    🔴 **这里试过加一句比较，然后撤回了** —— 记下来免得有人再试一遍：

     - 拿**参考圈**做基准：`projected = ref_lap + delta`，所以
       `projected - ref_lap` 恒等于 `delta` —— 而 `delta` 已经在同一圈里
       单独播报过了。加出来是纯重复。
     - 拿**你自己最好圈**（`best_actual_s`）做基准：数字是算得出来的，
       但它和同一圈里按参考圈口径说的 `lap_summary`（"慢 0.50"）
       **会在同一次播报里互相打架**（"慢 0.50" 之后紧接 "还快 2.09"）。
       实测就是这样：合成场次里 ref 是理想剖面、best_actual 是真实圈，
       两者差 2 秒，于是同一圈先说你慢、再说你快。
       **同一圈里出现两个基准 = 玩家不再相信任何一句。**

     结论：预测就报预测值本身。要"够不够好"的答案，`delta` 与
     `lap_summary` 已经用**同一个基准**（参考圈）回答过了。
    """
    return f"预计 {fmt_lap_time(f.get('projected_s'))}"


def fuel_range(f: dict[str, Any]) -> str:
    """续航：「油还够 2.4 圈」/「油只够 0.8 圈，这圈进站」。

    `unit` 由规则判定（电车说"电量"、油车说"油"）—— 说错一次就没人信了。
    🔴 只有**真的不够**（≤ `FUEL_CRIT_LAPS`）才给行动建议：不然每圈都在喊
       "进站"，那是狼来了。`invented_advice` 把这个阈值也用在云句上 ——
       模板不敢说的话，云也不许说。
    """
    unit = f.get("unit") or "油"
    left = float(f.get("laps_left") or 0.0)
    if left <= FUEL_CRIT_LAPS:
        return f"{unit}只够 {left:.1f} 圈，这圈进站"
    return f"{unit}还够 {left:.1f} 圈"


def next_focus(f: dict[str, Any]) -> str:
    """下一圈重点：「T3 连续 3 圈慢 0.40，注意刹车点」。

    🔴 与原文案（「下一圈重点：T3，最近亏 0.40」）的三点区别：

    1. 说「**连续 3 圈**」而不是「最近」—— 不报样本数的话，玩家不知道
       这是偶发还是习惯；而"习惯"正是这条规则和 `sector_loss`（单圈最慢段）
       的**全部区别**。
    2. 去掉了「下一圈重点：」这个前缀（8 个字），换成句尾的行动提示 ——
       前缀是主持人腔，句尾才是副驾会说的。
    3. ⚠️「注意刹车点」是**证据驱动的固定提示**，不是处方：
       它只在 facts 里 `ls_share`（刹车区窗口内的损失占比）≥0.5 时出现 ——
       `brake_late`/`brake_warn` 两条规则已经证明这个证据在本地是有的。
       证据不足时**只给事实，不给提示**：本模块不推断失误模式，
       没证据的处方是猜。真正"该怎么改"的处方留给 R2.2 的云。
    """
    txt = f"{f['label']} 连续 {f['laps']} 圈慢 {f['median_loss_s']:.2f}"
    if (f.get("ls_share") or 0.0) >= 0.5:
        txt += "，注意刹车点"
    return txt


def lap_advice(f: dict[str, Any]) -> str:
    """圈后**综合建议**（R2.4）：把 成绩 / 最慢段 / 续航 / 习惯 压成**一句**。

    🔴 为什么合并落在这一句、而不是让四条各说各的：
       `lap_summary` 的文档里记着"**故意不合并**"—— 四合一 26 字超 P_LOW 预算、
       嘈杂环境听不全。R2.4 的解法是把合并交给这一条**更长预算**的综合句
       （`TTL_CAP_S["lap_advice"] = 7.0`），并按优先级**贪心装填**：
       超预算就从队尾丢**整条**事实，而不是把数字截断（截断=编数字，见模块头）。

    优先级（安全 > 可执行 > 参考）：
      ① 圈速主体  ② 油见底（≤`FUEL_CRIT_LAPS`，含"进站"）  ③ 习惯弯
      ④ vs_ref 差  ⑤ 最慢段  ⑥ 一般续航（`FUEL_CRIT_LAPS` < left ≤ `FUEL_LOW_LAPS`）

    🔴 ①「永远保留」这条**不是只靠这里的顺序实现的** —— 顺序只在超预算时
       决定谁先让位，而云润色根本不看这个函数。真正让它在云端也成立的是
       `missing_mandatory()`：facts 里有 `lap_time_s` 而云句里没提 → 整句丢弃。
       两处必须一起改，改一处等于没改。

    🔴 圈速**主体**与 **vs_ref 差**拆成两段（`1:32.412` 与 `慢 0.37`）：
       否则"主体+差"（15 字）会先把预算吃满，把更值钱的习惯弯挤掉。
       拆开后习惯弯能和主体并排（8+15=23 ≤ 预算），差值反而先让位。

    数字全部来自 facts —— 与本地模板、云润色**共用同一份白名单**。
    """
    budget = char_budget(P_NORMAL, "lap_advice") or 28
    slots: list[str] = []

    t = f.get("lap_time_s")
    if t:
        slots.append(fmt_lap_time(t))          # ① 主体，永远保留

    left = f.get("laps_left")
    unit = f.get("unit") or "油"
    fuel_crit = left is not None and float(left) <= FUEL_CRIT_LAPS
    if fuel_crit:
        # 见底是安全信息：占高优先级，绝不被后面的低优先级挤掉。
        slots.append(f"{unit}只够 {float(left):.1f} 圈，这圈进站")

    lab = f.get("focus_label")
    if lab:
        slots.append(f"{lab} 连续 {f['focus_laps']} 圈慢 "
                     f"{float(f['focus_loss_s']):.2f}")

    d = f.get("vs_ref_s")
    if d is not None and abs(d) >= 0.05 and t:
        slots.append(f"{'慢' if d > 0 else '快'} {abs(d):.2f}")

    sec, loss = f.get("sector"), f.get("loss_s")
    if sec is not None and loss is not None:
        slots.append(f"S{sec} 慢 {float(loss):.2f}")

    if left is not None and not fuel_crit and float(left) <= FUEL_LOW_LAPS:
        slots.append(f"{unit}够 {float(left):.1f} 圈")

    out: list[str] = []
    for p in slots:
        if out and len("，".join(out + [p])) > budget:
            continue
        out.append(p)
    return "，".join(out)


# 🔴 有措辞层的 key（= R2.2 里"值得花一次云调用"的候选集）。
#    不在表里的 key 保持原样：A 档短句、以及本身已经够口语的几句
#    （"出界了，回到赛道" / "四轮打滑" / "换挡" / "给油晚了" …）。
#    这张表是 R2.2 的唯一入口 —— 云润色只走这里列出的 key。
RENDERERS = {
    "lap_advice": lap_advice,
    "lap_summary": lap_summary,
    "sector_loss": sector_loss,
    "projected_lap": projected_lap,
    "fuel_range": fuel_range,
    "next_focus": next_focus,
}


def render(key: str, facts: dict[str, Any]) -> str:
    """按 key 出句子。表里没有的 key 直接报错 —— 宁可炸在测试里，
    也不要在车上静默产出一句没经过措辞层的台词。"""
    return RENDERERS[key](facts)