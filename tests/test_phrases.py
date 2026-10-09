# -*- coding: utf-8 -*-
"""措辞层（R2.1）—— 验收的不是"好不好听"，是三条**可判定**的硬门。

 1. 数字只能来自 facts（否则就是"编数字"，本地和云两条路都不允许）
 2. 句子长度不超预算（超了要在写文案时改短，不能运行时截断）
 3. facts 不全会退化成短句，不能崩、不能吐 "-" 之外的怪东西

🔴 这组测试是 R2.2 云润色的**同一把尺**：`invented_numbers` 与 `over_budget`
   会被 prompt 后的校验直接复用。所以这里的用例就是将来"云编数字"的用例 ——
   只不过那时喂进来的是模型返回的句子。
"""
from __future__ import annotations

import pytest

from gt7coach import phrases
from gt7coach.phrases import (MAX_CHARS_DEFAULT, MAX_CHARS_OVERRIDE,
                              fact_allow, fmt_lap_time, invented_numbers,
                              numbers_in, over_budget, render)


# ===========================================================================
# 数字白名单 —— 全项目的唯一口径
# ===========================================================================

class TestFactAllow:
    def test_plain_number_variants(self):
        a = fact_allow({"x": 0.4})
        assert "0.4" in a and "0.40" in a and "0" in a

    def test_lap_time_splits_into_readable_form(self):
        """92.412 在句子里会是 "1:32.412"，碎成 1 与 32.412 两个 token。"""
        a = fact_allow({"lap_time_s": 92.412})
        assert "1:32.412" in a and "1" in a and "32.412" in a

    def test_walks_nested_structures(self):
        """facts 是嵌套的（sector_loss 的 best_each_s 是列表）。"""
        a = fact_allow({"sector": 2, "sectors": [31.0, 41.2],
                        "meta": {"lap": 5}})
        assert {"2", "31.0", "41.2", "5"} <= a

    def test_labels_contribute_their_number(self):
        """`T3` / `S2` 这类带编号的标签里那个数字也算"来自 facts"。"""
        a = fact_allow({"label": "T3"})
        assert "3" in a

    def test_bool_is_not_a_number(self):
        """True 是 int 的子类，但它不是数字 —— 混进来会让 "1" 变成万能钥匙。"""
        a = fact_allow({"ok": True})
        assert "1" not in a and "True" not in a

    def test_none_ignored(self):
        a = fact_allow({"vs_ref_s": None, "lap_time_s": 92.412})
        assert "0" not in a or "92.412" in a
        assert "None" not in a


class TestInventedNumbers:
    def test_clean_sentence_has_none(self, ):
        f = {"lap_time_s": 92.412, "vs_ref_s": 0.37}
        assert invented_numbers("1:32.412，慢 0.37", f) == []

    def test_catches_planted_number(self):
        """这是 R2.2 的核心护栏：模型/模板凭空写一个数就抓出来。"""
        f = {"laps_left": 2.4}
        assert invented_numbers("油还够 2.4 圈，还能再跑 3 圈", f) == ["3"]

    def test_catches_rounded_up_number(self):
        f = {"loss_s": 0.42}
        assert invented_numbers("慢 0.5", f) == ["0.5"], "四舍五入过的数字也是编的"

    def test_numbers_in_lists(self):
        assert numbers_in("T3 连续 3 圈慢 0.42") == ["3", "3", "0.42"]


# ===========================================================================
# 长度预算
# ===========================================================================

class TestBudget:
    def test_a_grade_has_no_budget(self):
        from gt7coach.contract import P_CRITICAL, P_HIGH
        long_txt = "刹车晚了 12 米" * 5
        assert not over_budget(long_txt, P_CRITICAL)
        assert not over_budget(long_txt, P_HIGH)

    def test_b_grade_budget_applies(self):
        from gt7coach.contract import P_LOW
        lim = MAX_CHARS_DEFAULT[P_LOW]
        assert not over_budget("字" * lim, P_LOW)
        assert over_budget("字" * (lim + 1), P_LOW)

    def test_override_bumps_up_for_rich_keys(self):
        from gt7coach.contract import P_LOW, P_NORMAL
        assert MAX_CHARS_OVERRIDE["lap_summary"] > MAX_CHARS_DEFAULT[P_LOW]
        assert not over_budget("1:32.412，慢 0.37", P_LOW, key="lap_summary")


# ===========================================================================
# 逐句措辞
# ===========================================================================

class TestLapSummary:
    def test_basic(self):
        assert phrases.lap_summary({"lap_time_s": 92.412}) == "1:32.412"

    def test_slower_includes_delta_without_engineering_words(self):
        t = phrases.lap_summary({"lap_time_s": 92.782, "vs_ref_s": 0.37})
        assert t == "1:32.782，慢 0.37"
        assert "参考" not in t

    def test_faster(self):
        t = phrases.lap_summary({"lap_time_s": 91.9, "vs_ref_s": -0.5})
        assert "快 0.50" in t

    def test_tiny_delta_dropped(self):
        """差值 <0.05 是噪声（采样精度就 0.001，说话不必带它）。"""
        assert phrases.lap_summary({"lap_time_s": 92.412,
                                    "vs_ref_s": 0.02}) == "1:32.412"

    def test_missing_time_is_dash(self):
        assert phrases.lap_summary({}) == "-"

    def test_within_budget(self):
        t = phrases.lap_summary({"lap_time_s": 92.782, "vs_ref_s": 0.37})
        assert not over_budget(t, 3, key="lap_summary")


class TestSectorLoss:
    def test_basic(self):
        assert phrases.sector_loss({"sector": 2, "loss_s": 0.31}) == "S2 慢 0.31"

    def test_gain_uses_daily_wording(self):
        t = phrases.sector_loss({"sector": 2, "loss_s": 0.31, "gain_s": 0.8})
        assert t == "S2 慢 0.31，还差 0.80"
        assert "潜在" not in t

    def test_small_gain_dropped(self):
        """gain 小于阈值时不说 —— "还差 0.02"没有可操作性，只是噪音。"""
        t = phrases.sector_loss({"sector": 1, "loss_s": 0.3, "gain_s": 0.02})
        assert t == "S1 慢 0.30"

    def test_within_budget(self):
        t = phrases.sector_loss({"sector": 2, "loss_s": 0.31, "gain_s": 0.8})
        assert not over_budget(t, 2, key="sector_loss")


class TestProjectedLap:
    def test_basic(self):
        assert phrases.projected_lap({"projected_s": 82.8}) == "预计 1:22.800"

    def test_no_comparison_ever(self):
        """🔴 预测句**只报预测值**，已经试过加比较并撤回了。

        账（见 phrases.projected_lap 文档）：
          · 按参考圈口径 → 差值恒等于 `delta`，而 delta 已单独播报 ⇒ 纯重复
          · 按自己最好圈口径 → 与同圈的 `lap_summary`（参考圈口径）
            在同一段播报里互相打架（先说"慢 0.50"、再说"还快 2.09"）
        所以这里断言：**给了任何比较用的 facts 也不许体现在文案里**。
        """
        assert phrases.projected_lap({"projected_s": 83.45}) == "预计 1:23.450"
        assert phrases.projected_lap({"projected_s": 83.45,
                                      "vs_best_s": 0.65}) == "预计 1:23.450"
        assert phrases.projected_lap({"projected_s": 83.45,
                                      "best_actual_s": 82.8}) == "预计 1:23.450"

    def test_no_best_actual_degrades(self):
        """还没跑出任何完整圈时不硬凑比较 —— 只给预测值。"""
        assert phrases.projected_lap({"projected_s": 82.8}) == "预计 1:22.800"

    def test_within_budget(self):
        assert not over_budget("预计 1:23.450", 3, key="projected_lap")


class TestFuelRange:
    def test_normal(self):
        assert phrases.fuel_range({"unit": "油", "laps_left": 2.4}) == "油还够 2.4 圈"

    def test_electric_says_battery(self):
        assert phrases.fuel_range({"unit": "电量",
                                   "laps_left": 2.4}) == "电量还够 2.4 圈"

    def test_truly_low_advises_pit(self):
        t = phrases.fuel_range({"unit": "油", "laps_left": 0.8})
        assert t == "油只够 0.8 圈，这圈进站"

    def test_boundary_one_lap_advises(self):
        assert "进站" in phrases.fuel_range({"unit": "油", "laps_left": 1.0})

    def test_just_above_one_lap_is_quiet(self):
        """1.1 圈不提进站 —— 阈值卡在这里就是为了不说"狼来了"。"""
        assert "进站" not in phrases.fuel_range({"unit": "油", "laps_left": 1.1})

    def test_default_unit(self):
        assert phrases.fuel_range({"laps_left": 2.0}).startswith("油")

    def test_within_budget(self):
        t = phrases.fuel_range({"unit": "油", "laps_left": 0.8})
        assert not over_budget(t, 2, key="fuel_range")


class TestNextFocus:
    def test_basic_reports_sample_size(self):
        t = phrases.next_focus({"label": "T3", "laps": 3, "median_loss_s": 0.42})
        assert t == "T3 连续 3 圈慢 0.42"
        assert "最近" not in t, "「最近」传不出『这是习惯』这个意思"

    def test_hint_only_with_brake_zone_evidence(self):
        t = phrases.next_focus({"label": "T3", "laps": 3, "median_loss_s": 0.42,
                                "ls_share": 0.72})
        assert t == "T3 连续 3 圈慢 0.42，注意刹车点"

    def test_weak_evidence_no_hint(self):
        t = phrases.next_focus({"label": "T3", "laps": 3, "median_loss_s": 0.42,
                                "ls_share": 0.31})
        assert "刹车点" not in t

    def test_within_budget(self):
        t = phrases.next_focus({"label": "T12", "laps": 12,
                                "median_loss_s": 1.234, "ls_share": 0.9})
        assert not over_budget(t, 2, key="next_focus")


# ===========================================================================
# 三条硬门（**覆盖全表**，新增一句忘了改就地失败）
# ===========================================================================

# 每种 key 的「典型最饱满 facts」—— 句子最长的那种输入。
# 🔴 新增 key 必须在这个表里加一行，否则 test_every_renderer_* 会因为
#    "RENDERERS 里有没被覆盖的 key"而失败（这正是我们要的提醒）。
RICH_CASES: dict[str, dict] = {
    "lap_summary": {"lap_time_s": 92.782, "vs_ref_s": 0.37,
                    "ref_lap_time_s": 92.412},
    "sector_loss": {"sector": 3, "loss_s": 0.31, "gain_s": 0.8,
                    "sectors": [33.4, 36.0, 31.6]},
    "projected_lap": {"projected_s": 83.45, "ref_lap_time_s": 82.9},
    "fuel_range": {"unit": "电量", "laps_left": 0.8, "per_lap": 8.0},
    "next_focus": {"label": "T12", "laps": 12, "median_loss_s": 1.234,
                   "ls_share": 0.9, "recent": [1.2, 1.3]},
}

PRIO = {"lap_summary": 3, "sector_loss": 2, "projected_lap": 3,
        "fuel_range": 2, "next_focus": 2}


def test_every_renderer_has_a_rich_case():
    assert set(RICH_CASES) == set(phrases.RENDERERS), \
        "RENDERERS 与 RICH_CASES 必须一一对应（新增措辞要同时加用例）"


@pytest.mark.parametrize("key", sorted(RICH_CASES))
def test_no_invented_numbers(key):
    """**硬门 1**：最饱满的 facts 也不允许句子里出现 facts 之外的数字。"""
    facts = RICH_CASES[key]
    txt = render(key, facts)
    bad = invented_numbers(txt, facts)
    assert not bad, f"{key} 编了数字 {bad}：{txt}"


@pytest.mark.parametrize("key", sorted(RICH_CASES))
def test_within_budget_hard_gate(key):
    """**硬门 2**：最长的那句也要在预算内（超了要改文案，不能运行时截断）。"""
    txt = render(key, RICH_CASES[key])
    assert not over_budget(txt, PRIO[key], key=key), \
        f"{key} 超预算 {len(txt)} 字：{txt}"


@pytest.mark.parametrize("key", sorted(RICH_CASES))
def test_renderer_is_pure_and_consistent(key):
    """**硬门 3**：同一份 facts 两次渲染必须一模一样（纯函数，无隐藏状态）。"""
    facts = RICH_CASES[key]
    assert render(key, facts) == render(key, facts)


@pytest.mark.parametrize("key", sorted(RICH_CASES))
def test_renderer_survives_minimal_facts(key):
    """facts 不全（只有必需字段）时要退化成短句，不能抛异常。"""
    txt = render(key, {"sector": 1, "label": "T1", "laps": 3,
                       "unit": "油", "lap_time_s": 60.0,
                       "loss_s": 0.3, "median_loss_s": 0.4,
                       "projected_s": 60.0, "laps_left": 2.0})
    assert isinstance(txt, str) and txt


# 各 key 的 ttl 容量（秒）—— 预算表（MAX_CHARS_*）的**唯一来源**。
# 🔴 这里不再"抄一份"：直接用 `phrases.TTL_CAP_S`，它就是单一事实源。
#    曾经把预算与 ttl 写成两个独立常数，结果必然分叉 ——
#    预算允许 24 字（≈5.33 s）而 ttl=5.0，最坏句子比 ttl 还长，
#    一旦补上过期检查第一个被丢的就是它。现在 `ttl_for` 负责让两者永远一致。
TTL_S = phrases.TTL_CAP_S


def test_speech_time_within_ttl_budget():
    """🔴 句子念出来要几秒 —— 这是 `ttl_s` 的下限（见 phrases.speech_s）。

    契约里写着"过期作废"，而 `ttl_s` 目前**还没有消费者**（只有声明与转递）。
    在这条链路补上过期检查之前，先把「句子不短于 ttl」变成可断言的量：
    这是最坏输入（最长的那种 facts），所以它守的是**上界**。
    """
    for key, facts in RICH_CASES.items():
        txt = render(key, facts)
        t = phrases.speech_s(txt)
        assert t <= TTL_S[key], (
            f"{key} 念出来要 {t:.1f} s，而 ttl 容量只有 {TTL_S[key]} s —— "
            f"话说一半就被判过期：{txt}")


def test_every_budget_fits_its_ttl_capacity():
    """🔴 预算与 ttl 容量**恒自洽**：每个 key 预算允许的最坏句子，
    都必须装得进它自己的 ttl 容量（`ttl_for` 算出来的下限 ≤ 容量）。

    这条是「预算/ttl 分叉」的**直接回归守卫**。曾经
    `MAX_CHARS_OVERRIDE["lap_summary"]=24` 而 ttl=5.0（24 字≈5.33 s
    >5.0 s），预算允许的句子被 ttl 判过期 —— 被 `brake_warn` 那个
    同类 bug 在真车上抓到（t_go≤1.0 整窗静默）之后才意识到是系统性问题。
    """
    from gt7coach.contract import P_LOW, P_NORMAL
    # 别名也要查：tyre_cold/tyre_hot 共享 tyre_temp 的容量。
    for key in list(TTL_S) + ["tyre_cold", "tyre_hot"]:
        canon = "tyre_temp" if key.startswith("tyre_") else key
        cap = TTL_S[canon]
        bud = phrases.char_budget(P_NORMAL, key) or phrases.char_budget(P_LOW, key)
        worst = "字" * bud
        need = phrases.ttl_for(worst, P_NORMAL, key)
        assert need <= cap + 0.001, (
            f"{key}(→{canon}): 预算 {bud} 字 ≈ {bud / phrases.CHAR_PER_S:.2f} s，"
            f"但 ttl 容量只有 {cap} s —— 预算允许的句子会被 ttl 判过期")


def test_speech_s_uses_same_rate_as_budget():
    """语速换算与长度预算是**同一个常数**，否则两个口径会各说各话。"""
    assert phrases.speech_s("字" * 45) == pytest.approx(10.0)
    assert phrases.CHAR_PER_S == 4.5


def test_unknown_key_raises():
    """表里没有的 key 直接报错 —— 宁可炸在测试里，也不要在车上静默出一句
    没经过措辞层的台词。"""
    with pytest.raises(KeyError):
        render("not_a_real_key", {})


def test_fmt_lap_time_still_canonical():
    assert fmt_lap_time(92.412) == "1:32.412"
    assert fmt_lap_time(59.999) == "0:59.999"
    assert fmt_lap_time(None) == "-"
    assert fmt_lap_time(0) == "-"


def test_numbers_shared_between_local_and_cloud():
    """🔴 本地模板与云润色必须用**同一把尺**。

    如果 R2.2 自己写一套白名单，就会出现"模板不许编数字、云可以"的漏法 ——
    这里断言两个入口（`invented_numbers` / `over_budget`）就是被云侧复用的那两个。
    """
    facts = RICH_CASES["lap_summary"]
    for txt in ("1:32.782，慢 0.37", "你跑了 1:32.782，慢了 0.37 秒",
                "1:30.000，快 1.5"):
        bad = invented_numbers(txt, facts)
        if "1:30.000" in txt or "1.5" in txt:
            assert bad, f"该抓到的没抓到：{txt}"
        else:
            assert not bad, f"误杀：{txt} → {bad}"