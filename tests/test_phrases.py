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
                              fact_allow, fmt_lap_time, invented_advice,
                              invented_numbers, lap_time_tokens,
                              misattributed, missing_mandatory, numbers_in,
                              over_budget, render, spell_digits)


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

    def test_two_decimal_lap_time_form_allowed(self):
        """`1:23.45`（两位小数）是同一个值的另一种精度写法，不是"编数字"。

        🔴 真机教训：不加这一条，一句完全正确的云句会**因为少打一个 0**
           被白名单整句丢掉 —— 症状是"回落率高得莫名其妙"。
           注意 `83.45` 的两位小数形是 `23.45`（秒数），不是 `83.45`。
        """
        f = {"lap_time_s": 83.45}
        assert invented_numbers("1:23.45", f) == []
        assert "23.45" in fact_allow(f)

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
# 闸②③④ —— 白名单**查不出**的三种毛病
# ===========================================================================
#
# 🔴 为什么白名单不够（真 key 冒烟抓到的缺口，都不是"编数字"）：
#
#   ① 丢了主体：`invented_numbers` 问的是"句中出现、但 facts 里没有的数字"，
#      对**少说了一个数**完全无感。实测云句两次都把圈速主体丢了，而且丢得
#      毫无痕迹 —— 不记违规、不回落、读起来还很顺。
#   ② 编了建议：模型改用**不带数字的指令**绕过白名单（实测「注意补油」，
#      而 facts 里 `laps_left=2.3`）。这条更危险：它是**指令**，
#      车手可能真去提前进站。
#   ③ 张冠李戴：把某个实体的值安到另一个实体头上（实测「第二段慢了0.42秒」，
#      而 0.42 是"与参考圈的差"、分段损失是 0.31）。两个数都在 facts 里，
#      白名单照样放行 —— 它判的是"数从哪来"，不是"数归谁"。
#
# 下面这几组用例就是"云句该被丢"的验收标准 —— 喂进去的都是模型返回的句子。

class TestLapTimeTokens:
    """圈速「说出来就算数」的写法 —— `fact_allow` 的子集，**减掉裸分位数**。

    🔴 两个真机教训，各错一个方向（都进过生产）：
       · **少了"原始秒数"形式** → 模型写「这圈83.45」被判"丢了主体"，
         6 圈 6 句全丢、`fallback_ratio` 恒为 1.0（云在 100% 空烧钱）；
       · **多了"裸分位数 1"** → 句子里任何 `1`（比如 `T1` 里的）都能
         让"丢了主体"的句子蒙混过关。
    """

    def test_minutes_form_accepted_via_seconds_part(self):
        assert "32.412" in lap_time_tokens(92.412)

    def test_raw_seconds_form_accepted(self):
        """真机同款：`83.45`（原始秒数）必须算"说了圈速"。"""
        toks = lap_time_tokens(83.45)
        assert "83.45" in toks and "83.450" in toks

    def test_bare_minute_token_is_excluded(self):
        """`1`（裸分位）**不算**说了圈速 —— 否则 `T1` 里的 `1` 成了万能通行证。"""
        assert "1" not in lap_time_tokens(92.412)

    def test_below_one_minute_has_no_minute_token(self):
        """不到 1 分钟的圈（卡丁车/短道）没有"分"位，不能强求一个 "0"。"""
        toks = lap_time_tokens(58.412)
        assert "0" not in toks and "58.412" in toks

    @pytest.mark.parametrize("v", [92.412, 69.7, 58.412, 101.5, 83.45])
    def test_is_subset_of_fact_allow(self, v):
        """逐项必须能在白名单里找到 —— 否则白名单先把它当"编数字"杀掉，
        这条检查根本没机会跑。"""
        allow = fact_allow({"lap_time_s": v})
        for tok in lap_time_tokens(v):
            assert tok in allow, f"{v} 的写法 {tok!r} 不在白名单里"


class TestMissingMandatory:
    """闸①：facts 里有圈速，云句里却没说 → 整句丢弃。

    🔴 这条才是「圈速主体永远保留」在**云端**的保证 —— `lap_advice` 的
       槽位顺序只在超预算时决定谁先让位，云润色根本不看那个函数。
    """

    F = {"lap_time_s": 92.412, "vs_ref_s": 0.37}

    def test_sentence_that_says_it_is_kept(self):
        assert missing_mandatory("1:32.412，慢 0.37", self.F) == []

    def test_two_decimal_form_counts_as_said(self):
        """`1:32.41` 也是"说了"—— 不能因为少一个 0 就把整句丢掉。"""
        assert missing_mandatory("1:32.41，慢 0.37", self.F) == []

    def test_sentence_omitting_it_is_dropped(self):
        """实测缺口①的复现：句子只给了 delta，主体不见了。"""
        assert missing_mandatory("慢 0.37", self.F) == ["lap_time_s"]

    def test_no_demand_when_facts_lacks_it(self):
        """facts 自己就没有圈速 → 无从要求（否则续航句会被误杀）。"""
        assert missing_mandatory("油还够 2.4 圈", {"laps_left": 2.4}) == []

    def test_sub_minute_lap_needs_no_minute_token(self):
        """不到 1 分钟的圈：不能因为句子里没有 "0" 就判它丢了主体。"""
        assert missing_mandatory("58.412，慢 0.37", {"lap_time_s": 58.412}) == []

    def test_only_minutes_without_seconds_is_dropped(self):
        """只说了 "1" 分（小数部分丢了）也算丢主体 —— 光报分钟没用。"""
        assert missing_mandatory("第 1 圈", self.F) == ["lap_time_s"]

    def test_bare_minute_from_a_label_is_not_enough(self):
        """`T1` 里的 `1` 不算圈速 —— 否则没报圈速的句子会靠这个 `1` 蒙混过关。"""
        f = {"lap_time_s": 92.412, "focus_label": "T1"}
        assert missing_mandatory("T1 这段慢 0.4", f) == ["lap_time_s"]

    def test_raw_seconds_reply_from_real_smoke_is_kept(self):
        """🔴 真机原始回归：3 号提示词下模型输出「这圈83.45，T1段慢了0.4秒」。

        它**说了圈速**、白名单也认 `83.45`（就是 facts 里那个值），
        却曾因"不是 M:SS.mmm 形式"被判丢主体 —— 结果 6 圈 6 句全丢。
        闸门只管"在不在"，不管"写成什么形式"。
        """
        f = {"lap_time_s": 83.45, "vs_ref_s": 0.42, "focus_label": "T1",
             "focus_laps": 3, "focus_loss_s": 0.40}
        sent = "这圈83.45，T1段慢了0.4秒"
        assert invented_numbers(sent, f) == []      # 白名单本来就放行
        assert missing_mandatory(sent, f) == []     # 不能说它丢了主体

    def test_known_limitation_chinese_numerals(self):
        """⚠️ 已知取舍：云句若用**中文数字**写圈速，会被误判成"丢了主体"。

        代价只是回落到本地模板（模板一定带阿拉伯数字圈速），且
        `_KEY_HINTS["lap_advice"]` 明确要求写成 M:SS.mmm，实测未出现。
        记在此处，免得将来当 bug 查。
        """
        assert missing_mandatory("一分三二秒四一二，慢 0.37",
                                 self.F) == ["lap_time_s"]

    def test_non_dict_facts_is_safe(self):
        assert missing_mandatory("随便一句话", None) == []


class TestInventedAdvice:
    """闸②：给出 facts **没有授权**的处方性指令 → 整句丢弃。

    授权阈值与 `fuel_range` 模板的 `FUEL_CRIT_LAPS` 对齐 ——
    模板不敢说的话（每圈都喊"进站"就是狼来了），云也不许说。
    """

    def test_unauthorized_advice_is_caught(self):
        """实测缺口②的复现：2.3 圈（不紧张）却说「注意补油」。"""
        f = {"lap_time_s": 92.412, "laps_left": 2.3}
        assert invented_advice("1:32.412，注意补油", f) == ["补油"]

    def test_all_offending_words_are_reported(self):
        hits = invented_advice("该进站换胎了", {"laps_left": 2.3})
        assert "进站" in hits and "换胎" in hits

    def test_no_laps_left_at_all_is_not_authorized(self):
        """facts 里根本没有续航信息 → 任何进站/补油指令都算编。"""
        assert invented_advice("注意补油", {"vs_ref_s": 0.3}) == ["补油"]

    def test_authorized_when_actually_critical(self):
        """真的只剩不到 1 圈 → 「这圈进站」是模板自己也会说的话，放行。"""
        f = {"lap_time_s": 92.412, "laps_left": 0.8, "unit": "油"}
        assert invented_advice("1:32.412，油只够 0.8 圈，这圈进站", f) == []

    def test_boundary_is_fuel_crit_laps(self):
        f = {"laps_left": phrases.FUEL_CRIT_LAPS}
        assert invented_advice("这圈进站", f) == []

    def test_just_above_boundary_is_not_authorized(self):
        f = {"laps_left": phrases.FUEL_CRIT_LAPS + 0.1}
        assert invented_advice("这圈进站", f) == ["进站"]

    def test_describing_is_not_advising(self):
        """只描述、不给指令的句子必须放行 —— 否则闸门会把好东西一起杀掉。"""
        f = {"label": "T3", "laps": 3, "median_loss_s": 0.42, "laps_left": 2.4}
        assert invented_advice("T3 连续 3 圈慢 0.42，注意刹车点", f) == []
        assert invented_advice("轮胎可以再撑两圈", f) == []

    def test_non_dict_facts_is_safe(self):
        assert invented_advice("注意补油", None) == []

    def test_threshold_is_single_sourced_with_template(self):
        """模板与闸门共用 `FUEL_CRIT_LAPS`：模板说"进站"的那一刻，闸门也必须放行。

        两处各写一个数 → 会出现"模板允许、闸门拒绝"，于是回落的模板句
        本身就是违规句。
        """
        assert phrases.fuel_range(
            {"unit": "油", "laps_left": phrases.FUEL_CRIT_LAPS}).endswith("这圈进站")
        assert invented_advice(
            "这圈进站", {"laps_left": phrases.FUEL_CRIT_LAPS}) == []


class TestMisattributed:
    """闸④：数字**归谁**（张冠李戴）。

    🔴 真机实测（PROMPT_VERSION=4）：facts 是 `sector=2 / loss_s=0.31 / vs_ref_s=0.42`，
       云句却是「这圈1:23.550，第二段慢了0.42秒」—— 0.42（与参考圈的差）
       被安到了"第二段"头上。两个数**都来自 facts**，所以前三道闸
       （编数字 / 丢主体 / 编建议）**全部放行**。这一闸判的是"是不是这个实体的"。

    设计上刻意保守：**只在句子明确点名了实体、且实体名在数字之前**时才判。
    没点名时一个数可以被合法地安在多个实体上，那种模糊性不该由代码猜。
    """

    F = {"lap_time_s": 83.45, "vs_ref_s": 0.42, "sector": 2, "loss_s": 0.31,
         "focus_label": "T1", "focus_laps": 3, "focus_loss_s": 0.40,
         "laps_left": 2.3, "unit": "油"}

    # —— 必须拦下 ————————————————————————————————

    def test_real_smoke_case(self):
        """真机原句：分段被安上了「与参考圈的差」。"""
        assert misattributed("这圈1:23.550，第二段慢了0.42秒",
                             self.F) == ["慢了0.42"]

    def test_focus_bound_to_sector_loss(self):
        """T1 是弯，0.31 是分段的损失 —— 张冠李戴。"""
        assert misattributed("1:23.450，T1 慢了0.31秒", self.F) == ["慢了0.31"]

    def test_focus_bound_to_lap_delta(self):
        assert misattributed("1:23.450，T1 慢了0.42秒", self.F) == ["慢了0.42"]

    def test_sector_named_but_not_authorized(self):
        """facts 只说过 S2，句子却点名 S3 —— 那个数不属于 S3。"""
        assert misattributed("1:23.450，S3 慢 0.31", self.F) == ["慢 0.31"]

    def test_unauthorized_focus(self):
        assert misattributed("1:23.450，T7 慢了0.31秒", self.F) == ["慢了0.31"]

    def test_fuel_bound_to_a_wrong_value(self):
        """laps_left=2.3，句子却说"油还够 0.42 圈"（0.42 是圈差）。"""
        assert misattributed("油还够 0.42 圈", self.F) == ["油还够 0.42 圈"]

    def test_entity_absent_from_facts(self):
        """facts 里压根没有分段 → 点名的任何段都是无授权的。"""
        assert misattributed("S1 慢 0.31", {"laps_left": 2.3}) == ["慢 0.31"]

    # —— 必须放行 ————————————————————————————————

    def test_correct_bindings(self):
        assert misattributed("1:23.450，第二段慢了0.31秒", self.F) == []
        assert misattributed("1:23.450，S2 慢 0.31", self.F) == []
        assert misattributed("1:23.450，T1 慢了0.40秒", self.F) == []
        assert misattributed("油还够 2.3 圈", self.F) == []

    def test_letter_prefixed_duan_is_a_corner_not_a_sector(self):
        """🔴 `T1段` 里的"段"是**弯**的段，不是分段。

        若按分段解 → 读成"第 1 段"，与 facts 的 `sector=2` 冲突 →
        真机的**合法句**被误杀（重演上轮 100% 误杀那个坑）。
        """
        assert misattributed("一圈1:23.450，T1段慢了0.4秒", self.F) == []
        assert misattributed("1:23.450，T1段慢了0.40秒", self.F) == []

    def test_number_before_the_entity_is_not_judged(self):
        """实体名在数字**之后** → 不判（那是另一个子句级的主语）。"""
        assert misattributed("慢了0.42秒的是第二段", self.F) == []

    def test_cross_clause_binding_is_not_judged(self):
        """`还差 0.42` 与实体名不在同一子句 → 不判。"""
        assert misattributed("S2 慢 0.31，还差 0.42", self.F) == []

    def test_no_entity_named_is_not_judged(self):
        """没点名实体 → 白名单已经够了，这一闸不猜。"""
        assert misattributed("1:23.450，慢了0.42秒", self.F) == []

    def test_next_focus_schema_uses_other_field_names(self):
        """🔴 同一件事在两个 key 里叫两个名字：`next_focus` 用
        `label`/`median_loss_s`，`lap_advice` 用 `focus_label`/`focus_loss_s`。
        只认一套会把另一个 key 的**本地模板**整句误杀（实测踩过）。
        """
        f = {"label": "T12", "laps": 12, "median_loss_s": 1.234}
        assert misattributed("T12 连续 12 圈慢 1.23", f) == []
        assert misattributed("T12 连续 12 圈慢 0.42", f) == ["慢 0.42"]

    def test_non_dict_facts_is_safe(self):
        assert misattributed("第二段慢了0.42秒", None) == []


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


class TestFuelRangeWithRaceDistance:
    """接上「还剩几圈到终点」之后的续航说法。

    🔴 有总圈数时优先回答"**够不够跑完这一局**"，而不是"还能跑几圈"——
       后者是半个答案：车手要据以决策的是前者（要不要进站、要不要省油）。
       总圈数未知（时间赛 / 练习赛 / 菜单态残留值）时退回旧说法，**不猜**。

    ⚠️ 这里的"余/差 N 圈"是**派生值**，它不在原始 facts 里。
       所以 `_fuel_range` 必须把它写进 evidence，否则数字白名单会
       把这一句判成"编的数字"（与 `brake_warn` 的 `over_kph` 同一个坑）。
    """

    def test_enough_to_finish_reports_the_margin(self):
        t = phrases.fuel_range({"unit": "油", "laps_left": 2.4,
                                "laps_to_go": 2, "margin_laps": 0.4})
        assert t == "油够到终点，余 0.4 圈"

    def test_enough_is_factual_not_advisory(self):
        """够的时候**不能**喊进站 —— 那就成了狼来了。"""
        t = phrases.fuel_range({"unit": "油", "laps_left": 5.0,
                                "laps_to_go": 2, "margin_laps": 3.0})
        assert "进站" not in t

    def test_short_of_the_finish_reports_the_gap(self):
        t = phrases.fuel_range({"unit": "油", "laps_left": 1.5,
                                "laps_to_go": 2, "margin_laps": -0.5})
        assert t == "油差 0.5 圈"

    def test_gap_without_advice(self):
        """不够的时候也**不**给指令 —— 还没到 FUEL_CRIT_LAPS，进站与否是
        车手自己的判断，本地只给事实（见 `fuel_range` 文档）。"""
        t = phrases.fuel_range({"unit": "油", "laps_left": 1.5,
                                "laps_to_go": 2, "margin_laps": -0.5})
        assert "进站" not in t and "省" not in t

    def test_crit_wins_over_race_distance(self):
        """真的见底（≤1 圈）时说"这圈进站"，其余说法让位。"""
        t = phrases.fuel_range({"unit": "油", "laps_left": 0.6,
                                "laps_to_go": 5, "margin_laps": -4.4})
        assert t == "油只够 0.6 圈，这圈进站"

    def test_unknown_race_distance_falls_back(self):
        """总圈数未知（时间赛 / 练习赛）→ 退回"还够 N 圈"。"""
        t = phrases.fuel_range({"unit": "油", "laps_left": 2.4,
                                "laps_to_go": None, "margin_laps": None})
        assert t == "油还够 2.4 圈"

    def test_missing_keys_fall_back(self):
        t = phrases.fuel_range({"unit": "油", "laps_left": 2.4})
        assert t == "油还够 2.4 圈"

    def test_margin_is_whitelisted(self):
        """派生余量在 facts 里 → 句子不该被数字白名单拦。"""
        f = {"unit": "油", "laps_left": 1.5, "laps_to_go": 2,
             "margin_laps": -0.5}
        assert invented_numbers(phrases.fuel_range(f), f) == []

    def test_margin_missing_from_facts_is_caught(self):
        """反证：余量没进 facts → 白名单必须抓出来。"""
        t = "油够到终点，余 0.4 圈"
        assert invented_numbers(t, {"unit": "油", "laps_left": 2.4}) == ["0.4"]

    def test_within_budget(self):
        for t in ("油够到终点，余 0.4 圈", "油差 0.6 圈"):
            assert not over_budget(t, 2, key="fuel_range")
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
    "lap_advice": {"lap_time_s": 92.782, "vs_ref_s": 0.37, "unit": "油",
                   "laps_left": 2.4, "sector": 3, "loss_s": 0.31,
                   "focus_label": "T12", "focus_laps": 12,
                   "focus_loss_s": 1.234},
    "lap_summary": {"lap_time_s": 92.782, "vs_ref_s": 0.37,
                    "ref_lap_time_s": 92.412},
    "sector_loss": {"sector": 3, "loss_s": 0.31, "gain_s": 0.8,
                    "sectors": [33.4, 36.0, 31.6]},
    "projected_lap": {"projected_s": 83.45, "ref_lap_time_s": 82.9},
    "fuel_range": {"unit": "电量", "laps_left": 0.8, "per_lap": 8.0},
    "next_focus": {"label": "T12", "laps": 12, "median_loss_s": 1.234,
                   "ls_share": 0.9, "recent": [1.2, 1.3]},
}

PRIO = {"lap_advice": 2, "lap_summary": 3, "sector_loss": 2, "projected_lap": 3,
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


@pytest.mark.parametrize("key", sorted(RICH_CASES))
def test_local_templates_pass_their_own_guards(key):
    """🔴 **本地模板不能踩自己的闸** —— 否则"回落模板"的那句本身就该被丢。

    云句被闸掉之后回落的正是这个模板句。模板若也过不了这几道闸
    （比如将来有人把 lap_advice 的圈速槽位挪到后面被预算挤掉，
    或者给某个模板句加了 facts 里没有的实体名），就会出现
    "丢了云句、换回一句同样不合格的模板" —— 白改。
    这条把四道闸与全部模板一次锁在一起。
    """
    facts = RICH_CASES[key]
    txt = render(key, facts)
    assert missing_mandatory(txt, facts) == [], f"{key} 模板丢了主体：{txt}"
    assert invented_advice(txt, facts) == [], f"{key} 模板编了建议：{txt}"
    assert misattributed(txt, facts) == [], f"{key} 模板张冠李戴：{txt}"


# ===========================================================================
# 数字逐位读音（无线电风格）—— 54 → 五四
# ===========================================================================

class TestSpellDigits:
    def test_plain_integer(self):
        assert spell_digits("刹车晚了 54 米") == "刹车晚了 五四 米"

    def test_two_digit_strips_place_value(self):
        # 正是用户点名要的：54 → 五四，不要「五十四」
        assert spell_digits("54") == "五四"
        assert spell_digits("12") == "一二"
        assert spell_digits("155") == "一五五"

    def test_decimal_reads_digit_by_digit(self):
        assert spell_digits("慢 0.37") == "慢 零点三七"
        assert spell_digits("还差 0.80") == "还差 零点八零"

    def test_signed_delta_keeps_direction(self):
        assert spell_digits("+0.37") == "正零点三七"
        assert spell_digits("-0.37") == "负零点三七"

    def test_lap_time_spoken_form(self):
        assert spell_digits("1:32.412") == "一分三二秒四一二"
        assert spell_digits("预计 1:22.800") == "预计 一分二二秒八零零"

    def test_labels_are_preserved(self):
        # S2 / T3 这类带字母的编号不动，只有独立数字被逐位读
        assert spell_digits("S2 慢 0.31，还差 0.80") == "S2 慢 零点三一，还差 零点八零"
        assert spell_digits("T3 连续 3 圈慢 0.40") == "T3 连续 三 圈慢 零点四零"

    def test_corner_short_form(self):
        assert spell_digits("晚 12") == "晚 一二"
        assert spell_digits("慢 54") == "慢 五四"

    def test_idempotent_on_chinese(self):
        # 已经没有阿拉伯数字的中文串，原样返回
        assert spell_digits("出界了，回到赛道") == "出界了，回到赛道"
        assert spell_digits("一分三二秒四一二") == "一分三二秒四一二"

    def test_char_count_is_preserved(self):
        # 字符数等价 → ttl 预算无需重算
        for t in ("刹车晚了 54 米", "慢 0.37", "1:32.412", "+0.37"):
            assert len(spell_digits(t)) == len(t), t

# ===========================================================================
# 名次 / 情绪向（R3.1）—— 本地专属模板
# ===========================================================================
#
# 🔴 这三条**不在 `RENDERERS` 里**，所以 `test_every_renderer_*` 那三条硬门
#    覆盖不到它们 —— 这里的用例是手动补的。新增情绪向文案时记得同步加。

class TestMoodPhrases:
    def test_position_text(self):
        assert phrases.position_now({"position": 10, "moved": 2}) == \
            "P10，追回 2 位"
        assert phrases.position_now({"position": 13, "moved": -3}) == \
            "P13，掉了 3 位"

    def test_leader_text(self):
        assert phrases.leader({"mode": "take"}) == \
            "已经是 P1，做得很好，稳扎稳打"
        assert phrases.leader({"mode": "hold"}) == "保持当前状态，稳扎稳打"

    def test_every_pool_entry_is_reachable(self):
        """词库里不能有"永远挑不到"的死词条 —— 那等于少写了一条。"""
        seen = set()
        for lap in range(1, 400):
            for pos in (11, 13, 17, 20):
                t = phrases.encourage({"position": pos, "num_cars": 20,
                                       "lap": lap})
                seen.add(t[len(f"还在 P{pos}，"):])
        assert seen == set(phrases.ENCOURAGE_POOL), \
            sorted(set(phrases.ENCOURAGE_POOL) - seen)

    def test_pool_entries_respect_the_tail_budget(self):
        """前缀「还在 P13，」是 7 字；尾句 ≤8 字才不超 `encourage` 的预算。"""
        budget = phrases.char_budget(3, "encourage")
        for tail in phrases.ENCOURAGE_POOL:
            txt = f"还在 P13，{tail}"
            assert len(txt) <= budget, (len(txt), budget, txt)

    def test_no_invented_numbers(self):
        for lap in range(1, 40):
            for pos in (11, 13, 17):
                ev = {"position": pos, "num_cars": 20, "lap": lap}
                assert phrases.invented_numbers(
                    phrases.encourage(ev), ev) == []
        ev = {"position": 7, "moved": 1, "num_cars": 20, "lap": 5}
        assert phrases.invented_numbers(phrases.position_now(ev), ev) == []

    def test_no_prescription_words(self):
        """🔴 尤其要挡住**「加油」**：它会撞上燃油处方闸（见 `ENCOURAGE_POOL` 注释）。"""
        bad: list[str] = []
        for words, _field, _lim in phrases._ADVICE_RULES:
            bad.extend(words)
        for tail in phrases.ENCOURAGE_POOL:
            assert not [w for w in bad if w in tail], (tail, bad)

    def test_pick_is_deterministic(self):
        """同一组种子 → 同一条（教练是确定性系统，回放必须可复现）。"""
        seeds = (7, 13, 20)
        assert phrases._pick(phrases.ENCOURAGE_POOL, *seeds) == \
            phrases._pick(phrases.ENCOURAGE_POOL, *seeds)

    def test_pick_varies_with_seed(self):
        got = {phrases._pick(phrases.ENCOURAGE_POOL, n) for n in range(50)}
        assert len(got) >= 3, got

    def test_mood_keys_have_a_budget_entry(self):
        """预算表的唯一来源是 `TTL_CAP_S` —— 没登记就会掉进 8 字的兜底。"""
        for key in ("position", "encourage", "leader"):
            assert key in phrases.TTL_CAP_S
            assert phrases.char_budget(3, key) == \
                phrases.MAX_CHARS_OVERRIDE[key]

    def test_mood_keys_are_local_only(self):
        """走云润色的代价：把"别急"扩写成 facts 里没有的处方。这里明确不走。"""
        for key in ("position", "encourage", "leader"):
            assert key not in phrases.RENDERERS
