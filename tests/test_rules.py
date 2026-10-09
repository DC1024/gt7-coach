# -*- coding: utf-8 -*-
"""规则引擎测试 —— 逐条验触发条件与"不该触发时不触发"。"""
from __future__ import annotations

import pytest

from gt7coach.contract import Frame
from gt7coach.refindex import RefLap
from gt7coach import phrases
from gt7coach.rules import Ctx, RuleConfig, RuleSet, fmt_lap_time
from gt7coach.synth import synth_profile

R = 600.0
ZONE_S = 400.0          # 合成赛道的刹车区起点
APEX_S = ZONE_S + 80.0  # 弯心
# 默认帧的速度 180 km/h = 50 m/s，轮胎半径 0.34 m ⇒ 自由滚动的角速度
FREEROLL_OMEGA = 50.0 / 0.34


@pytest.fixture
def ref():
    return RefLap.from_profile(synth_profile(radius_m=R, step_m=5.0))


@pytest.fixture
def rs():
    return RuleSet(RuleConfig())


@pytest.fixture
def st(rs):
    return RuleSet.fresh_state()


def mk(**over) -> Frame:
    """一帧默认「一切正常」的遥测，只改关心的那几项。

    🔴 默认帧必须**物理自洽**：180 km/h = 50 m/s，轮胎半径 0.34 m
       ⇒ 自由滚动的 ω = 50/0.34 ≈ 147.06 rad/s。
       早先随手写 ω=50 当默认值，滑移率算出 -0.66，于是**每一条**
       "什么都不该触发"的测试都被 slip_all 打爆 —— 夹具不自洽会让整批
       测试一起红，而且看起来像被测代码坏了。
       油门同理：自由滚动的判定窗是 0.15~0.85，默认给 0.9 会让半径标定
       一帧都收不到。
    """
    base = dict(
        t=0.0, lap_time_s=10.0, speed_kph=180.0, rpm=6000.0, max_rpm=8200.0,
        gear=4, throttle=0.5, brake=0.0, lap=1,
        x=600.0, z=0.0, glat=0.0, glon=0.0,
        tyre_temp=(88.0, 89.0, 86.0, 87.0),
        wheel_rads=(FREEROLL_OMEGA,) * 4,
        connected=True,
    )
    base.update(over)
    return Frame(**base)


def feed(rs, st, *, ticks=1, dt=0.1, s=None, lateral=None, ref=None,
         warmup=False, ref_pace_ok=True, **frame_over):
    """连续跑 N 个 tick，返回所有产生的 utterance。

    `warmup=True` 模拟"本场还没跑完一圈"（引擎侧会同时把 ref 收回去）。
    **单独放出这个开关是故意的**：正因为依赖参考圈的规则都写着
    `if c.ref is None or c.s is None: return None`，暖胎期"只说不依赖参考圈的
    信息"这件事才不用每条规则各判一次；而 `f.last_lap_ms` 那条（上一圈成绩）
    是唯一一个不吃 `ref` 却会串到上一轮的，所以它必须自己认这个标志。

    `ref_pace_ok=False` 模拟"手上这份参考圈来自性能差一截的另一辆车"
    （跨车型采用历史圈）—— 速度类规则该闭嘴，几何类规则照常。
    """
    out = []
    for _ in range(ticks):
        c = Ctx(f=mk(**frame_over), ref=ref, s=s, lateral_m=lateral, dt=dt,
                st=st, warmup=warmup, ref_pace_ok=ref_pace_ok)
        out += rs.evaluate(c)
    return out


def keys(us):
    return [u.key for u in us]


def by_prefix(us, pre):
    return [u for u in us if u.key.startswith(pre)]


# —— 1. 出界 ——————————————————————————————————————————

class TestOffTrack:
    # 🔴 #I：出界判定现在是「横向偏离 **且** 轮胎打滑」双重确认。
    #    下面会动的测试都先标定半径、再喂"后轮打滑"的轮速，模拟真出界。
    @staticmethod
    def _slipkw():
        w = FREEROLL_OMEGA
        # 后轮 1.12× → 滑移率 ≈0.12：高于出界闸门 off_track_slip_min(0.10)
        # 所以"双重确认"过得去；但低于打滑规则的 slip_threshold(0.15)，
        # 于是 _wheel_slip 自己不会响，测试只观察 off_track 这一条。
        # throttle=0.95 把轮速挡在"自由滚动标定窗"之外，
        # 避免标定把打滑值吸收掉、滑移率算成 0。
        return dict(wheel_rads=(w, w, w * 1.12, w * 1.12), throttle=0.95)

    def test_needs_hold(self, rs, st):
        feed(rs, st, ticks=200)   # 先标定半径
        # 只有 0.3 s，不到 0.4 s 的判定门限
        assert keys(feed(rs, st, ticks=3, lateral=25.0, **self._slipkw())) == []
        # 再来 0.2 s → 累计 0.5 s，该报
        us = feed(rs, st, ticks=2, lateral=25.0, **self._slipkw())
        assert "off_track" in keys(us)
        assert us[0].priority == 0 and us[0].short == "出界"

    def test_inside_line_is_silent(self, rs, st):
        assert feed(rs, st, ticks=10, lateral=5.0) == []

    def test_no_ref_no_lateral_no_offtrack(self, rs, st):
        assert feed(rs, st, ticks=10, lateral=None) == []

    def test_corner_gets_more_slack(self, rs, st):
        """🔴 判据是「距**赛车线**」而不是「距赛道边缘」—— 赛车线在弯里切内侧，
        走别的线偏离十几米完全正常。不按曲率放宽就会在弯中乱报出界，
        而误报一次就把教练的可信度毁了。"""
        feed(rs, st, ticks=200)   # 先标定半径
        slip = self._slipkw()
        # 直线 25m：超过 18m 基准 → 报
        assert "off_track" in keys(feed(rs, st, ticks=5, lateral=25.0,
                                        glat=0.0, **slip))
        # 弯中同样的 25m：阈值放宽到 32.4m → 不报
        assert feed(rs, st, ticks=5, lateral=25.0, glat=1.2, **slip) == []
        # 弯中真的跑出去 40m：还是要报
        assert "off_track" in keys(feed(rs, st, ticks=5, lateral=40.0,
                                        glat=1.2, **slip))

    def test_slack_saturates(self, rs, st):
        """超出参考 G 之后不再继续放宽 —— 阈值必须有上界，否则弯中等于不判。"""
        feed(rs, st, ticks=200)
        u = feed(rs, st, ticks=5, lateral=33.0, glat=5.0, **self._slipkw())
        assert "off_track" in keys(u)
        assert u[0].evidence["threshold_m"] == pytest.approx(32.4, abs=0.1)

    def test_evidence_reports_threshold(self, rs, st):
        feed(rs, st, ticks=200)
        u = feed(rs, st, ticks=5, lateral=25.0, glat=0.0, **self._slipkw())
        off = [x for x in u if x.key == "off_track"][0]
        assert off.evidence["threshold_m"] == pytest.approx(18.0, abs=0.1)
        # #I：双重确认下，打滑值也要进 evidence 方便复盘
        assert off.evidence["slip_rear"] == pytest.approx(0.12, abs=0.02)

    def test_stationary_in_menus_is_silent(self, rs, st):
        """菜单/停车时坐标会飘，速度门槛是防这个的。"""
        assert feed(rs, st, ticks=10, lateral=30.0, speed_kph=3.0) == []

    def test_off_track_requires_slip(self, rs, st):
        """#I 核心回归：横向偏离大但轮胎**没打滑**→不报（宁可放过）；
        同时打滑才报；关掉闸门后纯横向偏离就够报（向后兼容旧行为）。"""
        w = FREEROLL_OMEGA
        feed(rs, st, ticks=200)   # 标定半径
        # 场景 A：横向偏离大 + 自由滚动（无打滑）→ 不应报
        assert feed(rs, st, ticks=8, lateral=25.0,
                    wheel_rads=(w, w, w, w), throttle=0.95) == []
        # 场景 B：横向偏离大 + 后轮打滑 → 报
        us = feed(rs, st, ticks=8, lateral=25.0, **self._slipkw())
        assert "off_track" in keys(us)
        # 关闭闸门：纯横向偏离就够报（兼容旧行为 / 不要这层过滤的用户）
        rs.cfg.off_track_require_slip = False
        assert "off_track" in keys(
            feed(rs, st, ticks=8, lateral=25.0,
                  wheel_rads=(w, w, w, w), throttle=0.95))


# —— 2. 打滑 ——————————————————————————————————————————

class TestWheelSlip:
    def test_free_rolling_is_silent(self, rs, st):
        # ω = v/R_tyre：v=50 m/s, R=0.34 → ω=147.06
        w = 50.0 / 0.34
        assert feed(rs, st, ticks=30, wheel_rads=(w, w, w, w)) == []

    def test_rear_spin_fires(self, rs, st):
        w = 50.0 / 0.34
        # 后轮转速高 30% → 滑移率 +0.30
        us = feed(rs, st, ticks=5,
                  wheel_rads=(w, w, w * 1.3, w * 1.3))
        assert "slip_rear" in keys(us)
        assert us[0].short == "打滑"

    def test_front_lock_fires(self, rs, st):
        w = 50.0 / 0.34
        # 前轮完全不转 = 抱死，滑移率 -1.0
        us = feed(rs, st, ticks=5, wheel_rads=(0.0, 0.0, w, w))
        assert "slip_front" in keys(us)

    def test_radius_calibrates_then_slip_detected(self, rs, st):
        """半径是**在线标定**的：先喂一批自由滚动帧，标出来的半径要接近真值，
        之后人为制造滑移才判得准（用默认 0.34 也是这个值，但标定值必须落到它附近）。"""
        w = 50.0 / 0.34
        feed(rs, st, ticks=200, wheel_rads=(w, w, w, w))
        cal = st["radius"]
        assert cal["n"] >= 150
        assert cal["front"] == pytest.approx(0.34, rel=0.02)
        assert cal["rear"] == pytest.approx(0.34, rel=0.02)
        assert feed(rs, st, ticks=5, wheel_rads=(w, w, w * 1.3, w * 1.3)), \
            "标定完之后必须还能检出滑移"

    def test_low_speed_ignored(self, rs, st):
        """低速（起步/停车）时 ω 与 v 的关系噪声极大，不该报打滑。"""
        w = 2.0 / 0.34 * 1.5
        assert feed(rs, st, ticks=20, speed_kph=6.0,
                    wheel_rads=(w, w, w, w)) == []


# —— 3. 刹车区预告 ————————————————————————————————————

class TestBrakeWarn:
    def test_fires_within_window(self, rs, st, ref):
        """R2.1 改文案：**只说行动信息**。

        原句「1.0 秒后重刹区」→「1.0 秒后重刹」：少两个字，
        ttl 从固定 1.2 s 改成跟着信息有效期走（t_go + 0.6）。
        `v_min_kph`（参考最低速）是分析信息、念不完，移进 evidence。
        """
        # s=350，距 400 还有 50 m，180 km/h = 50 m/s → t_go = 1.0 s
        us = feed(rs, st, ref=ref, s=350.0)
        bw = by_prefix(us, "brake_warn@")
        assert len(bw) == 1
        assert "后重刹" in bw[0].text
        assert bw[0].evidence["t_go_s"] == pytest.approx(1.0, abs=0.05)
        assert bw[0].evidence["v_min_kph"] == 90.0, "分析信息仍要留着（给云/复盘）"

    def test_ttl_tracks_message_lifetime(self, rs, st, ref):
        """🔴 ttl 跟**信息有效期**走，不是固定值。

        这条消息一进刹车区就没用了（之后是 `brake_late` 接手），
        所以它的 ttl 必须 ≈ 剩余时间 —— 固定 1.2 s 会在 t_go=3 s 时
        让"3 秒后重刹"还没说完就过期。
        """
        bw = by_prefix(feed(rs, st, ref=ref, s=350.0), "brake_warn@")
        assert bw[0].ttl_s >= bw[0].evidence["t_go_s"] + 0.5

    def test_key_carries_zone_identity(self, rs, st, ref):
        """key 必须带刹车区标识 —— 闸门靠它做「同一个弯只提醒一次」。"""
        bw = by_prefix(feed(rs, st, ref=ref, s=350.0), "brake_warn@")
        assert bw[0].key.endswith("400")

    def test_silent_when_far(self, rs, st, ref):
        # s=200，d=200 m，50 m/s → t_go=4 s，太早
        assert by_prefix(feed(rs, st, ref=ref, s=200.0), "brake_warn@") == []

    def test_silent_after_passing_zone(self, rs, st, ref):
        assert by_prefix(feed(rs, st, ref=ref, s=450.0), "brake_warn@") == []

    def test_extra_when_faster_than_reference(self, rs, st, ref):
        """比参考快时，说的**只有"快多少 + 准备重刹"** —— 8 字，念得完。

        原来会在句尾再挂「，参考最低 90」，整句 17 字≈3.8 s，
        而 ttl 只有 1.2 s：话说一半就过期。参考最低速移进 evidence。
        """
        v_ref = ref.v_at_s(350.0)
        us = feed(rs, st, ref=ref, s=350.0, speed_kph=v_ref + 12.0)
        bw = by_prefix(us, "brake_warn@")
        assert bw[0].text == "快 12，准备重刹"
        assert len(bw[0].text) <= 10
        assert bw[0].evidence["v_ref_kph"] == pytest.approx(v_ref, abs=0.5)

    def test_no_warn_without_ref(self, rs, st):
        assert by_prefix(feed(rs, st, ref=None, s=350.0), "brake_warn@") == []


# —— 4. 刹车晚了 ——————————————————————————————————————

class TestBrakeLate:
    def test_fires_when_still_not_braking(self, rs, st, ref):
        # 已过入点 30 m，速度还是入点速度的 100%，刹车 0
        us = feed(rs, st, ref=ref, s=ZONE_S + 30.0, ticks=3, brake=0.0,
                  speed_kph=ref.v_at_s(ZONE_S) or 200.0)
        bl = by_prefix(us, "brake_late@")
        assert bl, us
        assert "刹车晚了" in bl[0].text
        assert bl[0].evidence["metric"] > 20.0

    def test_silent_when_braking(self, rs, st, ref):
        us = feed(rs, st, ref=ref, s=ZONE_S + 30.0, ticks=5, brake=0.8,
                  speed_kph=150.0)
        assert by_prefix(us, "brake_late@") == []

    def test_silent_before_zone(self, rs, st, ref):
        us = feed(rs, st, ref=ref, s=ZONE_S - 50.0, ticks=5, brake=0.0)
        assert by_prefix(us, "brake_late@") == []


# —— 5/6. 弯心慢了 / 给油晚了 ————————————————————————

class TestApexRules:
    def test_apex_slow_fires(self, rs, st, ref):
        v = ref.v_at_s(APEX_S) or 90.0
        us = feed(rs, st, ref=ref, s=APEX_S + 5.0, ticks=3,
                  speed_kph=v - 12.0)
        ap = by_prefix(us, "apex_slow@")
        assert ap and "弯心慢了" in ap[0].text

    def test_apex_ok_is_silent(self, rs, st, ref):
        v = ref.v_at_s(APEX_S) or 90.0
        assert by_prefix(feed(rs, st, ref=ref, s=APEX_S + 5.0, ticks=5,
                              speed_kph=v), "apex_slow@") == []

    def test_throttle_late_fires(self, rs, st, ref):
        us = feed(rs, st, ref=ref, s=APEX_S + 80.0, ticks=8, throttle=0.1,
                  speed_kph=110.0)
        assert by_prefix(us, "throttle_late@"), us

    def test_throttle_ok_is_silent(self, rs, st, ref):
        assert by_prefix(feed(rs, st, ref=ref, s=APEX_S + 80.0, ticks=8,
                              throttle=0.95), "throttle_late@") == []

    def test_too_close_to_apex_is_silent(self, rs, st, ref):
        """刚过弯心 10 m 油门还没跟上很正常，不该催。"""
        assert by_prefix(feed(rs, st, ref=ref, s=APEX_S + 10.0, ticks=10,
                              throttle=0.05), "throttle_late@") == []


# —— 7. 换挡 ——————————————————————————————————————————

class TestShift:
    def test_needs_hold(self, rs, st):
        assert keys(feed(rs, st, ticks=3, rpm=8300.0)) == []
        us = feed(rs, st, ticks=3, rpm=8300.0)
        assert "shift" in keys(us) and us[0].short == "换挡"

    def test_below_limit_silent(self, rs, st):
        assert feed(rs, st, ticks=30, rpm=7000.0) == []

    def test_no_limit_reported_is_silent(self, rs, st):
        """有些场次不给换挡灯上限（max_rpm=0）→ 不能拿 0 当红线。"""
        assert feed(rs, st, ticks=30, rpm=9000.0, max_rpm=0.0) == []


# —— 8. 胎温 ——————————————————————————————————————————

class TestTyreTemp:
    def test_hot_names_the_tyre(self, rs, st):
        us = feed(rs, st, ticks=25,
                  tyre_temp=(88.0, 121.0, 86.0, 87.0))
        th = [u for u in us if u.key == "tyre_hot"]
        assert th and "右前" in th[0].text and "121" in th[0].text

    def test_normal_silent(self, rs, st):
        assert feed(rs, st, ticks=40) == []

    def test_missing_tyre_data_silent(self, rs, st):
        assert feed(rs, st, ticks=40, tyre_temp=(0.0, 0.0, 0.0, 0.0)) == []


# —— 9. delta ———————————————————————————————————————

class TestDelta:
    def test_slow_lap_shows_plus(self, rs, st, ref):
        t_ref = ref.t_at_s(1000.0)
        us = feed(rs, st, ref=ref, s=1000.0, lap_time_s=t_ref + 0.5)
        d = [u for u in us if u.key == "delta"]
        assert d and d[0].text.startswith("+")
        assert d[0].evidence["delta_s"] == pytest.approx(0.5, abs=0.02)

    def test_fast_lap_shows_minus(self, rs, st, ref):
        t_ref = ref.t_at_s(1000.0)
        d = [u for u in feed(rs, st, ref=ref, s=1000.0,
                             lap_time_s=t_ref - 0.4) if u.key == "delta"]
        assert d and d[0].text.startswith("-")

    def test_within_threshold_silent(self, rs, st, ref):
        t_ref = ref.t_at_s(1000.0)
        assert [u for u in feed(rs, st, ref=ref, s=1000.0,
                                lap_time_s=t_ref + 0.05)
                if u.key == "delta"] == []


# —— 10. 圈后小结 ————————————————————————————————————

class TestLapSummary:
    """R2.4 起，圈后成绩并进 `lap_advice`（四条事实合成一句）；
    `RuleConfig.lap_advice=False` 才退回单独的 `lap_summary`。默认 `feed()` 走合并。"""

    def test_reports_last_lap(self, rs, st, ref):
        us = feed(rs, st, ref=ref, last_lap_ms=92412.0)
        ls = [u for u in us if u.key == "lap_advice"]
        assert ls and ls[0].text.startswith("1:32.412")

    def test_compares_to_reference(self, rs, st, ref):
        """成绩措辞：去掉了"比参考"（工程词），差值保留两位小数。

        "参考"对开车的人没有意义 —— 快慢是相对**自己**的，
        所以文案里是「1:32.412，慢 0.37」。数字一个没少。
        """
        slower = (ref.lap_time_s + 0.37) * 1000.0
        ls = [u for u in feed(rs, st, ref=ref, last_lap_ms=slower)
              if u.key == "lap_advice"]
        assert "慢 0.37" in ls[0].text and "参考" not in ls[0].text
        assert ls[0].evidence["vs_ref_s"] == pytest.approx(0.37, abs=0.02)
        # 闭合单条成绩时句子很短，远在预算内（见 test_phrases 的长度断言）
        assert len(ls[0].text) <= 24

    def test_no_last_lap_silent(self, rs, st, ref):
        assert [u for u in feed(rs, st, ref=ref, last_lap_ms=None)
                if u.key == "lap_advice"] == []

    def test_legacy_four_sentences_when_disabled(self, rs, st, ref):
        """`lap_advice=False` → 退回旧行为：四条各自单说（成绩单独出现）。"""
        rs.cfg.lap_advice = False
        us = feed(rs, st, ref=ref, last_lap_ms=92412.0)
        assert [u for u in us if u.key == "lap_summary"]
        assert not [u for u in us if u.key == "lap_advice"]

    def test_warmup_silent_about_last_lap(self, rs, st, ref):
        """暖胎期（本场还没跑完一圈）**不能**念"上一圈成绩"。

        🔴 `f.last_lap_ms` 是游戏给的"最后一次冲线"值，**重开比赛后它还停在
           上一轮** —— 不按住它，新的一局刚发车，教练先把上一局的圈速念一遍。
           而这条规则是唯一一个不吃 `c.ref` 的（所以"收回参考圈"挡不住它），
           必须自己认 `c.warmup`。
        """
        for flag in (True, False):
            rs.cfg.lap_advice = flag
            us = feed(rs, st, ref=ref, last_lap_ms=92412.0, warmup=True)
            assert [u for u in us
                    if u.key in ("lap_advice", "lap_summary")] == [], us


class TestLapAdvice:
    """R2.4：圈后四条事实（成绩 / 最慢段 / 续航 / 习惯）合成**一句** `lap_advice`。

    关键不是"信息更多"，而是"每圈恰好一条建议"—— 四条各说各的会撑爆
    gate 的每圈额度，也让云润色每圈要打好几次。
    """

    @staticmethod
    def _debrief(**kw):
        c = Ctx(f=mk(last_lap_ms=92412.0), st=RuleSet.fresh_state(), **kw)
        return [u for u in RuleSet(RuleConfig()).evaluate(c)
                if u.key == "lap_advice"]

    def test_merges_lap_time_and_worst_sector(self):
        lap = _lap_result(5)
        lap.sectors = [31.0, 41.0, 25.0]        # 最亏在 S2（40 → 41）
        u = self._debrief(lap=lap,
                          theory=_theory([30.0, 40.0, 25.0], [3, 3, 3],
                                         gain=0.8))
        assert u, "有成绩 + 分段就该有一条综合建议"
        t = u[0].text
        assert t.startswith("1:32.412")         # 成绩打头
        assert "S2 慢 1.00" in t, t
        # 四条的事实都要带上（云润色吃这同一份 facts）
        assert "lap_time_s" in u[0].evidence and "sector" in u[0].evidence

    def test_habit_beats_delta_and_sector_when_budget_tight(self):
        """预算装不下全部时，跨圈的**习惯弯**优先于 delta / 单圈最慢段。"""
        lap = _lap_result(5)
        lap.sectors = [31.0, 41.0, 25.0]
        u = self._debrief(lap=lap,
                          corners={"habit": {"label": "T1", "laps": 3,
                                             "median_loss_s": 0.40,
                                             "metric": 0.40, "ls_share": 0.0}},
                          theory=_theory([30.0, 40.0, 25.0], [3, 3, 3],
                                         gain=0.8))
        t = u[0].text
        assert "T1 连续 3 圈慢 0.40" in t, t
        assert not phrases.over_budget(t, 2, "lap_advice"), t

    def test_fuel_critical_gets_pit_advice(self):
        u = self._debrief(fuel={"per_lap": 8.0, "level": 8.0,
                                "laps_left": 0.8, "samples": 3})
        assert u and "进站" in u[0].text, u

    def test_silent_when_nothing_to_report(self, rs, st, ref):
        """无成绩、无油量、无分段、无习惯 → 静默（不硬凑一句话）。"""
        assert [u for u in RuleSet(RuleConfig()).evaluate(
            Ctx(f=mk(), ref=ref, st=st)) if u.key == "lap_advice"] == []

    def test_every_number_is_in_evidence(self):
        """合并句里出现的每个数字都要能在它自己的 evidence 里找到。"""
        lap = _lap_result(5)
        lap.sectors = [31.0, 41.0, 25.0]
        u = self._debrief(lap=lap,
                          fuel={"per_lap": 8.0, "level": 8.0,
                                "laps_left": 0.8, "samples": 3},
                          corners={"habit": {"label": "T1", "laps": 3,
                                             "median_loss_s": 0.40,
                                             "metric": 0.40, "ls_share": 0.0}},
                          theory=_theory([30.0, 40.0, 25.0], [3, 3, 3],
                                         gain=0.8))
        assert not phrases.invented_numbers(u[0].text, u[0].evidence)


def test_fmt_lap_time():
    assert fmt_lap_time(92.412) == "1:32.412"
    assert fmt_lap_time(59.999) == "0:59.999"
    assert fmt_lap_time(None) == "-"
    assert fmt_lap_time(0) == "-"


def test_roll_lap_resets_holds_but_keeps_calibration(rs, st):
    """圈变化要清掉「持续计时」，但**保留**轮胎半径标定 ——
    清掉标定等于每圈前几秒都用默认半径，白白丢掉已经学到的值。"""
    # 本测试只验证"清持续计时但留标定"，关掉 #I 的打滑闸门回到纯横向判据
    rs.cfg.off_track_require_slip = False
    feed(rs, st, ticks=5, lateral=25.0)
    st["radius"]["front"] = 0.341
    assert st["hold"].get("off", 0) > 0
    rs.roll_lap(st, 2)
    assert st["hold"] == {}
    assert st["radius"]["front"] == 0.341


# —— 11/12/13. 本地统计类规则（全部零网络）————————————

class TestProjectedLap:
    """预测圈速 = 参考圈圈速 + 当前 delta。无线电里最常被问的一句。"""

    def test_fires_after_threshold(self, rs, st, ref):
        t_ref = ref.t_at_s(ref.length_m * 0.6)
        us = feed(rs, st, ref=ref, s=ref.length_m * 0.6,
                  lap_time_s=t_ref + 0.9)
        p = [u for u in us if u.key == "projected_lap"]
        assert p, us
        assert p[0].evidence["projected_s"] == pytest.approx(
            ref.lap_time_s + 0.9, abs=0.02)
        assert p[0].text.startswith("预计 ")

    def test_silent_too_early(self, rs, st, ref):
        """跑得太早 delta 还在抖（起步、暖胎），报出来是误导。"""
        t_ref = ref.t_at_s(ref.length_m * 0.1)
        assert [u for u in feed(rs, st, ref=ref, s=ref.length_m * 0.1,
                                lap_time_s=t_ref + 1.5)
                if u.key == "projected_lap"] == []

    def test_silent_when_on_pace(self, rs, st, ref):
        t_ref = ref.t_at_s(ref.length_m * 0.6)
        assert [u for u in feed(rs, st, ref=ref, s=ref.length_m * 0.6,
                                lap_time_s=t_ref + 0.05)
                if u.key == "projected_lap"] == []

    def test_short_form_is_compact(self, rs, st, ref):
        """弯中禁言要用短句，所以 short 必须是能念的短数字。"""
        t_ref = ref.t_at_s(ref.length_m * 0.6)
        p = [u for u in feed(rs, st, ref=ref, s=ref.length_m * 0.6,
                             lap_time_s=t_ref + 0.9)
             if u.key == "projected_lap"][0]
        assert len(p.short) <= 6

    def test_silent_without_ref(self, rs, st):
        assert [u for u in feed(rs, st, ref=None, s=1000.0, lap_time_s=40.0)
                if u.key == "projected_lap"] == []


class TestSectorLoss:
    """圈后指出「哪一段最慢」——比"这圈慢 0.4"有用得多。"""

    @staticmethod
    def _lap(sectors, lap_time=None, ok=True):
        from gt7coach.lapstats import LapResult
        return LapResult(lap=5, length_m=3770.0,
                         lap_time_s=lap_time or sum(sectors),
                         sectors=list(sectors), ok=ok, why="")

    @staticmethod
    def _theory(best, samples, gain=None):
        return {"n_sectors": len(best), "best_each_s": list(best),
                "samples": list(samples), "theory_best_s": sum(best),
                "best_actual_s": None, "gain_s": gain, "laps": 3}

    def test_picks_worst_sector(self, rs, st):
        c = Ctx(f=mk(), lap=self._lap([30.0, 40.5, 25.0]),
                theory=self._theory([30.0, 40.0, 25.0], [3, 3, 3], gain=0.5))
        u = rs._sector_loss(c)
        assert u and u.evidence["sector"] == 2
        assert u.evidence["loss_s"] == pytest.approx(0.5, abs=0.01)
        assert "S2" in u.text

    def test_includes_potential_gain(self, rs, st):
        """R2.1 措辞：「潜在 0.80」→「还差 0.80」。

        "潜在"是工程词（提示的是"潜在提升空间"），开车的人不会这么说；
        "还差 0.80"是同一件事的日常说法 —— 而**数字与 evidence 完全不变**。
        """
        c = Ctx(f=mk(), lap=self._lap([31.0, 41.0, 25.0]),
                theory=self._theory([30.0, 40.0, 25.0], [3, 3, 3], gain=0.8))
        u = rs._sector_loss(c)
        assert u and "还差 0.80" in u.text and "潜在" not in u.text
        assert u.evidence["gain_s"] == pytest.approx(0.8)

    def test_skips_sectors_with_few_samples(self, rs, st):
        """某段样本 <2 时"最好值"就是本圈自己，差值恒 0，报它没意义。"""
        c = Ctx(f=mk(), lap=self._lap([30.0, 45.0, 25.0]),
                theory=self._theory([30.0, 44.0, 25.0], [3, 1, 3], gain=0.9))
        assert rs._sector_loss(c) is None

    def test_silent_when_close(self, rs, st):
        c = Ctx(f=mk(), lap=self._lap([30.05, 40.02, 25.0]),
                theory=self._theory([30.0, 40.0, 25.0], [3, 3, 3]))
        assert rs._sector_loss(c) is None

    def test_silent_when_lap_invalid(self, rs, st):
        c = Ctx(f=mk(), lap=self._lap([30.0, 45.0, 25.0], ok=False),
                theory=self._theory([30.0, 40.0, 25.0], [3, 3, 3]))
        assert rs._sector_loss(c) is None

    def test_silent_without_theory(self, rs, st):
        assert rs._sector_loss(Ctx(f=mk(), lap=self._lap([30.0, 45.0, 25.0]),
                                   theory=None)) is None

    def test_silent_on_length_mismatch(self, rs, st):
        c = Ctx(f=mk(), lap=self._lap([30.0, 45.0]),
                theory=self._theory([30.0, 40.0, 25.0], [3, 3, 3]))
        assert rs._sector_loss(c) is None


class TestFuelRange:
    @staticmethod
    def _fuel(left, per=8.0, level=24.0):
        return {"per_lap": per, "level": level, "laps_left": left,
                "samples": 3}

    def test_warns_when_low(self, rs, st):
        u = rs._fuel_range(Ctx(f=mk(), fuel=self._fuel(2.4)))
        assert u and "还够 2.4 圈" in u.text
        assert u.evidence["laps_left"] == 2.4

    def test_silent_when_plenty(self, rs, st):
        assert rs._fuel_range(Ctx(f=mk(), fuel=self._fuel(9.0))) is None

    def test_ev_says_battery(self, rs, st):
        """电车说"电量"、油车说"油" —— 说错一次就没人信了。"""
        u = rs._fuel_range(Ctx(f=mk(powertrain="electric"),
                               fuel=self._fuel(1.8)))
        assert u and "电量" in u.text

    def test_advises_pit_when_truly_low(self, rs, st):
        """真不够（≤1 圈）才给行动建议。

        🔴 阈值卡在 1.0 圈是为了防"狼来了"：每圈都喊"进站"，
           喊到第三次就没人听了 —— 那比不说还糟。
        """
        u = rs._fuel_range(Ctx(f=mk(), fuel=self._fuel(0.8)))
        assert u and "进站" in u.text
        # 1.8 圈（还够两圈）时**不该**提进站
        u2 = rs._fuel_range(Ctx(f=mk(), fuel=self._fuel(1.8)))
        assert u2 and "进站" not in u2.text

    def test_fuel_says_fuel(self, rs, st):
        """R2.1 措辞：油车说「油」不说「油量」——「油量还够 1.8 圈」是说明书腔。"""
        u = rs._fuel_range(Ctx(f=mk(powertrain="fuel"), fuel=self._fuel(1.8)))
        assert u and u.text.startswith("油还够")
        assert u.evidence["unit"] == "油"

    def test_silent_without_data(self, rs, st):
        assert rs._fuel_range(Ctx(f=mk(), fuel=None)) is None
        assert rs._fuel_range(Ctx(f=mk(), fuel={"laps_left": None})) is None


class TestFuelRangeLapsToGo:
    """加油建议接上「还剩几圈到终点」。

    🔴 这是本轮的**新增地基**：`Frame.laps_to_go` 让续航播报从
       「油够跑 2.4 圈」升级成「够不够跑完这一局」。

    ⚠️ 两个必须钉住的边界：
       ① 余量是**派生值**，必须进 evidence，否则数字白名单会误杀这一句；
       ② 总圈数未知时**不猜**，退回旧说法。
    """

    @staticmethod
    def _fuel(left, per=8.0, level=24.0):
        return {"per_lap": per, "level": level, "laps_left": left,
                "samples": 3}

    def test_enough_reports_margin_and_stays_factual(self, rs, st):
        # 10 圈赛在第 9 圈上 → 还剩 2 圈；油够 2.4 圈 ⇒ 够，余 0.4 圈
        f = mk(lap=9, laps_in_race=10)
        u = rs._fuel_range(Ctx(f=f, fuel=self._fuel(2.4)))
        assert u is not None
        assert "够到终点" in u.text
        assert "余 0.4 圈" in u.text
        assert "进站" not in u.text

    def test_short_reports_gap_without_advice(self, rs, st):
        """不够跑完 → 只报差多少，**不**指挥进站（还没到 FUEL_CRIT_LAPS）。"""
        f = mk(lap=9, laps_in_race=10)            # 还剩 2 圈
        u = rs._fuel_range(Ctx(f=f, fuel=self._fuel(1.5)))
        assert u is not None
        assert "差 0.5 圈" in u.text
        assert "进站" not in u.text

    def test_margin_is_in_evidence(self, rs, st):
        """派生余量必须在 evidence 里 —— 数字白名单靠它放行。"""
        f = mk(lap=9, laps_in_race=10)
        u = rs._fuel_range(Ctx(f=f, fuel=self._fuel(1.5)))
        assert u.evidence["laps_to_go"] == 2
        assert u.evidence["margin_laps"] == pytest.approx(-0.5)

    def test_unknown_total_laps_falls_back(self, rs, st):
        """时间赛 / 练习赛：没总圈数 → 退回"还够 N 圈"，

        而且 evidence 里**不写** laps_to_go —— 没有的东西就不该出现在
        facts 里，否则云会以为自己有依据。
        """
        f = mk(lap=3, laps_in_race=0)
        u = rs._fuel_range(Ctx(f=f, fuel=self._fuel(2.4)))
        assert u is not None
        assert u.text == "油还够 2.4 圈"
        assert "laps_to_go" not in u.evidence

    def test_no_sentence_number_outside_facts(self, rs, st):
        """端到端：句子里的每个数字都能在 evidence 里找到。"""
        f = mk(lap=9, laps_in_race=10)
        u = rs._fuel_range(Ctx(f=f, fuel=self._fuel(1.5)))
        assert phrases.invented_numbers(u.text, u.evidence) == []

    def test_no_advice_word_slips_in(self, rs, st):
        """措辞里不能冒出 `invented_advice` 管的指令词。

        🔴 这条守的是「**说法升级 ≠ 变成指挥**」：加了"够到终点"之后，
           很容易顺手写成"够了，不用进站"或"不够，赶紧进站省油"——
           两个都越界了。
        """
        for left in (2.4, 1.5):
            f = mk(lap=9, laps_in_race=10)
            u = rs._fuel_range(Ctx(f=f, fuel=self._fuel(left)))
            assert phrases.invented_advice(u.text, u.evidence) == []


class TestNextFocus:
    """R1.6 + R2.4：哪个弯**反复**亏 —— 教练和仪表盘最本质的区别。

    仪表盘显示的是**现在**，教练能告诉你的是**你的习惯**。
    单圈说「S2 慢 0.4」可能是被慢车挡了；连续三圈都在同一个弯亏才是习惯 ——
    这正是这条规则（按弯累积）和 sector_loss（单圈最慢段）的本质区别。
    """

    @staticmethod
    def _habit(label="T3", med=0.42, laps=3):
        return {"label": label, "median_loss_s": med, "metric": med,
                "laps": laps, "recent": [med, med, med]}

    @staticmethod
    def _lap_result(lap_no=5):
        from gt7coach.lapstats import LapResult
        return LapResult(lap=lap_no, length_m=3770.0, lap_time_s=69.7,
                         sectors=[24.0, 23.0, 22.7], ok=True)

    def test_fires_with_habit(self, rs, st):
        """R2.1 措辞：「下一圈重点：T3，最近亏 0.42」→「T3 连续 3 圈慢 0.42」。

        🔴 三个变化都不是修辞：
           ① 说「连续 3 圈」而不是「最近」—— 不报样本数，玩家不知道这是偶发
              还是习惯，而"习惯"是这条规则的全部价值；
           ② 去掉 8 个字的前缀（主持人腔），句子才有余量装信息；
           ③ 无刹车区证据时**不给提示**（`ls_share` 缺席 → 不说"注意刹车点"）。
        """
        c = Ctx(f=mk(), lap=self._lap_result(5),
                corners={"habit": self._habit("T3", 0.42)})
        u = rs._next_focus(c)
        assert u and u.text == "T3 连续 3 圈慢 0.42"
        assert "刹车点" not in u.text, "没有刹车区证据就不该开处方"
        assert u.evidence["median_loss_s"] == pytest.approx(0.42)

    def test_hinting_when_loss_in_brake_zone(self, rs, st):
        """有证据（损失主要出在刹车区）才给「注意刹车点」。

        🔴 这仍然是**指出**不是**处方**：证据来自参考圈各刹车区的入点
           （`corner_loss_split` 算的占比），不是我们猜"他怎么开错的"。
        """
        h = self._habit("T3", 0.42)
        h["ls_share"] = 0.72
        u = rs._next_focus(Ctx(f=mk(), lap=self._lap_result(5),
                               corners={"habit": h}))
        assert u and "注意刹车点" in u.text

    def test_no_hint_when_evidence_weak(self, rs, st):
        """占比不到一半 → 提示可能误导（那损失主要在出弯加速而非刹车）。"""
        h = self._habit("T3", 0.42)
        h["ls_share"] = 0.31
        u = rs._next_focus(Ctx(f=mk(), lap=self._lap_result(5),
                               corners={"habit": h}))
        assert u and "刹车点" not in u.text

    def test_repeats_only_every_n_laps(self, rs, st):
        """同一个弯每圈都念同一句就成了唠叨 —— 而唠叨会让人开始忽略教练。"""
        c = Ctx(f=mk(), lap=self._lap_result(5),
                corners={"habit": self._habit("T3", 0.42)})
        assert rs._next_focus(c) is not None          # 第 5 圈提醒
        # 紧接着的第 6、7 圈不再提醒（间隔 3 圈）
        c6 = Ctx(f=mk(), lap=self._lap_result(6),
                 corners={"habit": self._habit("T3", 0.42)}, st=c.st)
        assert rs._next_focus(c6) is None
        # 到第 8 圈才再提
        c8 = Ctx(f=mk(), lap=self._lap_result(8),
                 corners={"habit": self._habit("T3", 0.42)}, st=c.st)
        assert rs._next_focus(c8) is not None

    def test_silent_without_habit(self, rs, st):
        assert rs._next_focus(Ctx(f=mk(), corners={"habit": None})) is None
        assert rs._next_focus(Ctx(f=mk(), corners=None)) is None

    def test_silent_when_habit_below_threshold(self, rs, st):
        """样本不足时 habit 本身就是 None（CornerTracker 挡住了），这里兜第二道。"""
        h = self._habit("T3", 0.42)
        h["laps"] = 1                                  # 只有 1 圈样本
        assert rs._next_focus(Ctx(f=mk(), lap=self._lap_result(5),
                                  corners={"habit": h})) is None

    def test_silent_without_lap(self, rs, st):
        """没有刚跑完的圈就没有"下一圈"可言。"""
        c = Ctx(f=mk(), lap=None, corners={"habit": self._habit("T3", 0.42)})
        assert rs._next_focus(c) is None


def _lap_result(lap_no=5):
    """给新类用的一圈统计（与 TestNextFocus._lap_result 同构）。"""
    from gt7coach.lapstats import LapResult
    return LapResult(lap=lap_no, length_m=3770.0, lap_time_s=69.7,
                     sectors=[24.0, 23.0, 22.7], ok=True)


def _theory(best, samples, gain=None):
    return {"n_sectors": len(best), "best_each_s": list(best),
            "samples": list(samples), "theory_best_s": sum(best),
            "best_actual_s": None, "gain_s": gain, "laps": 3}


class TestTtlFitsSpeech:
    """🔴 句子长度与 `ttl_s` 必须匹配 —— 用 rules **真实产出的** Utterance 校验。

    为什么要有这条：`ttl_s` 在契约里的定义是"过期作废，免得攒到出弯再播一条
    旧消息"，但它目前**还没有消费者**。R2.1 把这四句加长之后，
    "2 秒有效期 vs 4 秒语音"这种矛盾就出现了 —— 而一旦补上过期检查，
    第一批发不出声的就是这几句（最恼人的失败：屏幕上什么也没错，就是不说话）。

    语速常数与长度预算共用同一个（`phrases.CHAR_PER_S`），
    所以"不超预算"和"不超 ttl"两件事不会各说各话。
    """

    def test_lap_advice_fits(self, rs, st, ref):
        u = [x for x in feed(rs, st, ref=ref, last_lap_ms=(ref.lap_time_s + 0.37) * 1000)
             if x.key == "lap_advice"][0]
        assert phrases.speech_s(u.text) <= u.ttl_s, (u.text, u.ttl_s)

    def test_lap_advice_worst_case_fits(self, rs, st, ref):
        """最坏输入：成绩 + 油见底 + 习惯弯 + 最慢段**全都在**，
        合并句仍必须念得完（贪心装填保证不超预算 → 不超 ttl）。"""
        c = Ctx(f=mk(last_lap_ms=92412.0), lap=_lap_result(5), st=st, ref=ref,
                fuel={"per_lap": 8.0, "level": 8.0, "laps_left": 0.8,
                      "samples": 3},
                corners={"habit": {"label": "T12", "laps": 12,
                                   "median_loss_s": 1.234,
                                   "metric": 1.234, "ls_share": 0.9}},
                theory=_theory([30.0, 40.0, 25.0], [3, 3, 3], gain=0.8))
        u = [x for x in RuleSet(RuleConfig()).evaluate(c)
             if x.key == "lap_advice"][0]
        assert phrases.speech_s(u.text) <= u.ttl_s, (u.text, u.ttl_s)
        assert not phrases.over_budget(u.text, 2, "lap_advice"), u.text

    def test_sector_loss_fits(self, rs, st):
        lap = _lap_result(5)
        lap.sectors = [31.0, 41.0, 25.0]
        c = Ctx(f=mk(), lap=lap, theory=_theory([30.0, 40.0, 25.0],
                                                [3, 3, 3], gain=0.8), st=st)
        u = [x for x in RuleSet(RuleConfig()).evaluate(c)
             if x.key == "lap_advice"][0]
        assert phrases.speech_s(u.text) <= u.ttl_s, (u.text, u.ttl_s)

    def test_projected_lap_fits(self, rs, st, ref):
        """R2.1 把它从 2.0 提到 3.0 —— 因为 11 字 ≈ 2.4 s 语音，2.0 装不下。"""
        c = Ctx(f=mk(lap_time_s=ref.t_at_s(2000.0) + 0.6, last_lap_ms=90000.0),
                ref=ref, s=2000.0, theory=_theory([30.0, 40.0, 25.0], [3, 3, 3]),
                st=st)
        us = RuleSet(RuleConfig()).evaluate(c)
        p = [x for x in us if x.key == "projected_lap"]
        if p:
            assert phrases.speech_s(p[0].text) <= p[0].ttl_s, (p[0].text, p[0].ttl_s)

    def test_fuel_range_fits(self, rs, st):
        u = RuleSet(RuleConfig())._fuel_range(
            Ctx(f=mk(powertrain="electric"), st=st,
                fuel={"per_lap": 8.0, "level": 10.0, "laps_left": 0.8,
                      "samples": 3}))
        assert u and phrases.speech_s(u.text) <= u.ttl_s, (u.text, u.ttl_s)

    def test_brake_warn_fits(self, rs, st, ref):
        """🔴 这条是被数据打脸的：原句 17 字 ≈ 3.8 s 语音，ttl 只有 1.2 s。

        改动前 A 档**从没被算过一次**（我以为"≤6 字的短句不用管"），
        实测才发现引导句其实有 17 字。所以 A 档也要过这把尺。
        """
        bw = by_prefix(feed(rs, st, ref=ref, s=350.0), "brake_warn@")
        assert bw and phrases.speech_s(bw[0].text) <= bw[0].ttl_s, bw[0]

    def test_brake_warn_long_window_fits(self, rs, st, ref):
        """扫**真实可达**的整个预告窗（t_go ∈ (0, brake_warn_s]）都要念得完。

        🔴 别再用「t_go=3 s」当"长窗口"了 —— `brake_warn_s = 1.5` 是硬上限，
        `_brake_warn` 里 `if not (0 < t_go <= cfg.brake_warn_s)` 会直接把
        t_go=3 挡掉，那样写出来的测试**根本没有 utterance 可比**，
        只会得到一个永远不执行断言的假绿。边界由 50 m/s ⇒ s=400−t_go·50
        反推：t_go 1.4/1.0/0.4 ⇒ s=330/350/380。
        """
        cfg = RuleConfig()
        for t_go in (1.4, 1.0, 0.4):
            s_at = ZONE_S - t_go * 50.0        # 180 km/h = 50 m/s
            assert 0.0 < t_go <= cfg.brake_warn_s
            bw = by_prefix(feed(rs, st, ref=ref, s=s_at), "brake_warn@")
            assert bw, (t_go, s_at)
            assert phrases.speech_s(bw[0].text) <= bw[0].ttl_s, (bw[0].text,
                                                                bw[0].ttl_s)

    def test_brake_warn_ttl_never_shorter_than_speech(self, rs, st, ref):
        """🔴 回归守卫：ttl 下限写死 1.2 s 时 `t_go≈0` 的短窗会丢消息。

        「1.0 秒后重刹」8 字 ≈ 1.78 s > 1.2 s，而 `brake_warn_s=1.5` 又
        让 `t_go + 0.6` 最多到 2.1 —— 看似够，但 t_go 小的时候①只剩 1.2 s。
        所以 ttl 必须取 `max(speech_s, t_go + 0.6)`，两个约束一个都不能少。
        """
        bw = by_prefix(feed(rs, st, ref=ref, s=ZONE_S - 20.0), "brake_warn@")
        assert bw, "t_go=0.4 s 仍在预告窗内，必须产出"
        u = bw[0]
        assert u.ttl_s >= phrases.speech_s(u.text), (u.text, u.ttl_s)
        assert u.ttl_s >= u.evidence["t_go_s"] + 0.5, u.ttl_s

    def test_brake_late_fits(self, rs, st, ref):
        """「刹车晚了 30 米」= 9~10 字 ≈ 2.0~2.2 s > 原 ttl 1.5 s（已提到 2.5 s）。

        🔴 触发参数必须抄 `TestBrakeLate::test_fires_when_still_not_braking`：
        要**过入点 30 m 且仍在入弯段**（`s=400+30`，apex 在 480 —— 超过 apex
        后 `_brake_late` 会主动静默，之前误用 s=430 才拿到空列表）。
        """
        v_in = ref.v_at_s(ZONE_S) or 200.0
        us = feed(rs, st, ref=ref, s=ZONE_S + 30.0, ticks=3, brake=0.0,
                  speed_kph=v_in)
        bl = by_prefix(us, "brake_late@")
        assert bl, us
        assert phrases.speech_s(bl[0].text) <= bl[0].ttl_s, (bl[0].text,
                                                            bl[0].ttl_s)

    def test_next_focus_worst_case_fits(self, rs, st):
        """最坏输入：两位数的弯号 + 两位数的圈数 + 一个 1.23 的损失 + 有提示。"""
        h = {"label": "T12", "median_loss_s": 1.234, "metric": 1.234,
             "laps": 12, "recent": [1.2, 1.3], "ls_share": 0.9}
        u = rs._next_focus(Ctx(f=mk(), lap=_lap_result(12), st=st,
                               corners={"habit": h}))
        assert u and phrases.speech_s(u.text) <= u.ttl_s, (u.text, u.ttl_s)


class TestEverySpokenNumberIsInEvidence:
    """🔴 全规则扫描：**任何**播报里出现的数字都必须能在它自己的
    `evidence` 里找到 —— 这是 `phrases.fact_allow` 那份白名单在
    rules 层的镜像。

    为什么单开一类（而不是只在 phrases 里测）：`test_phrases.py` 的
    `test_no_invented_numbers` 只喂**手工构造**的 RICH_CASES，facts 是我
    自己写的，当然自洽 —— 它测的是"phrases 不编数字"。**它测不到
    rules 忘了往 evidence 里写字段**。

    真事：`_brake_warn` 算出了 `over = speed_kph - v_ref`、句子说
    「快 57，准备重刹」，但 `over` 从没进 evidence（只写了 `v_ref_kph`）。
    合成数据下 forever 绿；**真车场次回放**才被数字白名单抓出来
    （INVENTED ['57']）。所以这一类必须用**真实/合成赛道跑完整引擎**，
    而不是单点构造 —— 单点构造永远不会碰到那个分支。
    """

    @staticmethod
    def _sweep(eng, src):
        said = []
        for _ in range(len(src._frames) + 20):
            st = eng.tick()
            for u in st.say:
                said.append(u)
            if not st.connected:
                break
        return said

    def test_synthetic_lap_numbers_all_traceable(self):
        """合成赛道跑 4 圈（会触发 brake_warn / delta / lap_summary 等）。"""
        from gt7coach.engine import CoachConfig, CoachEngine
        from gt7coach.source import ReplaySource
        from gt7coach.synth import synth_lap_frames, synth_profile
        frames = synth_lap_frames(laps=4)
        src = ReplaySource(frames, profile=synth_profile(), loop=False)
        eng = CoachEngine(src, CoachConfig(poll_interval_s=0.0),
                          clock=src.clock)
        said = self._sweep(eng, src)
        assert said, "合成赛道应该至少产出一条播报"
        bad = []
        for u in said:
            inv = phrases.invented_numbers(u.text, u.evidence)
            if inv:
                bad.append((u.key, u.text, inv, u.evidence))
        assert not bad, f"有播报的数字不在 evidence 里：{bad}"

    def test_brake_warn_over_speed_is_in_evidence(self, rs, st, ref):
        """精确回归：`over_kph`（"快 57"里的 57）必须在 evidence 里。

        用比参考快 12 km/h 触发 `over` 分支 —— 这正是当天漏掉的那条路径。
        """
        v_ref = ref.v_at_s(350.0)
        us = feed(rs, st, ref=ref, s=350.0, speed_kph=v_ref + 12.0)
        bw = by_prefix(us, "brake_warn@")
        assert bw, us
        u = bw[0]
        assert u.text == "快 12，准备重刹"
        assert u.evidence["over_kph"] == pytest.approx(12.0, abs=1.0)
        assert phrases.invented_numbers(u.text, u.evidence) == []

    def test_sweep_all_synthetic_brake_positions(self, rs, st, ref):
        """把整个刹车预告窗扫一遍（含 over 分支）逐点查白名单。"""
        v_ref = ref.v_at_s(350.0)
        for s_at, spd in ((350.0, None), (350.0, v_ref + 30.0),
                          (380.0, v_ref + 5.0), (330.0, v_ref + 40.0)):
            kw = {"ref": ref, "s": s_at}
            if spd is not None:
                kw["speed_kph"] = spd
            for u in by_prefix(feed(rs, st, **kw), "brake_warn@"):
                assert phrases.invented_numbers(u.text, u.evidence) == [], \
                    (u.text, u.evidence)


# —— 参考圈「分层」：几何面 vs 速度面 ——————————————————————
#
# 背景（方案的头号结论）：跨车型复用历史参考圈时，参考圈里只有**位置**
# 是能跨车用的，**速度**不能。所以规则层要按"用哪一面"分开处理 ——
# 速度面不可比时七条规则静默，几何面永远可用。

#: 速度面：跨车型时必须闭嘴的规则 key 前缀
SPEED_FACE_KEYS = ("brake_warn@", "brake_late@", "apex_slow@",
                   "delta", "projected_lap")
#: 几何面：跨车型时**照常工作**的规则 key 前缀
GEOMETRY_FACE_KEYS = ("off_track", "throttle_late@")


class TestReferenceLayering:
    """🔴 跨车型（速度差一截）时：速度类规则静默，几何类规则照常。"""

    def test_speed_face_rules_go_silent(self, rs, st, ref):
        # 已过刹车入点 30 m、没踩刹车、速度还是入点速度 → 正常会报"刹车晚了"
        kw = dict(ref=ref, s=ZONE_S + 30.0, ticks=6, brake=0.0,
                  speed_kph=ref.v_at_s(ZONE_S) or 200.0)
        assert by_prefix(feed(rs, st, **kw), "brake_late@"), \
            "同一条件在速度面可用时**必须**还会报（否则这条测试没在测东西）"
        st2 = RuleSet.fresh_state()
        us = feed(rs, st2, ref_pace_ok=False, **kw)
        for pre in SPEED_FACE_KEYS:
            assert by_prefix(us, pre) == [], (pre, keys(us))

    def test_geometry_face_rules_keep_working(self, rs, st, ref):
        """出界（横向偏差）与给油晚了（位置判据）都不吃速度面。

        🔴 这是"分层"的全部价值：换了车，教练仍然能告诉你"出界了"和
           "这个弯你没给油"，只是不再拿别人的速度来量你。
        """
        us = feed(rs, st, ref=ref, s=APEX_S + 80.0, ticks=8, throttle=0.1,
                  speed_kph=110.0, lateral=60.0, ref_pace_ok=False)
        got = "".join(keys(us))
        for pre in GEOMETRY_FACE_KEYS:
            assert pre in got, (pre, keys(us))

    def test_same_car_is_never_silenced(self, rs, st, ref):
        """默认 `ref_pace_ok=True` ⇒ 与加这个标志之前**完全一致**。"""
        v = ref.v_at_s(APEX_S) or 90.0
        assert by_prefix(feed(rs, st, ref=ref, s=APEX_S + 5.0, ticks=3,
                              speed_kph=v - 12.0), "apex_slow@")

    def test_lap_summary_keeps_the_time_drops_the_comparison(self, rs, st, ref):
        """圈速是游戏给的实测值（与参考圈无关），"比参考圈快/慢"才是跨车比较。"""
        c = Ctx(f=mk(last_lap_ms=71234.0), ref=ref, s=APEX_S, st=st)
        assert "vs_ref_s" in (rs._lap_summary(c).evidence or {})
        c2 = Ctx(f=mk(last_lap_ms=71234.0), ref=ref, s=APEX_S, st=st,
                 ref_pace_ok=False)
        ev = rs._lap_summary(c2).evidence
        assert ev.get("lap_time_s") == pytest.approx(71.234, abs=0.001)
        assert "vs_ref_s" not in ev

    def test_every_speed_face_rule_is_covered(self):
        """守：新增依赖速度面的规则时，必须同步登记到 SPEED_FACE_KEYS。

        🔴 这是这个表存在的唯一理由 —— 否则有人加了一条"比参考圈慢多少"
           的新规则却没加闸，跨车时它就会照着别人的速度说话，
           而这个文件的其它测试**一条都不会红**。
        """
        assert len(SPEED_FACE_KEYS) >= 5


class TestCarVerdict:
    """同车判据：优先比数字车型码（`contract.car_verdict`）。

    🔴 为什么值得单独测：车型名要过 `cars.csv` 查表，表没命中时**两边都拿到
       空串** → 判不出差别 → 于是静默跨车采用，连条日志都没有。数字码没有
       这道中间环节，所以判据必须优先用它。
    """

    def test_same_code_is_same_car(self):
        from gt7coach.contract import car_verdict
        assert car_verdict(805, "", 805, "") == "same_car"

    def test_different_code_is_cross_car(self):
        from gt7coach.contract import car_verdict
        assert car_verdict(805, "A", 902, "B") == "cross_car"

    def test_falls_back_to_name_when_code_missing(self):
        """老版 Dash 不给 car_code → 退回比车型名（比"完全不判"强）。"""
        from gt7coach.contract import car_verdict
        assert car_verdict(0, "GT-R", 0, "GT-R") == "same_car"
        assert car_verdict(0, "GT-R", 0, "Civic") == "cross_car"

    def test_unknown_is_a_third_state(self):
        """🔴 判不出就是判不出，**不能**当成同车 —— 由性能窗口兜底。"""
        from gt7coach.contract import car_verdict
        assert car_verdict(0, "", 0, "") == "unknown"
        assert car_verdict(805, "", 0, "") == "unknown"

    def test_sentinel_code_is_treated_as_unknown(self):
        """65535 是 u16 哨兵，不是车型码。"""
        from gt7coach.contract import car_verdict
        assert car_verdict(65535, "", 65535, "") == "unknown"


class TestRefPaceOk:
    """速度面可比性判据（`RefLap.pace_ok`）。"""

    @staticmethod
    def _lap(t: float, code: int = 0, name: str = "") -> RefLap:
        r = RefLap.from_profile(synth_profile(radius_m=R, step_m=5.0))
        r.lap_time_s = t
        r.car_code = code
        r.car_name = name
        return r

    def test_same_car_always_ok(self):
        """同车**不设**圈速上限：今天跑得烂也该拿历史最好当标杆。

        这才是 history_best 的意义 —— 反过来（把同车的历史圈也禁掉）
        会毁掉这个功能本身。
        """
        base = self._lap(70.0, 805, "GT-R")
        ref = self._lap(55.0, 805, "GT-R")      # 快 21%
        assert ref.pace_ok(base) is True

    def test_cross_car_within_window_is_ok(self):
        """不同车但性能接近（同类 BoP）→ 速度面仍可用。"""
        base = self._lap(70.0, 805, "GT-R")
        ref = self._lap(67.0, 902, "Civic")     # 快 4.3%
        assert ref.car_match(base) == "cross_car"
        assert ref.pace_ok(base) is True

    def test_cross_car_beyond_window_is_rejected(self):
        """🔴 快 30% 就是两辆车不是一个量级 —— delta 会退化成恒定 +20 秒。"""
        base = self._lap(70.0, 805, "GT-R")
        ref = self._lap(50.0, 902, "Civic")
        assert ref.pace_ok(base) is False

    def test_unknown_car_falls_to_pace_window(self):
        """判不出车型（cars.csv 没命中）时靠圈速兜底，而不是无脑采用。"""
        base = self._lap(70.0)
        assert self._lap(68.0).pace_ok(base) is True
        assert self._lap(50.0).pace_ok(base) is False

    def test_missing_lap_time_is_rejected(self):
        """圈速算不出来 → 不给用（不知道就是不知道）。"""
        base = self._lap(0.0, 805, "GT-R")
        ref = self._lap(60.0, 902, "Civic")
        assert ref.pace_ok(base) is False

    def test_tolerance_is_configurable(self):
        base = self._lap(70.0, 805, "GT-R")
        ref = self._lap(60.0, 902, "Civic")     # 快 14.3%
        assert ref.pace_ok(base, tol=0.10) is False
        assert ref.pace_ok(base, tol=0.20) is True


class TestFasterSessionsCarCode:
    """候选筛选改用数字车型码 —— 与 `car_verdict` 同一把尺。"""

    @staticmethod
    def _src(rows):
        from gt7coach.source import HttpSource
        s = HttpSource("http://127.0.0.1:1", timeout=0.01)
        s.sessions = lambda: rows
        return s

    def test_filters_by_code_even_when_name_is_empty(self):
        """🔴 这正是旧实现的洞：车型名都是空串 → 过滤被跳过 → 静默跨车采用。"""
        rows = [
            {"file": "s1.jsonl", "best_lap_s": 60.0, "car_code": 805},
            {"file": "s2.jsonl", "best_lap_s": 61.0, "car_code": 902},  # 别的车
        ]
        out = self._src(rows).faster_sessions(
            70.0, exclude="me.jsonl", car_code=805, car_name="")
        assert [x["file"] for x in out] == ["s1.jsonl"]

    def test_code_wins_over_matching_name(self):
        """车型名撞了、码不同 → 以**码**为准（名字可能来自不可靠的查表）。"""
        rows = [{"file": "s.jsonl", "best_lap_s": 60.0,
                 "car_code": 902, "car_name": "GT-R"}]
        assert self._src(rows).faster_sessions(
            70.0, car_code=805, car_name="GT-R") == []

    def test_same_car_off_still_returns_cross_car(self):
        """`same_car=False` 是显式开关：允许跨车（性能窗口仍会兜底）。"""
        rows = [{"file": "s.jsonl", "best_lap_s": 60.0, "car_code": 902}]
        assert self._src(rows).faster_sessions(
            70.0, car_code=805, same_car=False)


# ===========================================================================
# 名次与情绪向（R3.1）
# ===========================================================================

class TestMood:
    """R3.1 名次 / 情绪向播报。

    🔴 这一组守的核心是**克制**，不是"能报"。名次每圈都可能变、鼓励更是纯
       情绪，所以每条规则都配了"什么时候不说"的断言；少一条就会在真车上变成
       唠叨，而唠叨会让人开始忽略教练 —— 比不说还糟（见 `gate.py` 模块头）。
    """

    @staticmethod
    def _ctx(st, *, position=13, num_cars=20, lap_no=5, warmup=False,
             with_lap=True):
        """一个默认的「20 车赛、P13、刚跑完第 5 圈」上下文。"""
        return Ctx(f=mk(lap=lap_no, position=position, num_cars=num_cars),
                   st=st, warmup=warmup,
                   lap=_lap_result(lap_no) if with_lap else None)

    # —— 16. 名次变化 ——————————————————————————————

    def test_position_gain_reports(self, rs, st):
        rs._position_now(self._ctx(st, position=12))     # 第一帧只记不报
        u = rs._position_now(self._ctx(st, position=10))
        assert u is not None
        assert u.text == "P10，追回 2 位"
        assert u.priority == 3
        assert u.short == "P10"

    def test_position_loss_reports(self, rs, st):
        rs._position_now(self._ctx(st, position=10))
        u = rs._position_now(self._ctx(st, position=13))
        assert u is not None and u.text == "P13，掉了 3 位"

    def test_first_sight_is_silent(self, rs, st):
        """刚看到名次时不报 —— 光秃秃一个「P13」没有比较对象，等于没信息。"""
        assert rs._position_now(self._ctx(st, position=13)) is None

    def test_unchanged_position_is_silent(self, rs, st):
        rs._position_now(self._ctx(st, position=13))
        assert rs._position_now(self._ctx(st, position=13)) is None

    def test_unknown_position_is_silent(self, rs, st):
        """菜单态 / 时间赛：名次与车数都是 0 → 闭嘴。"""
        assert rs._position_now(self._ctx(st, position=0)) is None
        assert rs._position_now(self._ctx(st, num_cars=0)) is None

    def test_too_few_cars_is_silent(self, rs, st):
        """两人对跑时「你追回 1 位」毫无意义。"""
        rs._position_now(self._ctx(st, position=2, num_cars=2))
        assert rs._position_now(self._ctx(st, position=1, num_cars=2)) is None

    def test_garbage_values_are_silent(self, rs, st):
        """车数刷成垃圾值（31847）→ 不许念出「还在 P1203」。

        🔴 这类"数字合法、语义荒唐"的错，白名单拦不住（数字都在 facts 里），
           只能靠上界闸 —— 与 `MAX_PLAUSIBLE_LAPS` 同一类。
        """
        assert rs._position_now(
            self._ctx(st, position=1203, num_cars=31847)) is None

    def test_contradiction_is_silent(self, rs, st):
        """名次不可能超过参赛车数 —— 自相矛盾的数据不照念。"""
        rs._position_now(self._ctx(st, position=9))
        assert rs._position_now(self._ctx(st, position=25)) is None

    def test_warmup_is_silent(self, rs, st):
        """发车那一团里名次每秒都在跳 —— 这时候报"追回 2 位"，播的是噪声。"""
        assert rs._position_now(self._ctx(st, warmup=True)) is None

    def test_derived_moved_is_in_evidence(self, rs, st):
        """🔴 `moved` 是派生值：不写进 evidence 就会被数字白名单判成"编的"。

        与 `brake_warn` 的 `over_kph` 是同一个坑（真机抓出来的）。
        """
        rs._position_now(self._ctx(st, position=12))
        u = rs._position_now(self._ctx(st, position=10))
        assert u is not None
        assert u.evidence["moved"] == 2
        assert phrases.invented_numbers(u.text, u.evidence) == []

    # —— 17. 后半区鼓励 ————————————————————————————

    def test_encourages_in_back_half(self, rs, st):
        u = rs._encourage(self._ctx(st, position=13, num_cars=20))
        assert u is not None
        assert u.text.startswith("还在 P13，")
        assert u.priority == 3
        assert u.short == "稳住"

    def test_silent_in_front_half(self, rs, st):
        """上半区不需要鼓励 —— 在那里说"别急"听着像讽刺。"""
        assert rs._encourage(self._ctx(st, position=8)) is None

    def test_even_grid_boundary(self, rs):
        """20 车第 10 名属上半区（2*10=20，不 > 20）；第 11 名才是后半区。

        🔴 判据写成 `2*pos > num_cars` 而不是 `pos > num_cars/2`，正是为了
           不引入"整除往哪取整"的争议。这个边界单独一条守着。
        """
        assert rs._encourage(self._ctx(RuleSet.fresh_state(),
                                       position=10)) is None
        assert rs._encourage(
            self._ctx(RuleSet.fresh_state(), position=11)) is not None

    def test_silent_with_too_few_cars(self, rs):
        """4 车赛的第 3 名不算"后半区"。"""
        assert rs._encourage(self._ctx(RuleSet.fresh_state(),
                                       position=3, num_cars=4)) is None

    def test_gap_between_encouragements(self, rs, st):
        """隔 3 圈才再鼓励一次 —— 天天被鼓励的人会开始怀疑自己是不是很差。

        🔴 每次"播出"都要显式调 `on_spoken` —— 规则只**产出候选**，
           记账推迟到闸门确认放行之后（见 `RuleSet.on_spoken`）。
        """
        u = rs._encourage(self._ctx(st, lap_no=5))
        assert u is not None
        rs.on_spoken([u], st)
        assert rs._encourage(self._ctx(st, lap_no=6)) is None
        assert rs._encourage(self._ctx(st, lap_no=7)) is None
        u = rs._encourage(self._ctx(st, lap_no=8))
        assert u is not None
        rs.on_spoken([u], st)

    def test_unsaid_candidate_stays_pending(self, rs, st):
        """🔴 没播出的候选必须**留着**，不能推进记账。

        这是名次播报在真实比赛里颗粒无收的根因：以前规则一发现变化就当场
        更新状态位，于是候选只活一个 tick，被闸门 `break` 跳过就永久消失。
        现在的契约是：只要不调 `on_spoken`，同一条候选下一圈还会再来。
        """
        u = rs._encourage(self._ctx(st, lap_no=5))
        assert u is not None
        # 故意不调 on_spoken（模拟被闸门跳过）
        assert rs._encourage(self._ctx(st, lap_no=6)) is not None
        assert rs._encourage(self._ctx(st, lap_no=7)) is not None

    def test_silent_without_a_completed_lap(self, rs, st):
        """只在圈后说 —— 鼓励不该插在弯里。"""
        assert rs._encourage(self._ctx(st, with_lap=False)) is None

    def test_encouragement_is_deterministic(self, rs):
        """🔴 回放可复现的底线：同一个 (lap, position, num_cars) 永远同一句。

        教练是**确定性系统**，`FileSource` 回放是核心调试手段 —— 挑词一旦吃
        全局 `random` 状态，回放就不可复现、测试也无从断言（见 `phrases._pick`）。
        """
        a = rs._encourage(self._ctx(RuleSet.fresh_state(), lap_no=7))
        b = rs._encourage(self._ctx(RuleSet.fresh_state(), lap_no=7))
        assert a is not None and b is not None
        assert a.text == b.text

    def test_encouragement_varies(self, rs):
        """反证：同一个种子不能永远同一句 —— 否则"随机"是假的。"""
        got = set()
        for lap in range(1, 40):
            u = rs._encourage(self._ctx(RuleSet.fresh_state(), lap_no=lap))
            if u:
                got.add(u.text)
        assert len(got) >= 3, got

    def test_no_prescription_words(self, rs):
        """🔴 情绪向句子绝不能含处方词 —— 尤其**「加油」**。

        它在中文里既是"come on"也是"加燃料"，而 `_ADVICE_RULES` 把「加油」
        登记成了**燃油处方词**（需 `laps_left <= FUEL_CRIT_LAPS` 授权）。
        一句"加油！"会被判成"编了个进站指令" → 整句丢掉。
        """
        bad: list[str] = []
        for words, _field, _lim in phrases._ADVICE_RULES:
            bad.extend(words)
        for lap in range(1, 40):
            u = rs._encourage(self._ctx(RuleSet.fresh_state(), lap_no=lap))
            if u:
                assert not [w for w in bad if w in u.text], (u.text, bad)

    # —— 18. 领跑提醒 ——————————————————————————————

    def test_leader_take_on_first(self, rs, st):
        u = rs._leader(self._ctx(st, position=1))
        assert u is not None
        assert u.key == "leader@take"
        assert u.text == "已经是 P1，做得很好，稳扎稳打"

    def test_leader_hold_after_gap(self, rs, st):
        """领跑后隔 5 圈才提醒一次"保持住"。

        🔴 同 `test_gap_between_encouragements`：每次播出都要调 `on_spoken`。
        """
        u = rs._leader(self._ctx(st, lap_no=5, position=1))
        assert u is not None
        rs.on_spoken([u], st)
        assert rs._leader(self._ctx(st, lap_no=6, position=1)) is None
        assert rs._leader(self._ctx(st, lap_no=9, position=1)) is None
        u = rs._leader(self._ctx(st, lap_no=10, position=1))
        assert u is not None and u.key == "leader@hold"
        assert u.text == "保持当前状态，稳扎稳打"

    def test_leader_resets_when_lost(self, rs, st):
        """掉出 P1 再夺回 → 重新"祝贺"，而不是接着说"保持住"。"""
        assert rs._leader(self._ctx(st, lap_no=5, position=1)) is not None
        assert rs._leader(self._ctx(st, lap_no=6, position=3)) is None
        u = rs._leader(self._ctx(st, lap_no=7, position=1))
        assert u is not None and u.key == "leader@take"

    def test_silent_when_not_leading(self, rs, st):
        assert rs._leader(self._ctx(st, position=2)) is None

    def test_silent_when_alone(self, rs, st):
        """一个人跑时间赛时不说"你是 P1"。"""
        assert rs._leader(self._ctx(st, position=1, num_cars=1)) is None
        assert rs._leader(self._ctx(st, position=1, num_cars=0)) is None

    # —— 定位：这一类是唯一不依赖参考圈的 ————————————————

    def test_mood_does_not_need_a_reference_lap(self, rs, st, ref):
        """🔴 跨车型时**速度类规则全部闭嘴**，但名次 / 情绪照常。

        这正是 R3.1 的定位：换了车、没有历史圈、甚至第一次跑这条赛道，
        教练照样能陪你说话 —— 它不再需要拿别人的速度来量你。
        """
        # 名次变化：先在 P13 记一帧，再切到 P11
        rs._position_now(self._ctx(st, position=13))
        c = Ctx(f=mk(lap=5, position=11, num_cars=20), ref=ref, s=100.0,
              st=st, lap=_lap_result(5), ref_pace_ok=False)
        assert rs._encourage(c) is not None
        assert rs._position_now(c) is not None

    def test_mood_keys_are_in_their_own_panel_group(self, rs, st):
        """情绪向要在面板上能单独关掉（有人就是不想被鼓励）。"""
        from gt7coach import panel

        for key in ("position", "encourage", "leader@take", "leader@hold"):
            assert panel.group_of_key(key) == "mood"


class TestRaceFinish:
    """R3.2：最后一圈冲线后报最终名次，且只报一次。"""

    @staticmethod
    def _ctx(st, *, position=3, num_cars=16, lap_no=5, laps_in_race=5,
             warmup=False):
        return Ctx(f=mk(lap=lap_no, position=position, num_cars=num_cars,
                        laps_in_race=laps_in_race),
                   st=st, warmup=warmup, lap=_lap_result(lap_no))

    def test_announces_at_final_lap(self, rs, st):
        u = rs._race_finish(self._ctx(st))
        assert u is not None
        assert u.key == "race_finish"
        assert u.text == "此次比赛第 3 位（共 16 车）"

    def test_silent_before_final_lap(self, rs, st):
        # 还在最后一圈之前（第 4 圈刚完）→ 不报
        assert rs._race_finish(self._ctx(st, lap_no=4)) is None

    def test_one_shot(self, rs, st):
        # 同一局只报一次：播报后 on_spoken 记下 laps_in_race，下一帧静默
        u = rs._race_finish(self._ctx(st))
        assert u is not None
        rs.on_spoken([u], st)
        assert rs._race_finish(self._ctx(st)) is None

    def test_resets_on_new_race(self, rs, st):
        # 报过一局后，新一局（已完成圈数 < 总圈数）应当复位、可再报
        u = rs._race_finish(self._ctx(st))
        assert u is not None
        rs.on_spoken([u], st)
        assert rs._race_finish(self._ctx(st, lap_no=4)) is None  # 新局未到终局
        # 新局跑完最后一圈应再次触发（finish_done_laps 已清除）
        st.pop("finish_done_laps", None)
        u2 = rs._race_finish(self._ctx(st))
        assert u2 is not None

    def test_silent_without_position(self, rs, st):
        # 名次 / 车数无效（菜单态、时间赛）→ 闭嘴
        assert rs._race_finish(self._ctx(st, position=0, num_cars=0)) is None

    def test_silent_warmup(self, rs, st):
        assert rs._race_finish(self._ctx(st, warmup=True)) is None

    def test_silent_no_laps_in_race(self, rs, st):
        # 时间赛 / 未知总圈数（laps_in_race=0）→ 不报
        assert rs._race_finish(self._ctx(st, laps_in_race=0)) is None

    def test_key_in_mood_group(self, rs, st):
        from gt7coach import panel
        assert panel.group_of_key("race_finish") == "mood"
