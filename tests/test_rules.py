# -*- coding: utf-8 -*-
"""规则引擎测试 —— 逐条验触发条件与"不该触发时不触发"。"""
from __future__ import annotations

import pytest

from gt7coach.contract import Frame
from gt7coach.refindex import RefLap
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
         **frame_over):
    """连续跑 N 个 tick，返回所有产生的 utterance。"""
    out = []
    for _ in range(ticks):
        c = Ctx(f=mk(**frame_over), ref=ref, s=s, lateral_m=lateral, dt=dt,
                st=st)
        out += rs.evaluate(c)
    return out


def keys(us):
    return [u.key for u in us]


def by_prefix(us, pre):
    return [u for u in us if u.key.startswith(pre)]


# —— 1. 出界 ——————————————————————————————————————————

class TestOffTrack:
    def test_needs_hold(self, rs, st):
        # 只有 0.3 s，不到 0.4 s 的判定门限
        assert keys(feed(rs, st, ticks=3, lateral=25.0)) == []
        # 再来 0.2 s → 累计 0.5 s，该报
        us = feed(rs, st, ticks=2, lateral=25.0)
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
        # 直线 25m：超过 18m 基准 → 报
        assert "off_track" in keys(feed(rs, st, ticks=5, lateral=25.0,
                                        glat=0.0))
        # 弯中同样的 25m：阈值放宽到 32.4m → 不报
        assert feed(rs, st, ticks=5, lateral=25.0, glat=1.2) == []
        # 弯中真的跑出去 40m：还是要报
        assert "off_track" in keys(feed(rs, st, ticks=5, lateral=40.0,
                                        glat=1.2))

    def test_slack_saturates(self, rs, st):
        """超出参考 G 之后不再继续放宽 —— 阈值必须有上界，否则弯中等于不判。"""
        u = feed(rs, st, ticks=5, lateral=33.0, glat=5.0)
        assert "off_track" in keys(u)
        assert u[0].evidence["threshold_m"] == pytest.approx(32.4, abs=0.1)

    def test_evidence_reports_threshold(self, rs, st):
        u = feed(rs, st, ticks=5, lateral=25.0, glat=0.0)
        assert u[0].evidence["threshold_m"] == pytest.approx(18.0, abs=0.1)

    def test_stationary_in_menus_is_silent(self, rs, st):
        """菜单/停车时坐标会飘，速度门槛是防这个的。"""
        assert feed(rs, st, ticks=10, lateral=30.0, speed_kph=3.0) == []


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
        # s=350，距 400 还有 50 m，180 km/h = 50 m/s → t_go = 1.0 s
        us = feed(rs, st, ref=ref, s=350.0)
        bw = by_prefix(us, "brake_warn@")
        assert len(bw) == 1
        assert "重刹区" in bw[0].text
        assert bw[0].evidence["t_go_s"] == pytest.approx(1.0, abs=0.05)

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
        v_ref = ref.v_at_s(350.0)
        us = feed(rs, st, ref=ref, s=350.0, speed_kph=v_ref + 12.0)
        bw = by_prefix(us, "brake_warn@")
        assert "比参考快" in bw[0].text
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
    def test_reports_last_lap(self, rs, st, ref):
        us = feed(rs, st, ref=ref, last_lap_ms=92412.0)
        ls = [u for u in us if u.key == "lap_summary"]
        assert ls and ls[0].text.startswith("1:32.412")

    def test_compares_to_reference(self, rs, st, ref):
        slower = (ref.lap_time_s + 0.37) * 1000.0
        ls = [u for u in feed(rs, st, ref=ref, last_lap_ms=slower)
              if u.key == "lap_summary"]
        assert "比参考慢 0.37" in ls[0].text
        assert ls[0].evidence["vs_ref_s"] == pytest.approx(0.37, abs=0.02)

    def test_no_last_lap_silent(self, rs, st, ref):
        assert [u for u in feed(rs, st, ref=ref, last_lap_ms=None)
                if u.key == "lap_summary"] == []


def test_fmt_lap_time():
    assert fmt_lap_time(92.412) == "1:32.412"
    assert fmt_lap_time(59.999) == "0:59.999"
    assert fmt_lap_time(None) == "-"
    assert fmt_lap_time(0) == "-"


def test_roll_lap_resets_holds_but_keeps_calibration(rs, st):
    """圈变化要清掉「持续计时」，但**保留**轮胎半径标定 ——
    清掉标定等于每圈前几秒都用默认半径，白白丢掉已经学到的值。"""
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
        c = Ctx(f=mk(), lap=self._lap([31.0, 41.0, 25.0]),
                theory=self._theory([30.0, 40.0, 25.0], [3, 3, 3], gain=0.8))
        u = rs._sector_loss(c)
        assert u and "潜在 0.80" in u.text

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
        """电车说"电量"、油车说"油量" —— 说错一次就没人信了。"""
        u = rs._fuel_range(Ctx(f=mk(powertrain="electric"),
                               fuel=self._fuel(1.8)))
        assert u and "电量" in u.text

    def test_fuel_says_fuel(self, rs, st):
        u = rs._fuel_range(Ctx(f=mk(powertrain="fuel"), fuel=self._fuel(1.8)))
        assert u and "油量" in u.text

    def test_silent_without_data(self, rs, st):
        assert rs._fuel_range(Ctx(f=mk(), fuel=None)) is None
        assert rs._fuel_range(Ctx(f=mk(), fuel={"laps_left": None})) is None
