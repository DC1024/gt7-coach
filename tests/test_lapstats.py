# -*- coding: utf-8 -*-
"""本地圈统计测试 —— 分段用时与油耗。

这一层是 R1.5 的核心：**完全不打接口**，全部从手上的实时帧算。
所以每个量都要能用合成数据手算验证，不能"看着差不多"。
"""
from __future__ import annotations

import math

import pytest

from gt7coach.contract import Frame
from gt7coach.lapstats import (FuelTracker, SectorTracker, lap_result,
                               ref_sector_times, sector_times)
from gt7coach.refindex import RefLap
from gt7coach.synth import synth_lap_frames, synth_profile

R = 600.0
L = 2.0 * math.pi * R


class TestSectorTimes:
    def test_uniform_is_evenly_split(self):
        """匀速跑完 300m，分 3 段，每段必须正好 10s。"""
        sec = sector_times([0.0, 100.0, 200.0, 300.0], [0.0, 10.0, 20.0, 30.0],
                           300.0, 3)
        assert sec == pytest.approx([10.0, 10.0, 10.0])

    def test_boundary_interpolates(self):
        """段边界落在两个采样点之间时要插值，不是取最近点。

        采样点在 90/210，而段边界在 100/200 —— 两个边界都必须插出来。
        （顺便：采样点少于 4 个会被护栏拒掉，所以不能拿 3 个点测这件事。）
        """
        sec = sector_times([0.0, 90.0, 210.0, 300.0], [0.0, 9.0, 21.0, 30.0],
                           300.0, 3)
        assert sec == pytest.approx([10.0, 10.0, 10.0])

    def test_sum_equals_total(self):
        """段用时之和必须恒等于整圈用时 —— 边界时刻插值再作差，不该有累积误差。"""
        s = [0.0, 37.0, 91.0, 150.0, 226.0, 300.0]
        t = [0.0, 3.1, 8.0, 13.9, 21.0, 28.5]
        for n in (2, 3, 4, 5):
            sec = sector_times(s, t, 300.0, n)
            assert sum(sec) == pytest.approx(28.5, abs=1e-9), n

    def test_rejects_partial_lap(self):
        """跑不满整圈时最后一段会被算成 0，看着像"这段飞快" —— 必须拒绝。"""
        assert sector_times([0.0, 100.0, 200.0], [0.0, 10.0, 20.0], 300.0, 3) is None

    def test_rejects_degenerate_input(self):
        assert sector_times([0.0], [0.0], 300.0, 3) is None
        assert sector_times([0.0, 100.0, 200.0, 300.0], [0.0, 1.0, 2.0, 3.0],
                            300.0, 1) is None
        assert sector_times([0.0, 100.0, 200.0, 300.0], [0.0, 1.0, 2.0, 3.0],
                            0.0, 3) is None

    def test_mismatched_lengths_rejected(self):
        assert sector_times([0.0, 100.0, 200.0, 300.0], [0.0, 1.0, 2.0],
                            300.0, 3) is None

    def test_ref_sector_times_from_profile(self):
        ref = RefLap.from_profile(synth_profile(radius_m=R, step_m=5.0))
        sec = ref_sector_times(ref, 3)
        assert sec and len(sec) == 3
        # 三段之和 = 整圈用时
        assert sum(sec) == pytest.approx(ref.lap_time_s, abs=1e-6)
        # 🔴 减速弯在 400~560m，而圈长 3770m 分 3 段的边界是 1257/2513 ——
        #    所以它落在**第一段**里，第一段必然最慢。
        #    （第一版我写成"中段最慢"，是把它当成"赛道中段"想当然了。）
        assert sec[0] == max(sec), sec


class TestFuelTracker:
    def _feed(self, ft, used_values):
        """按引擎的用法喂：先 end_lap 观察，再 record 记进统计。"""
        for used in used_values:
            ft.start_lap(100.0)
            ft.record(ft.end_lap(100.0 - used))

    def test_per_lap_is_median(self):
        ft = FuelTracker()
        self._feed(ft, (8.0, 8.1, 40.0, 8.2, 8.05))   # 40 是异常值
        # 中位数应当是 8.1（排序后 8.0/8.05/8.1/8.2/40 —— 奇数个取中间）
        assert ft.per_lap == pytest.approx(8.1)

    def test_median_ignores_one_outlier(self):
        """单圈油耗会被慢车/暖胎圈带偏，"还能跑几圈"要稳到能据以决策。"""
        ft = FuelTracker()
        self._feed(ft, (8.0, 8.0, 8.0, 8.0, 30.0))
        assert ft.per_lap == pytest.approx(8.0)

    def test_rejects_refuel(self):
        ft = FuelTracker()
        ft.start_lap(50.0)
        assert ft.end_lap(80.0) is None, "剩余量上升 = 加过油，本圈油耗不可信"
        # 但标记要跟着更新，否则下一圈会连着算错
        assert ft.mark == 80.0

    def test_rejects_absurd_jump(self):
        ft = FuelTracker()
        ft.start_lap(60.0)
        assert ft.end_lap(0.0) is None, "整圈掉 60% 超出量程，多半读到了菜单值"
        ft.start_lap(60.0)
        assert ft.end_lap(59.99) is None, "掉 0.01 不算一圈油耗"

    def test_rejects_without_mark(self):
        assert FuelTracker().end_lap(90.0) is None

    def test_laps_left(self):
        ft = FuelTracker()
        self._feed(ft, (8.0, 8.0, 8.0))
        assert ft.laps_left(30.0) == pytest.approx(3.75)
        assert ft.laps_left(0.0) is None
        assert ft.to_dict()["per_lap"] == pytest.approx(8.0)

    def test_laps_left_before_any_lap(self):
        assert FuelTracker().laps_left(50.0) is None


class TestLapResult:
    @staticmethod
    def _lap_frames(**kw):
        return [f for f in synth_lap_frames(hz=10.0, laps=1, **kw)]

    def test_ok_path(self):
        fs = self._lap_frames()
        ft = FuelTracker()
        ft.start_lap(fs[0].fuel_pct)
        res = lap_result(fs, lap=1, n_sectors=3, expected_len_m=L, fuel=ft)
        assert res.ok, res.why
        assert res.length_m == pytest.approx(L, rel=0.01)
        assert res.lap_time_s > 60.0
        assert len(res.sectors) == 3
        # 整圈用时 = 各段之和（统一到公共基准圈长）——这是刻意的，
        # 否则"三段加起来 ≠ 整圈"看起来就是算错了
        assert sum(res.sectors) == pytest.approx(res.lap_time_s, abs=1e-9)
        # 与真实圈速的差应当只来自采样（10Hz 下最后一帧可能落在冲线之后）
        assert abs(res.lap_time_s - fs[-1].lap_time_s) < 0.5
        assert res.fuel_used == pytest.approx(8.0, abs=0.05)

    def test_rejects_joined_mid_lap(self):
        """中途接入时弧长起点不是起跑线，段边界全错 —— 必须整圈丢掉。

        这是**最隐蔽**的一种错：算出来的分段看着很合理，但每一段都错位。
        """
        fs = self._lap_frames()[200:]           # 掐掉开头 20 秒
        res = lap_result(fs, lap=1, n_sectors=3, expected_len_m=L)
        assert not res.ok
        assert "中途接入" in res.why

    def test_rejects_wrong_length(self):
        """圈长与基准差 >3% = 残缺圈（跑错路/被切断），分段跨圈不可比。"""
        fs = self._lap_frames()
        res = lap_result(fs, lap=1, n_sectors=3, expected_len_m=L * 1.10)
        assert not res.ok
        assert "残缺圈" in res.why

    def test_rejects_no_coords(self):
        fs = [Frame(t=f.t, lap_time_s=f.lap_time_s, speed_kph=f.speed_kph,
                    lap=f.lap, x=0.0, z=0.0) for f in self._lap_frames()]
        res = lap_result(fs, lap=1, n_sectors=3, expected_len_m=L)
        assert not res.ok and "坐标" in res.why

    def test_rejects_too_few_frames(self):
        assert not lap_result(self._lap_frames()[:3], lap=1, n_sectors=3,
                              expected_len_m=L).ok

    def test_works_without_expected_length(self):
        """第一圈还没参考圈时也要能算 —— 用本圈自己的圈长当基准。"""
        res = lap_result(self._lap_frames(), lap=1, n_sectors=3,
                         expected_len_m=None)
        assert res.ok
        assert res.length_m > 0

    def test_to_dict_is_json_safe(self):
        import json
        res = lap_result(self._lap_frames(), lap=1, n_sectors=3,
                         expected_len_m=L)
        json.dumps(res.to_dict(), ensure_ascii=False)


class TestSectorTracker:
    def test_theory_is_sum_of_bests(self):
        st = SectorTracker(3)
        st.add([30.0, 40.0, 25.0], 95.0)
        st.add([28.0, 42.0, 26.0], 96.0)
        assert st.best == [28.0, 40.0, 25.0]
        assert st.theory_s() == pytest.approx(93.0)
        assert st.best_actual_s() == pytest.approx(95.0)
        assert st.gain_s() == pytest.approx(2.0)

    def test_gain_never_negative(self):
        """理论值来自真实圈的各段最好值，不可能比实际最快圈还慢；
        采样噪声导致的负数不报，免得输出"潜在 -0.02"自己打自己脸。"""
        st = SectorTracker(3)
        st.add([30.0, 40.0, 25.0], 95.0)
        assert st.gain_s() == pytest.approx(0.0)

    def test_incomplete_sector_gives_no_theory(self):
        """段数对不上（配置改了/数据残缺）时宁可不给，也不给半成品。"""
        st = SectorTracker(3)
        st.add([30.0, 40.0], 70.0)          # 只有 2 段
        assert st.theory_s() is None
        assert st.to_dict() is None

    def test_samples_counted_per_sector(self):
        st = SectorTracker(2)
        st.add([10.0, 20.0], 30.0)
        st.add([11.0, 19.0], 30.0)
        d = st.to_dict()
        assert d["samples"] == [2, 2]
        assert d["laps"] == 2
        assert d["best_each_s"] == [10.0, 19.0]

    def test_to_dict_is_json_safe(self):
        import json
        st = SectorTracker(3)
        st.add([10.0, 20.0, 30.0], 60.0)
        json.dumps(st.to_dict(), ensure_ascii=False)

    def test_lap_totals_bounded(self):
        st = SectorTracker(1)
        for i in range(100):
            st.add([10.0 + i], 10.0 + i)
        assert len(st.lap_totals) <= 30


class TestFuelMarkAlwaysAdvances:
    """被拒的圈也必须推进油量标记。

    圈统计会因为"中途接入 / 残缺圈"被整体拒绝，但那一圈**真实消耗了油**。
    标记不推进的话，下一个有效圈量到的是**两圈**的消耗 —— 8 变 16，
    而且刚好落在合法区间（0.05~50）里，**没有任何护栏会挡住它**。
    这类"错得刚好合法"的 bug 最难查。
    """

    def test_end_lap_only_observes(self):
        """`end_lap` 只推进标记并返回消耗，不记统计 ——
        记不记由调用方在确认这一圈有效后决定。"""
        ft = FuelTracker()
        ft.start_lap(100.0)
        assert ft.end_lap(92.0) == pytest.approx(8.0)
        assert ft.per_lap is None, "还没 record，不该进统计"
        assert ft.mark == pytest.approx(92.0), "但标记必须推进"
        ft.record(8.0)
        assert ft.per_lap == pytest.approx(8.0)

    def test_next_lap_measures_one_lap_only(self):
        ft = FuelTracker()
        ft.start_lap(100.0)
        ft.end_lap(92.0)                        # 这一圈被拒（比如中途接入）
        ft.start_lap(92.0)
        used = ft.end_lap(84.0)                 # 这一圈有效
        assert used == pytest.approx(8.0), "不该量成两圈的 16"
        ft.record(used)
        assert ft.per_lap == pytest.approx(8.0)

    def test_rejected_lap_does_not_feed_stats(self):
        """`lap_result` 拒绝的圈不能把油耗喂进中位数。"""
        fs = [f for f in synth_lap_frames(hz=10.0, laps=1)]
        ft = FuelTracker()
        ft.start_lap(fs[0].fuel_pct)
        res = lap_result(fs[200:], lap=1, n_sectors=3, expected_len_m=L,
                         fuel=ft)          # 中途接入 → 被拒
        assert not res.ok
        assert ft.values == [], "被拒的圈不该进统计"
        assert ft.mark is not None, "但标记要推进到本圈末"

    def test_accepted_lap_feeds_stats(self):
        fs = [f for f in synth_lap_frames(hz=10.0, laps=1)]
        ft = FuelTracker()
        ft.start_lap(fs[0].fuel_pct)
        res = lap_result(fs, lap=1, n_sectors=3, expected_len_m=L, fuel=ft)
        assert res.ok
        assert ft.values, "有效圈要进统计"
        assert res.fuel_used == pytest.approx(8.0, abs=0.05)


class TestFuelMarkAlwaysAdvances:
    """被拒的圈也必须推进油量标记。

    圈统计会因为"中途接入 / 残缺圈"被整体拒绝，但那一圈**真实消耗了油**。
    标记不推进的话，下一个有效圈量到的是**两圈**的消耗 —— 8 变 16，
    而且刚好落在合法区间（0.05~50）里，**没有任何护栏会挡住它**。
    这类"错得刚好合法"的 bug 最难查。
    """

    def test_end_lap_only_observes(self):
        """`end_lap` 只推进标记并返回消耗，不记统计 ——
        记不记由调用方在确认这一圈有效后决定。"""
        ft = FuelTracker()
        ft.start_lap(100.0)
        assert ft.end_lap(92.0) == pytest.approx(8.0)
        assert ft.per_lap is None, "还没 record，不该进统计"
        assert ft.mark == pytest.approx(92.0), "但标记必须推进"
        ft.record(8.0)
        assert ft.per_lap == pytest.approx(8.0)

    def test_next_lap_measures_one_lap_only(self):
        ft = FuelTracker()
        ft.start_lap(100.0)
        ft.end_lap(92.0)                        # 这一圈被拒（比如中途接入）
        ft.start_lap(92.0)
        used = ft.end_lap(84.0)                 # 这一圈有效
        assert used == pytest.approx(8.0), "不该量成两圈的 16"
        ft.record(used)
        assert ft.per_lap == pytest.approx(8.0)

    def test_rejected_lap_does_not_feed_stats(self):
        """`lap_result` 拒绝的圈不能把油耗喂进中位数。"""
        fs = [f for f in synth_lap_frames(hz=10.0, laps=1)]
        ft = FuelTracker()
        ft.start_lap(fs[0].fuel_pct)
        res = lap_result(fs[200:], lap=1, n_sectors=3, expected_len_m=L,
                         fuel=ft)          # 中途接入 → 被拒
        assert not res.ok
        assert ft.values == [], "被拒的圈不该进统计"
        assert ft.mark is not None, "但标记要推进到本圈末"

    def test_accepted_lap_feeds_stats(self):
        fs = [f for f in synth_lap_frames(hz=10.0, laps=1)]
        ft = FuelTracker()
        ft.start_lap(fs[0].fuel_pct)
        res = lap_result(fs, lap=1, n_sectors=3, expected_len_m=L, fuel=ft)
        assert res.ok
        assert ft.values, "有效圈要进统计"
        assert res.fuel_used == pytest.approx(8.0, abs=0.05)
