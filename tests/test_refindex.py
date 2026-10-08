# -*- coding: utf-8 -*-
"""参考圈索引测试 —— 几何定位是整条链路的立足点，错一点就全都错。"""
from __future__ import annotations

import math

import pytest

from gt7coach.contract import Frame
from gt7coach.refindex import RefLap
from gt7coach.synth import synth_lap_frames, synth_profile

R = 600.0
L = 2.0 * math.pi * R


class TestFromProfile:
    def test_accepts_synthetic_profile(self):
        ref = RefLap.from_profile(synth_profile(radius_m=R))
        assert ref.source == "profile"
        assert len(ref.grid_m) == len(ref.speed_kph) == len(ref.xs)
        assert abs(ref.length_m - L) / L < 0.001
        assert ref.brake_in and ref.apex

    def test_rejects_misaligned_channel(self):
        """通道长度与网格对不上必须**拒绝**，不能凑合 ——
        错位一格的曲线在图上看着完全正常，但每一个数都是隔壁的。"""
        p = synth_profile(radius_m=R)
        p["speed_kph"] = p["speed_kph"][:-3]
        with pytest.raises(ValueError, match="长度"):
            RefLap.from_profile(p)

    def test_rejects_without_geometry(self):
        """没有折线就没法实时定位，必须拒绝（宁可退回自攒）。"""
        p = synth_profile(radius_m=R)
        p["pt"] = {"x": [], "z": []}
        with pytest.raises(ValueError, match="坐标"):
            RefLap.from_profile(p)

    def test_rejects_error_payload(self):
        with pytest.raises(ValueError):
            RefLap.from_profile({"error": "第 99 圈没有可用数据"})
        with pytest.raises(ValueError):
            RefLap.from_profile({})


class TestFromFrames:
    """降级路径：用 10Hz 实时帧自己攒。

    用 10Hz（而不是合成时的 60Hz）是刻意的 —— 那就是真实情况下的采样密度。
    """

    def test_length_and_time(self):
        fs = synth_lap_frames(radius_m=R, hz=10.0)
        ref = RefLap.from_frames(fs, lap=1, step_m=5.0)
        assert ref.source == "self"
        assert abs(ref.length_m - L) / L < 0.005
        assert ref.warnings, "降级路径必须自己声明精度不如 Dash 的 60Hz 口径"
        # 时间轴单调
        assert ref.t_rel_s == sorted(ref.t_rel_s)

    def test_markers_derived(self):
        ref = RefLap.from_frames(synth_lap_frames(radius_m=R, hz=10.0))
        assert len(ref.brake_in) == 1
        z = ref.brake_in[0]
        assert abs(z["s_in_m"] - 400.0) < 40.0, z
        assert z["v_min_kph"] == pytest.approx(90.0, abs=4.0)
        assert ref.apex, "必须能从刹车区后面找到弯心"
        assert ref.apex[0]["speed_kph"] == pytest.approx(90.0, abs=6.0)
        assert ref.throttle_on, "必须能找到出弯给油点"

    def test_rejects_too_few_frames(self):
        with pytest.raises(ValueError, match="帧太少"):
            RefLap.from_frames(synth_lap_frames()[:10])

    def test_rejects_no_coords(self):
        fs = [Frame(t=f.t, speed_kph=f.speed_kph, lap=f.lap, x=0.0, z=0.0)
              for f in synth_lap_frames(radius_m=R, hz=10.0)]
        with pytest.raises(ValueError, match="帧太少"):
            RefLap.from_frames(fs)


class TestQueries:
    @pytest.fixture
    def ref(self):
        return RefLap.from_profile(synth_profile(radius_m=R, step_m=5.0))

    def test_t_at_s_monotonic(self, ref):
        ts = [ref.t_at_s(s) for s in range(0, int(L) - 10, 50)]
        assert ts == sorted(ts)
        assert ts[0] == pytest.approx(0.0, abs=0.5)
        assert ref.t_at_s(L) == pytest.approx(ref.lap_time_s, abs=0.5)

    def test_t_at_s_clamps_outside(self, ref):
        assert ref.t_at_s(-100) == ref.t_rel_s[0]
        assert ref.t_at_s(L * 3) == pytest.approx(ref.t_rel_s[-1], abs=1e-6)

    def test_apex_min_speed_matches_reference(self, ref):
        """弯心处的参考速度应当就是剖面里那一段的最低速。"""
        apex = ref.apex[0]
        v = ref.v_at_s(apex["s_m"])
        assert v == pytest.approx(90.0, abs=3.0)

    def test_nearest_locates_point_on_circle(self, ref):
        """已知角度的点，最近点必须落在对应弧长处。"""
        for frac in (0.0, 0.25, 0.5, 0.75):
            theta = frac * 2 * math.pi
            # 稍微往外偏 3 m，模拟走线外移
            x = (R + 3.0) * math.cos(theta)
            z = (R + 3.0) * math.sin(theta)
            _i, s, lat = ref.nearest(x, z)
            assert s == pytest.approx(frac * L, abs=15.0), frac
            assert lat == pytest.approx(3.0, abs=1.5), frac

    def test_next_brake_wraps_around(self, ref):
        """圈末的「下一个刹车点」必须是起点前那个 —— 不回绕的话最后一段路
        完全没提示，而那往往是最需要提示的地方（最后一弯冲线）。"""
        z = ref.next_brake(L - 100.0)
        assert z is not None
        assert z["s_in_m"] == pytest.approx(400.0, abs=20.0)

    def test_next_brake_mid_lap(self, ref):
        z = ref.next_brake(100.0)
        assert z["s_in_m"] == pytest.approx(400.0, abs=20.0)

    def test_to_dict_is_slim(self, ref):
        d = ref.to_dict()
        assert d["source"] == "profile" and d["points"] > 100
        assert "grid_m" not in d, "摘要里不该塞整条曲线"
