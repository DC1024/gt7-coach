# -*- coding: utf-8 -*-
"""契约层测试：解析 GT7 Dash `/api/v1/live`。

重点守两件事：
  1. **`timing` 是顶层字段**，不在 `car` 里。写成 `car["current_lap_time_s"]`
     会静默拿到 0 —— 而 0 是个完全合法的值（圈首），看不出错了。
  2. **菜单态哨兵 `0xFFFF` 必须归一**，否则会出现「第 65535 圈」。
"""
from __future__ import annotations

from gt7coach.contract import Frame, Utterance, has_coords


def _live(**over) -> dict:
    """一份最小可用的 /api/v1/live 响应。"""
    car = {
        "speed_kph": 180.0, "rpm": 6500.0, "gear": 4,
        "throttle": 0.8, "brake": 0.0,
        "position_m": {"x": 100.0, "y": 0.0, "z": -250.0},
        "g_force": {"longitudinal": -0.4, "lateral": 0.9, "magnitude": 0.98},
        "tyre_temp_c": [88.0, 89.0, 86.0, 87.0],
        "wheel_rad_per_s": [60.0, 61.0, 59.0, 60.0],
        "shift_alert": {"min_rpm": 7000.0, "max_rpm": 8200.0,
                        "shift_now": False},
    }
    timing = {"current_lap": 3, "current_lap_time_s": 41.25,
              "last_lap_ms": 92412.0}
    d = {"connected": True, "car": car, "timing": timing}
    d.update(over)
    return d


def test_reads_timing_from_top_level_not_car():
    f = Frame.from_v1_live(_live(), t=11.0)
    assert f.lap_time_s == 41.25, "timing 是顶层字段；从 car 里读会静默得到 0"
    assert f.lap == 3
    assert f.last_lap_ms == 92412.0


def test_lap_sentinel_normalized():
    d = _live()
    d["timing"]["current_lap"] = 0xFFFF
    assert Frame.from_v1_live(d, t=0.0).lap == 0


def test_last_lap_ms_absent_is_none():
    d = _live()
    d["timing"].pop("last_lap_ms")
    assert Frame.from_v1_live(d, t=0.0).last_lap_ms is None


def test_wheel_key_prefers_correct_name_over_deprecated():
    """旧键 `wheel_rev_per_s` 名字是错的（单位是 rad/s）。两个键取值相同时
    用哪个都行，但**新键优先**——将来 Dash 修正旧键语义时不能被它带偏。"""
    d = _live()
    d["car"]["wheel_rev_per_s"] = [1.0, 1.0, 1.0, 1.0]
    f = Frame.from_v1_live(d, t=0.0)
    assert f.wheel_rads[:1] == (60.0,)

    d["car"].pop("wheel_rad_per_s")
    assert Frame.from_v1_live(d, t=0.0).wheel_rads[:1] == (1.0,)


def test_missing_fields_do_not_crash():
    f = Frame.from_v1_live({}, t=5.0)
    assert f.speed_kph == 0.0 and f.lap == 0 and f.connected is False
    assert f.tyre_temp == () and f.wheel_rads == ()
    assert f.coords_ok is False


def test_g_force_and_derived():
    f = Frame.from_v1_live(_live(), t=0.0)
    assert f.glon == -0.4 and f.glat == 0.9      # g_force[0]=纵向 [1]=横向
    assert abs(f.g_mag - (0.4 ** 2 + 0.9 ** 2) ** 0.5) < 1e-9
    assert abs(f.speed_ms - 50.0) < 1e-9


def test_has_coords():
    assert has_coords(1.0, 0.0, 0.0) is True
    assert has_coords(0.0, 0.0, 0.0) is False
    assert has_coords(1e-9, 0.0, 0.0) is False


def test_utterance_defaults_short_to_text():
    u = Utterance(key="k", text="刹车晚了 12 米")
    assert u.short == "刹车晚了 12 米"
    d = u.to_dict()
    assert d["key"] == "k" and d["priority"] == 2
