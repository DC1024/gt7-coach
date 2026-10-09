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


class TestRaceInfo:
    """比赛信息（名次 / 车数 / 总圈数 / 车型码）。

    🔴 这些字段有三个共同陷阱，各有一条测试守着：

      ① `race.grid_position` **名字带 grid 却不是发车位** —— 比赛进行中它随
         排名实时变，真正的发车位是 `grid_start`。取错会得到"整场名次不动"。
      ② 菜单态把它们写成 **65535**（u16 哨兵），`int()` 出来是个完全合法的
         "第 65535 名"，必须归一到 0。
      ③ **0 表示"未知"**而不是"第 0 名" —— 练习赛/时间赛不给这些值。
    """

    def _race_live(self, **race) -> dict:
        d = _live()
        d["car"]["race"] = {"time_of_day_ms": 0, "grid_position": 7,
                            "grid_start": 12, "num_cars": 20, **race}
        d["timing"]["laps_in_race"] = 10
        d["car"]["car_code"] = 805
        return d

    def test_race_block_parsed(self):
        f = Frame.from_v1_live(self._race_live(), t=0.0)
        assert f.position == 7
        assert f.num_cars == 20
        assert f.laps_in_race == 10
        assert f.car_code == 805

    def test_absent_race_block_is_all_zero(self):
        """没有 race 块（老 Dash / 非比赛）→ 全 0，不是 KeyError。"""
        f = Frame.from_v1_live(_live(), t=0.0)
        assert (f.position, f.num_cars, f.laps_in_race, f.car_code) == (0, 0, 0, 0)

    def test_u16_sentinel_is_not_a_rank(self):
        """65535 名次/车数必须归 0 —— 否则会播报"第 65535 名"。"""
        f = Frame.from_v1_live(
            self._race_live(grid_position=65535, num_cars=65535), t=0.0)
        assert f.position == 0
        assert f.num_cars == 0

    def test_negative_is_rejected(self):
        for bad in (-1, -7):
            f = Frame.from_v1_live(self._race_live(grid_position=bad), t=0.0)
            assert f.position == 0

    def test_non_numeric_is_rejected(self):
        for bad in ("7", None, True, [7]):
            f = Frame.from_v1_live(self._race_live(grid_position=bad), t=0.0)
            assert f.position == 0


class TestLapsToGo:
    """`laps_to_go` = 到终点还剩几圈，**含当前这一圈**。

    🔴 含当前圈是与"油够跑几圈"对齐的唯一口径：10 圈赛在第 3 圈上，
       还得跑 3~10 共 8 圈；写成 `laps_in_race - lap` 只会给 7，
       少算的这一圈恰好落在"油够 7.5 圈、差 0.5 圈"这种最要命的判断上。
    """

    def test_includes_the_lap_being_driven(self):
        base = Frame.from_v1_live(_live(), t=0.0)      # 夹具里 current_lap=3
        assert base.lap == 3
        f = Frame(lap=base.lap, laps_in_race=10)
        assert f.laps_to_go == 8

    def test_last_lap_is_one(self):
        assert Frame(lap=10, laps_in_race=10).laps_to_go == 1

    def test_unknown_total_laps_is_none(self):
        """时间赛 / 练习赛不给总圈数 → None，绝不能猜。"""
        assert Frame(lap=3, laps_in_race=0).laps_to_go is None

    def test_menu_state_is_none(self):
        assert Frame(lap=0, laps_in_race=10).laps_to_go is None

    def test_past_the_end_does_not_go_negative(self):
        """超时/甩尾圈：不给负数，钳到 0。"""
        assert Frame(lap=12, laps_in_race=10).laps_to_go == 0


class TestNormU16:
    """u16 遥测字段的**统一**归一（`contract.norm_u16`）。

    🔴 三个入口共用它（`/live` 解析、jsonl 解析、缓存 meta），各写一份就
       会出现"第 65535 名"从其中一个口子漏进来。这里把它当契约来测。
    """

    def test_keeps_ordinary_values(self):
        from gt7coach.contract import norm_u16
        assert norm_u16(7) == 7 and norm_u16(20.0) == 20

    def test_sentinel_is_unknown(self):
        """菜单态把 lap / num_cars / quali_pos 写成 65535。"""
        from gt7coach.contract import norm_u16
        assert norm_u16(65535) == 0

    def test_negative_is_unknown(self):
        from gt7coach.contract import norm_u16
        assert norm_u16(-1) == 0

    def test_non_numeric_is_unknown(self):
        from gt7coach.contract import norm_u16
        assert norm_u16(None) == 0
        assert norm_u16("12") == 0
        assert norm_u16(True) == 0      # bool 不是名次

    def test_float_is_truncated_not_rounded(self):
        """名次/圈数是整数语义；向下截断而不是四舍五入。"""
        from gt7coach.contract import norm_u16
        assert norm_u16(7.9) == 7
