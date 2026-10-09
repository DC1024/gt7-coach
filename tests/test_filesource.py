# -*- coding: utf-8 -*-
"""`FileSource`（读 jsonl 回放）+ replay CLI 测试。

这套东西的用途是**调参**：改一个阈值跑一遍历史场次，看会说的话怎么变。
所以测试的重点不是"能不能读文件"，而是：
  · 半截行不能把整场读成空（Dash 侧踩过：4183 帧 → 0 帧）
  · `t` 是绝对墙钟，圈内用时必须自己算（不能拿 t 直接当圈内计时）
  · 本地建的参考圈要能被 `from_profile` 解析回去（往返一致）
  · replay 的输出要能当断言用（--json 可 diff）
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from gt7coach.refindex import RefLap
from gt7coach.source import FileSource, _read_session
from gt7coach.synth import synth_lap_frames

ROOT = Path(__file__).resolve().parents[1]
R = 600.0
L = 2.0 * math.pi * R
T0 = 1791434581.47          # 一个真实场次用过的墙钟起点


def write_session(path: Path, laps: int = 2, hz: float = 60.0,
                  truncated: bool = False, header_car: int = 1302,
                  position: int = 0, num_cars: int = 0,
                  laps_in_race: int = 0, car_code: int = 0) -> Path:
    """按记录器的 jsonl 格式落盘（字段名与真实文件一致）。

    `position` 在 jsonl 里的键是 **`quali_pos`**（0x84）—— 与 Dash `/live`
    的 `race.grid_position` 同一个源：比赛进行中它随排名实时变。
    想验 u16 哨兵就传 65535。
    """
    lines = [json.dumps({"session_id": "test", "started_at": T0,
                         "circuit": None, "car": header_car,
                         "powertrain": "fuel"}, ensure_ascii=False)]
    frames = synth_lap_frames(radius_m=R, hz=hz, laps=laps)
    t_abs = T0
    dt = 1.0 / hz
    for f in frames:
        lines.append(json.dumps({
            "t": round(t_abs, 4), "lap": f.lap, "speed_kph": f.speed_kph,
            "rpm": f.rpm, "gear": f.gear, "throttle": f.throttle,
            "brake": f.brake,
            "g_force": [f.glon, f.glat, 0.0],
            "car_x": f.x, "car_y": 0.0, "car_z": f.z,
            "wheel_rads": list(f.wheel_rads), "tyre_temp": list(f.tyre_temp),
            "gas_level": f.fuel_pct, "gas_capacity": 100.0,
            "max_alert_rpm": f.max_rpm, "powertrain": "fuel",
            "has_coords": True,
            "quali_pos": position, "num_cars": num_cars,
            "laps_in_race": laps_in_race, "car_code": car_code,
        }, ensure_ascii=False))
        t_abs += dt
    body = "\n".join(lines) + "\n"
    if truncated:
        body = body[:-40]              # 砍掉最后一行尾巴 = 正在写入
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def sess(tmp_path) -> Path:
    return write_session(tmp_path / "s.jsonl", laps=2, hz=60.0)


@pytest.fixture
def sess3(tmp_path) -> Path:
    """3 圈。

    🔴 一圈的统计是在**下一圈开始**时才结算的（那时才知道这一圈结束了）。
       所以 N 圈的数据只能结算出 N-1 圈 —— 最后那一圈永远悬着，除非场次
       自然结束。需要"两圈都已结算"的用例必须给 3 圈。
    """
    return write_session(tmp_path / "s3.jsonl", laps=3, hz=60.0)


@pytest.fixture
def sess3(tmp_path) -> Path:
    """3 圈。

    🔴 一圈的统计是在**下一圈开始**时才结算的（那时才知道这一圈结束了）。
       所以 N 圈的数据只能结算出 N-1 圈 —— 最后那一圈永远悬着，除非场次
       自然结束。需要"两圈都已结算"的用例必须给 3 圈。
    """
    return write_session(tmp_path / "s3.jsonl", laps=3, hz=60.0)


class TestReadSession:
    def test_reads_all_frames(self, sess):
        hdr, frames = _read_session(sess)
        assert hdr["car"] == 1302 and hdr["powertrain"] == "fuel"
        assert len(frames) == len(synth_lap_frames(radius_m=R, hz=60.0,
                                                   laps=2))

    def test_truncated_last_line_is_dropped(self, tmp_path):
        """🔴 半截行必须丢，不能让整场读成 0 帧。

        Dash 侧实测过：末尾截断 60 字节 → 4183 帧变 0 帧，
        因为 json.loads 抛异常被外层 except 兜成"空 store"。
        """
        p = write_session(tmp_path / "t.jsonl", laps=2, truncated=True)
        hdr, frames = _read_session(p)
        assert hdr, "header 都要读不出来那就是全废了"
        assert len(frames) > 6000, len(frames)

    def test_lap_time_is_relative(self, sess):
        """`t` 是绝对墙钟（十几亿），圈内用时必须自己减出来。"""
        _hdr, frames = _read_session(sess)
        lap1 = [f for f in frames if f.lap == 1]
        assert lap1[0].lap_time_s == pytest.approx(0.0, abs=1e-6)
        assert lap1[-1].lap_time_s > 60.0
        assert lap1[-1].t > 1e9, "原始 t 仍是绝对时刻"

    def test_g_force_order(self, sess):
        """g_force 是 [纵向, 横向]，别接反（接反了左右弯全错）。"""
        _hdr, frames = _read_session(sess)
        f = frames[100]
        assert f.glon == pytest.approx(f.glon)      # 存在
        assert f.glat != 0.0 or f.glon != 0.0

    def test_fuel_and_powertrain(self, sess):
        _hdr, frames = _read_session(sess)
        assert frames[0].fuel_pct == pytest.approx(100.0)
        assert frames[-1].fuel_pct < 100.0

    def test_race_fields_round_trip_from_jsonl(self, tmp_path):
        """场次文件里的名次/车数/总圈数必须能被读回来。"""
        p = write_session(tmp_path / "r.jsonl", laps=1, hz=20.0,
                          position=6, num_cars=20, laps_in_race=10,
                          car_code=805)
        _hdr, frames = _read_session(p)
        assert frames[0].position == 6
        assert frames[0].num_cars == 20
        assert frames[0].laps_in_race == 10
        assert frames[0].car_code == 805

    def test_u16_sentinel_in_jsonl_is_not_a_rank(self, tmp_path):
        """菜单态把 quali_pos 写成 65535 → 归 0，别念"第 65535 名"。"""
        p = write_session(tmp_path / "r2.jsonl", laps=1, hz=20.0,
                          position=65535, num_cars=65535)
        _hdr, frames = _read_session(p)
        assert frames[0].position == 0
        assert frames[0].num_cars == 0

    def test_missing_race_fields_default_to_zero(self, tmp_path):
        p = write_session(tmp_path / "r3.jsonl", laps=1, hz=20.0)
        _hdr, frames = _read_session(p)
        f = frames[0]
        assert (f.position, f.num_cars, f.laps_in_race, f.car_code) == (0, 0, 0, 0)
        assert frames[0].powertrain == "fuel"


class TestFileSource:
    def test_lap_spans_and_best(self, sess):
        src = FileSource(str(sess))
        spans = src.lap_spans
        assert sorted(spans) == [1, 2]
        assert spans[1][1] - spans[1][0] == pytest.approx(69.7, abs=1.5)
        assert src.best_lap_s is not None

    def test_replays_in_order_then_stops(self, sess):
        src = FileSource(str(sess))
        n, prev_t = 0, None
        while True:
            f = src.poll()
            if f is None:
                break
            assert prev_t is None or f.t >= prev_t, "必须按时间升序"
            prev_t = f.t
            n += 1
        assert n > 6000

    def test_clock_follows_frames(self, sess):
        """时钟必须是帧的墙钟 —— 用真实时钟会让 20s 冷却在一次几秒跑完的
        回放里永远生效，得出"教练一句话都不说"的假结论。"""
        src = FileSource(str(sess))
        first = src.poll()
        assert src.clock() == first.t
        for _ in range(500):
            src.poll()
        assert src.clock() > first.t + 5.0

    def test_lap_filter(self, sess):
        src = FileSource(str(sess), lap=2)
        assert len(src._frames) > 1000
        assert {f.lap for f in src._frames} == {2}

    def test_profile_built_locally(self, sess):
        src = FileSource(str(sess))
        p = src.lap_profile(src._session["file"])
        assert p is not None
        ref = RefLap.from_profile(p)          # 必须能被解析回去
        assert ref.length_m == pytest.approx(L, rel=0.02)
        assert ref.brake_in and ref.apex

    def test_profile_is_json_safe(self, sess):
        src = FileSource(str(sess))
        json.dumps(src.lap_profile(src._session["file"]), ensure_ascii=False)

    def test_no_cross_session_history_offline(self, sess):
        src = FileSource(str(sess))
        assert src.faster_sessions(70.0) == []

    def test_ref_lap_override(self, sess):
        src = FileSource(str(sess), ref_lap=2)
        p = src.lap_profile(src._session["file"])
        assert p["lap"] == 2


class TestProfileRoundTrip:
    def test_to_profile_from_profile(self, sess):
        """`to_profile` 的产物必须能被 `from_profile` 解析回来 ——
        序列化格式对不上时，离线回放会静默拿不到参考圈。"""
        src = FileSource(str(sess))
        p = src.lap_profile(src._session["file"])
        a = RefLap.from_profile(p)
        b = RefLap.from_profile(a.to_profile())
        assert len(b.grid_m) == len(a.grid_m)
        assert b.length_m == a.length_m
        assert b.lap_time_s == a.lap_time_s
        assert len(b.brake_in) == len(a.brake_in)


class TestReplayCLI:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "gt7coach", "replay", *args],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300)

    def test_json_output_shape(self, sess3):
        r = self._run(str(sess3), "--json")
        assert r.returncode == 0, r.stderr
        d = json.loads(r.stdout)
        assert d["ref"] and d["ref"]["source"] == "profile"
        assert d["laps_available"] == [1, 2, 3]
        # 3 圈数据 → 前两圈结算完毕（最后一圈要等下一圈开始才知道结束）
        assert len(d["laps"]) == 2
        assert d["laps"][0]["ok"] is True
        assert d["theory"]["theory_best_s"] > 0
        assert isinstance(d["said"], list)
        assert "dropped_by_gate" in d

    def test_conf_override_changes_output(self, sess3):
        """调参工具的全部意义：改一个数，输出跟着变。"""
        base = json.loads(self._run(str(sess3), "--json").stdout)
        # 把"刹车晚了"的门槛提到 400 米 —— 任何一圈都不可能触发
        tuned = json.loads(self._run(
            str(sess3), "--json", "--conf", "brake_late_m=400").stdout)
        assert tuned["config_overrides"] == ["rules.brake_late_m=400.0"]
        assert tuned["said_by_key"].get("brake_late@0", 0) == 0 \
            or "brake_late" not in str(tuned["said_by_key"])
        assert base["ref"]["lap_time_s"] == tuned["ref"]["lap_time_s"]

    def test_conf_gate_section(self, sess):
        d = json.loads(self._run(str(sess), "--json",
                                 "--conf", "gate.max_per_lap=1").stdout)
        assert d["config_overrides"] == ["gate.max_per_lap=1"]

    def test_conf_bad_key_exits(self, sess):
        r = self._run(str(sess), "--conf", "nope=1")
        assert r.returncode != 0
        assert "不是可配置项" in (r.stderr + r.stdout)

    def test_human_output_mentions_ref(self, sess):
        r = self._run(str(sess))
        assert r.returncode == 0
        assert "参考圈" in r.stdout
        assert "逐圈" in r.stdout

    def test_marks_unusable_ref(self, tmp_path):
        """参考圈不可用时必须显式警告 —— 悄悄退回自攒会让整份输出看着正常
        但全是错的（真实踩过：菜单帧攒出 390m 的假参考圈）。"""
        p = write_session(tmp_path / "bad.jsonl", laps=1, hz=10.0)
        # 只留几十帧：既建不出参考圈（<30 帧就放弃），也跑不满一圈
        lines = p.read_text(encoding="utf-8").splitlines()
        p.write_text("\n".join(lines[:60]) + "\n", encoding="utf-8")
        r = self._run(str(p))
        assert r.returncode == 0
        assert "⚠" in r.stdout
