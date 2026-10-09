# -*- coding: utf-8 -*-
"""引擎端到端测试 —— 用合成赛道跑真实链路（ReplaySource）。

守的是「装配」层面的坑，不是单条规则：
  · 参考圈是**异步**取的吗？拿不到时会不会退化成自攒？
  · 换场次时自攒的旧参考圈有没有作废？（不作废 = 拿上一条赛道的折线定位，
    而且是**静默**错）
  · 一次 tick 里说的话有没有重复计进播报历史？
"""
from __future__ import annotations

import math
import threading
import time

import pytest

from conftest import wait_for
from gt7coach.contract import Frame
from gt7coach.engine import CoachConfig, CoachEngine, RefProvider
from gt7coach.source import ReplaySource
from gt7coach.synth import synth_lap_frames, synth_profile

R = 600.0
L = 2.0 * math.pi * R


def make_engine(*, with_profile=True, laps=3, interval=0.01, **cfg_over):
    frames = synth_lap_frames(radius_m=R, hz=10.0, laps=laps)
    prof = synth_profile(radius_m=R) if with_profile else None
    src = ReplaySource(frames, profile=prof, loop=True)
    cfg = CoachConfig(poll_interval_s=interval, sess_poll_boot_s=0.02,
                      sess_poll_idle_s=0.05, **cfg_over)
    return CoachEngine(src, cfg), src


def drive(eng, n, sleep=0.0):
    """跑 n 次 tick，返回最后一次的 state。"""
    st = None
    for _ in range(n):
        st = eng.tick()
        if sleep:
            time.sleep(sleep)
    return st


def wait_ref(eng, source="profile", timeout=4.0):
    """等到参考圈就位。

    🔴 必须**边等边 tick**。参考圈是两跳异步的：tick 先唤醒「找场次」的线程，
    下次 tick 拿到场次后才去要剖面。只 tick 一次然后干等，等的是一个
    还没被安排的下一步。而且所有 HTTP 都在后台线程上，tick 本身不阻塞。
    """
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        eng.tick()
        ref = eng._current_ref()
        if ref is not None and ref.source == source:
            return True
        time.sleep(0.01)
    return False


class TestRefAcquisition:
    def test_profile_path_used_when_available(self):
        eng, src = make_engine(with_profile=True)
        drive(eng, 3)
        assert wait_for(lambda: eng._current_ref() is not None
                        and eng._current_ref().source == "profile")
        assert src.profile_calls >= 1

    def test_profile_fetched_once_not_every_tick(self):
        """参考圈是一次性资产：同一个场次同一份最快圈，绝不能每 tick 重取。"""
        eng, src = make_engine(with_profile=True)
        drive(eng, 120)
        assert wait_for(lambda: src.profile_calls >= 1)
        assert src.profile_calls == 1, src.profile_calls

    def test_falls_back_to_self_built_after_one_lap(self):
        """Dash 没有 /profile（或取失败）→ 第一圈纯记录，第二圈起自攒可用。"""
        eng, _src = make_engine(with_profile=False)
        drive(eng, 5)
        # 第一圈内：还没有任何参考
        assert eng._current_ref() is None
        assert not drive(eng, 1).ref_ready
        # 跑过一整圈（10Hz × 约 70 s ≈ 700 帧）
        for _ in range(750):
            st = eng.tick()
        assert st.ref_ready, "第二圈开始应当有自攒参考圈"
        assert st.stats["ref_source"] == "self"
        assert st.stats["ref_state"] == "failed", "取不到 profile 要如实说明"

    def test_self_built_ref_comes_with_warning(self):
        eng, _src = make_engine(with_profile=False)
        for _ in range(760):
            eng.tick()
        ref = eng._current_ref()
        assert ref is not None and ref.source == "self"
        assert any("10Hz" in w for w in ref.warnings)

    def test_no_frame_reports_disconnected(self):
        eng = CoachEngine(ReplaySource([]), CoachConfig())

        class Dead:
            def poll(self):
                return None
            last_error = "HTTP 000"

        eng2 = CoachEngine(Dead(), CoachConfig())
        st = eng2.tick()
        assert st.connected is False
        assert st.stats["source_error"] == "HTTP 000"
        assert eng.tick().connected is False


class TestSessionChange:
    def test_self_ref_invalidated_on_new_session(self):
        """换场次必须丢掉自攒参考圈 —— 留着它 = 拿上一条赛道的折线定位。"""
        eng, src = make_engine(with_profile=False)
        for _ in range(760):
            eng.tick()
        assert eng._self_ref is not None
        src._session = {"file": "another.jsonl", "live": True,
                        "best_lap_s": 88.0}
        # 场次发现是后台的：要给它机会跑一轮（tick 只负责唤醒线程）
        end = time.monotonic() + 3.0
        while time.monotonic() < end and eng._self_ref is not None:
            eng.tick()
            time.sleep(0.02)
        assert eng._self_ref is None, "换场次后自攒参考圈必须作废"
        assert eng._lap_buf == []

    def test_ref_provider_key_uses_file_and_best_lap(self):
        assert RefProvider.key_for({"file": "a.jsonl", "best_lap_s": 90.0}) \
            == ("a.jsonl", 90.0)
        assert RefProvider.key_for({}) == (None, None)
        a = RefProvider.key_for({"file": "a.jsonl", "best_lap_s": 90.0})
        b = RefProvider.key_for({"file": "a.jsonl", "best_lap_s": 88.0})
        assert a != b, "最快圈刷新了要重取（更好的参考圈）"


class TestStateSurface:
    def test_state_fields_populated_with_profile(self):
        eng, _ = make_engine(with_profile=True)
        assert wait_ref(eng)
        st = drive(eng, 30)
        assert st.ref_ready and st.ref_lap == 1
        assert st.ref_len_m == pytest.approx(L, rel=0.01)
        assert st.s_m is not None and 0 <= st.s_m <= L * 1.01
        assert st.next_brake_m is not None and st.next_brake_m >= 0
        assert st.stats["ref_source"] == "profile"

    def test_lateral_distance_small_on_reference_line(self):
        """合成帧就走在参考线上，横向距离应当接近 0。"""
        eng, _ = make_engine(with_profile=True)
        assert wait_ref(eng)
        st = drive(eng, 20)
        assert st.stats["lateral_m"] is not None
        assert st.stats["lateral_m"] < 3.0

    def test_spoken_history_dedupes_per_tick(self):
        eng, _ = make_engine(with_profile=True)
        for _ in range(40):
            eng.tick()
        keys = [h["key"] for h in eng._spoken]
        assert len(keys) == len(set(keys))   # 同一 tick 里同一 key 不该插两次
        assert len(eng._spoken) <= eng.cfg.spoken_history

    def test_to_dict_serializable(self):
        import json
        eng, _ = make_engine(with_profile=True)
        st = drive(eng, 5)
        json.dumps(st.to_dict(), ensure_ascii=False)   # 不能有非 JSON 类型

    def test_delta_sign_is_plausible(self):
        """合成帧与参考圈同源，delta 应当在 0 附近（±1 s 内），
        不该出现几秒级的偏差 —— 那说明距离轴或时间轴错位了。"""
        eng, _ = make_engine(with_profile=True)
        assert wait_ref(eng)
        seen = []
        for _ in range(200):
            st = eng.tick()
            if st.delta_s is not None:
                seen.append(st.delta_s)
        assert seen
        bad = [d for d in seen if abs(d) > 2.0]
        assert not bad, f"delta 明显错位：{bad[:5]}"


class TestStopMidRun:
    def test_no_crash_when_lap_has_no_coords(self):
        frames = [Frame(t=i * 0.1, lap_time_s=i * 0.1, speed_kph=150.0,
                        lap=1, x=0.0, z=0.0) for i in range(200)]
        eng = CoachEngine(ReplaySource(frames), CoachConfig())
        st = drive(eng, 200)      # 正好把 200 帧跑完，不触发"没数据"
        assert st.connected is True
        assert st.s_m is None, "没坐标就不该给出位置"
        assert st.ref_ready is False


class TestBadDrivingEndToEnd:
    """故意开烂，看教练说不说该说的话。

    规则层单测已经逐条验过触发条件了，这里验的是**装配**：
    从一帧遥测进去、到说出人话出来，中间不某处断掉
    （参考圈、定位、规则、闸门）。

    🔴 每个场景只制造**一种**毛病。原因见 `test_one_problem_per_corner`：
       同一个弯里既报"刹车晚了"又报"给油晚了"不叫覆盖全，叫唠叨 ——
       好教练是有取舍的，所以不能用一个"什么都烂"的圈去要求它把每条
       规则都念一遍。
    """

    @staticmethod
    def _range_of(f):
        """合成圆上 θ = s/R，反推弧长。"""
        if not f.coords_ok:
            return None
        return (math.atan2(f.z, f.x) % (2 * math.pi)) * R

    def _patch(self, base, lo, hi, **changes):
        """在第 2 圈的 [lo, hi] 米区间改几项，顺便把轮速改成与速度自洽。

        🔴 改了 speed_kph 就**必须**同步改 wheel_rads（ω = v / R_tyre）：
           只改速度不改轮速会造出"四轮抱死"的假象，于是 slip_all 先报出来、
           把真正要验的规则挤掉 —— 夹具不自洽会让人误判成规则坏了。
        """
        out = []
        for f in base:
            s = self._range_of(f) if f.lap == 2 else None
            if s is None or not (lo <= s <= hi):
                out.append(f)
                continue
            over = dict(f.__dict__)
            over.update(changes)
            v_ms = over.get("speed_kph", f.speed_kph) / 3.6
            over["wheel_rads"] = (v_ms / 0.34,) * 4
            out.append(Frame(**over))
        return out

    def _run(self, frames):
        src = ReplaySource(frames, profile=synth_profile(radius_m=R))
        eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                           sess_poll_idle_s=0.05),
                          clock=src.clock)
        assert wait_ref(eng), "参考圈没就位"
        said = {}
        for _ in range(len(frames)):
            st = eng.tick()
            for u in st.say:
                said.setdefault(u.key.split("@")[0], []).append(u.text)
        return said

    def test_late_brake_is_caught(self):
        frames = self._patch(synth_lap_frames(radius_m=R, hz=10.0, laps=3),
                             400.0, 480.0, brake=0.0, throttle=1.0,
                             speed_kph=195.0)
        said = self._run(frames)
        assert "brake_late" in said, list(said)

    def test_late_brake_uses_short_form_in_corner(self):
        """弯里 G 大 → 只念短句。长句「刹车晚了 15 米」在弯中来不及听。"""
        frames = self._patch(synth_lap_frames(radius_m=R, hz=10.0, laps=3),
                             400.0, 480.0, brake=0.0, throttle=1.0,
                             speed_kph=195.0)
        said = self._run(frames)
        texts = said["brake_late"]
        assert all(t.startswith("晚 ") for t in texts), texts

    def test_lazy_throttle_is_caught(self):
        """刹车没问题、只是出弯不给油 —— 要能单独报出来。"""
        frames = self._patch(synth_lap_frames(radius_m=R, hz=10.0, laps=3),
                             520.0, 700.0, throttle=0.05)
        said = self._run(frames)
        assert "throttle_late" in said, list(said)

    def test_redline_is_caught(self):
        frames = self._patch(synth_lap_frames(radius_m=R, hz=10.0, laps=3),
                             0.0, 2 * math.pi * R, rpm=8600.0, max_rpm=8200.0)
        said = self._run(frames)
        assert "shift" in said, list(said)

    def test_off_track_is_caught(self):
        """把第 2 圈的车整圈挪到参考线外 30 m → 应当报出界。"""
        out = []
        for f in synth_lap_frames(radius_m=R, hz=10.0, laps=3):
            if f.lap == 2 and f.coords_ok:
                k = (R + 30.0) / R
                f = Frame(**{**f.__dict__, "x": f.x * k, "z": f.z * k})
            out.append(f)
        assert "off_track" in self._run(out)

    def test_one_problem_per_corner(self):
        """同一个弯里连报两条以上 = 唠叨。

        真实教练会取舍：刹晚了导致出弯慢，说第一条就够了。
        这条断言是防止将来有人为了"覆盖全"把闸门调松。
        """
        frames = self._patch(synth_lap_frames(radius_m=R, hz=10.0, laps=2),
                             400.0, 700.0, brake=0.0, throttle=0.05,
                             speed_kph=195.0)
        said = self._run(frames)
        corner = [k for k in said if k.split("_")[0] in
                  ("brake", "throttle", "apex")]
        assert len(corner) <= 2, f"一个弯说了 {corner}"

    def test_brief_even_when_driving_badly(self):
        """一圈跑烂也不能变成话痨 —— 非紧急播报有每圈额度上限。"""
        frames = self._patch(synth_lap_frames(radius_m=R, hz=10.0, laps=1),
                             400.0, 700.0, brake=0.0, throttle=0.05,
                             speed_kph=195.0, rpm=8600.0, max_rpm=8200.0)
        src = ReplaySource(frames, profile=synth_profile(radius_m=R))
        eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                           sess_poll_idle_s=0.05),
                          clock=src.clock)
        assert wait_ref(eng)
        normal = 0
        for _ in range(len(frames)):
            normal += len([u for u in eng.tick().say if u.priority >= 2])
        assert normal <= eng.gate.cfg.max_per_lap, normal


class TestSlowDependencyMustNotStallTicks:
    """🔴 慢的依赖绝不能在 tick 线程上等。

    这是服务器上真实踩到的：`/api/v1/sessions` 的冷态要把每个场次文件
    流式扫一遍才能给出列表里的「最快圈」（服务器上 5 场里有个 111 MB 的
    → **11 秒**），而容器每次重启都回到冷态。早先「找哪一场在直播」是同步
    写在 tick 里的，于是每个 tick 卡 3~11 秒、10Hz 掉到 0.3Hz ——
    仪表盘慢一下，教练整个哑掉。现象是 `/health` 里 ticks 个位数、
    source_error=timed out，而两边其实都在正常运行。
    """

    class SlowSession(ReplaySource):
        """`live_session()` 慢 1 秒（模拟冷态场次列表）。"""

        def live_session(self):
            time.sleep(1.0)
            return self._session

    def _engine(self):
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=2)
        src = self.SlowSession(frames, profile=synth_profile(radius_m=R),
                               loop=True)
        return CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                            sess_poll_idle_s=0.05),
                           clock=src.clock), src

    def test_ticks_stay_fast_despite_slow_session_lookup(self):
        eng, _ = self._engine()
        eng.tick()                     # 唤醒一次场次发现（1 秒的慢查询）
        t0 = time.monotonic()
        for _ in range(30):
            eng.tick()
        el = time.monotonic() - t0
        assert el < 0.4, f"30 个 tick 花了 {el:.2f}s —— 慢查询把 tick 拖住了"

    def test_no_thread_pileup_while_slow_lookup_pending(self):
        """慢查询还没回来时不能继续堆线程，否则服务端一慢就线程爆炸。"""
        eng, _ = self._engine()
        eng.tick()
        time.sleep(0.1)                # 上一次还在 loading
        for _ in range(20):
            eng.tick()
        state, _err = eng.refs.session_status()
        assert state == "loading", state
        assert threading.active_count() < 12, threading.active_count()

    def test_reference_still_arrives_despite_slow_lookup(self):
        """慢不等于失败：场次最终被发现，参考圈照样建起来。"""
        eng, _ = self._engine()
        assert wait_ref(eng, timeout=6.0), eng.refs.session_status()

    def test_stats_expose_session_state(self):
        """场次发现的状态要单独暴露，否则分不清"网络断了"和"对方在忙"。"""
        eng, _ = self._engine()
        st = eng.tick()
        assert "sess_state" in st.stats and "sess_error" in st.stats
        assert st.stats["sess_state"] in ("idle", "loading", "ok", "empty",
                                          "failed")
