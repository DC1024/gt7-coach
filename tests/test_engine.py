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
        rp = RefProvider(ReplaySource([]), policy="session_best")
        assert rp.key_for({"file": "a.jsonl", "best_lap_s": 90.0}) \
            == ("a.jsonl", 90.0, "session_best")
        assert rp.key_for({}) == (None, None, "session_best")
        a = rp.key_for({"file": "a.jsonl", "best_lap_s": 90.0})
        b = rp.key_for({"file": "a.jsonl", "best_lap_s": 88.0})
        assert a != b, "最快圈刷新了要重取（更好的参考圈）"

    def test_key_includes_policy(self):
        """改了参考圈策略必须重取 —— 否则切到 history_best 也不会有任何变化。"""
        a = RefProvider(ReplaySource([]), policy="session_best")
        b = RefProvider(ReplaySource([]), policy="history_best")
        sess = {"file": "a.jsonl", "best_lap_s": 90.0}
        assert a.key_for(sess) != b.key_for(sess)


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


class TestLocalLapStats:
    """R1.5：分段用时 / 油耗 / 预测圈速 —— **全部本地算，不打任何接口**。

    不调 Dash 的 `/sectors` 与 `/pitstops` 的理由：直播场次上调它们每次都要
    重解析整场 jsonl（缓存按 mtime/size，而直播文件一直在变），一场 20 万帧
    要 2s+ 且越来越贵；而这些数从手上的实时帧就能算。
    """

    @staticmethod
    def _engine(laps=3, ref_faster=1.0, **cfg):
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=laps)
        prof = synth_profile(radius_m=R)
        if ref_faster != 1.0:
            # 把参考圈"改快"一点，制造一个稳定的正 delta
            prof["t_rel_s"] = [t / ref_faster for t in prof["t_rel_s"]]
            prof["lap_time_s"] /= ref_faster
        src = ReplaySource(frames, profile=prof, loop=True)
        eng = CoachEngine(src,
                          CoachConfig(sess_poll_boot_s=0.02,
                                      sess_poll_idle_s=0.05,
                                      poll_interval_s=0.01, **cfg),
                          clock=src.clock)
        return eng, src

    def test_sectors_and_theory_after_laps(self):
        eng, _ = self._engine(laps=3)
        assert wait_ref(eng)
        # 跑完两整圈（10Hz × 约 70s ≈ 700 帧/圈）
        for _ in range(1500):
            eng.tick()
        st = eng.tick()
        assert st.last_lap is not None, "应当已经有跑完的圈"
        assert st.last_lap["ok"], st.last_lap
        secs = st.last_lap["sectors"]
        assert len(secs) == 3
        # 容差 5ms：state 里这三个数各自四舍五入到 3 位小数，
        # "舍入后的和"与"和的舍入"本来就可以差 1~2ms（不是算错）
        assert sum(secs) == pytest.approx(st.last_lap["lap_time_s"], abs=0.005)
        assert st.theory_best_s is not None
        # 理论最快 = 各段最好值之和，必然 ≤ 实际最快圈
        assert st.theory_best_s <= st.last_lap["lap_time_s"] + 1e-6
        assert st.potential_gain_s is not None and st.potential_gain_s >= 0.0

    def test_fuel_per_lap_and_range(self):
        """合成数据每圈正好耗 8.0（按里程线性），所以续航要能算准。"""
        eng, _ = self._engine(laps=3)
        assert wait_ref(eng)
        for _ in range(1500):
            eng.tick()
        st = eng.tick()
        assert st.fuel_per_lap == pytest.approx(8.0, abs=0.2)
        assert st.last_lap["fuel_used"] == pytest.approx(8.0, abs=0.2)
        # 剩余量 / 每圈油耗
        assert st.fuel_laps_left is not None
        assert st.fuel_laps_left > 0

    def test_projected_lap_appears_when_behind(self):
        eng, _ = self._engine(laps=3, ref_faster=1.015)
        assert wait_ref(eng)
        seen_projected, seen_state = False, False
        for _ in range(1200):
            st = eng.tick()
            if st.projected_lap_s is not None:
                seen_state = True
            if any(u.key == "projected_lap" for u in st.say):
                seen_projected = True
            if seen_projected:
                break
        assert seen_state, "落后期应当给出预测圈速"
        assert seen_projected, "预测圈速也要真的说出口"

    def test_projected_lap_consistent_with_delta(self):
        """预计圈速 = 参考圈圈速 + 当前 delta，三个数必须自洽。"""
        eng, _ = self._engine(laps=3, ref_faster=1.015)
        assert wait_ref(eng)
        for _ in range(1200):
            st = eng.tick()
            if st.projected_lap_s is not None and st.delta_s is not None:
                ref = eng._current_ref()
                assert st.projected_lap_s == pytest.approx(
                    ref.lap_time_s + st.delta_s, abs=0.01)
                return
        pytest.fail("始终没有同时拿到 projected 与 delta")

    def test_no_extra_endpoint_calls_for_lap_stats(self):
        """分段/油耗是本地算的 —— 跑完几圈不该多出任何接口调用。

        参考圈剖面仍然只取一次（它只能来自服务端）。
        """
        eng, src = self._engine(laps=3)
        assert wait_ref(eng)
        for _ in range(1500):
            eng.tick()
        assert src.profile_calls == 1, src.profile_calls
        assert eng._sectors.lap_totals, "本地分段应当已经算出来了"

    def test_partial_lap_does_not_pollute_theory(self):
        """中途接入时第一圈是残圈，不能进理论最快圈的统计。"""
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=2)
        # 从第 1 圈中途开始（掐掉前 25 秒）
        src = ReplaySource(frames[250:], profile=synth_profile(radius_m=R),
                           loop=True)
        eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                           sess_poll_idle_s=0.05),
                          clock=src.clock)
        assert wait_ref(eng)
        for _ in range(1600):
            eng.tick()
        # 第一圈（残）不该被计入；后续完整的圈才计
        assert eng._sectors.lap_totals, "后面完整的圈应当被计入"
        assert eng._prev_lap is not None
        # 若最后一条是残圈，它必须带着可读的原因而不是静默算错
        if not eng._prev_lap.ok:
            assert eng._prev_lap.why


class TestBestLapRefreshDoesNotWipeLapBuffer:
    """🔴 直播场次里「最快圈刷新」不等于「换场次」。

    每跑完一圈，场次列表里的 `best_lap_s` 就会变；而参考圈的 key 里含
    `best_lap_s`（刷新了要重取剖面，这是对的）。但如果把这个 key 变化
    当成"换场次"去处理，就会**清空帧缓冲与自攒参考圈** ——
    直播时每圈跑到 ~15 秒（下一次轮询）缓冲就被清，于是：
      · 分段统计永远是 0 圈
      · 自攒参考圈永远建不起来
    而表面上一切正常（指令照回、状态照出）—— 最坏的那种 bug。
    """

    @staticmethod
    def _engine():
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=3)
        src = ReplaySource(frames, profile=None, loop=True)
        src._session = {"file": "live.jsonl", "live": True, "best_lap_s": 69.7,
                        "car_name": "X"}
        eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                           sess_poll_idle_s=0.02,
                                           ref_policy="session_best"),
                          clock=src.clock)
        return eng, src

    def test_buffer_survives_best_lap_change(self):
        eng, src = self._engine()
        for _ in range(40):
            eng.tick()
        assert eng._lap_buf, "先得有帧"
        # 同一场次、只是最快圈刷新了
        src._session = {"file": "live.jsonl", "live": True,
                        "best_lap_s": 68.9, "car_name": "X"}
        for _ in range(20):
            eng.tick()
        assert eng._lap_buf, "最快圈刷新不该清空帧缓冲"
        assert eng._ref_sess_key[0] == "live.jsonl"
        # 但参考圈该被重取（key 变了）
        assert eng._ref_sess_key[1] == 68.9

    def test_real_session_change_still_wipes(self):
        """真的换场次（换文件）时，缓冲与自攒参考圈仍然必须作废。"""
        eng, src = self._engine()
        for _ in range(40):
            eng.tick()
        before = len(eng._lap_buf)
        assert before >= 30, before
        # 假装已经攒了一份自攒参考圈（要是个真 RefLap —— 引擎会拿它去定位）
        from gt7coach.refindex import RefLap
        eng._self_ref = RefLap(lap=99, source="self", grid_m=[0.0, 1.0],
                               speed_kph=[0.0, 0.0], throttle=[0.0, 0.0],
                               brake=[0.0, 0.0], t_rel_s=[0.0, 0.0],
                               xs=[0.0, 0.0], zs=[0.0, 0.0],
                               length_m=1.0, lap_time_s=1.0)
        src._session = {"file": "other.jsonl", "live": True,
                        "best_lap_s": 70.0, "car_name": "X"}
        for _ in range(20):
            eng.tick()
        # 🔴 清空之后**后续 tick 又会攒新帧**，所以判据是"缓冲变回新场次的长度"
        #    而不是"空数组"（第一版就是拿 `== []` 断的，必假）。
        assert len(eng._lap_buf) <= 21, f"{before} → {len(eng._lap_buf)}"
        assert eng._self_ref is None, "换文件必须作废自攒参考圈"
        assert eng._sector_len_m is None, "换赛道分段基准要重来"

    def test_lap_stats_accumulate_across_laps(self):
        """连跑三圈，分段统计必须真的累起来（这是上面那个 bug 的可见症状）。"""
        eng, _ = self._engine()
        for _ in range(2400):
            eng.tick()
        assert len(eng._sectors.lap_totals) >= 1, \
            f"分段统计没累起来（lap_samples={len(eng._sectors.lap_totals)}）"
        assert eng._sector_len_m is not None


class TestCornerHabitEndToEnd:
    """R1.6：跨圈累积「哪个弯反复亏」，第 3 圈起主动播报。

    这是整条链路里唯一需要跨圈累积的规则，也是教练和仪表盘最本质的区别：
    仪表盘显示**现在**，教练告诉你**你的习惯**。
    """

    def test_fires_after_three_laps_of_same_corner(self):
        # 直播车整体比参考圈慢 ~5% → 每个弯都亏，累积 3 圈后应当主动指出
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=4,
                                  base_kph=190.0, dip_kph=85.0)
        src = ReplaySource(frames, profile=synth_profile(radius_m=R), loop=True)
        eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                           sess_poll_idle_s=0.05),
                          clock=src.clock)
        assert wait_ref(eng)
        fired, habit = [], None
        for _ in range(2900):
            st = eng.tick()
            for u in st.say:
                if u.key.startswith("corner_habit"):
                    fired.append((st.lap, u.text))
            if eng._corners.habit():
                habit = eng._corners.habit()
        assert habit is not None, "4 圈之后应当已经形成习惯"
        assert habit["label"] == "T1"
        assert habit["median_loss_s"] > 0.3
        assert fired, f"习惯形成了却没播报；losses={eng._corners.losses}"
        # 只有一个弯 → 文案里必须点名那个弯
        assert "T1" in fired[0][1], fired

    def test_does_not_fire_every_lap(self):
        """同一个弯隔 3 圈才提醒第二次 —— 每圈念同一句就成了唠叨。"""
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=4,
                                  base_kph=190.0, dip_kph=85.0)
        src = ReplaySource(frames, profile=synth_profile(radius_m=R), loop=True)
        eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                           sess_poll_idle_s=0.05),
                          clock=src.clock)
        assert wait_ref(eng)
        laps_spoken = []
        for _ in range(2900):
            st = eng.tick()
            for u in st.say:
                if u.key.startswith("corner_habit"):
                    laps_spoken.append(st.lap)
        assert laps_spoken, "应当至少播报一次"
        if len(laps_spoken) > 1:
            assert laps_spoken[1] - laps_spoken[0] >= 3, laps_spoken

    def test_stats_expose_corner_state(self):
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=2)
        src = ReplaySource(frames, profile=synth_profile(radius_m=R), loop=True)
        eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                           sess_poll_idle_s=0.05),
                          clock=src.clock)
        assert wait_ref(eng)
        for _ in range(700):
            eng.tick()
        st = eng.tick()
        assert "corners" in st.stats and "corner_habit" in st.stats
        assert st.stats["corners"]["corners"] >= 1
