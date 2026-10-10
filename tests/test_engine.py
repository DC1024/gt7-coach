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


def make_engine(*, with_profile=True, laps=6, interval=0.01, **cfg_over):
    """默认 laps=6 —— 故意给足圈数。

    🔴 回放源 `loop=True` 绕回来时，"下一帧"就是第 1 圈第 0 帧，与**重开比赛**
       在引擎眼里**一模一样**（圈号倒退 → `_reset_run`，清掉自攒参考圈、
       分段基准、每弯累积）。那是引擎正确的行为，但会让"只想看跑了 4 圈以后
       长什么样"的测试在第 5 圈悄悄被清空。给足圈数就不会撞上这件事。
    """
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


def drive_to_lap(eng, lap_n, max_iter=4000):
    """一直 tick 到**跑到第 lap_n 圈**为止，返回那一刻的 state。

    比数帧数稳健：一圈几帧会随合成参数漂移，而这些测试真正关心的时刻是
    「第 1 圈跑完之后」（参考圈从这一圈起才拿得出手）。
    """
    st = None
    for _ in range(max_iter):
        st = eng.tick()
        if st.lap >= lap_n:
            return st
    return st


def drive_until_lap(eng, lap_n, max_iter=4000):
    """一直 tick 到「第 lap_n 圈已结算」为止（_prev_lap.lap >= lap_n）。

    比数帧数稳健：合成一圈的帧数不固定，靠「圈号」判定才不会因为
    多跑/少跑几十帧而误判。
    """
    for _ in range(max_iter):
        eng.tick()
        if eng._prev_lap is not None and eng._prev_lap.lap >= lap_n:
            return True
    return False


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

    def test_falls_back_to_self_built_after_two_laps(self):
        """Dash 没有 /profile（或取失败）→ 前两圈（含出场圈）纯记录，
        第 2 圈跑完才开始自攒可用。

        🔴 这正是「前几圈瞎播报」的根治点：`self_ref_min_lap=2` 让第 1 圈
        （出场 / 暖胎圈，慢且不具代表性）不参与自攒参考圈的标定，
        否则第 2~3 圈的刹车点 / 弯心速度全建在慢圈上 → 用户感知的瞎播报。
        """
        eng, _src = make_engine(with_profile=False)
        drive(eng, 5)
        # 第一圈内：还没有任何参考
        assert eng._current_ref() is None
        assert not drive(eng, 1).ref_ready
        # 跑过第 1 圈：它刚结算，但按 self_ref_min_lap=2 它**不能**成为参考圈
        assert drive_until_lap(eng, 1)
        assert eng._self_ref is None, "第 1 圈不应进入自攒参考圈"
        # 再跑过第 2 圈：自攒参考圈就位
        assert drive_until_lap(eng, 2)
        st = eng.tick()
        assert st.ref_ready, "第 2 圈跑完应当有自攒参考圈"
        assert st.stats["ref_source"] == "self"
        assert st.stats["ref_state"] == "failed", "取不到 profile 要如实说明"

    def test_self_built_ref_comes_with_warning(self):
        eng, _src = make_engine(with_profile=False)
        # 自攒参考圈最早在第 2 圈跑完才建（self_ref_min_lap=2）
        assert drive_until_lap(eng, 2)
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
        # 自攒参考圈最早在第 2 圈跑完才建（self_ref_min_lap=2）
        assert drive_until_lap(eng, 2)
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
        # 🔴 第 1 圈只播报不依赖参考圈的信息（见 TestWarmupNoReferenceLap），
        #    `ref_ready` 在这一圈必须是 False —— 要验"字段填对了"就得先跑完
        #    第 1 圈。
        st = drive_to_lap(eng, 2)
        assert st.ref_ready and st.ref_lap == 1
        assert st.ref_len_m == pytest.approx(L, rel=0.01)
        assert st.s_m is not None and 0 <= st.s_m <= L * 1.01
        assert st.next_brake_m is not None and st.next_brake_m >= 0
        assert st.stats["ref_source"] == "profile"

    def test_lateral_distance_small_on_reference_line(self):
        """合成帧就走在参考线上，横向距离应当接近 0。"""
        eng, _ = make_engine(with_profile=True)
        assert wait_ref(eng)
        st = drive_to_lap(eng, 2)
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
        不该出现几秒级的偏差 —— 那说明距离轴或时间轴错位了。

        🔴 第 1 圈不收（暖胎期不把参考圈交给规则层），所以要多跑一点。
        """
        eng, _ = make_engine(with_profile=True)
        assert wait_ref(eng)
        seen = []
        for _ in range(1400):
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
        """把第 2 圈的车整圈挪到参考线外 30 m → 应当报出界。

        🔴 #I：出界现在是「横向偏离 **且** 轮胎打滑」双重确认。真车冲出
        柏油必然伴随打滑，所以把这一圈轮速也改成"打滑态"（1.12× 自由滚动），
        否则纯挪坐标只会让横向偏离达标、却因无打滑而不报，反把测试带偏。
        """
        out = []
        for f in synth_lap_frames(radius_m=R, hz=10.0, laps=3):
            if f.lap == 2 and f.coords_ok:
                k = (R + 30.0) / R
                v_ms = f.speed_kph / 3.6
                w = v_ms / 0.34 * 1.12   # 真实出界 = 跑出柏油 = 轮胎打滑
                f = Frame(**{**f.__dict__, "x": f.x * k, "z": f.z * k,
                             "wheel_rads": (w, w, w, w), "throttle": 0.95})
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
        # laps=3（不是 2）：回放绕回圈号倒退 = 引擎眼里的"重开比赛"（见
        # make_engine 的说明），圈数给足，避免在观察窗内撞上那次清空。
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=3)
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
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=5)
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
        # laps=5：弯窗口在第 1 圈跑完时才依据参考圈生成，随后每圈记一条损失；
        # 攒够 corner_min_laps=3 条要到第 4 圈末，而回放绕回会被当成重开比赛
        # （见 make_engine），所以圈数要留出余量。
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=5,
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
                # R2.4 起习惯弯并进 `lap_advice`（不再单独发 corner_habit@）；
                # 用"文本点名 T1"识别含习惯的那条合并句。
                if u.key == "lap_advice" and "T1" in u.text:
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
        # laps=5：弯窗口在第 1 圈跑完时才依据参考圈生成，随后每圈记一条损失；
        # 攒够 corner_min_laps=3 条要到第 4 圈末，而回放绕回会被当成重开比赛
        # （见 make_engine），所以圈数要留出余量。
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=5,
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
                # 习惯弯并进 lap_advice（见上一条）；按"点名 T1"识别。
                if u.key == "lap_advice" and "T1" in u.text:
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
        # 900 帧 > 一圈（约 700 帧）—— 弯窗口是在第 1 圈**跑完**那一刻依参考圈
        # 生成的，700 帧只是勉强够到边界（wait_ref 已经吃掉几十帧）。
        for _ in range(900):
            eng.tick()
        st = eng.tick()
        assert "corners" in st.stats and "corner_habit" in st.stats
        assert st.stats["corners"]["corners"] >= 1


class TestCornerAccountingOncePerLap:
    """🔴 回归守卫：**每圈只能记一次**每弯损失。

    这条测试是被一个真 bug 逼出来的：`_finalize_lap` 里那段"每弯累积"的代码
    因为生成脚本跑了两遍而**重复出现了两次**，后果是每圈的损失被记两遍、
    `laps` 翻倍 —— "连续 3 圈"实际是"连续 6 圈"。
    而当时 231 项测试**全绿**：因为断言只看了"有没有播报"，
    没看样本数。样本数是这条规则唯一的信息量所在（"偶发"vs"习惯"），
    翻倍等于把偶发说成习惯 —— 这是最坏的一种错。
    """

    def test_lap_counted_once_per_lap(self):
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=5,
                                  base_kph=190.0, dip_kph=85.0)
        src = ReplaySource(frames, profile=synth_profile(radius_m=R), loop=True)
        eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                           sess_poll_idle_s=0.05),
                          clock=src.clock)
        assert wait_ref(eng)
        for _ in range(2900):
            eng.tick()
        losses = eng._corners.losses.get("T1") or []
        laps_done = eng._prev_lap.lap if eng._prev_lap else 0
        assert losses, "跑了好几圈却一条损失都没记"
        # 5 圈 → 最多 5 条（首圈可能不完整）。翻倍会是 8~10 条。
        assert len(losses) <= laps_done, (
            f"记了 {len(losses)} 条损失，但只跑了 {laps_done} 圈 —— "
            f"每弯累积被记了多次（见本类文档）")
        assert eng._corners.habit()["laps"] == len(losses)

    def test_laps_count_matches_actual_laps_in_speech(self):
        """播报里说的圈数必须等于真实样本数 —— 玩家听到的"连续 N 圈"要可信。"""
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=6,
                                  base_kph=185.0, dip_kph=85.0)
        src = ReplaySource(frames, profile=synth_profile(radius_m=R), loop=True)
        eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                           sess_poll_idle_s=0.05),
                          clock=src.clock)
        assert wait_ref(eng)
        spoken = []
        for _ in range(len(frames) + 50):
            st = eng.tick()
            for u in st.say:
                # 习惯弯并进 lap_advice；含习惯的合并句 evidence 里带 focus_*。
                if u.key == "lap_advice" and "focus_label" in u.evidence:
                    spoken.append(u)
        assert spoken, "6 圈之后应当播报过"
        for u in spoken:
            n = u.evidence["focus_laps"]
            # 🔴 只能校验「文本自洽」与「当时至少够门槛」，**不能**拿
            #    `eng._corners.losses` 的**当前长度**去比 —— 那是"事后状态"，
            #    而播报是第 3 圈那一刻发生的（之后样本还在继续增长）。
            #    初版就这么写错了，于是断言 3 == 5 失败，看起来像代码有问题。
            #    「用活的、会继续增长的状态去校验过去的事件」是个通用陷阱。
            assert n >= 3, f"样本不足就不该播报：{u.evidence}"
            assert f"连续 {n} 圈" in u.text, u.text
            assert u.evidence["focus_label"] in u.text


class TestWarmupNoReferenceLap:
    """开局第 1 圈：**只播报不需要参考圈的信息**。

    这是用户的明确要求：「跑完一圈（参考圈从第 1 圈变成第 2 圈）才开始播报」。
    它背后是两个真 bug：
      · Dash 的 `/profile` 在本场只有一圈时会把**正在跑的那一圈**回给我们
        （那边 `usable = trimmed or usable`），于是第 1 圈在拿半圈跟自己比，
        报出「1.4 秒后重刹」「给油晚了」；
      · 重开比赛 / 换场次之后教练还在接着播报（上一轮的参考圈与每弯累积）。

    判据因此从"有没有 ref 对象"改成"本场有没有**跑完**的圈"。
    """

    # 依赖参考圈的规则键（引擎在暖胎期把 ref/s 收回去，这些自然全都出不来）
    REF_KEYS = ("off_track", "brake_warn", "brake_late", "apex_slow",
                "throttle_late", "delta", "projected_lap")

    BAD = dict(brake=0.0, throttle=1.0, speed_kph=195.0)

    @classmethod
    def _frames(cls, laps, bad=(1, 2)):
        """把 `bad` 里那几圈的 400~480 m 段改成「该刹不刹」，并顺手拉爆转速。

        🔴 参考圈**始终**是同一份完整的合成剖面，所以"第 1 圈不说、第 2 圈说"
           只可能由暖胎期解释，不是夹具在偏心。转速拉爆是为了留一个
           **不依赖参考圈**的正对照（`shift`）—— 否则"第 1 圈什么都没说"
           和"教练整个哑掉了"分不出来。
        """
        out = []
        for f in synth_lap_frames(radius_m=R, hz=10.0, laps=laps):
            s = (math.atan2(f.z, f.x) % (2 * math.pi)) * R if f.coords_ok else None
            if f.lap in bad and s is not None and 400.0 <= s <= 480.0:
                over = dict(f.__dict__)
                over.update(cls.BAD, rpm=8600.0)
                v = over.get("speed_kph", f.speed_kph) / 3.6
                over["wheel_rads"] = (v / 0.34,) * 4
                f = Frame(**over)
            out.append(f)
        return out

    @staticmethod
    def _engine(frames, **cfg_over):
        src = ReplaySource(frames, profile=synth_profile(radius_m=R), loop=False)
        cfg = CoachConfig(sess_poll_boot_s=0.02, sess_poll_idle_s=0.05, **cfg_over)
        return CoachEngine(src, cfg, clock=src.clock), src

    def _drive(self, eng, n):
        """跑完 n 帧，返回 [(tick, lap, key)]。"""
        said = []
        for i in range(n):
            st = eng.tick()
            for u in st.say:
                said.append((i, st.lap, u.key.split("@")[0]))
        return said

    def test_first_lap_says_nothing_ref_dependent(self):
        frames = self._frames(3)
        eng, _ = self._engine(frames)
        said = self._drive(eng, len(frames))
        lap1 = {k for (_i, lap, k) in said if lap == 1}
        assert lap1, "第 1 圈不是不能说话，只是不能说依赖参考圈的话"
        assert not (lap1 & set(self.REF_KEYS)), sorted(lap1 & set(self.REF_KEYS))
        # 正对照：同样的开法，第 2 圈就该说了
        lap2 = {k for (_i, lap, k) in said if lap == 2}
        assert lap2 & set(self.REF_KEYS), sorted(lap2)

    def test_state_admits_no_reference_during_warmup(self):
        """仪表盘那个「参考圈：车载 60Hz · 第 N 圈」标签的数据源必须诚实。

        🔴 关键场景是"手上**已经**有一份 ref 对象，但还不能用"：
           暖胎期 `ref_ready` 必须是 False，`ref_lap`/`ref_len_m` 必须一起为
           None（三个字段描述同一个 ref）。只改 `ref_ready` 的话，标签会变成
           "建立中…·第 1 圈"这种自相矛盾的串。
        """
        frames = self._frames(3)
        eng, _ = self._engine(frames)
        assert wait_ref(eng), "手上得先有一份 ref —— 否则测不出「有但不用」"
        st = eng.tick()
        assert st.lap == 1, f"这一刻应当还在第 1 圈（实际第 {st.lap} 圈）"
        assert eng._current_ref() is not None, "手上明明有 ref"
        assert st.ref_ready is False
        assert st.ref_lap is None and st.ref_len_m is None
        # `ref_source` 说的是"**手上这份**从哪来"（排障用），与"能不能用"分开：
        # 此刻手上确实有一份 profile，只是被暖胎期那道闸按住。
        assert st.stats["ref_source"] == "profile"
        assert st.stats["run_laps"] == 0
        assert "暖胎" in (st.stats["ref_blocked"] or "")
        # 跑完第 1 圈 → 参考圈才"拿得出手"，且三个字段说的是同一个 ref
        for _ in range(len(frames)):
            st = eng.tick()
            if st.ref_ready:
                break
        assert st.ref_ready, "跑完一圈之后应当有可用参考圈"
        assert st.ref_lap is not None and st.ref_len_m is not None
        assert st.ref_len_m == pytest.approx(L, rel=0.05)

    def test_in_progress_profile_is_never_published(self):
        """Dash 明说"这一圈还在跑" → 当没取到处理，**绝不发布**。

        发布出去就是三重污染：拿去定位（凭空的"出界了"）、存进本地缓存
        （把半圈存成"这条赛道的最好圈"）、当形状比对基准（把后面每次
        "是不是同一条赛道"的判断一起带偏）。
        """
        prof = dict(synth_profile(radius_m=R))
        prof["meta"] = {"file": "replay.jsonl", "in_progress": True}
        frames = self._frames(2)
        src = ReplaySource(frames, profile=prof, loop=False)
        eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                           sess_poll_idle_s=0.05),
                          clock=src.clock)
        for _ in range(150):
            eng.tick()
        assert src.profile_calls >= 1, "得真去取过，才谈得上拒绝"
        assert eng._current_ref() is None, "半圈剖面绝不能被发布成参考圈"
        st = eng.tick()
        assert st.ref_ready is False
        assert st.stats["ref_state"] == "failed"
        # 错误得说得清：这是"内容还不能用"，不是网络/服务端的问题
        assert "还在跑" in (st.stats["ref_error"] or ""), st.stats["ref_error"]

    def test_restart_wipes_previous_run(self):
        """重开比赛（圈号倒退）→ 上一轮的东西一件都不能留。

        用户看到的现象是「切换场次 / 重新比赛，赛道工程师还在继续工作」——
        因为旧代码只把"圈号变了"当成"跑完一圈"，圈号倒退回 1 时它既不结算、
        也不作废，参考圈与每弯累积全是上一轮的，于是新一轮第 1 圈继续播报。
        """
        frames = self._frames(3) + self._frames(2)     # 后一段：圈号回到 1
        eng, _ = self._engine(frames)
        seen_run, restarted_at = 0, None
        said = []
        for i in range(len(frames)):
            st = eng.tick()
            seen_run = max(seen_run, eng._run_laps)
            if restarted_at is None and seen_run >= 2 and eng._run_laps == 0:
                restarted_at = i
                assert eng._self_ref is None, "自攒参考圈是上一轮的，必须作废"
                assert eng._prev_lap is None, "上一圈成绩是上一轮的"
                assert eng._lap_buf == [] or len(eng._lap_buf) <= 1
            for u in st.say:
                said.append((i, st.lap, u.key.split("@")[0]))
        assert restarted_at is not None, "夹具没造出「圈号倒退」"
        after1 = [(i, k) for (i, lap, k) in said
                  if i > restarted_at and lap == 1]
        assert after1, "重开之后第 1 圈仍该说不需要参考圈的话"
        assert not ({k for _i, k in after1} & set(self.REF_KEYS)), after1
        # 新一轮第 2 圈：参考圈该回来了（不是被永久关掉）
        after2 = {k for (i, lap, k) in said if i > restarted_at and lap == 2}
        assert after2 & set(self.REF_KEYS), sorted(after2)


class TestSpeechDigits:
    """🔴 语音串（speech）必须随 utterance 一起产出，且是逐位中文。

    这是「语音报数字 54→五四」的端到端落点：引擎在闸门之后给每条要说的话贴
    上 `speech`（屏幕用的 `text` 仍是原样 54 / 115，便于扫读）。用不依赖参考圈的
    胎温过热规则触发，它产出带数字的话「左前胎 115 度过热」。
    """

    def test_engine_attaches_digit_by_digit_speech(self):
        frames = [Frame(t=i * 0.1, lap_time_s=i * 0.1, lap=1,
                        tyre_temp=(115.0, 100.0, 100.0, 100.0),
                        speed_kph=30.0) for i in range(40)]
        # session=None → 后台不去取参考圈，测试完全确定性、不依赖异步线程。
        src = ReplaySource(frames, profile=None, session=None, loop=False)
        eng = CoachEngine(src, CoachConfig(poll_interval_s=0.1,
                                           sess_poll_boot_s=999,
                                           sess_poll_idle_s=999))
        got = None
        for _ in range(50):
            st = eng.tick()
            if st.say:
                got = st.say[0]
                break
        assert got is not None, "应触发胎温过热播报"
        assert got.text == "左前胎 115 度过热"
        # 语音走逐位中文，屏幕 text 保持原样
        assert got.speech == "左前胎 一一五 度过热"
        # 序列化到 /api/v1/coach/state 时也带 speech，仪表盘据此播报
        assert got.to_dict()["speech"] == "左前胎 一一五 度过热"
        # 字符数等价 → ttl 预算无需因这一层重算
        assert len(got.speech) == len(got.text)


class TestDisconnectStopsSpeech:
    """游戏断开 / 待机：dash 仍回最后一帧但 connected=False，
    引擎必须早退、不再产生新播报。"""

    def test_no_utterance_when_disconnected(self):
        frames = [Frame(t=i * 0.1, lap_time_s=i * 0.1, lap=1,
                        connected=False,
                        tyre_temp=(115.0, 100.0, 100.0, 100.0),
                        speed_kph=30.0) for i in range(40)]
        src = ReplaySource(frames, profile=None, session=None, loop=False)
        eng = CoachEngine(src, CoachConfig(poll_interval_s=0.1,
                                           sess_poll_boot_s=999,
                                           sess_poll_idle_s=999))
        for _ in range(20):
            st = eng.tick()
            assert st.connected is False
            assert st.say == []   # 断开后不应有任何播报


# ===========================================================================
# 名次 / 情绪向（R3.1）—— 端到端装配
# ===========================================================================
#
# 🔴 守的是**装配**，不是规则本身（规则单测在 `tests/test_rules.py::TestMood`）：
#    `Frame` 上有 `position` 字段 ≠ 它会被念出来。中间还隔着两道 ——
#    ① `RuleSet.evaluate` 里有没有注册；② 闸门有没有把它竞争掉。
#    少任何一道的表现都是"静默"，而不是报错。

class TestMoodEndToEnd:
    @staticmethod
    def _frames(position_fn, *, laps=6):
        """合成帧 + 名次：`position_fn(i)` 给出第 i 帧的名次。"""
        import dataclasses

        base = synth_lap_frames(radius_m=R, hz=10.0, laps=laps)
        return [
            dataclasses.replace(f, position=position_fn(i), num_cars=20,
                                laps_in_race=laps)
            for i, f in enumerate(base)
        ]

    @staticmethod
    def _drive(frames, *, step_s=0.1):
        """跑完全部帧，收集说出口的播报（key 去掉位置后缀归并）。

        🔴 用**可控时钟**推进，而不是真等：闸门有 6 s 跨类冷却，而测试循环
           比真车快几百倍。不推时钟的话，「名次变化」会一直被上一条播报的
           冷却挡住 —— 那测出来的不是规则坏了，是**测法错了**。
        """
        # session=None → 后台不取参考圈，测试完全确定性、不依赖异步线程。
        src = ReplaySource(frames, profile=None, session=None, loop=False)
        tick_t = {"v": 1000.0}

        def clock():
            tick_t["v"] += step_s
            return tick_t["v"]

        eng = CoachEngine(src, CoachConfig(poll_interval_s=step_s,
                                           sess_poll_boot_s=999,
                                           sess_poll_idle_s=999),
                          clock=clock)
        said: dict[str, list[str]] = {}
        for _ in range(len(frames) + 10):
            for u in eng.tick().say:
                said.setdefault(u.key.split("@")[0], []).append(u.text)
        return said

    def test_position_change_is_broadcast(self):
        """20 车赛里从 P13 追到 P11 → 会念出来（规则已注册 + 闸门放行）。

        🔴 名次变化与鼓励同为 P_LOW，会**互相竞争** —— 这是闸门"不打扰优先"
           的正确行为，不是 bug。所以这条用例把名次变化放在离圈首足够远的
           位置（30% 处），让两条话各自的 6 s 跨类冷却都过得去。
        """
        base = synth_lap_frames(radius_m=R, hz=10.0, laps=6)
        switch = int(len(base) * 0.3)
        frames = self._frames(lambda i: 13 if i < switch else 11)
        said = self._drive(frames)
        assert "position" in said, list(said)
        assert any("追回 2 位" in t for t in said["position"]), \
            said["position"]

    def test_position_is_silent_when_unknown(self):
        """合成帧默认不带名次（0）→ 闭嘴，不能念"P0"。"""
        said = self._drive(self._frames(lambda i: 0))
        assert "position" not in said, said

    def test_leader_reminder_fires(self):
        """一路领跑 → 至少念一次"已经是 P1"。"""
        said = self._drive(self._frames(lambda i: 1))
        assert "leader" in said, list(said)
        assert any(t.startswith("已经是 P1") for t in said["leader"]), \
            said["leader"]

    def test_encouragement_fires_in_back_half(self):
        """一直卡在 P13/20 → 会鼓励。"""
        said = self._drive(self._frames(lambda i: 13))
        assert "encourage" in said, list(said)
        assert all(t.startswith("还在 P13，") for t in said["encourage"]), \
            said["encourage"]

    def test_no_encouragement_in_front_half(self):
        """上半区（P8/20）不鼓励 —— 在那里说"别急"听着像讽刺。"""
        said = self._drive(self._frames(lambda i: 8))
        assert "encourage" not in said, said

    def test_stats_expose_position(self):
        """名次两个数进 stats —— 排障第一现场。

        🔴 教练念了句"还在 P13"，第一反应是"它读到的是不是 13"。
           不暴露出来，就得去翻服务端日志；而"教练在瞎编"和
           "教练读到的数就是错的"是两种完全不同的故障。
        """
        frames = self._frames(lambda i: 13)
        src = ReplaySource(frames, profile=None, session=None, loop=False)
        eng = CoachEngine(src, CoachConfig(poll_interval_s=0.1,
                                           sess_poll_boot_s=999,
                                           sess_poll_idle_s=999))
        st = eng.tick()
        assert st.stats["position"] == 13
        assert st.stats["num_cars"] == 20

    def test_back_half_and_leadership_do_not_both_fire(self):
        """🔴 情绪向两条**互斥**：领跑时不会同时被鼓励"别急"。

        这不是巧合而是设计：名次数据只有一份，P1 与"后半区"不可能同时成立。
        留着这条是为了防止将来有人把鼓励的判据改成"落后于某人"之类 ——
        那样 P1 也可能"落后于理论最快圈"，就会同时挨上两句。
        """
        said = self._drive(self._frames(lambda i: 1))
        assert "leader" in said
        assert "encourage" not in said
