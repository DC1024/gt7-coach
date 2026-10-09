# -*- coding: utf-8 -*-
"""跨场次历史参考圈 —— 「拿你自己的最好成绩当标杆」。

开一场慢的，用本场最快圈当参考只会拿"你今天最烂的那一圈"来夸你。
所以要多找一步：**本场之外**更快的圈。

风险也随之而来：候选可能根本不是同一条赛道（换过赛道、反向布局、
不同线路变体），拿它当参考会让实时定位整体失准 —— 而且那种失准
是"偶尔算错"而不是报错，最难查。所以必须**几何形状比对**确认同赛道。

为什么不用赛道识别（`track_id`）：要靠服务端逐个算指纹，实测 10 个场次里
只有 2 个已识别，靠它等于大部分时候用不上。而两条折线的中位最近点距离，
同赛道几米、不同赛道几十上百米，判别力干净。
"""
from __future__ import annotations

import math

import pytest

from conftest import wait_for
from gt7coach.engine import CoachConfig, CoachEngine, RefProvider
from gt7coach.refindex import RefLap
from gt7coach.rules import Ctx, RuleConfig, RuleSet
from gt7coach.source import ReplaySource
from gt7coach.synth import synth_lap_frames, synth_profile
from test_rules import SPEED_FACE_KEYS, ZONE_S, mk

R = 600.0
L = 2.0 * math.pi * R


# —— 形状比对 ————————————————————————————————————————

class TestShapeDistance:
    @staticmethod
    def _ref(radius=600.0):
        return RefLap.from_profile(synth_profile(radius_m=radius))

    def test_same_track_is_near_zero(self):
        d = self._ref(600.0).shape_distance(self._ref(600.0))
        assert d == pytest.approx(0.0, abs=1.0)

    def test_different_track_is_large(self):
        d = self._ref(600.0).shape_distance(self._ref(400.0))
        assert d > 100.0, d

    def test_two_same_length_but_different_tracks(self):
        """🔴 圈长一样但形状不同 —— 这正是不能用圈长判断的原因。

        （真实场景：同一个赛道的不同布局变体、或反向跑。）
        """
        a = self._ref(600.0)
        # 造一个"同样圈长但不是同一条圈"的折线：把圆压扁成椭圆，
        # 周长大致相同但形状完全不同
        import copy
        b = copy.deepcopy(a)
        b.xs = [x * 1.6 for x in a.xs]
        b.zs = [z * 0.5 for z in a.zs]
        d = a.shape_distance(b)
        assert d is not None and d > 50.0, d

    def test_none_when_no_polyline(self):
        a = self._ref(600.0)
        b = self._ref(600.0)
        b.xs, b.zs = [], []
        assert a.shape_distance(b) is None


# —— 候选筛选 ————————————————————————————————————————

class TestFasterSessions:
    def test_picks_only_faster_and_sorted(self):
        from gt7coach.source import HttpSource
        src = HttpSource("http://127.0.0.1:1", timeout=0.01)
        rows = [
            {"file": "s1.jsonl", "best_lap_s": 69.0, "car_name": "X"},
            {"file": "s2.jsonl", "best_lap_s": 68.0, "car_name": "X"},
            {"file": "s3.jsonl", "best_lap_s": 71.0, "car_name": "X"},  # 更慢
            {"file": "me.jsonl", "best_lap_s": 60.0, "car_name": "X"},  # 自己
            {"file": "s4.jsonl", "best_lap_s": 67.5, "car_name": "Y"},  # 换车了
        ]
        src.sessions = lambda: rows
        out = src.faster_sessions(70.0, exclude="me.jsonl", car_name="X",
                                  limit=5)
        assert [x["file"] for x in out] == ["s2.jsonl", "s1.jsonl"]

    def test_same_car_filter_can_be_off(self):
        from gt7coach.source import HttpSource
        src = HttpSource("http://127.0.0.1:1", timeout=0.01)
        src.sessions = lambda: [
            {"file": "s.jsonl", "best_lap_s": 60.0, "car_name": "Y"}]
        assert src.faster_sessions(70.0, car_name="X", same_car=False)
        assert src.faster_sessions(70.0, car_name="X", same_car=True) == []

    def test_no_car_name_disables_filter(self):
        """车型表没命中时别因为查不到车型就完全不用历史数据。"""
        from gt7coach.source import HttpSource
        src = HttpSource("http://127.0.0.1:1", timeout=0.01)
        src.sessions = lambda: [
            {"file": "s.jsonl", "best_lap_s": 60.0, "car_name": "Y"}]
        assert src.faster_sessions(70.0, car_name="") == [{"file": "s.jsonl",
                                                           "best_lap_s": 60.0,
                                                           "car_name": "Y"}]

    def test_respects_limit(self):
        from gt7coach.source import HttpSource
        src = HttpSource("http://127.0.0.1:1", timeout=0.01)
        src.sessions = lambda: [{"file": f"s{i}.jsonl", "best_lap_s": 60.0 + i}
                                for i in range(10)]
        assert len(src.faster_sessions(70.0, limit=3)) == 3


# —— Provider 级：采用 / 拒绝 ————————————————————————

def _profiles(live_best=69.7):
    """本场剖面 + 一条更快的历史剖面 + 一条别的赛道的剖面。"""
    live = synth_profile(radius_m=R)
    live["lap_time_s"] = live_best
    live["t_rel_s"] = [t * live_best / max(live["t_rel_s"][-1], 1e-6)
                       for t in live["t_rel_s"]]

    hist = synth_profile(radius_m=R)
    k = 0.97                                   # 历史圈快 3%
    hist["t_rel_s"] = [t * k for t in hist["t_rel_s"]]
    hist["lap_time_s"] *= k

    other = synth_profile(radius_m=400.0)      # 另一条赛道（形状差 200m）
    return live, hist, other


def _engine(policy="history_best", candidates=("hist.jsonl",),
            profiles=None):
    frames = synth_lap_frames(radius_m=R, hz=10.0, laps=2)
    src = ReplaySource(frames, profile=None, loop=True)
    live, hist, other = _profiles()
    src.profiles = {"live.jsonl": live, "hist.jsonl": hist,
                    "other.jsonl": other}
    src._session = {"file": "live.jsonl", "live": True, "best_lap_s": 69.7,
                    "car_name": "TestCar"}
    src.history_candidates = [
        {"file": f, "best_lap_s": 67.6 if f == "hist.jsonl" else 60.0,
         "car_name": "TestCar"} for f in candidates]
    eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                       sess_poll_idle_s=0.05,
                                       ref_policy=policy),
                      clock=src.clock)
    return eng, src


def _wait_source(eng, want, timeout=4.0):
    end = time_mono() + timeout
    while time_mono() < end:
        eng.tick()
        ref = eng._current_ref()
        if ref is not None and ref.source == want:
            return True
    return False


def time_mono():
    import time
    return time.monotonic()


class TestHistoryRefAdoption:
    def test_adopts_same_track_history(self):
        eng, _ = _engine()
        assert _wait_source(eng, "history"), eng.refs.history_status()
        ref = eng._current_ref()
        assert ref.source == "history"
        # 历史圈快 3%，参考圈速应当是那个更快的
        assert ref.lap_time_s < 69.7 * 0.99
        st = eng.refs.history_status()
        assert st["adopted"]["file"] == "hist.jsonl"
        assert st["adopted"]["shape_m"] < 60.0

    def test_rejects_other_track_and_keeps_session_best(self):
        """🔴 候选是别的赛道时必须拒绝 —— 宁可没有历史参考，也不能拿错赛道。"""
        eng, _ = _engine(candidates=("other.jsonl",))
        # 先等本场剖面到位（它一定会被发出来，保证教练马上能用）
        assert _wait_source(eng, "profile")
        # 再给它时间尝试历史候选
        for _ in range(120):
            eng.tick()
            if eng.refs.history_status().get("rejected"):
                break
        st = eng.refs.history_status()
        assert st["rejected"], st
        assert "形状" in st["rejected"][0]["why"]
        assert st["rejected"][0]["shape_m"] > 60.0
        assert eng._current_ref().source == "profile", "必须退回本场最快圈"

    def test_session_best_policy_skips_search(self):
        eng, _ = _engine(policy="session_best")
        assert _wait_source(eng, "profile")
        for _ in range(60):
            eng.tick()
        st = eng.refs.history_status()
        assert not st.get("candidates"), "session_best 不该去找历史圈"
        assert eng._current_ref().source == "profile"

    def test_first_candidate_tried_before_second(self):
        """候选按最快圈升序试，第一个不行才试第二个。"""
        eng, _ = _engine(candidates=("other.jsonl", "hist.jsonl"))
        assert _wait_source(eng, "history"), eng.refs.history_status()
        st = eng.refs.history_status()
        assert st["candidates"] == ["other.jsonl", "hist.jsonl"]
        assert st["adopted"]["file"] == "hist.jsonl"

    def test_profile_is_published_before_history_search(self):
        """历史搜索要额外几次 /profile，可能慢几秒 —— 期间必须已有参考圈可用。

        否则为了"找更好的"让用户干等，反而比不找更糟。
        """
        eng, _ = _engine()
        seen_early = False
        for _ in range(200):
            eng.tick()
            ref = eng._current_ref()
            if ref is not None and ref.source == "profile":
                seen_early = True
                break
        assert seen_early, "应当先发布本场剖面"

    def test_stats_expose_history_attempt(self):
        eng, _ = _engine()
        assert _wait_source(eng, "history")
        st = eng.tick()
        h = st.stats["ref_history"]
        assert h["adopted"]["file"] == "hist.jsonl"
        assert st.stats["ref_source"] == "history"


class TestHistoryRefRobustness:
    def test_missing_profile_is_skipped(self):
        """候选的剖面取不到就跳过，不该整个失败。"""
        eng, src = _engine(candidates=("missing.jsonl", "hist.jsonl"))
        assert _wait_source(eng, "history"), eng.refs.history_status()
        st = eng.refs.history_status()
        assert any("取不到" in r["why"] for r in st["rejected"])

    def test_no_candidates_keeps_working(self):
        eng, _ = _engine(candidates=())
        assert _wait_source(eng, "profile")
        assert eng._current_ref().source == "profile"
        assert eng.refs.history_status()["adopted"] is None

    def test_source_error_does_not_break_ref(self):
        """拿历史候选时源抛异常 → 记下来继续用本场剖面，不能整体挂掉。"""
        eng, src = _engine()

        def boom(*_a, **_k):
            raise RuntimeError("场次列表挂了")
        src.faster_sessions = boom
        assert _wait_source(eng, "profile")
        for _ in range(80):
            eng.tick()
        st = eng.refs.history_status()
        assert "error" in st and "RuntimeError" in st["error"]
        assert eng._current_ref() is not None, "本场剖面必须还在用"


# —— 跨车型：几何面照用，速度面禁用 ————————————————————————
#
# 🔴 这是方案 §7 的落地。两条防线：
#     ① 挑候选时按 car_code 过滤（默认开，同车才用）；
#     ② 车型判不出来时（cars.csv 没命中 → 两边都是空串），
#        用**圈速窗口**兜底，而不是像旧实现那样静默跨车采用。

def _engine_unknown_car(hist_lap_s: float):
    """本场 69.7s；历史候选指定圈速，且**两边车型都查不到**（空名 / 0 码）。"""
    frames = synth_lap_frames(radius_m=R, hz=10.0, laps=2)
    src = ReplaySource(frames, profile=None, loop=True)
    live, hist, _other = _profiles()
    src.profiles = {"live.jsonl": live}
    # 历史剖面：形状与本场同赛道，但圈速是指定的
    src.profiles["hist.jsonl"] = _retime(synth_profile(radius_m=R, step_m=5.0),
                                         hist_lap_s)
    src._session = {"file": "live.jsonl", "live": True,
                    "best_lap_s": 69.7, "car_name": ""}
    src.history_candidates = [
        {"file": "hist.jsonl", "best_lap_s": hist_lap_s, "car_name": ""}]
    eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                       sess_poll_idle_s=0.05,
                                       ref_policy="history_best"),
                      clock=src.clock)
    return eng, src


def _retime(prof: dict, lap_s: float) -> dict:
    """把一份剖面的时间轴整体缩放成 `lap_s`（形状不变、只变快慢）。

    用来造"同一条赛道、但明显更快/更慢的另一辆车"的参考圈。
    """
    import copy
    p = copy.deepcopy(prof)
    cur = float(p.get("lap_time_s") or 0.0) or float(p["t_rel_s"][-1] or 0.0)
    k = lap_s / max(cur, 1e-6)
    p["lap_time_s"] = round(lap_s, 3)
    p["t_rel_s"] = [round(t * k, 3) for t in p["t_rel_s"]]
    return p


class TestHistoryRefCrossCar:
    """跨车型复用历史圈时的「分层」行为。"""

    def test_same_car_history_keeps_the_speed_face(self):
        """同车 → 速度面照用（这正是 history_best 的意义）。"""
        eng, _ = _engine()
        assert _wait_source(eng, "history"), eng.refs.history_status()
        st = eng.tick()
        assert st.stats["ref_car_match"] == "same_car"
        assert st.stats["ref_pace_ok"] is True

    def test_cross_car_within_window_stays_usable(self):
        """不同车但性能接近 → 速度面仍可用。"""
        eng, _ = _engine_unknown_car(67.0)      # 快 3.9%
        assert _wait_source(eng, "history"), eng.refs.history_status()
        st = eng.tick()
        assert st.stats["ref_car_match"] == "unknown"
        assert st.stats["ref_pace_ok"] is True

    def test_cross_car_beyond_window_kills_only_the_speed_face(self):
        """🔴 快 36% ⇒ 速度面禁用，但**几何面必须还活着**。

        出了这条，就会退回"要么全都用、要么全都别用" —— 而全别用意味着
        换了车以后教练连"你出界了"都不说了。
        """
        eng, _ = _engine_unknown_car(45.0)      # 69.7 / 45 ≈ 1.55
        assert _wait_source(eng, "history"), eng.refs.history_status()
        st = eng.tick()
        assert st.stats["ref_pace_ok"] is False
        assert st.stats["ref_car_match"] == "unknown"
        # 🔴 关键：参考圈**对象**还在（几何面照用），只是速度面被禁。
        #    上面那条 assert 若改成"没有参考圈"，换了车教练就彻底哑了。
        assert eng._current_ref() is not None
        ref = eng._current_ref()

        def _run(s_at: float, lateral: float, **over):
            """同一状态连跑 8 tick（够过所有 hold 阈值），返回说过的 key。"""
            stt = RuleSet.fresh_state()
            rsx = RuleSet(RuleConfig())
            out: list[str] = []
            for _ in range(8):
                c = Ctx(f=mk(**over), ref=ref, s=s_at, lateral_m=lateral,
                        dt=0.1, st=stt, ref_pace_ok=False)
                out += [u.key for u in rsx.evaluate(c)]
            return out

        # 🔴 速度面：七条规则全部静默（正对照见 test_rules.py 里同一条件）
        got = "".join(_run(ZONE_S + 30.0, 0.0, speed_kph=200.0,
                           lap_time_s=30.0, brake=0.0))
        for pre in SPEED_FACE_KEYS:
            assert pre not in got, (pre, got)
        # 🔴 几何面：出界照样报 —— 换了车也不能把这条一起关掉
        assert "off_track" in _run(ZONE_S + 30.0, 60.0, speed_kph=200.0)

    def test_stats_expose_the_two_numbers_behind_the_verdict(self):
        """排障要能直接看到判据的两个数，不用再去服务端翻。"""
        eng, _ = _engine_unknown_car(45.0)
        assert _wait_source(eng, "history")
        st = eng.tick()
        assert st.stats["ref_lap_time_s"] == pytest.approx(45.0, abs=0.5)
        assert st.stats["base_lap_time_s"] == pytest.approx(69.7, abs=0.5)

    def test_same_car_filter_is_code_first(self):
        """车型名都查不到、码不同 → 不许跨车采用（旧实现在这里是静默放行的）。

        直接打 `faster_sessions`，因为它就是那道闸。
        """
        from gt7coach.source import HttpSource
        src = HttpSource("http://127.0.0.1:1", timeout=0.01)
        src.sessions = lambda: [
            {"file": "s1.jsonl", "best_lap_s": 50.0, "car_code": 805},
            {"file": "s2.jsonl", "best_lap_s": 51.0, "car_code": 902},
        ]
        out = src.faster_sessions(70.0, exclude="live.jsonl",
                                  car_code=902, car_name="")
        assert [x["file"] for x in out] == ["s2.jsonl"]
