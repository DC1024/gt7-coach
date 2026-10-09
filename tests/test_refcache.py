# -*- coding: utf-8 -*-
"""参考圈本地缓存 —— 单元与集成测试。

两条杠杆（根治「前几圈瞎播报」）之一：把跑过的最好一圈按「赛道指纹 + 车型」
存盘，下次同赛道直接复用，不用等服务端历史搜索的冷启动延迟。

安全底线（与 history_best 的 shape_distance 校验同一把尺）：缓存**绝不**在
没核对赛道形状之前被采用。本文件既测往返，也测「形状不对就拒绝」「目录建不了
就静默禁用」，并有一个端到端断言：引擎在缓存命中更快参考圈时优先采用它。
"""
from __future__ import annotations

import time

import pytest

from gt7coach.engine import CoachConfig, CoachEngine
from gt7coach.refcache import RefCache
from gt7coach.refindex import RefLap
from gt7coach.source import ReplaySource
from gt7coach.synth import synth_lap_frames, synth_profile

R = 600.0


def _ref(radius=600.0):
    return RefLap.from_profile(synth_profile(radius_m=radius))


class TestRefCacheRoundTrip:
    def test_save_then_load(self, tmp_path):
        ref = _ref()
        c = RefCache(str(tmp_path / "cache"))
        assert c.enabled
        c.save(ref, "CarX")
        loaded = c.load("CarX", ref.track_fingerprint())
        assert loaded is not None
        assert loaded.lap_time_s == pytest.approx(ref.lap_time_s)
        assert len(loaded.xs) == len(ref.xs)
        assert loaded.source == "profile"

    def test_load_wrong_fingerprint_returns_none(self, tmp_path):
        ref = _ref(600.0)
        other = _ref(400.0)
        c = RefCache(str(tmp_path / "cache"))
        c.save(ref, "CarX")
        # 不同赛道 → 指纹不同 → 不应返回（否则就是误用别的赛道的参考圈）
        assert c.load("CarX", other.track_fingerprint()) is None

    def test_load_missing_file_returns_none(self, tmp_path):
        c = RefCache(str(tmp_path / "cache"))
        assert c.load("CarX", "deadbeefdeadbeef") is None

    def test_save_is_idempotent_same_key(self, tmp_path):
        ref = _ref()
        c = RefCache(str(tmp_path / "cache"))
        fp = ref.track_fingerprint()
        c.save(ref, "CarX")
        c.save(ref, "CarX")            # 覆盖同 key，不应抛、不应出错
        assert c.load("CarX", fp) is not None

    def test_corrupt_payload_rejected(self, tmp_path):
        ref = _ref()
        c = RefCache(str(tmp_path / "cache"))
        fp = ref.track_fingerprint()
        path = c._path("CarX", fp)
        path.write_text("{ this is not valid json", encoding="utf-8")
        assert c.load("CarX", fp) is None


class TestFingerprintStability:
    def test_same_track_stable_across_builds(self):
        a = _ref(600.0).track_fingerprint()
        b = _ref(600.0).track_fingerprint()
        assert a and a == b

    def test_different_track_differs(self):
        a = _ref(600.0).track_fingerprint()
        b = _ref(400.0).track_fingerprint()
        assert a != b

    def test_tiny_track_rejected(self):
        # 点数不够无法算指纹 → 空串（缓存自动跳过这类参考圈）
        ref = RefLap(lap=1, source="self", grid_m=[0.0, 1.0],
                     speed_kph=[0.0, 0.0], throttle=[0.0, 0.0],
                     brake=[0.0, 0.0], t_rel_s=[0.0, 0.0],
                     xs=[0.0, 0.0], zs=[0.0, 0.0],
                     length_m=1.0, lap_time_s=1.0)
        assert ref.track_fingerprint() == ""


class TestRefCacheDisabled:
    def test_none_dir_is_disabled(self):
        c = RefCache(None)
        assert not c.enabled
        c.save(_ref(), "CarX")          # 静默 no-op
        assert c.load("CarX", "x") is None

    def test_uncreatable_dir_disables_silently(self, tmp_path):
        # 把一个已存在的文件当父目录 → mkdir 必然失败 → 禁用且不抛
        blocker = tmp_path / "afile"
        blocker.write_text("x")
        c = RefCache(str(blocker / "sub" / "dir"))
        assert not c.enabled
        c.save(_ref(), "CarX")
        assert c.load("CarX", "x") is None


class TestEngineAdoptsCachedRef:
    """端到端：预置一条更快的同赛道参考圈进缓存，引擎应当优先采用它。

    🔴 这把「读本地盘替换服务端历史搜索」的承诺真正验出来——
    否则缓存只是摆设（旧实现会在读缓存之前先把 base 写盘覆盖掉）。
    """

    def test_adopts_faster_cached_ref(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        # 预置一条更快（60s）的同赛道参考圈
        faster = _ref(600.0)
        faster.lap_time_s = 60.0
        RefCache(cache_dir).save(faster, "TestCar")

        # 本场剖面故意更慢（69.7s），模拟"今天这趟跑得不如历史最好"
        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=2)
        prof = synth_profile(radius_m=R)
        prof["lap_time_s"] = 69.7
        prof["t_rel_s"] = [t * 69.7 / max(prof["t_rel_s"][-1], 1e-6)
                           for t in prof["t_rel_s"]]
        src = ReplaySource(frames, profile=prof, loop=True)
        src._session = {"file": "live.jsonl", "live": True,
                        "best_lap_s": 69.7, "car_name": "TestCar"}
        src.history_candidates = []      # 不找服务端历史，单独验证本地缓存
        eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                           sess_poll_idle_s=0.05,
                                           ref_policy="history_best",
                                           ref_cache_dir=cache_dir),
                          clock=src.clock)
        end = time.monotonic() + 6.0
        adopted = None
        while time.monotonic() < end:
            eng.tick()
            ref = eng._current_ref()
            if ref is not None:
                adopted = ref
                if ref.lap_time_s <= 61.0:   # 命中缓存（60s）即停
                    break
            time.sleep(0.01)
        assert adopted is not None, "参考圈应就位"
        assert adopted.lap_time_s == pytest.approx(60.0, abs=0.5), \
            adopted.lap_time_s
        assert eng.refs.history_status().get("cached_adopted") is True

    def test_does_not_adopt_slower_cached_ref(self, tmp_path):
        """缓存比本场更慢时绝不能采用 —— 用更慢的反而会带偏教练。"""
        cache_dir = str(tmp_path / "cache")
        slower = _ref(600.0)
        slower.lap_time_s = 120.0          # 比本场 69.7 慢得多
        RefCache(cache_dir).save(slower, "TestCar")

        frames = synth_lap_frames(radius_m=R, hz=10.0, laps=2)
        prof = synth_profile(radius_m=R)
        prof["lap_time_s"] = 69.7
        prof["t_rel_s"] = [t * 69.7 / max(prof["t_rel_s"][-1], 1e-6)
                           for t in prof["t_rel_s"]]
        src = ReplaySource(frames, profile=prof, loop=True)
        src._session = {"file": "live.jsonl", "live": True,
                        "best_lap_s": 69.7, "car_name": "TestCar"}
        src.history_candidates = []
        eng = CoachEngine(src, CoachConfig(sess_poll_boot_s=0.02,
                                           sess_poll_idle_s=0.05,
                                           ref_policy="history_best",
                                           ref_cache_dir=cache_dir),
                          clock=src.clock)
        assert drive_until_ref(eng, timeout=6.0)
        ref = eng._current_ref()
        assert ref.lap_time_s == pytest.approx(69.7, abs=0.5), ref.lap_time_s
        assert eng.refs.history_status().get("cached_adopted") is not True


def drive_until_ref(eng, timeout=6.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        eng.tick()
        if eng._current_ref() is not None:
            return True
        time.sleep(0.01)
    return False
