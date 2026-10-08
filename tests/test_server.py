# -*- coding: utf-8 -*-
"""HTTP 服务测试 —— 这是仪表盘跟赛道工程师联通的**唯一**接口，形状不能乱。"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from gt7coach.engine import CoachConfig, CoachEngine
from gt7coach.server import CoachService, make_server
from gt7coach.source import ReplaySource
from gt7coach.synth import synth_lap_frames, synth_profile

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(url: str):
    with _OPENER.open(url, timeout=5) as r:
        return r.status, json.loads(r.read().decode("utf-8")), dict(r.headers)


def _post(url: str, body: dict):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=5) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


@pytest.fixture
def server():
    frames = synth_lap_frames(laps=2)
    src = ReplaySource(frames, profile=synth_profile(), loop=True)
    eng = CoachEngine(src, CoachConfig(poll_interval_s=0.02,
                                       sess_poll_boot_s=0.0,
                                       sess_poll_idle_s=0.0))
    svc = CoachService(eng, interval_s=0.02)
    srv = make_server(svc, host="127.0.0.1", port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    svc.start()
    time.sleep(0.3)                     # 让参考圈线程落地
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    svc.stop()
    srv.shutdown()
    srv.server_close()


class TestRoutes:
    def test_health(self, server):
        code, d, _ = _get(server + "/api/v1/coach/health")
        assert code == 200 and d["ok"] is True
        assert d["ticks"] > 0
        assert "ref_ready" in d

    def test_state_shape(self, server):
        code, d, _ = _get(server + "/api/v1/coach/state")
        assert code == 200
        for k in ("api_version", "connected", "ref_ready", "lap", "s_m",
                  "delta_s", "next_brake_m", "next_brake_s", "say", "spoken",
                  "stats"):
            assert k in d, k
        assert d["api_version"] == 1
        assert d["ref_ready"] is True
        assert d["stats"]["ref_source"] == "profile"

    def test_say_is_edge_triggered(self, server):
        """`/say` 取走就没了；再取应当为空（除非这期间又说了新的）。"""
        _get(server + "/api/v1/coach/state")     # 先跑起来
        time.sleep(0.2)
        _code, first, _ = _get(server + "/api/v1/coach/say")
        assert isinstance(first["say"], list)
        _code, second, _ = _get(server + "/api/v1/coach/say")
        assert isinstance(second["say"], list)
        # 拿走的那些不会再出现
        taken = {json.dumps(u, sort_keys=True) for u in first["say"]}
        again = {json.dumps(u, sort_keys=True) for u in second["say"]}
        assert not (taken & again), "边沿接口不能把已经取走的又给一遍"

    def test_history_route(self, server):
        code, d, _ = _get(server + "/api/v1/coach/history?n=5")
        assert code == 200 and isinstance(d["spoken"], list)
        assert len(d["spoken"]) <= 5

    def test_config_roundtrip(self, server):
        code, d, _ = _get(server + "/api/v1/coach/config")
        assert code == 200 and "gate" in d and "rules" in d and "coach" in d
        assert d["gate"]["max_per_lap"] >= 1

        code, d2 = _post(server + "/api/v1/coach/config",
                         {"gate": {"max_per_lap": 3}})
        assert code == 200 and d2["config"]["gate"]["max_per_lap"] == 3

    def test_config_rejects_unknown_field(self, server):
        with pytest.raises(urllib.error.HTTPError) as ei:
            _post(server + "/api/v1/coach/config", {"gate": {"nope": 1}})
        assert ei.value.code == 400

    def test_config_rejects_non_numeric(self, server):
        """把数值项改成字符串会让之后所有比较静默变 False，必须整条拒绝。"""
        with pytest.raises(urllib.error.HTTPError) as ei:
            _post(server + "/api/v1/coach/config", {"gate": {"max_per_lap": "x"}})
        assert ei.value.code == 400

    def test_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as ei:
            _get(server + "/api/v1/coach/nope")
        assert ei.value.code == 404

    def test_cors_headers_for_dashboard(self, server):
        """仪表盘是**另一个源**，没有 CORS 头浏览器直接拦掉。"""
        _code, _d, headers = _get(server + "/api/v1/coach/state")
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_demo_page_served(self, server):
        with _OPENER.open(server + "/", timeout=5) as r:
            html = r.read().decode("utf-8")
        assert "赛道工程师" in html
        assert "/api/v1/coach/state" in html


class TestServiceRobustness:
    def test_tick_exception_does_not_kill_service(self):
        """tick 里抛异常不能让服务整体死掉 —— 顶多这一轮没数据。"""

        class Boom:
            last_error = None

            def poll(self):
                raise RuntimeError("炸了")

            def sessions(self):
                return []

            def live_session(self):
                return None

        svc = CoachService(CoachEngine(Boom(), CoachConfig()), interval_s=0.02)
        svc.start()
        time.sleep(0.15)
        try:
            d = svc.state()
            assert d["connected"] is False
            assert "tick_error" in d["stats"]
            assert svc.health()["ticks"] > 0, "服务应当还在跑"
        finally:
            svc.stop()
