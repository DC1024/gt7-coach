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

    def test_config_tts_change_over_http(self, server):
        """改 `tts_*` 要**重建** TtsEngine 并把新状态回给调用方。"""
        code, d = _post(server + "/api/v1/coach/config",
                        {"coach": {"tts_max_chars": 33}})
        assert code == 200
        assert d["applied"]["coach"] == ["tts_max_chars"]
        assert d["tts_rebuilt"] is True
        assert d["config"]["coach"]["tts_max_chars"] == 33
        assert d["tts"]["max_chars"] == 33      # 真的进了引擎，不只是进了 cfg

    def test_config_other_change_does_not_rebuild_tts(self, server):
        _code, d = _post(server + "/api/v1/coach/config",
                         {"gate": {"max_per_lap": 4}})
        assert "tts_rebuilt" not in d

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
        # 面板内嵌在自检页里（用户自选播报内容）
        assert "/api/v1/coach/panel" in html


class TestBroadcastPanel:
    """播报开关面板：用户自选"什么内容播报、什么不播报"。"""

    def test_panel_lists_groups(self, server):
        code, d, _ = _get(server + "/api/v1/coach/panel")
        assert code == 200
        ids = [g["id"] for g in d["groups"]]
        assert ids == ["safety", "driving", "tyres", "pace", "debrief"]
        assert d["muted"] == []                 # 默认全开
        assert all("muted" in g and "label" in g for g in d["groups"])

    def test_panel_set_and_readback(self, server):
        code, d = _post(server + "/api/v1/coach/panel",
                        {"muted": ["tyres", "pace"]})
        assert code == 200 and d["ok"] is True
        assert d["muted"] == ["pace", "tyres"]  # 归一化：排序
        _c, d2, _ = _get(server + "/api/v1/coach/panel")
        assert d2["muted"] == ["pace", "tyres"]
        # 生效：闸门配置同步更新
        _c, cfg, _ = _get(server + "/api/v1/coach/config")
        assert sorted(cfg["gate"]["muted"]) == ["pace", "tyres"]

    def test_panel_rejects_unknown_group(self, server):
        with pytest.raises(urllib.error.HTTPError) as ei:
            _post(server + "/api/v1/coach/panel", {"muted": ["nope"]})
        assert ei.value.code == 400

    def test_panel_rejects_non_list(self, server):
        with pytest.raises(urllib.error.HTTPError) as ei:
            _post(server + "/api/v1/coach/panel", {"muted": "tyres"})
        assert ei.value.code == 400

    def test_panel_can_reopen_all(self, server):
        _post(server + "/api/v1/coach/panel", {"muted": ["tyres"]})
        _c, d = _post(server + "/api/v1/coach/panel", {"muted": []})
        assert d["muted"] == []


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


class TestTtsConfigHotReload:
    """🔴 `POST /config` 改 `tts_*` 必须**重建** TtsEngine 才生效。

    `TtsEngine.__init__` 把 `tts_config()` 的**快照**收进 `self.cfg`，
    之后再不回看 `CoachConfig` —— 所以不重建就会"存进 cfg 但不生效"：
    **读回配置像是改了、行为一点没变**（最糟的一种失灵，因为它看不出来）。
    """

    class _Src:
        last_error = None

        def poll(self):
            return None

    def _svc(self):
        return CoachService(CoachEngine(self._Src(), CoachConfig()))

    def test_numeric_tts_change_swaps_the_engine(self):
        svc = self._svc()
        old = svc.engine.tts
        out = svc.update_config({"coach": {"tts_max_chars": 20}})
        assert out["applied"]["coach"] == ["tts_max_chars"]
        assert out["tts_rebuilt"] is True
        new = svc.engine.tts
        assert new is not old, "必须换一台新引擎 —— 旧引擎读的是旧快照"
        assert new.cfg.max_chars == 20          # 新值真的进了引擎
        assert out["tts"]["max_chars"] == 20
        assert new.enabled is False             # 没配 workspace → 静默禁用

    def test_rebuild_keeps_budget_counters(self):
        """改配置不能白拿一份配额（`chars_per_day` 是唯一的成本硬顶）。"""
        svc = self._svc()
        svc.engine.tts.note_lap(3)                 # 🔴 先 note_lap：它会清零本圈计数
        svc.engine.tts._st["chars_day"] = 123
        svc.engine.tts._st["chars_lap"] = 45
        svc.update_config({"coach": {"tts_max_chars": 30}})
        st = svc.engine.tts.status()
        assert st["chars_today"] == 123
        assert st["chars_lap"] == 45

    def test_other_changes_leave_tts_alone(self):
        svc = self._svc()
        old = svc.engine.tts
        out = svc.update_config({"gate": {"max_per_lap": 4}})
        assert "tts_rebuilt" not in out
        assert svc.engine.tts is old

    def test_tts_string_fields_are_still_rejected(self):
        """`tts_workspace_id` 是字符串 —— 仍走"只允许改数值项"的硬规则。

        放开它没有好处：`tts_config()` 会把值原样收进快照，一个笔误
        （比如把 workspace 写成别的空间）在状态里只表现为"合成 403"。
        """
        svc = self._svc()
        with pytest.raises(ValueError):
            svc.update_config({"coach": {"tts_workspace_id": "ws-x"}})


class TestHealthDiagnostics:
    """`/health` 必须能一眼回答「是谁的问题」。

    排障时第一个要回答的问题是：网络断了？仪表盘在忙？还是真的没在录？
    这三种情况在 health 里长得**一模一样**（都是 connected=false），
    所以场次发现与参考圈取数的状态必须各自单独暴露。
    """

    def test_health_exposes_both_states(self, server):
        code, d, _ = _get(server + "/api/v1/coach/health")
        assert code == 200
        for k in ("sess_state", "ref_state", "ref_source",
                  "sess_error", "ref_error"):
            assert k in d, k

    def test_sess_state_reaches_ok(self, server):
        """ReplaySource 的场次发现很快 → 应落到 ok（不是一直 loading）。"""
        from conftest import wait_for
        assert wait_for(lambda: _get(server + "/api/v1/coach/health")[1]
                        .get("sess_state") == "ok", timeout=3.0), \
            _get(server + "/api/v1/coach/health")[1]


class TestVoicePreemption:
    """P0 要能**打断**正在念的闲话。

    浏览器 TTS 默认是排队制：一句 delta 会把随后的"出界"堵在它后面，
    等念完黄花菜都凉了。真赛车无线电是抢麦，不是排队。
    """

    def test_demo_page_cancels_on_p0(self, server):
        with _OPENER.open(server + "/", timeout=5) as r:
            html = r.read().decode("utf-8")
        assert "speechSynthesis.cancel()" in html
        assert "prio === 0" in html, "只有 P0 才抢占，否则会互相打断"
        # 播报时必须把优先级传进去（不传就等于永远不抢占）
        assert "say(u.speech || u.text, u.priority)" in html
        # 🔴 R3 回归：A 档不得进"等云 TTS"分支 —— 云合成要 0.5~2 s，
        #    让「出界了」等 0.9 s 就完全失去意义了。
        assert "u.priority < P_NORMAL" in html
