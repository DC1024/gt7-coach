# -*- coding: utf-8 -*-
"""R2.2 云接入测试 —— 全部用本地假 OpenAI 服务器，零真实外呼。

覆盖：正常 / 500 / 超时 / usage 缺失 / 编数字 / 丢主体 / 编建议 /
主备切换 / 预算闸 / 熔断退避 / 引擎接入状态位 / 降级回落模板。

🔴 设计红线：本文件不得 import requests，不得访问任何外网。
所有「云」都是 127.0.0.1 上 `ThreadingHTTPServer` 起的假端点。
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gt7coach import cloud, phrases
from gt7coach.cloud import CloudError
from gt7coach.narrate import CloudConfig, Narrator

# 测试用 key 环境变量（Narrator 只认变量名，不认明文）
os.environ.setdefault("GT7_TEST_KEY", "test-key")

FACTS = {"lap_time_s": 83.412, "vs_ref_s": 0.37}
# 一句「只用 facts 里的数字」的合法云回复
OK_REPLY = "1:23.412 慢 0.37"
# 一句「编了数字（15 不在 facts 里）」的违规云回复
BAD_REPLY = "快 15 米注意刹车"


# —— 假 OpenAI 服务器 ————————————————————————————————————————

class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):                       # noqa: N802
        srv = self.server
        if srv.behavior == "error500":
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if srv.behavior == "timeout":
            time.sleep(5.0)                 # 超过客户端 timeout_s
            return
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n)
        # 顺手记下请求体 —— 「用户填的 model 到底有没有真发出去」只能从这里看
        try:
            srv.seen.append(json.loads(raw.decode("utf-8") or "{}"))
        except (ValueError, UnicodeDecodeError):
            srv.seen.append({})
        if srv.behavior == "nousage":
            resp = {"choices": [{"message": {"content": srv.reply_text}}]}
        else:
            resp = {
                "choices": [{"message": {"content": srv.reply_text}}],
                "usage": {"prompt_tokens": 3,
                          "completion_tokens": len(srv.reply_text)},
            }
        body = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):              # 静默
        pass


@contextmanager
def fake_llm(behavior="ok", reply_text=OK_REPLY):
    """起一个 127.0.0.1 假端点，yield (base_url, server)。可中途改 behavior/reply_text。"""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.behavior = behavior
    srv.reply_text = reply_text
    srv.seen = []                       # 收到的请求体（含 model 字段），供断言
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", srv
    finally:
        srv.shutdown()
        srv.server_close()


def _cfg(base_url, behavior="ok", reply_text=OK_REPLY, **kw):
    return CloudConfig(
        enabled=True, provider="test", base_url=base_url, model="m",
        api_key_env="GT7_TEST_KEY", timeout_s=1.0, **kw)


# —— cloud.chat 单测 ————————————————————————————————————————

def test_chat_ok():
    with fake_llm("ok") as (url, _):
        r = cloud.chat(url, "m", "test-key", [{"role": "user", "content": "x"}])
    assert isinstance(r, cloud.CloudReply)
    assert r.content == OK_REPLY
    assert r.usage.prompt == 3 and r.usage.completion == len(OK_REPLY)
    assert r.latency_s >= 0


def test_chat_500():
    with fake_llm("error500") as (url, _):
        with pytest.raises(CloudError) as e:
            cloud.chat(url, "m", "test-key", [])
    assert e.value.kind == "http"


def test_chat_timeout():
    with fake_llm("timeout") as (url, _):
        with pytest.raises(CloudError) as e:
            cloud.chat(url, "m", "test-key", [], timeout_s=0.4)
    assert e.value.kind == "timeout"


def test_chat_nousage_estimated():
    with fake_llm("nousage") as (url, _):
        r = cloud.chat(url, "m", "test-key", [])
    assert r.usage.estimated is True
    assert r.usage.completion == len(OK_REPLY)


def test_chat_nokey():
    with fake_llm("ok") as (url, _):
        with pytest.raises(CloudError) as e:
            cloud.chat(url, "m", "", [])
    assert e.value.kind == "nokey"


# —— Narrator 单测 ————————————————————————————————————————

def test_disabled_returns_template():
    """禁用态 = 纯本地模板，与 phrases.render 完全一致（R2.1 行为）。"""
    n = Narrator(cfg=CloudConfig(enabled=False))
    out = n.render("lap_summary", FACTS)
    assert out == phrases.render("lap_summary", FACTS)
    assert n.status()["enabled"] is False


# —— R2.4 圈后综合建议走同一条云链路 ————————————————————————————

ADVICE_FACTS = {"lap_time_s": 92.782, "vs_ref_s": 0.37, "unit": "油",
                "laps_left": 2.4, "sector": 3, "loss_s": 0.31,
                "focus_label": "T12", "focus_laps": 12, "focus_loss_s": 1.234}


def test_lap_advice_goes_through_cloud():
    """`lap_advice` 在 RENDERERS 里 → 走云润色（白名单按同一份 facts 把关）。"""
    assert "lap_advice" in phrases.RENDERERS
    with fake_llm("ok", reply_text="1:32.782 慢 0.37") as (url, _):
        n = Narrator(cfg=_cfg(url))
        out = n.render("lap_advice", ADVICE_FACTS)
    assert out == "1:32.782 慢 0.37"
    assert n.status()["calls"] == 1


def test_lap_advice_falls_back_to_template_on_bad_number():
    """云编了数字 → 丢弃、回落**本地合并模板**（绝不静默）。"""
    with fake_llm("ok", reply_text="快 15 米") as (url, _):
        n = Narrator(cfg=_cfg(url))
        out = n.render("lap_advice", ADVICE_FACTS)
    assert out == phrases.render("lap_advice", ADVICE_FACTS)
    assert n.status()["num_violations"] == 1


# —— 第二、三道闸：云句**拿到了但不敢用** ————————————————————————————
#
# 🔴 真 key 冒烟（2026-10-09）抓到的两个缺口都不是"编数字"，白名单放行：
#     ① 丢了圈速主体（只说了 delta）
#     ② 编了一句不带数字的指令（「注意补油」，而 facts 里 2.4 圈）
#    这两类现在都由代码层的闸拦下，并按同一路径丢弃 + 回落模板。

def test_cloud_dropping_lap_time_is_discarded():
    """缺口①：云句只说了 delta、丢了圈速主体 → 丢弃、回落模板。

    `invented_numbers` 对"少说一个数"完全无感，所以必须由
    `missing_mandatory` 拦 —— 这是「圈速主体永远保留」在云端的唯一保证。
    """
    with fake_llm("ok", reply_text="慢 0.37") as (url, _):
        n = Narrator(cfg=_cfg(url))
        out = n.render("lap_summary", FACTS)
    assert out == phrases.render("lap_summary", FACTS)   # 回落模板
    assert "1:23.412" in out                             # 模板把主体带回来了
    st = n.status()
    assert st["calls"] == 1
    assert st["num_violations"] == 0                     # 不是"编数字"
    assert st["drop_missing"] == 1
    assert st["drop_advice"] == 0
    assert st["cloud_rejects"] == 1
    assert st["fallback_ratio"] == 1.0                   # 拿到的句子全被丢
    assert st["last_guard"]["missing"] == ["lap_time_s"]
    assert st["last_guard"]["advice"] == []
    assert st["degraded"] is True


def test_cloud_inventing_advice_is_discarded():
    """缺口②：facts 里 2.4 圈（不紧张），云却输出「注意补油」。

    它**一个数字都没编** → 白名单放行；必须由 `invented_advice` 拦。
    这条比缺口①危险：它是指令，车手可能真去提前进站。
    """
    with fake_llm("ok", reply_text="1:32.782，注意补油") as (url, _):
        n = Narrator(cfg=_cfg(url))
        out = n.render("lap_advice", ADVICE_FACTS)
    assert out == phrases.render("lap_advice", ADVICE_FACTS)
    assert "补油" not in out                       # 模板不给这个指令
    st = n.status()
    assert st["num_violations"] == 0
    assert st["drop_advice"] == 1
    assert st["drop_missing"] == 0
    assert st["last_guard"]["advice"] == ["补油"]


def test_guard_fallback_template_keeps_the_core():
    """回落的目标是**模板**，而模板保证带圈速 —— 「丢主体」不会真的丢。"""
    with fake_llm("ok", reply_text="注意补油") as (url, _):
        n = Narrator(cfg=_cfg(url))
        out = n.render("lap_advice", ADVICE_FACTS)
    assert out == phrases.render("lap_advice", ADVICE_FACTS)
    assert "1:32.782" in out
    # 模板自身也必须过同样的闸（否则回落的那句本身就是违规的）
    assert phrases.missing_mandatory(out, ADVICE_FACTS) == []
    assert phrases.invented_advice(out, ADVICE_FACTS) == []


def test_cloud_advice_authorized_when_fuel_critical():
    """真的只剩不到 1 圈 → 「这圈进站」是模板自己也会说，放行用云句。"""
    facts = {"lap_time_s": 92.782, "laps_left": 0.8, "unit": "油"}
    with fake_llm("ok", reply_text="1:32.782，油只够 0.8 圈进站") as (url, _):
        n = Narrator(cfg=_cfg(url))
        out = n.render("lap_advice", facts)
    assert out == "1:32.782，油只够 0.8 圈进站"       # 用了云句
    st = n.status()
    assert st["drop_advice"] == 0 and st["cloud_rejects"] == 0
    assert st["fallback_ratio"] == 0.0


def test_guarded_sentences_count_toward_circuit_breaker():
    """连续被闸掉 = 连续失败 → 触发熔断。

    ⚠️ 这是**有意**的：模型连续不守规矩就别再花钱了。代价（云播报整体停
       60 s）与收益都从 `/cloud` 的 `fallback_ratio` 看得见。
    """
    clock = {"t": 1000.0}
    with fake_llm("ok", reply_text="慢 0.37") as (url, _):   # 每次都丢主体
        n = Narrator(cfg=_cfg(url), clock=lambda: clock["t"])
        for _ in range(3):
            assert n.render("lap_summary", FACTS) == \
                phrases.render("lap_summary", FACTS)
        st = n.status()
        assert st["drop_missing"] == 3
        assert st["degraded"] is True
        assert n._st["cooldown_until"] > clock["t"]          # 已进入冷却


def test_fallback_ratio_zero_when_all_cloud_sentences_are_usable():
    with fake_llm("ok") as (url, _):                          # OK_REPLY 合法
        n = Narrator(cfg=_cfg(url))
        n.render("lap_summary", FACTS)
    st = n.status()
    assert st["cloud_rejects"] == 0 and st["fallback_ratio"] == 0.0
    assert st["last_guard"] is None


# —— 提示词版本 —— 改了提示就必须升版本，否则旧缓存会一直命中 ————————————

def test_prompt_version_bumped_for_the_two_guards():
    """🔴 版本号历史 —— 每一版都是真机抓出来的：
       "3" = 把「必须保留圈速主体」「不许编建议」写进提示；
       "4" = 要求圈速写成 M:SS.mmm（3 号版本实测模型写了原始秒数「83.45」）。

    不升版本号的话，同一份 facts 会一直命中上一版产出的旧句子，
    你会以为新提示词没生效 —— 这是最容易白忙一场的坑。
    """
    from gt7coach import prompts
    assert prompts.PROMPT_VERSION == "4"
    assert "不要给事实里没有的建议" in prompts.SYSTEM_PROMPT
    hint = prompts._KEY_HINTS["lap_advice"]
    assert "必须说出来" in hint and "M:SS.mmm" in hint


def test_cloud_raw_seconds_lap_time_is_accepted():
    """🔴 真机回归：模型把圈速写成**原始秒数**（`83.45`）而不是 `1:23.450`。

    它**说了主体**、白名单也认（`83.45` 就是 facts 里的值），绝不能被判成
    "丢了主体" —— 否则 `fallback_ratio` 恒为 1.0，云在 100% 空烧钱。
    实测就这样：改正之前 6 圈 6 句全部被丢。**形式是提示词的活，闸门只查在不在。**
    """
    facts = {"lap_time_s": 83.45, "vs_ref_s": 0.42, "focus_label": "T1",
             "focus_laps": 3, "focus_loss_s": 0.40, "laps_left": 2.3,
             "unit": "油"}
    with fake_llm("ok", reply_text="这圈83.45，T1段慢了0.4秒") as (url, _):
        n = Narrator(cfg=_cfg(url))
        out = n.render("lap_advice", facts)
    assert out == "这圈83.45，T1段慢了0.4秒"        # 用了云句，没有回落
    st = n.status()
    assert st["cloud_rejects"] == 0
    assert st["fallback_ratio"] == 0.0
    assert st["degraded"] is False


def test_bare_minute_token_does_not_satisfy_the_guard():
    """`T1` 里的 `1` 不能冒充圈速 —— 否则"丢了主体"的句子会蒙混过关。"""
    facts = {"lap_time_s": 83.45, "focus_label": "T1", "focus_laps": 3,
             "focus_loss_s": 0.40}
    with fake_llm("ok", reply_text="T1 这段慢了0.40") as (url, _):
        n = Narrator(cfg=_cfg(url))
        out = n.render("lap_advice", facts)
    assert out == phrases.render("lap_advice", facts)   # 回落模板
    assert n.status()["drop_missing"] == 1


# —— 第四道闸：数字归谁（张冠李戴）—————————————————————————

_ATTR_FACTS = {"lap_time_s": 83.45, "vs_ref_s": 0.42, "sector": 2,
               "loss_s": 0.31, "focus_label": "T1", "focus_laps": 3,
               "focus_loss_s": 0.40, "laps_left": 2.3, "unit": "油"}


def test_cloud_misattribution_is_discarded():
    """真机实测：facts 是 `sector=2 / loss_s=0.31 / vs_ref_s=0.42`，
    云句却是「这圈1:23.450，第二段慢了0.42秒」—— 0.42（与参考圈的差）
    被安到了"第二段"头上。

    🔴 圈速必须**与 facts 一致**，否则先被白名单当"编数字"拦下，
       就验证不到"只有归属校验能拦它"这件事。前三道闸在这里全部放行。
    """
    with fake_llm("ok", reply_text="这圈1:23.450，第二段慢了0.42秒") as (url, _):
        n = Narrator(cfg=_cfg(url))
        out = n.render("lap_advice", _ATTR_FACTS)
    assert out == phrases.render("lap_advice", _ATTR_FACTS)   # 回落模板
    st = n.status()
    assert st["num_violations"] == 0, "两个数都在 facts 里 —— 白名单本就放行"
    assert st["drop_missing"] == 0 and st["drop_advice"] == 0
    assert st["drop_misattr"] == 1
    assert st["cloud_rejects"] == 1 and st["fallback_ratio"] == 1.0
    assert st["last_guard"]["misattr"] == ["慢了0.42"]


def test_cloud_correct_attribution_is_used():
    """同一条 facts，把数安对了（0.4 是 T1 的损失）→ 用云句，不回落。

    这是**真机第 1/3/5 圈实际产出的句子形态**（`T1段` 要按弯解），
    必须放行 —— 否则又变成"每一句都被丢"。
    """
    with fake_llm("ok", reply_text="一圈1:23.450，T1段慢了0.4秒") as (url, _):
        n = Narrator(cfg=_cfg(url))
        out = n.render("lap_advice", _ATTR_FACTS)
    assert out == "一圈1:23.450，T1段慢了0.4秒"
    st = n.status()
    assert st["drop_misattr"] == 0
    assert st["cloud_rejects"] == 0 and st["fallback_ratio"] == 0.0
    assert st["degraded"] is False


def test_cache_key_includes_prompt_version(monkeypatch):
    """升级提示词版本必须让旧缓存**自动失效**（否则新提示永远不生效）。"""
    from gt7coach import prompts
    before = Narrator._cache_key("lap_summary", FACTS)
    monkeypatch.setattr(prompts, "PROMPT_VERSION", "99")
    assert Narrator._cache_key("lap_summary", FACTS) != before


def test_no_api_key_falls_back():
    """enabled 但环境变量没设 → 回落模板，记 no_key，不报错。"""
    os.environ.pop("GT7_TEST_KEY", None)
    try:
        n = Narrator(cfg=_cfg("http://127.0.0.1:1"))
        out = n.render("lap_summary", FACTS)
        assert out == phrases.render("lap_summary", FACTS)
        assert n.status()["no_key"] == 1
    finally:
        os.environ["GT7_TEST_KEY"] = "test-key"


def test_polish_ok_and_stats():
    with fake_llm("ok") as (url, _):
        n = Narrator(cfg=_cfg(url))
        out = n.render("lap_summary", FACTS)
    assert out == OK_REPLY                       # 用了云句
    st = n.status()
    assert st["calls"] == 1
    assert st["tokens"] == 3 + len(OK_REPLY)
    assert st["degraded"] is False
    assert st["num_violations"] == 0


def test_invented_numbers_fall_back():
    """云句编了数字（15 不在 facts）→ 丢弃云句、回落模板、记 num_violations。"""
    with fake_llm("ok", reply_text=BAD_REPLY) as (url, _):
        n = Narrator(cfg=_cfg(url))
        out = n.render("lap_summary", FACTS)
    assert out == phrases.render("lap_summary", FACTS)   # 回落模板
    assert n.status()["num_violations"] == 1
    assert n.status()["degraded"] is True


def test_budget_per_lap_gate():
    with fake_llm("ok") as (url, srv):
        n = Narrator(cfg=_cfg(url, limits={"per_lap": 1, "per_session": 30,
                                            "per_day": 300}))
        # 第一圈第一次 → 走云（回复需与 facts 数字一致，否则过不了白名单）
        n.note_lap(1)
        srv.reply_text = "1:23.412 慢 0.37"
        out1 = n.render("lap_summary", {"lap_time_s": 83.412, "vs_ref_s": 0.37})
        assert out1 == "1:23.412 慢 0.37"
        # 同一圈第二次（facts 不同以避开缓存）→ 预算闸拦截，回落模板，不再打外呼
        srv.reply_text = "1:24.000 慢 1.00"
        out2 = n.render("lap_summary", {"lap_time_s": 84.0, "vs_ref_s": 1.0})
        assert out2 == phrases.render("lap_summary",
                                      {"lap_time_s": 84.0, "vs_ref_s": 1.0})
        assert n.status()["budget_drops"] == 1
        assert n.status()["calls"] == 1          # 仍是 1 次真实调用
        # 下一圈 → 预算重置，又能走云
        n.note_lap(2)
        srv.reply_text = "1:25.000 慢 2.00"
        out3 = n.render("lap_summary", {"lap_time_s": 85.0, "vs_ref_s": 2.0})
        assert out3 == "1:25.000 慢 2.00"
        assert n.status()["calls"] == 2


def test_failover_primary_to_fallback():
    """主家连不上 → 顺位顶上备家，记 switches。"""
    with fake_llm("ok") as (ok_url, _):
        # 主家用一个必然失败的地址，备家指向假 ok 端点
        n = Narrator(cfg=CloudConfig(
            enabled=True, provider="p1", base_url="http://127.0.0.1:1",
            model="m", api_key_env="GT7_TEST_KEY", timeout_s=0.4,
            fallbacks=["p2"],
        ))
        # 把 fallback 的 base_url 指到真 ok 端点：用 monkeypatch 改预设
        from gt7coach import cloud as cloud_mod
        cloud_mod.PROVIDERS["p2"] = {"base_url": ok_url}
        try:
            out = n.render("lap_summary", FACTS)
        finally:
            del cloud_mod.PROVIDERS["p2"]     # 不能泄漏，污染后面的测试
    assert out == OK_REPLY
    st = n.status()
    assert st["switches"] == 1
    assert st["last_switch_to"] == "p2"


def test_circuit_breaker_trips_and_cools_down():
    """连续 3 次失败 → 进入冷却，冷却期内回落模板且不再打外呼。"""
    clock = {"t": 1000.0}

    def fake_clock():
        return clock["t"]

    with fake_llm("error500") as (url, _):
        n = Narrator(cfg=_cfg(url), clock=fake_clock)
        # 3 次失败 → 触发熔断
        for _ in range(3):
            out = n.render("lap_summary", FACTS)
            assert out == phrases.render("lap_summary", FACTS)  # 回落模板
        assert n.status()["degraded"] is True
        assert n._st["cooldown_until"] > clock["t"]             # 进入冷却
        calls_after_trip = n.status()["calls"]
        # 冷却期内：再调用不应打外呼（calls 不增）
        _ = n.render("lap_summary", FACTS)
        assert n.status()["calls"] == calls_after_trip
    # 推进时钟越过冷却期 → 半开放 1 次探测（这次会再失败并续期冷却）
    clock["t"] += 1000.0
    with fake_llm("error500") as (url, _):
        n2 = Narrator(cfg=_cfg(url), clock=fake_clock)
        for _ in range(3):
            n2.render("lap_summary", FACTS)
        before_err = n2.status()["errors"]
        clock["t"] += 1000.0
        n2.render("lap_summary", FACTS)     # 半开放探测（仍失败 → 续期冷却）
        # 失败的外呼也会记 errors：证明冷却期结束后确实又打了一次探测
        assert n2.status()["errors"] == before_err + 1


def test_cache_hit_avoids_call():
    """同 facts 二次渲染命中缓存，不重复打外呼。"""
    with fake_llm("ok") as (url, _):
        n = Narrator(cfg=_cfg(url))
        out1 = n.render("lap_summary", FACTS)
        calls1 = n.status()["calls"]
        out2 = n.render("lap_summary", FACTS)     # 同一 facts → 缓存
        assert out2 == out1
        assert n.status()["calls"] == calls1      # 没再打外呼
        assert n.status()["cache_hits"] == 1


# —— 引擎接入 ————————————————————————————————————————

class _DummySrc:
    """返回 None 帧的假源，仅用于验证 tick 把 cloud 状态塞进 state。"""
    last_error = None

    def poll(self):
        return None


def test_engine_wires_cloud_status_into_state():
    from gt7coach.engine import CoachEngine, CoachConfig
    eng = CoachEngine(_DummySrc(), CoachConfig())
    assert isinstance(eng.narrator, Narrator)
    st = eng.tick()
    assert "cloud" in st.stats
    assert st.stats["cloud"]["enabled"] is False


def test_engine_narrator_disabled_matches_template():
    """禁用态 narrator 渲染结果与本地模板一致（回归：判断逻辑不变）。"""
    from gt7coach.engine import CoachEngine, CoachConfig
    eng = CoachEngine(_DummySrc(), CoachConfig())
    assert eng.narrator.render("lap_summary", FACTS) == \
        phrases.render("lap_summary", FACTS)


def test_server_cloud_endpoint():
    """GET /api/v1/coach/cloud 真实跑通，返回禁用态的云状态。"""
    import json as _json
    import threading
    import urllib.request

    from gt7coach.engine import CoachConfig, CoachEngine
    from gt7coach.server import CoachService, make_server

    eng = CoachEngine(_DummySrc(), CoachConfig())
    svc = CoachService(eng)
    srv = make_server(svc, host="127.0.0.1", port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    port = srv.server_address[1]
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/v1/coach/cloud", timeout=3) as r:
            data = _json.loads(r.read())
        assert data["enabled"] is False
        assert "degraded" in data and "num_violations" in data
        assert "est_cost_yuan" in data
    finally:
        srv.shutdown()
        srv.server_close()


# —— 费用估算：输入/输出**分档**计价 ——————————————————————————————
#
# 🔴 回归背景：旧实现写死 `tokens / 1e4 * 1.4`，把方案 §5 里 **TTS 的
#    「1.4 元/万字符」** 当成了 token 单价 —— 单位错、量级错，实测真机一次
#    调用报 ¥0.042 而真实约 ¥0.00007。以下是锁住正确口径的护栏。

def test_cost_uses_two_tier_pricing():
    """est_cost = prompt/1e6*price_in + completion/1e6*price_out。"""
    with fake_llm("ok") as (url, _):
        n = Narrator(cfg=_cfg(url))
        n.render("lap_summary", FACTS)
    st = n.status()
    assert st["tokens_prompt"] == 3
    assert st["tokens_completion"] == len(OK_REPLY)
    assert st["tokens"] == 3 + len(OK_REPLY)      # 合计仍保留（兼容旧消费者）
    assert st["est_cost_yuan"] == round(
        3 / 1e6 * 0.15 + len(OK_REPLY) / 1e6 * 1.5, 6)


def test_cost_respects_configured_prices():
    """换厂商/换模型时单价可配，/cloud 端点报的费用随之变。"""
    with fake_llm("ok") as (url, _):
        n = Narrator(cfg=_cfg(url, price_in_yuan_per_mtok=1.0,
                              price_out_yuan_per_mtok=2.0))
        n.render("lap_summary", FACTS)
    st = n.status()
    assert st["est_cost_yuan"] == round(
        3 / 1e6 * 1.0 + len(OK_REPLY) / 1e6 * 2.0, 6)
    assert st["price_yuan_per_mtok"] == {"in": 1.0, "out": 2.0}


def test_cost_not_overestimated():
    """护栏：这一次调用真实花费在 1e-5 元量级。

    旧公式 `(3+15)/1e4*1.4 = ¥0.00252` 会在这里失败（差约 110 倍）；
    prompt 更长时（真机 283 token）差距达 600 倍。
    """
    with fake_llm("ok") as (url, _):
        n = Narrator(cfg=_cfg(url))
        n.render("lap_summary", FACTS)
    assert n.status()["est_cost_yuan"] < 1e-4


def test_no_calls_means_zero_cost():
    """一次都没走云（无 key）→ 费用必须是 0，不能凭空计。"""
    n = Narrator(cfg=CloudConfig(
        enabled=True, provider="test", base_url="http://127.0.0.1:1",
        model="m", api_key_env="GT7_NO_SUCH_KEY_ENV", timeout_s=1.0))
    n.render("lap_summary", FACTS)
    assert n.status()["est_cost_yuan"] == 0.0


def test_cloud_config_parses_price_fields():
    """cloud.json 里单价写成字符串也要能吃（float 转换）。"""
    cfg = CloudConfig.from_dict(
        {"enabled": True, "price_in_yuan_per_mtok": "0.5",
         "price_out_yuan_per_mtok": 2})
    assert cfg.price_in_yuan_per_mtok == 0.5
    assert cfg.price_out_yuan_per_mtok == 2.0
    # 缺字段 → 回落 qwen-flash 官方默认价
    d = CloudConfig.from_dict({})
    assert (d.price_in_yuan_per_mtok, d.price_out_yuan_per_mtok) == (0.15, 1.5)


# —— 无预设模型 / 无免费承诺（2026-10-10 用户要求）—————————————————
#
# 起因：状态里显示「qwen3.8-flash（厂商预设）· 免费额度内」，用户指出
# **它可能今天免费，明天就收费或者被删除了** —— 项目不该向玩家做任何
# 免费承诺。要求：
#   ① 端点表/UI 预设里**没有任何模型名**，模型名一律玩家自己填；
#   ② 「免费额度名单」与一切免费/计费提示**整体删除**；
#   ③ ollama 本质是 OpenAI 兼容端点，label 直接叫「OpenAI 兼容」。

class TestNoModelPreset:
    """预设里不许有任何模型名 —— 改预设时这些测试会红。"""

    def test_providers_have_no_model_field(self):
        for name, p in cloud.PROVIDERS.items():
            assert "model" not in p, f"端点表 {name} 不许再带模型预设"

    def test_ui_presets_have_no_model_field(self):
        for name, p in cloud.PROVIDER_PRESETS.items():
            assert "model" not in p, f"UI 预设 {name} 不许再带模型预设"

    def test_ollama_is_labeled_openai_compatible(self):
        assert cloud.PROVIDER_PRESETS["ollama"]["label"] == "OpenAI 兼容"

    def test_free_registry_is_gone(self):
        """免费名单与三个免费提示函数必须整体删除，不许留半套。"""
        for attr in ("FREE_MODELS", "free_info", "is_free_model",
                     "model_warning"):
            assert not hasattr(cloud, attr), attr


class TestUserSuppliedModel:
    """模型名玩家自己填 —— 填了必须**真的用上**，留空则云措辞不可用。"""

    def _narrator(self, url, **kw):
        """把 dashscope 预设指到本地假端点，模拟真实的预设厂商。"""
        real = dict(cloud.PROVIDERS["dashscope"])
        cloud.PROVIDERS["dashscope"] = {**real, "base_url": url}
        return real, Narrator(cfg=CloudConfig(
            enabled=True, provider="dashscope", api_key_env="GT7_TEST_KEY",
            timeout_s=1.0, **kw))

    def test_user_model_is_sent_as_is(self):
        with fake_llm("ok") as (url, srv):
            real, n = self._narrator(url, model="kimi-k3")
            try:
                out = n.render("lap_summary", FACTS)
                assert out == OK_REPLY
                assert srv.seen[-1]["model"] == "kimi-k3"
            finally:
                cloud.PROVIDERS["dashscope"] = real

    def test_empty_model_disables_cloud_calls(self):
        """🔴 留空 = 云措辞不可用（回落本地模板），**没有**默认模型兜底。"""
        with fake_llm("ok") as (url, srv):
            real, n = self._narrator(url, model="")
            try:
                n.render("lap_summary", FACTS)
                assert not srv.seen, "模型名留空竟还发了外呼"
            finally:
                cloud.PROVIDERS["dashscope"] = real

    def test_fallback_uses_the_user_model_too(self):
        """备用厂商没有自己的模型预设 —— 主备只切换 base_url，
        模型名统一用玩家填的那个。"""
        with fake_llm("ok") as (url, srv):
            real = dict(cloud.PROVIDERS["dashscope"])
            cloud.PROVIDERS["dashscope"] = {**real, "base_url": "http://127.0.0.1:1"}
            cloud.PROVIDERS["__fb_test__"] = {"base_url": url}
            try:
                n = Narrator(cfg=CloudConfig(
                    enabled=True, provider="dashscope", model="main-model",
                    api_key_env="GT7_TEST_KEY", timeout_s=0.4,
                    fallbacks=["__fb_test__"]))
                n.render("lap_summary", FACTS)
                assert srv.seen[-1]["model"] == "main-model"
            finally:
                cloud.PROVIDERS["dashscope"] = real
                del cloud.PROVIDERS["__fb_test__"]

    def test_status_reports_the_user_model_only(self):
        """状态只透出玩家填的模型名；model_from_user / model_is_free /
        model_warning 等免费标注字段已整体删除。"""
        with fake_llm("ok") as (url, _):
            real, n = self._narrator(url, model="kimi-k3")
            try:
                st = n.status()
                assert st["model"] == "kimi-k3"
                for gone in ("model_from_user", "model_is_free",
                             "model_free", "model_warning"):
                    assert gone not in st, gone
                # 只报变量名与"有没有设"，绝不回显 key
                assert st["api_key_env"] == "GT7_TEST_KEY"
                assert st["has_key"] is True
            finally:
                cloud.PROVIDERS["dashscope"] = real

    def test_status_empty_model_reports_empty(self):
        with fake_llm("ok") as (url, _):
            real, n = self._narrator(url, model="")
            try:
                st = n.status()
                assert st["model"] == ""
            finally:
                cloud.PROVIDERS["dashscope"] = real
