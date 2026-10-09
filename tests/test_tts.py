# -*- coding: utf-8 -*-
"""R3 云 TTS 测试 —— 全部用本地假 CosyVoice 服务器，零真实外呼。

覆盖四条红线 + 所有失败路径：

  1. **A 档永不上云**：只有 priority >= P_NORMAL 的句子才进这条链路。
  2. **绝不阻塞 tick**：`request()` O(1)；合成在后台线程。
  3. **失败永远回落浏览器 TTS**：无 key / 无 workspace / HTTP 错 / 超时 /
     下载失败 / 目录写不了 —— 全部只是"拿不到 tts_url"，绝不外抛到 tick。
  4. **音频落盘缓存**：命中即微秒级本地读，零网络零费用。

🔴 设计红线：本文件不得访问任何外网。所有"云"都是 127.0.0.1 上
`ThreadingHTTPServer` 起的假端点；provider 表里临时注册一个 `test` 适配器
（顺带验证"换厂商只改一个函数"这个抽象是真的成立）。
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gt7coach import tts as tts_mod
from gt7coach.tts import TtsConfig, TtsEngine, TtsError

# 测试用 key 环境变量（引擎只认变量名，不认明文）
os.environ.setdefault("GT7_TEST_TTS_KEY", "test-tts-key")

# 假音频载荷：开头带 ID3 便于人工辨认，长度固定便于断言
AUDIO = b"ID3" + b"\x00" * 61

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

P_NORMAL = 2                       # 与 contract.py 对齐：priority < 2 即 A 档


# —— HTTP 小工具 ————————————————————————————————————————————

def _get(url: str):
    with _OPENER.open(url, timeout=5) as r:
        return r.status, json.loads(r.read().decode("utf-8")), dict(r.headers)


def _get_bytes(url: str):
    with _OPENER.open(url, timeout=5) as r:
        return r.status, r.read(), dict(r.headers)


def _post(url: str, body: dict):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=5) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _post_bytes(url: str, body: dict):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=5) as r:
        return r.status, r.read()


# —— 假 CosyVoice 服务器 ————————————————————————————————————

class _Handler(BaseHTTPRequestHandler):
    """两步协议：POST /synth 拿音频 URL → GET 那个 URL 拿字节。"""

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, body: bytes, ctype="application/octet-stream"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):                       # noqa: N802
        srv = self.server
        with srv.lock:
            srv.calls += 1
        if srv.delay:
            time.sleep(srv.delay)
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n)
        try:
            srv.last_body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            srv.last_body = None
        srv.last_auth = self.headers.get("Authorization")

        b = srv.behavior
        if b == "error500":
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if b == "timeout":
            time.sleep(srv.delay or 2.0)     # 超过客户端 timeout_s
            return
        if b == "notjson":
            self._raw(b"<html>502 Bad Gateway</html>", "text/html")
            return

        text = ((srv.last_body or {}).get("input") or {}).get("text") or ""
        if not isinstance(text, str):
            text = ""
        port = srv.server_address[1]
        audio_url = f"http://127.0.0.1:{port}/audio/{len(text)}.mp3"

        if b == "badshape":                  # output 里没有 audio.url
            self._json({"output": {"audio": {}}, "usage": {"characters": 1}})
            return
        if b == "audio_not_dict":            # audio 是字符串而不是对象
            self._json({"output": {"audio": audio_url}})
            return
        if b == "noout":                     # 整个 output 缺失
            self._json({"usage": {"characters": 1}})
            return
        if b == "badscheme":                 # 音频 URL 不是 http(s)
            self._json({"output": {"audio": {"url": "file:///C:/secret"}}})
            return
        if b == "empty":
            audio_url = f"http://127.0.0.1:{port}/audio/empty.mp3"

        chars = srv.force_chars if srv.force_chars is not None else len(text)
        payload: dict = {"output": {"audio": {"url": audio_url}}}
        if not srv.no_usage:
            payload["usage"] = {"characters": chars}
        self._json(payload)

    def do_GET(self):                        # noqa: N802
        srv = self.server
        with srv.lock:
            srv.audio_calls += 1
        if srv.audio_behavior == "error404":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = b"" if (srv.audio_behavior == "empty" or "empty" in self.path) \
            else AUDIO
        self._raw(data, "audio/mpeg")

    def log_message(self, *a):               # 静默
        pass


def _register_test_provider(port: int, audio_host=None) -> None:
    """注册一个本地 provider —— 复用百炼的 body/parse/chars 纯函数。"""
    tts_mod.PROVIDERS["test"] = {
        "label": "本地假 CosyVoice",
        "model": "cosyvoice-v3-flash",
        "voice": "longanyang",
        "endpoint": lambda cfg: f"http://127.0.0.1:{port}/synth",
        "build": tts_mod._bailian_build,
        "parse": tts_mod._bailian_parse,
        "chars": tts_mod._bailian_chars,
        "audio_host": audio_host,
    }


@contextmanager
def fake_tts(behavior="ok", *, audio_behavior="ok", delay=0.0,
             audio_host=None, force_chars=None, no_usage=False):
    """起 127.0.0.1 假端点并注册 provider。可中途改 behavior（测线程存活）。"""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.lock = threading.Lock()
    srv.calls = 0
    srv.audio_calls = 0
    srv.behavior = behavior
    srv.audio_behavior = audio_behavior
    srv.delay = delay
    srv.force_chars = force_chars
    srv.no_usage = no_usage
    srv.last_body = None
    srv.last_auth = None
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _register_test_provider(port, audio_host)
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        tts_mod.PROVIDERS.pop("test", None)


def _cfg(tmp_path=None, **kw) -> TtsConfig:
    base = dict(enabled=True, provider="test", workspace_id="ws-test",
                cache_dir=str(tmp_path) if tmp_path else None,
                api_key_env="GT7_TEST_TTS_KEY", timeout_s=1.0)
    base.update(kw)
    return TtsConfig(**base)


@contextmanager
def engine(tmp_path, **kw):
    """起一个引擎并保证退出时停掉后台线程（否则线程泄漏会污染后续测试）。"""
    e = TtsEngine(_cfg(tmp_path, **kw))
    try:
        yield e
    finally:
        e.close()


# —— 1. 端点与响应解析（纯函数，零网络）——————————————————————

class TestBailianAdapter:
    def test_endpoint_uses_workspace_domain(self):
        """🔴 语音合成**不在** dashscope 上 —— 照抄 chat 的域名只会拿到 404。"""
        url = tts_mod._bailian_endpoint(TtsConfig(workspace_id="ws-abc"))
        assert url == ("https://ws-abc.cn-beijing.maas.aliyuncs.com"
                       "/api/v1/services/audio/tts/SpeechSynthesizer")
        assert "dashscope" not in url
        assert "cn-beijing" in url, "该服务只在华北2（北京）地域可用"

    def test_endpoint_without_workspace_is_nokey(self):
        with pytest.raises(TtsError) as e:
            tts_mod._bailian_endpoint(TtsConfig(workspace_id=""))
        assert e.value.kind == "nokey", "报成 http 会被误读为'key 没权限'"

    def test_default_model_has_system_voices(self):
        """🔴 回归闸：v3.5 系列**不支持系统音色**，默认必须留在 v3-flash。

        踩过一次：默认写 v3.5-flash + longanyang，第一次真调用就会报
        "音色不存在"，而单测全绿 —— 因为默认值没人断言。
        """
        default = tts_mod.PROVIDERS["bailian"]["model"]
        assert default == "cosyvoice-v3-flash"
        assert not default.startswith("cosyvoice-v3.5")
        assert tts_mod.PROVIDERS["bailian"]["voice"] == "longanyang"

    def test_parse_ok(self):
        assert tts_mod._bailian_parse(
            {"output": {"audio": {"url": "http://x/a.mp3"}}}) == "http://x/a.mp3"

    def test_parse_missing_url(self):
        with pytest.raises(TtsError) as e:
            tts_mod._bailian_parse({"output": {"audio": {}}})
        assert e.value.kind == "badjson"

    def test_parse_audio_not_dict(self):
        with pytest.raises(TtsError) as e:
            tts_mod._bailian_parse({"output": {"audio": "http://x/a.mp3"}})
        assert e.value.kind == "badjson"

    @pytest.mark.parametrize("raw,expect", [
        ({"usage": {"characters": 42}}, 42),
        ({"usage": {"characters": "42"}}, 42),
        ({}, 0),
        ({"usage": {}}, 0),
        ({"usage": {"characters": None}}, 0),
        ({"usage": {"characters": "abc"}}, 0),
        ({"usage": None}, 0),
    ])
    def test_chars(self, raw, expect):
        assert tts_mod._bailian_chars(raw) == expect


# —— 2. 禁用态：静默降级，绝不报错 ————————————————————————————
#
# 🔴 这一组用**真实 provider 名**（bailian）而不是测试用的 `test`：
#    `disabled_reason` 会先判 provider 是否存在，用 `test` 会被
#    "未知 provider" 抢先命中，把真正要验的那条原因盖掉。

class TestDisabled:
    def test_not_enabled(self, tmp_path):
        e = TtsEngine(_cfg(tmp_path, provider="bailian", enabled=False))
        assert e.enabled is False
        assert "未开启" in e.disabled_reason
        assert e.request("随便说点什么") is None
        with pytest.raises(TtsError) as ei:
            e.synthesize_now("随便说点什么")
        assert ei.value.kind == "disabled"

    def test_missing_workspace(self, tmp_path):
        e = TtsEngine(_cfg(tmp_path, provider="bailian", workspace_id=""))
        assert e.enabled is False
        assert e.disabled_reason == "缺 workspace_id"
        assert e.request("x") is None

    def test_missing_cache_dir(self):
        """没缓存 = 每句都真花钱，所以直接禁用而不是"能跑但不缓存"。"""
        e = TtsEngine(_cfg(provider="bailian"))
        assert e.enabled is False
        assert "cache_dir" in e.disabled_reason

    def test_unwritable_cache_dir(self, tmp_path):
        bad = tmp_path / "afile"
        bad.write_text("not a dir")
        e = TtsEngine(_cfg(bad, provider="bailian"))
        assert e.enabled is False
        assert "不可写" in e.disabled_reason

    def test_unknown_provider(self, tmp_path):
        e = TtsEngine(_cfg(tmp_path, provider="nope"))
        assert e.enabled is False
        assert "未知 provider" in e.disabled_reason

    def test_disabled_engine_never_opens_network(self, tmp_path):
        """没起任何假服务器 —— 禁用态下就算 request 也不该有出网动作。"""
        e = TtsEngine(_cfg(tmp_path, provider="bailian", enabled=False))
        assert e.request("一" * 10) is None
        assert e.status()["calls"] == 0
        assert e.status()["errors"] == 0


# —— 3. 同步合成 + 落盘缓存 ————————————————————————————————

class TestSynthesize:
    def test_bytes_and_cache(self, tmp_path):
        with fake_tts() as srv, engine(tmp_path) as e:
            data = e.synthesize_now("本圈比参考慢零点四秒")
            assert data == AUDIO
            assert srv.calls == 1 and srv.audio_calls == 1
            assert e.is_cached("本圈比参考慢零点四秒")
            url = e.url_for("本圈比参考慢零点四秒")
            handle = f"{e._hash('本圈比参考慢零点四秒')}.mp3"
            assert url == f"{tts_mod.TTS_HANDLE_PATH}/{handle}"
            assert e.read_cached(handle) == AUDIO

    def test_second_call_is_local(self, tmp_path):
        """缓存命中 = 零外呼零费用。这是整套设计里最省钱的一条。"""
        with fake_tts() as srv, engine(tmp_path) as e:
            e.synthesize_now("这句话会说很多次")
            e.synthesize_now("这句话会说很多次")
            assert srv.calls == 1, "第二次不该再出网"
            assert e.status()["cache_hits"] == 1

    def test_no_part_file_left(self, tmp_path):
        """先写 .part 再 rename —— 绝不能留下半截文件在缓存目录里。"""
        with fake_tts(), engine(tmp_path) as e:
            e.synthesize_now("原子落盘")
            assert list(tmp_path.rglob("*.part")) == []

    def test_hash_is_16_hex(self, tmp_path):
        with engine(tmp_path) as e:
            h = e._hash("x")
            assert len(h) == 16
            int(h, 16)                       # 不是十六进制就抛

    @pytest.mark.parametrize("field,value", [
        ("voice", "longxiaochun"),
        ("model", "cosyvoice-v3-plus"),
        ("audio_format", "wav"),
        ("sample_rate", 16000),
    ])
    def test_hash_covers_voice_and_format(self, tmp_path, field, value):
        """换音色/采样率必须重新合成 —— 否则会出现"换了声音还是旧音色"。"""
        with engine(tmp_path) as e:
            a = e._hash("同一句话")
        e2 = TtsEngine(_cfg(tmp_path, **{field: value}))
        try:
            b = e2._hash("同一句话")
        finally:
            e2.close()
        assert a != b, f"{field} 没进缓存键"

    def test_usage_chars_come_from_server(self, tmp_path):
        """计费以**服务端回报**的字符数为准，不是 len(text)（对账才站得住）。"""
        with fake_tts(force_chars=50), engine(tmp_path) as e:
            e.synthesize_now("短句")
            st = e.status()
            assert st["chars"] == 50
            assert st["chars_lap"] == 50
            assert st["est_cost_yuan"] == pytest.approx(50 / 1000.0 * 0.1)

    def test_no_usage_block(self, tmp_path):
        with fake_tts(no_usage=True), engine(tmp_path) as e:
            assert e.synthesize_now("没有 usage 块") == AUDIO
            assert e.status()["chars"] == 0

    def test_request_body_shape(self, tmp_path):
        """请求体形状对齐百炼文档；Authorization 必须是 Bearer。"""
        with fake_tts() as srv, engine(tmp_path) as e:
            e.synthesize_now("念这句")
            assert srv.last_auth == "Bearer test-tts-key"
            assert srv.last_body == {"model": "cosyvoice-v3-flash",
                                     "input": {"text": "念这句",
                                               "voice": "longanyang",
                                               "format": "mp3",
                                               "sample_rate": 24000}}

    def test_empty_text_rejected(self, tmp_path):
        with engine(tmp_path) as e:
            for bad in ("", "   ", None):
                with pytest.raises(TtsError) as ei:
                    e.synthesize_now(bad)
                assert ei.value.kind == "badjson"

    def test_too_long_text_rejected(self, tmp_path):
        with fake_tts() as srv, engine(tmp_path, max_chars=10) as e:
            with pytest.raises(TtsError) as ei:
                e.synthesize_now("字" * 11)
            assert ei.value.kind == "badjson"
            assert srv.calls == 0, "超长句应该在出网前就被拦掉"


# —— 4. 失败路径：全部只是"拿不到音频"，绝不炸 ————————————————————

class TestFailures:
    def test_http_500(self, tmp_path):
        with fake_tts("error500"), engine(tmp_path) as e:
            with pytest.raises(TtsError) as ei:
                e.synthesize_now("会挂")
            assert ei.value.kind == "http"
            assert not e.is_cached("会挂")

    def test_timeout(self, tmp_path):
        with fake_tts("timeout", delay=2.0), engine(tmp_path, timeout_s=0.3) as e:
            with pytest.raises(TtsError) as ei:
                e.synthesize_now("超时")
            assert ei.value.kind == "timeout"

    def test_no_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GT7_TEST_TTS_KEY", raising=False)
        with fake_tts() as srv, engine(tmp_path) as e:
            with pytest.raises(TtsError) as ei:
                e.synthesize_now("没 key")
            assert ei.value.kind == "nokey"
            assert srv.calls == 0, "没 key 就不该浪费一次出网"
            assert not e.status()["has_key"]

    def test_bad_response_shape(self, tmp_path):
        with fake_tts("badshape"), engine(tmp_path) as e:
            with pytest.raises(TtsError) as ei:
                e.synthesize_now("形状不对")
            assert ei.value.kind == "badjson"

    def test_audio_field_not_dict(self, tmp_path):
        """`audio` 是字符串而不是对象 —— 有专门的 AttributeError 分支。"""
        with fake_tts("audio_not_dict"), engine(tmp_path) as e:
            with pytest.raises(TtsError) as ei:
                e.synthesize_now("audio 不是对象")
            assert ei.value.kind == "badjson"

    def test_output_missing(self, tmp_path):
        with fake_tts("noout"), engine(tmp_path) as e:
            with pytest.raises(TtsError) as ei:
                e.synthesize_now("没有 output")
            assert ei.value.kind == "badjson"

    def test_non_json_body(self, tmp_path):
        with fake_tts("notjson"), engine(tmp_path) as e:
            with pytest.raises(TtsError) as ei:
                e.synthesize_now("回了个 HTML")
            assert ei.value.kind == "badjson"

    def test_audio_download_404(self, tmp_path):
        with fake_tts(audio_behavior="error404"), engine(tmp_path) as e:
            with pytest.raises(TtsError) as ei:
                e.synthesize_now("音频下不到")
            assert ei.value.kind == "http"
            assert not e.is_cached("音频下不到"), "没下到就不能留缓存"

    def test_audio_download_empty(self, tmp_path):
        with fake_tts("empty"), engine(tmp_path) as e:
            with pytest.raises(TtsError) as ei:
                e.synthesize_now("空音频")
            assert ei.value.kind == "badjson"

    def test_audio_url_bad_scheme(self, tmp_path):
        """🔴 那个 URL 来自云端响应，不能照单全收。"""
        with fake_tts("badscheme"), engine(tmp_path) as e:
            with pytest.raises(TtsError) as ei:
                e.synthesize_now("file:// 想都别想")
            assert ei.value.kind == "badjson"

    def test_audio_url_host_not_whitelisted(self, tmp_path):
        """主机不在白名单 → 拒绝，避免被畸形响应牵着访问内网地址。"""
        with fake_tts(audio_host="aliyuncs.com"), engine(tmp_path) as e:
            with pytest.raises(TtsError) as ei:
                e.synthesize_now("主机不在白名单")
            assert ei.value.kind == "badjson"
            assert "白名单" in str(ei.value)

    def test_failure_leaves_no_cache_entry(self, tmp_path):
        with fake_tts("error500"), engine(tmp_path) as e:
            with pytest.raises(TtsError):
                e.synthesize_now("失败不留痕")
            assert e.read_cached(e._hash("失败不留痕") + ".mp3") is None


# —— 5. 预算闸 ————————————————————————————————————————————
#
# 🔴 注意：`request()` **成功排队也返回 None**（"还没就绪"）。所以判断
#    "是被拦下还是已排队"不能看返回值，必须看 `budget_drops` 计数器。
#    这个坑踩过一次：写成 `assert request(...) is not None` 结果永远是假。

def _lim(**kw) -> dict:
    """构造预算闸配置。

    🔴 三个上限量的都是**字符数**（键名带 `chars_` 前缀就是为了不搞错）。
       与 R2.2 `narrate.CloudConfig.limits` 同名不同量纲 —— 那边比的是
       calls_*，所以 `per_lap=5` 是"每圈最多 5 次云调用"。
    """
    base = {"chars_per_lap": 10 ** 9, "chars_per_session": 10 ** 9,
            "chars_per_day": 10 ** 9}
    base.update(kw)
    return base


class TestBudget:
    def test_per_lap_gate_blocks(self, tmp_path):
        lim = _lim(chars_per_lap=5)
        with fake_tts(), engine(tmp_path, limits=lim) as e:
            e.note_lap(1)
            e.synthesize_now("三字句")        # 3 字符
            e.synthesize_now("两字")          # +2 → 5，已达上限
            assert e.request("本圈新的一句") is None
            assert e.status()["budget_drops"] == 1

    def test_note_lap_resets_lap_budget(self, tmp_path):
        lim = _lim(chars_per_lap=3)
        with fake_tts(), engine(tmp_path, limits=lim) as e:
            e.note_lap(1)
            e.synthesize_now("三字句")
            e.request("本圈新的")
            assert e.status()["budget_drops"] == 1
            e.note_lap(2)                     # 换圈 → 每圈预算归零
            e.request("换圈之后")
            assert e.status()["budget_drops"] == 1, "换圈后不该再被拦"

    def test_cache_hit_ignores_budget(self, tmp_path):
        """不花钱的事不该被闸住 —— 缓存检查在预算检查之前。"""
        lim = _lim(chars_per_lap=1)
        with fake_tts(), engine(tmp_path, limits=lim) as e:
            text = "已经存过的句子"
            e.synthesize_now(text)
            assert e.status()["chars_lap"] > lim["chars_per_lap"]
            assert e.request(text) is not None, "命中缓存必须照常返回 URL"
            assert e.status()["cache_hits"] >= 1


    def test_default_limits_are_character_scale(self):
        """🔴 回归闸：单位是**字符**，不是次数。

        踩过一次：默认照抄 R2.2 的 `{"per_lap": 6}`（那边的意思是
        "每圈 6 次云调用"），可这边比的是字符数 —— 一句 29 字的播报
        就超预算近 5 倍，同一圈之后所有 B 档句被静默丢弃。真机表现是
        "开是开了，但一整圈只念一句"，而日志里什么都看不出来
        （只在 /tts 的 budget_drops 里）。
        """
        lim = TtsConfig().limits
        assert set(lim) == {"chars_per_lap", "chars_per_session",
                            "chars_per_day"}
        assert lim["chars_per_lap"] >= 3 * 30, "每圈至少装得下 3 句 30 字的播报"
        assert lim["chars_per_lap"] <= lim["chars_per_session"]
        assert lim["chars_per_session"] <= lim["chars_per_day"]
        assert (lim["chars_per_day"] / 1000.0
                * TtsConfig().price_yuan_per_kchar <= 5.0), "日预算不该大到失控"

    def test_real_sentence_fits_in_one_lap(self, tmp_path):
        """一句真实的播报（~29 字）绝不该把一圈的预算吃光。"""
        with fake_tts(force_chars=29), engine(tmp_path) as e:
            e.note_lap(1)
            e.synthesize_now("本圈比参考慢零点四秒，注意补油")
            assert e.request("第二句播报") is None    # 排队也返回 None，正常
            assert e.status()["budget_drops"] == 0, "默认预算不该被一句撑爆"
            assert e.status()["inflight"] + e.status()["queue"] >= 1

    def test_session_limit(self, tmp_path):
        lim = _lim(chars_per_session=4)
        with fake_tts(force_chars=4), engine(tmp_path, limits=lim) as e:
            e.synthesize_now("短句")
            e.request("新的")
            assert e.status()["budget_drops"] == 1

    def test_day_limit_and_rollover(self, tmp_path):
        lim = _lim(chars_per_day=4)
        with fake_tts(force_chars=4), engine(tmp_path, limits=lim) as e:
            e.synthesize_now("短句")
            e.request("新的")
            assert e.status()["budget_drops"] == 1
            e._st["day"] = "2000-01-01"       # 假装跨天了
            e.request("新的一天")
            assert e.status()["budget_drops"] == 1, "跨天应当重置日额度"
            assert e.status()["chars_today"] == 0


# —— 6. 异步 request()：tick 里 O(1)，合成在后台 ——————————————

class TestAsyncRequest:
    def test_request_is_cheap_then_url_appears(self, tmp_path):
        """tick 侧：先返回 None（还没做好），稍后再问就有 URL 了。"""
        from conftest import wait_for
        with fake_tts(), engine(tmp_path) as e:
            text = "本圈节奏不错"
            assert e.request(text) is None, "第一次必然还没就绪"
            assert wait_for(lambda: e.is_cached(text), timeout=3.0,
                            interval=0.02), e.status()
            url = e.request(text)
            assert url is not None and url.startswith(tts_mod.TTS_HANDLE_PATH)
            assert e.status()["calls"] == 1

    def test_inflight_dedup(self, tmp_path):
        """tick 是 10Hz，同一句一秒被问十次 —— 不能排十次队。"""
        with fake_tts(delay=0.4) as srv, engine(tmp_path) as e:
            text = "同一句话"
            for _ in range(10):
                assert e.request(text) is None
            assert e.status()["inflight"] <= 1
            assert srv.calls <= 1

    def test_queue_cap_drops(self, tmp_path):
        """在制品上限：菜单里长期不动时不能无限堆任务。"""
        with fake_tts(delay=0.5), engine(tmp_path, max_queue=1) as e:
            for i in range(6):
                e.request(f"第{i}句不一样的")
            assert e.status()["dropped"] > 0

    def test_max_chars_dropped_in_request(self, tmp_path):
        with fake_tts() as srv, engine(tmp_path, max_chars=10) as e:
            assert e.request("字" * 11) is None
            assert e.status()["dropped"] == 1
            assert srv.calls == 0

    def test_blank_text(self, tmp_path):
        with fake_tts() as srv, engine(tmp_path) as e:
            for bad in ("", "   ", None):
                assert e.request(bad) is None
            assert srv.calls == 0

    def test_worker_survives_failures(self, tmp_path):
        """🔴 worker 线程绝不能因为异常而死 —— 死了会表现成"配了但从不生效"。"""
        from conftest import wait_for
        with fake_tts("error500") as srv, engine(tmp_path) as e:
            e.request("先失败一次")
            assert wait_for(lambda: e.status()["errors"] >= 1, timeout=3.0), \
                e.status()
            srv.behavior = "ok"                  # 云恢复了
            text = "恢复之后这句"
            assert e.request(text) is None
            assert wait_for(lambda: e.is_cached(text), timeout=3.0,
                            interval=0.02), e.status()
            assert e.status()["calls"] == 1

    def test_worker_survives_unexpected_exception(self, tmp_path):
        """连 TtsError 之外的意外异常也不能弄死线程。"""
        from conftest import wait_for
        with fake_tts(), engine(tmp_path) as e:
            real = e._synth_and_store
            calls = {"n": 0}

            def patched(text, **kw):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("意料之外")
                return real(text, **kw)

            e._synth_and_store = patched          # type: ignore[method-assign]
            e.request("先炸一下")
            assert wait_for(lambda: e.status()["errors"] >= 1, timeout=3.0), \
                e.status()
            assert "RuntimeError" in (e.status()["last_error"] or "")
            text = "炸完还得能用"
            assert e.request(text) is None
            assert wait_for(lambda: e.is_cached(text), timeout=3.0,
                            interval=0.02), e.status()

    def test_close_stops_worker(self, tmp_path):
        with fake_tts():
            e = TtsEngine(_cfg(tmp_path))
            assert e._worker is not None and e._worker.is_alive()
            e.close()
            assert e._worker is None
            e.request("关掉之后")                 # 只要求不抛异常

    def test_close_is_idempotent(self, tmp_path):
        with fake_tts():
            e = TtsEngine(_cfg(tmp_path))
            e.close()
            e.close()                            # 重复关不该炸


# —— 7. read_cached 的目录穿越防护 ————————————————————————

class TestReadCached:
    def test_path_traversal_rejected(self, tmp_path):
        with fake_tts(), engine(tmp_path) as e:
            h = e._hash("x")
            for bad in ("../../etc/passwd", "short.mp3",
                        "x" * 16 + ".wav", "zzzzzzzzzzzzzzzz.mp3",
                        f"{h}.mp3/../x", h, f"{h}.",
                        "0123456789abcdef.mp3/../../evil.mp3"):
                assert e.read_cached(bad) is None, bad

    def test_valid_handle_reads(self, tmp_path):
        with fake_tts(), engine(tmp_path) as e:
            e.synthesize_now("能读回来")
            assert e.read_cached(f"{e._hash('能读回来')}.mp3") == AUDIO

    def test_missing_valid_handle(self, tmp_path):
        with engine(tmp_path) as e:
            assert e.read_cached("0123456789abcdef.mp3") is None

    def test_no_cache_dir_means_nothing_readable(self):
        e = TtsEngine(_cfg(cache_dir=None))
        assert e.read_cached("0123456789abcdef.mp3") is None
        assert e.path_for("x") is None
        assert e.is_cached("x") is False
        assert e.url_for("x") is None


# —— 8. status()：可观测，且绝不回显 key ———————————————————————

class TestStatus:
    def test_shape(self, tmp_path):
        with fake_tts(), engine(tmp_path) as e:
            e.synthesize_now("看看状态")
            st = e.status()
            for k in ("enabled", "disabled_reason", "provider", "model", "voice",
                      "format", "sample_rate", "api_key_env", "has_key",
                      "workspace_id", "cache_dir", "cache_files", "calls",
                      "cache_hits", "errors", "budget_drops", "dropped",
                      "chars", "chars_today", "chars_lap", "est_cost_yuan",
                      "price_yuan_per_kchar", "inflight", "queue",
                      "last_latency_s", "last_error", "limits"):
                assert k in st, k
            assert st["enabled"] is True
            assert st["disabled_reason"] is None
            assert st["cache_files"] == 1
            assert st["last_latency_s"] is not None
            assert st["last_error"] is None

    def test_never_leaks_key(self, tmp_path):
        with fake_tts(), engine(tmp_path) as e:
            blob = json.dumps(e.status(), ensure_ascii=False)
            assert "test-tts-key" not in blob
            assert e.status()["api_key_env"] == "GT7_TEST_TTS_KEY"

    def test_disabled_status_reports_reason(self, tmp_path):
        e = TtsEngine(_cfg(tmp_path, enabled=False))
        st = e.status()
        assert st["enabled"] is False
        assert st["disabled_reason"]


# —— 8.5 配置热更新：重建引擎要继承**计费计数** ————————————————————
#
# 🔴 `TtsEngine.cfg` 是 `tts_config()` 的**快照**，改配置只能整体重建
#    （见 `CoachEngine.rebuild_tts`）。若重建顺手把预算清零，用户每改一次设置
#    就白拿一份日/场/圈配额 —— 而 `chars_per_day` 是我们**唯一**的成本硬顶。

class TestCarryOver:
    def test_counters_are_inherited_and_new_cfg_takes_effect(self, tmp_path):
        with fake_tts(), engine(tmp_path) as old:
            old.note_lap(7)
            old.synthesize_now("一句中文")
            before = old.status()
            assert before["chars_today"] > 0

            new = TtsEngine(_cfg(tmp_path, max_chars=20))
            new.carry_over(old)
            after = new.status()
            assert after["chars_today"] == before["chars_today"]
            assert after["chars"] == before["chars"]
            assert after["calls"] == before["calls"]
            assert after["chars_lap"] == before["chars_lap"]
            assert new.cfg.max_chars == 20     # 新配置生效 = 重建的目的
            assert new._st["lap"] == 7         # 圈号也继承，每圈预算不被误重置
            new.close()

    def test_daily_cap_survives_a_rebuild(self, tmp_path):
        """配额不能因改配置而重置：日上限触顶后，重建出来的引擎照样拒绝。

        在设置页滑一下 `tts_max_chars` 就能重置配额的话，成本闸形同虚设。
        """
        lim = {"chars_per_day": 4, "chars_per_session": 100,
               "chars_per_lap": 100}
        with fake_tts(force_chars=4), engine(tmp_path, limits=lim) as old:
            old.synthesize_now("四个字")
            assert old.status()["chars_today"] == 4          # 正好触顶
            new = TtsEngine(_cfg(tmp_path, limits=lim))
            new.carry_over(old)
            # ⚠️ 必须在 with 里调：`enabled` 每次求值都会查 provider 表，
            #    而 `fake_tts` 退出时会注销本地 provider —— 出去再调就变成
            #    "未启用"（在更早的分支返回），根本走不到预算闸。
            assert new.request("再来一句") is None            # 日配额已满
            assert new.status()["budget_drops"] == 1
            new.close()

    def test_carry_over_does_not_share_the_queue(self, tmp_path):
        """只继承计数，**不继承**在制品 —— 重建本就是换一条新链路。"""
        with fake_tts(), engine(tmp_path) as old:
            new = TtsEngine(_cfg(tmp_path))
            new.carry_over(old)
            assert new._inflight == set()
            assert new._q.qsize() == 0
            new.close()


# —— 9. 引擎接入：A 档永不上云 ————————————————————————————————
#
# 🔴 R3 的头号红线。用 spy 包住 `tts.request`，看得见"谁被送去云上"。
#    两条互补的用例：
#      · 桩版 —— 把「一句 A 档 + 一句 B 档」稳定塞进每个 tick，精确命中
#        那条 `priority >= P_NORMAL` 分支，不依赖任何数据文件；
#      · 赛道版 —— 真跑合成赛道，验证真实数据下 A 档的 tts_url 恒为空。
#    只留赛道版是不行的：合成赛道很可能一句 B 档都不产（实测只出 brake_warn），
#    那样"B 档确实上了云"就没人验证了。

class TestEngineWiring:
    A_TEXT = "出界了"
    B_TEXT = "本圈慢了零点四"

    @staticmethod
    def _spy(eng):
        """包住 request，记录每一句被送进云的文本。"""
        asked: list[str] = []
        real = eng.tts.request

        def spy(text):
            asked.append(text)
            return real(text)

        eng.tts.request = spy                     # type: ignore[method-assign]
        return asked

    def _cfg(self, tmp_path):
        from gt7coach.engine import CoachConfig
        return CoachConfig(poll_interval_s=0.0, tts_enabled=True,
                           tts_provider="test", tts_workspace_id="ws-test",
                           tts_api_key_env="GT7_TEST_TTS_KEY",
                           tts_cache_dir=str(tmp_path))

    def test_only_b_tier_requested(self, tmp_path):
        from gt7coach.contract import P_CRITICAL, Utterance
        from gt7coach.engine import CoachEngine
        from gt7coach.source import ReplaySource
        from gt7coach.synth import synth_lap_frames, synth_profile

        frames = synth_lap_frames(laps=1)
        src = ReplaySource(frames, profile=synth_profile(), loop=True)
        with fake_tts():
            eng = CoachEngine(src, self._cfg(tmp_path), clock=src.clock)
            # 桩掉规则与闸门：每 tick 稳定产出一句 A 档 + 一句 B 档
            eng.rules.evaluate = lambda ctx: [            # type: ignore[method-assign]
                Utterance(key="a-tier", text=self.A_TEXT, priority=P_CRITICAL),
                Utterance(key="b-tier", text=self.B_TEXT, priority=P_NORMAL),
            ]
            eng.gate.filter = lambda cands, **kw: list(cands)  # type: ignore[method-assign]
            asked = self._spy(eng)
            said: list = []
            try:
                for _ in range(30):
                    st = eng.tick()
                    said += list(st.say)
            finally:
                eng.tts.close()

        assert said, "桩应当每 tick 都产出句子"
        a_seen = [u for u in said if u.priority < P_NORMAL]
        b_seen = [u for u in said if u.priority >= P_NORMAL]
        assert a_seen and b_seen, "两档都该出现，否则这个测试没意义"
        # 红线：送进云的文本集合里绝不能有 A 档那句
        assert set(asked) == {self.B_TEXT}, \
            f"A 档漏到云上了：{set(asked) - {self.B_TEXT}}"
        # A 档的 tts_url 恒为空
        assert all(u.tts_url is None for u in a_seen)
        # 端到端接线成立：后台合成完成后 B 档拿到 URL
        assert any(u.tts_url for u in b_seen), "B 档应当最终拿到云音频 URL"

    def test_a_tier_tts_url_always_none_on_real_laps(self, tmp_path):
        """真实合成赛道的对照用例：A 档一律不参与云 TTS。"""
        from gt7coach.engine import CoachEngine
        from gt7coach.source import ReplaySource
        from gt7coach.synth import synth_lap_frames, synth_profile

        frames = synth_lap_frames(laps=3)
        src = ReplaySource(frames, profile=synth_profile(), loop=False)
        with fake_tts():
            eng = CoachEngine(src, self._cfg(tmp_path), clock=src.clock)
            self._spy(eng)
            a_tier: list = []
            try:
                for _ in range(len(frames) + 20):
                    st = eng.tick()
                    a_tier += [u for u in st.say if u.priority < P_NORMAL]
                    if not st.connected:
                        break
            finally:
                eng.tts.close()
        assert a_tier, "合成赛道应当产出 A 档句子（刹车点/打滑/出界）"
        assert all(u.tts_url is None for u in a_tier)

    def test_state_exposes_tts_stats(self):
        from gt7coach.engine import CoachConfig, CoachEngine
        from gt7coach.source import ReplaySource
        from gt7coach.synth import synth_lap_frames, synth_profile

        src = ReplaySource(synth_lap_frames(laps=1),
                           profile=synth_profile(), loop=True)
        eng = CoachEngine(src, CoachConfig(poll_interval_s=0.0), clock=src.clock)
        try:
            d = eng.tick().to_dict()
            assert "tts" in d["stats"]
            assert d["stats"]["tts"]["enabled"] is False
            assert d["stats"]["tts"]["disabled_reason"]
        finally:
            eng.tts.close()

    def test_tts_url_in_utterance_dict(self):
        """契约层字段：`say[].tts_url` 必须在，null 不等于出错。"""
        from gt7coach.contract import Utterance
        u = Utterance(key="k", text="t", priority=P_NORMAL, speech="s")
        assert u.tts_url is None
        assert u.to_dict()["tts_url"] is None


# —— 10. HTTP 路由 ————————————————————————————————————————

class TestRoutes:
    @pytest.fixture
    def server(self, tmp_path):
        from gt7coach.engine import CoachConfig, CoachEngine
        from gt7coach.server import CoachService, make_server
        from gt7coach.source import ReplaySource
        from gt7coach.synth import synth_lap_frames, synth_profile

        with fake_tts() as fake:
            frames = synth_lap_frames(laps=2)
            src = ReplaySource(frames, profile=synth_profile(), loop=True)
            eng = CoachEngine(src, CoachConfig(
                poll_interval_s=0.02, sess_poll_boot_s=0.0, sess_poll_idle_s=0.0,
                tts_enabled=True, tts_provider="test",
                tts_workspace_id="ws-test",
                tts_api_key_env="GT7_TEST_TTS_KEY",
                tts_cache_dir=str(tmp_path)), clock=src.clock)
            svc = CoachService(eng, interval_s=0.02)
            srv = make_server(svc, host="127.0.0.1", port=0)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            svc.start()
            time.sleep(0.25)
            try:
                yield f"http://127.0.0.1:{srv.server_address[1]}", fake, eng
            finally:
                svc.stop()
                srv.shutdown()
                srv.server_close()
                eng.tts.close()

    def test_tts_status_route(self, server):
        base, _fake, _eng = server
        code, d, _ = _get(base + "/api/v1/coach/tts")
        assert code == 200 and d["enabled"] is True
        assert d["model"] == "cosyvoice-v3-flash"
        assert d["voice"] == "longanyang"
        assert d["has_key"] is True

    def test_post_synthesize_and_get_audio(self, server):
        base, fake, eng = server
        code, body = _post_bytes(base + "/api/v1/coach/tts",
                                 {"text": "服务端合成一句"})
        assert code == 200 and body == AUDIO
        assert fake.calls >= 1

        handle = f"{eng.tts._hash('服务端合成一句')}.mp3"
        code, data, headers = _get_bytes(base + f"/api/v1/coach/tts/{handle}")
        assert code == 200 and data == AUDIO
        assert headers.get("Content-Type") == "audio/mpeg"

    def test_audio_is_long_cacheable(self, server):
        """内容由 hash 唯一决定，永不变化 → 可以放长缓存。"""
        base, _fake, eng = server
        _post_bytes(base + "/api/v1/coach/tts", {"text": "缓存头"})
        _c, _d, headers = _get_bytes(
            base + f"/api/v1/coach/tts/{eng.tts._hash('缓存头')}.mp3")
        assert "max-age" in (headers.get("Cache-Control") or "")

    def test_missing_audio_404(self, server):
        base, _fake, _eng = server
        with pytest.raises(urllib.error.HTTPError) as ei:
            _get(base + "/api/v1/coach/tts/0123456789abcdef.mp3")
        assert ei.value.code == 404

    def test_post_rejects_empty_text(self, server):
        base, fake, _eng = server
        with pytest.raises(urllib.error.HTTPError) as ei:
            _post(base + "/api/v1/coach/tts", {"text": "  "})
        assert ei.value.code == 400
        assert fake.calls == 0

    def test_post_cloud_down_is_503(self, server):
        """云不可用是**预期内**降级，不是 500 —— 前端据此回落浏览器 TTS。"""
        base, fake, _eng = server
        fake.behavior = "error500"
        with pytest.raises(urllib.error.HTTPError) as ei:
            _post(base + "/api/v1/coach/tts", {"text": "云挂了"})
        assert ei.value.code == 503
        assert json.loads(ei.value.read().decode("utf-8"))["kind"] == "http"

    def test_post_disabled_is_503(self):
        """TTS 没配时 POST 也是 503（不是 400/500）—— 语义是"暂时不可用"。"""
        from gt7coach.engine import CoachConfig, CoachEngine
        from gt7coach.server import CoachService, make_server
        from gt7coach.source import ReplaySource
        from gt7coach.synth import synth_lap_frames, synth_profile

        src = ReplaySource(synth_lap_frames(laps=1),
                           profile=synth_profile(), loop=True)
        eng = CoachEngine(src, CoachConfig(poll_interval_s=0.0), clock=src.clock)
        svc = CoachService(eng, interval_s=0.02)
        srv = make_server(svc, host="127.0.0.1", port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        svc.start()
        time.sleep(0.1)
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            with pytest.raises(urllib.error.HTTPError) as ei:
                _post(base + "/api/v1/coach/tts", {"text": "没配"})
            assert ei.value.code == 503
            _c, d, _ = _get(base + "/api/v1/coach/tts")
            assert d["enabled"] is False and d["disabled_reason"]
        finally:
            svc.stop()
            srv.shutdown()
            srv.server_close()
            eng.tts.close()


# —— 11. CLI 参数解析（守的是"提示说得对不对"）——————————————————
#
# 🔴 实测踩过：只给 `--tts-workspace` 而忘了 `--tts-cache-dir` 时，启动横幅说的是
#    "未开启（cfg.enabled=false）" —— 用户明明想开却被告知没开，真正缺的那一项
#    完全看不见。判据从 `ws and cache` 改成 `ws or cache` 之后，引擎才会报准确原因。

class _NS:
    """最小 argparse.Namespace 替身（不必为几个字段去建真 parser）。"""

    def __init__(self, **kw):
        self.tts = kw.get("tts")
        self.tts_workspace = kw.get("ws")
        self.tts_cache_dir = kw.get("cache")
        self.tts_voice = kw.get("voice")
        self.tts_model = kw.get("model")


def _clean_env(monkeypatch):
    for k in ("GT7_COACH_TTS_WORKSPACE", "GT7_COACH_TTS_CACHE_DIR"):
        monkeypatch.delenv(k, raising=False)


class TestResolveTts:
    def test_nothing_configured_stays_off(self, monkeypatch):
        _clean_env(monkeypatch)
        from gt7coach.cli import _resolve_tts
        r = _resolve_tts(_NS())
        assert r["tts_enabled"] is False
        assert r["tts_workspace_id"] == "" and r["tts_cache_dir"] is None

    def test_workspace_only_still_means_intent(self, monkeypatch):
        """给了一半也算"想开" —— 这样引擎才能报出准确的缺失项。"""
        _clean_env(monkeypatch)
        from gt7coach.cli import _resolve_tts
        r = _resolve_tts(_NS(ws="ws-x"))
        assert r["tts_enabled"] is True
        assert r["tts_cache_dir"] is None

    def test_cache_only_still_means_intent(self, monkeypatch):
        _clean_env(monkeypatch)
        from gt7coach.cli import _resolve_tts
        assert _resolve_tts(_NS(cache="/tmp/x"))["tts_enabled"] is True

    def test_both_configured_auto_on(self, monkeypatch):
        _clean_env(monkeypatch)
        from gt7coach.cli import _resolve_tts
        r = _resolve_tts(_NS(ws="ws-x", cache="/tmp/x"))
        assert r["tts_enabled"] is True and r["tts_workspace_id"] == "ws-x"

    def test_no_tts_wins(self, monkeypatch):
        _clean_env(monkeypatch)
        from gt7coach.cli import _resolve_tts
        assert _resolve_tts(_NS(tts=False, ws="ws-x",
                                cache="/tmp/x"))["tts_enabled"] is False

    def test_explicit_tts_without_anything(self, monkeypatch):
        _clean_env(monkeypatch)
        from gt7coach.cli import _resolve_tts
        assert _resolve_tts(_NS(tts=True))["tts_enabled"] is True

    def test_env_vars_used(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GT7_COACH_TTS_WORKSPACE", "ws-env")
        monkeypatch.setenv("GT7_COACH_TTS_CACHE_DIR", str(tmp_path))
        from gt7coach.cli import _resolve_tts
        r = _resolve_tts(_NS())
        assert r["tts_enabled"] is True
        assert r["tts_workspace_id"] == "ws-env"
        assert r["tts_cache_dir"] == str(tmp_path)

    def test_cli_beats_env(self, monkeypatch):
        monkeypatch.setenv("GT7_COACH_TTS_WORKSPACE", "ws-env")
        from gt7coach.cli import _resolve_tts
        assert _resolve_tts(_NS(ws="ws-cli"))["tts_workspace_id"] == "ws-cli"

    def test_replay_style_auto_false(self, monkeypatch, tmp_path):
        """replay 是离线调参：配齐也不联网，只有显式 --tts 才开。"""
        _clean_env(monkeypatch)
        from gt7coach.cli import _resolve_tts
        assert _resolve_tts(_NS(ws="ws-x", cache=str(tmp_path)),
                            auto=False)["tts_enabled"] is False
        assert _resolve_tts(_NS(tts=True, ws="ws-x", cache=str(tmp_path)),
                            auto=False)["tts_enabled"] is True

    def test_half_config_reports_the_real_missing_piece(self, monkeypatch,
                                                        capsys):
        """只给 workspace：必须说"缺 cache_dir"，不能说"未开启"。"""
        _clean_env(monkeypatch)
        from gt7coach.cli import _print_tts_state, _resolve_tts
        from gt7coach.engine import CoachConfig, CoachEngine
        from gt7coach.source import ReplaySource
        from gt7coach.synth import synth_lap_frames, synth_profile

        src = ReplaySource(synth_lap_frames(laps=1),
                           profile=synth_profile(), loop=True)
        eng = CoachEngine(src, CoachConfig(poll_interval_s=0.0,
                                           **_resolve_tts(_NS(ws="ws-x"))),
                          clock=src.clock)
        try:
            assert eng.tts.enabled is False, "引擎仍然拦住（没缓存=每句都花钱）"
            assert "cache_dir" in eng.tts.disabled_reason
            _print_tts_state(eng)
            out = capsys.readouterr().out
            assert "cache_dir" in out
            assert "未开启" not in out
        finally:
            eng.tts.close()

    def test_fully_unconfigured_is_silent(self, monkeypatch, capsys):
        """从没打算用，就别在启动日志里制造噪音。"""
        _clean_env(monkeypatch)
        monkeypatch.delenv("GT7_COACH_LLM_KEY", raising=False)
        from gt7coach.cli import _print_tts_state, _resolve_tts
        from gt7coach.engine import CoachConfig, CoachEngine
        from gt7coach.source import ReplaySource
        from gt7coach.synth import synth_lap_frames, synth_profile

        src = ReplaySource(synth_lap_frames(laps=1),
                           profile=synth_profile(), loop=True)
        eng = CoachEngine(src, CoachConfig(poll_interval_s=0.0,
                                           **_resolve_tts(_NS())),
                          clock=src.clock)
        try:
            _print_tts_state(eng)
            assert capsys.readouterr().out == ""
        finally:
            eng.tts.close()
