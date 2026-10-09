# -*- coding: utf-8 -*-
"""
R3 云 TTS —— 只给 **B 档**句子的「嗓子」。
==========================================

🔴 四条红线（违反任何一条都要返工）：

 1. **A 档永不上云**。只有 `priority >= P_NORMAL`（P_NORMAL / P_LOW）的句子才进
    这条链路。A 档（出界 / 打滑 / 刹车点 / 换挡）继续用浏览器 `speechSynthesis`，
    延迟 0 —— 云合成一次 0.5~2 s，250 km/h 时 2 s = 139 米，来不及。
 2. **绝不阻塞 tick**。合成放在**后台线程**里，`request()` 只往队列丢一个任务就返回。
    谁要是把 `synthesize_now()` 直接写进 `_tick()`，10Hz 主循环会当场掉到 0.5Hz，
    整个教练连带仪表盘一起废掉。那条路只留给 HTTP 的 `POST`（请求自带线程）。
 3. **失败永远回落浏览器 TTS**。无 key / 无 workspace / HTTP 错 / 超时 / 下载失败 /
    目录写不了 —— 任何一种都表现为"拿不到 `tts_url`"，前端就用 Web Speech API 念。
    玩家不该感知云的存在，更不该因为云挂了而静音。
 4. **音频落盘缓存**。「T1 连续三圈慢 0.4」这类句子高度重复，缓存命中率极高；
    命中就是一次本地读盘（微秒级），零网络、零费用、零延迟。

只用标准库 `urllib` —— 与 `cloud.py` 同一套纪律（显式剥代理、不新增依赖、
容器是 `python:3.12-slim`）。

🔴 与 R2.2 的 `cloud.py` 最大的不同，是**端点不一样**：
   语音合成**不在** `dashscope.aliyuncs.com` 上，而是走**业务空间专属域名**
   `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com`，且**只在华北2（北京）地域可用**。
   照抄 chat 的 base_url 只会拿到 404。而且它是**两步**：先合成拿到一个 24 小时有效的
   音频 URL，再 GET 那个 URL 才拿到字节。
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from urllib.parse import urlparse

# HTTP 路径前缀：state 里给出的 tts_url 指向这里，前端直接 GET 拿 mp3。
# 放在本模块是为了让 engine / server 用同一个常量，不至于两处各写一遍。
TTS_HANDLE_PATH = "/api/v1/coach/tts"


class TtsError(Exception):
    """云 TTS 失败的统一异常。上层据此回落浏览器 TTS，绝不外抛到 tick。"""

    def __init__(self, msg: str, *, kind: str = "error"):
        super().__init__(msg)
        self.kind = kind     # "http" | "timeout" | "net" | "nokey" | "badjson" | "disabled"


# —— 厂商适配 ——————————————————————————————————————————————
#
# 「换厂商只改一个函数」的落点：每家只要给出 4 个纯函数 + 2 个默认值，
# 其余（缓存、队列、预算、重试、统计）全部复用。
#
#   endpoint(cfg) -> 完整 URL
#   build(text, cfg) -> 请求体
#   parse(raw) -> 音频 URL（有的厂商直接回 base64，那就换成 get_bytes 直接要字节）
#   chars(raw) -> 本次计费的字符数（用于预算与费用对账）
#   audio_host -> 允许下载音频的主机后缀（防御：响应里的 URL 也照单全收不安全）


def _bailian_endpoint(cfg: "TtsConfig") -> str:
    """百炼 CosyVoice / Qwen-Audio-TTS 的语音合成端点。

    🔴 必须带 `WorkspaceId`，且只有华北2（北京）有。没配 workspace 直接判失败 ——
       落到 `dashscope.aliyuncs.com` 上是 404，报出来的错会被人当成"key 没权限"。
    """
    if not cfg.workspace_id:
        raise TtsError("workspace_id 未配置（百炼语音合成走业务空间专属域名）",
                       kind="nokey")
    return (f"https://{cfg.workspace_id}.cn-beijing.maas.aliyuncs.com"
            "/api/v1/services/audio/tts/SpeechSynthesizer")


def _bailian_build(text: str, cfg: "TtsConfig") -> dict[str, Any]:
    """🔴 必须用 `resolved_model` / `resolved_voice`，不能用裸 `cfg.model`。

    踩过一次：写成 `cfg.model` 时，配置留空（= 用默认）会发出 `"model": ""`，
    真机上直接 400；而 `status()` 报的却是 `resolved_model`（正确的那一个），
    于是"状态显示正常、一调用就失败"。单测是比对完整请求体才抓出来的。
    """
    return {
        "model": cfg.resolved_model,
        "input": {
            "text": text,
            "voice": cfg.resolved_voice,
            "format": cfg.audio_format,
            "sample_rate": cfg.sample_rate,
        },
    }


def _bailian_parse(raw: dict[str, Any]) -> str:
    try:
        url = ((raw.get("output") or {}).get("audio") or {}).get("url") or ""
    except AttributeError as e:                       # output.audio 不是 dict
        raise TtsError(f"bad response shape: {e}", kind="badjson") from e
    if not url:
        # 非流式**必须**给 url；给了 base64 的是流式（本文档不用流式）。
        raise TtsError("响应里没有音频 URL", kind="badjson")
    return url


def _bailian_chars(raw: dict[str, Any]) -> int:
    try:
        return int((raw.get("usage") or {}).get("characters") or 0)
    except (TypeError, ValueError):
        return 0


PROVIDERS: dict[str, dict[str, Any]] = {
    "bailian": {
        "label": "阿里云百炼 CosyVoice",
        # 🔴 默认模型用 **v3-flash 而不是 v3.5-flash**：
        #    v3.5 系列**不支持系统音色**，只能用声音复刻/设计出来的音色 ID；
        #    而本项目要的是"开箱即用、不需要先录一段音"，所以选 v3-flash。
        #    代价是单价略高（1 元/万字符 vs 0.8），但省掉整个音色定制流程。
        #    若将来要用 v3.5，必须同时把 voice 换成复刻/设计音色，否则报"音色不存在"。
        "model": "cosyvoice-v3-flash",
        "voice": "longanyang",      # 龙安洋：阳光大男孩，语速适中，适合播报
        "endpoint": _bailian_endpoint,
        "build": _bailian_build,
        "parse": _bailian_parse,
        "chars": _bailian_chars,
        "audio_host": "aliyuncs.com",
    },
}

DEFAULT_PROVIDER = "bailian"


def resolve_provider(name: str) -> dict[str, Any]:
    return PROVIDERS.get(name, {})


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    """🔴 显式关掉代理（与 cloud.py 同一条教训：本机代理会把请求拐走而超时）。"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


# —— 配置 ————————————————————————————————————————————————

@dataclass
class TtsConfig:
    """云 TTS 配置。默认**禁用** —— 不配就是纯浏览器 TTS（= R2.2 的行为）。"""

    enabled: bool = False
    provider: str = DEFAULT_PROVIDER
    # 百炼业务空间 ID（`ws-` 开头）。不是密钥，可以进配置文件。
    workspace_id: str = ""
    # 🔴 只存**变量名**，明文 key 绝不进文件。默认与 R2.2 的 LLM 共用一把 key
    #    （百炼的 key 是业务空间级的，语音与语言模型不分开授权）。
    api_key_env: str = "GT7_COACH_LLM_KEY"
    model: str = ""            # 空 → 用 provider 的默认
    voice: str = ""            # 空 → 用 provider 的默认
    audio_format: str = "mp3"
    sample_rate: int = 24000
    timeout_s: float = 8.0     # B 档不抢麦，可以比 chat 的 2s 宽松
    # 单句最长字符数（防呆）。超了直接不合成：一句 200 字的播报既不合理也烧钱。
    max_chars: int = 60
    cache_dir: str | None = None    # 建不了 → 静默禁用（无缓存=每次都花钱）
    max_queue: int = 8              # 在制品上限，防菜单里长期不动时堆任务
    # 计价口径（元 / 千字符）。默认 = 百炼 cosyvoice-v3-flash 官方价
    # 1 元/万字符 = 0.1 元/千字符（2026-10 查证；免费额度 1 万字符/模型）。
    # 换模型必须同步改，否则 /tts 报的费用是错的。
    price_yuan_per_kchar: float = 0.1
    # 预算闸。🔴 **单位是字符数，不是调用次数** —— 这一点与 R2.2 的
    #    `narrate.CloudConfig.limits` 同名不同量纲（那边比的是 calls_*，
    #    所以 per_lap=5 的意思是"每圈最多 5 次云调用"）。这里比的是
    #    chars_lap / chars_session / chars_day，所以键名带 `chars_` 前缀，
    #    让量纲一眼可见。
    #    踩过的坑：照抄 R2.2 写成 `{"per_lap": 6}`，于是一句 29 字的播报
    #    就超预算近 5 倍，同一圈之后所有 B 档句被**静默丢弃**（只在
    #    `/api/v1/coach/tts` 的 budget_drops 里看得见）—— 真机表现是
    #    "开是开了，但一整圈只念一句"，极难往预算闸上想。
    #    取值按真实量级：一句 B 档播报 15~30 字，一圈最多说 ~6 句 → 200 字/圈。
    limits: dict[str, int] = field(default_factory=lambda: {
        "chars_per_lap": 200,          # ≈6 句中长播报
        "chars_per_session": 12000,    # ≈100 圈的长场次也够用
        "chars_per_day": 20000,        # 硬顶 ¥2.00/天（0.1 元/千字符）
    })

    @property
    def resolved_model(self) -> str:
        return self.model or (resolve_provider(self.provider).get("model") or "")

    @property
    def resolved_voice(self) -> str:
        return self.voice or (resolve_provider(self.provider).get("voice") or "")


# —— 引擎 ————————————————————————————————————————————————

class TtsEngine:
    """B 档句子的云合成器：后台线程 + 磁盘缓存 + 预算闸。"""

    def __init__(self, cfg: TtsConfig | None = None, *,
                 opener: urllib.request.OpenerDirector | None = None,
                 start_worker: bool = True):
        self.cfg = cfg or TtsConfig()
        self._opener = opener or _no_proxy_opener()
        self._lock = threading.Lock()
        self._q: queue.Queue[tuple[str, str]] = queue.Queue()
        self._inflight: set[str] = set()
        self._stop = threading.Event()
        self._st = self._fresh_state()
        self._cache_dir: str | None = None
        self._cache_error: str | None = None
        self._prepare_cache_dir()
        self._worker: threading.Thread | None = None
        if self.enabled and start_worker:
            self._worker = threading.Thread(target=self._run, daemon=True,
                                            name="gt7-tts")
            self._worker.start()

    # —— 状态 ——————————————————————————————————————————

    @staticmethod
    def _fresh_state() -> dict[str, Any]:
        return {
            "calls": 0, "errors": 0, "cache_hits": 0, "budget_drops": 0,
            "dropped": 0,                # 队列满/在制品重复而丢弃的
            "chars": 0,
            "last_latency_s": None,
            "last_error": None,
            "day": date.today().isoformat(),
            "chars_day": 0, "chars_session": 0,
            "lap": None, "chars_lap": 0,
        }

    def _prepare_cache_dir(self) -> None:
        d = self.cfg.cache_dir
        if not d:
            self._cache_error = "未配置 cache_dir（无缓存会让每句都真花钱）"
            return
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".probe")
            with open(probe, "wb") as f:
                f.write(b"ok")
            os.remove(probe)
        except OSError as e:
            self._cache_error = f"cache_dir 不可写: {e}"
            return
        self._cache_dir = d

    @property
    def enabled(self) -> bool:
        """真正可用吗 —— 三个条件缺一不可，缺任何一个都静默降级到浏览器 TTS。"""
        return bool(self.cfg.enabled and self.cfg.workspace_id
                    and self._cache_dir and resolve_provider(self.cfg.provider))

    @property
    def disabled_reason(self) -> str | None:
        if not self.cfg.enabled:
            return "未开启（cfg.enabled=false）"
        if not self.cfg.workspace_id:
            return "缺 workspace_id"
        if not resolve_provider(self.cfg.provider):
            return f"未知 provider: {self.cfg.provider}"
        return self._cache_error

    # —— 缓存键与路径 ————————————————————————————————————

    def _hash(self, text: str) -> str:
        """缓存键含 model/voice/format/rate —— 换音色或换采样率必须重新合成，
        否则会出现"换了声音但还是旧音色"这种查半天查不出的怪事。"""
        blob = "|".join([self.cfg.provider, self.cfg.resolved_model,
                         self.cfg.resolved_voice, self.cfg.audio_format,
                         str(self.cfg.sample_rate), text])
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    def path_for(self, text: str) -> str | None:
        if not self._cache_dir:
            return None
        h = self._hash(text)
        # 二级目录散列：同目录几千个文件在 FAT/exFAT 卷上会明显变慢
        return os.path.join(self._cache_dir, h[:2],
                            f"{h}.{self.cfg.audio_format}")

    def is_cached(self, text: str) -> bool:
        p = self.path_for(text)
        return bool(p and os.path.exists(p))

    def url_for(self, text: str) -> str | None:
        """这句的音频在哪。None = 还没有 → 前端应回落浏览器 TTS。"""
        if not self.is_cached(text):
            return None
        return f"{TTS_HANDLE_PATH}/{self._hash(text)}.{self.cfg.audio_format}"

    def read_cached(self, handle: str) -> bytes | None:
        """按 URL 里的 handle（`<hash>.<ext>`）读音频。找不到 → None（路由转 404）。"""
        if not self._cache_dir:
            return None
        h, _, ext = handle.partition(".")
        # 🔴 只允许 16 位十六进制 + 已知后缀：handle 来自 URL 路径，
        #    不校验就能用 `../../etc/passwd` 之类做目录穿越。
        if len(h) != 16 or ext != self.cfg.audio_format:
            return None
        try:
            int(h, 16)
        except ValueError:
            return None
        p = os.path.join(self._cache_dir, h[:2], f"{h}.{ext}")
        try:
            with open(p, "rb") as f:
                return f.read()
        except OSError:
            return None

    # —— 预算闸 ————————————————————————————————————————

    def note_lap(self, lap: int) -> None:
        """圈变化时由引擎调用：重置每圈预算。"""
        with self._lock:
            if self._st["lap"] != lap:
                self._st["lap"] = lap
                self._st["chars_lap"] = 0

    def _roll_day(self) -> None:
        today = date.today().isoformat()
        if self._st["day"] != today:
            self._st["day"] = today
            self._st["chars_day"] = 0

    def _budget_ok(self) -> bool:
        """还允许合成吗。**缓存命中不消耗预算**（不花钱的事不该被闸住）。

        🔴 三个上限量的都是**字符数**（键名带 `chars_` 前缀就是为了不搞错）。
        """
        self._roll_day()
        lim = self.cfg.limits or {}
        big = 10 ** 9
        return (self._st["chars_day"] < lim.get("chars_per_day", big)
                and self._st["chars_session"] < lim.get("chars_per_session", big)
                and self._st["chars_lap"] < lim.get("chars_per_lap", big))

    # —— 对外主入口 ————————————————————————————————————

    def request(self, text: str) -> str | None:
        """tick 里调用：**立刻返回**，绝不阻塞。

        返回：已缓存 → 音频 URL；否则 None（已排队 / 已超预算 / 不可用 / 已在制）。
        前端拿到 None 时不必区分原因 —— 一律先用浏览器 TTS 顶上。
        """
        text = (text or "").strip()
        if not text or not self.enabled:
            return None
        if len(text) > self.cfg.max_chars:
            with self._lock:
                self._st["dropped"] += 1
            return None

        cached = self.url_for(text)
        if cached:
            with self._lock:
                self._st["cache_hits"] += 1
            return cached

        h = self._hash(text)
        with self._lock:
            # 在制品去重：tick 是 10Hz，同一句一秒会被问十次，不能排队十次
            if h in self._inflight:
                return None
            if self._q.qsize() + len(self._inflight) >= self.cfg.max_queue:
                self._st["dropped"] += 1
                return None
            if not self._budget_ok():
                self._st["budget_drops"] += 1
                return None
            self._inflight.add(h)
        self._q.put((text, h))
        return None

    def synthesize_now(self, text: str, *, timeout_s: float | None = None
                       ) -> bytes:
        """**同步**合成并返回音频字节。只给 HTTP 的 `POST /tts` 用。

        🔴 绝不要在 `_tick()` 里调它 —— 一次 0.5~2 s，主循环会当场卡死。
        HTTP handler 跑在独立线程（`ThreadingHTTPServer`），阻塞是安全的。
        """
        text = (text or "").strip()
        if not text:
            raise TtsError("text 为空", kind="badjson")
        if len(text) > self.cfg.max_chars:
            raise TtsError(f"text 超过 {self.cfg.max_chars} 字符", kind="badjson")
        if not self.enabled:
            raise TtsError(self.disabled_reason or "TTS 不可用", kind="disabled")
        p = self.path_for(text)
        if p and os.path.exists(p):
            with self._lock:
                self._st["cache_hits"] += 1
            with open(p, "rb") as f:
                return f.read()
        return self._synth_and_store(text, timeout_s=timeout_s)

    # —— 后台 worker ————————————————————————————————————

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text, h = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if not self.is_cached(text):
                    self._synth_and_store(text)
            except TtsError as e:
                with self._lock:
                    self._st["errors"] += 1
                    self._st["last_error"] = f"{e.kind}: {e}"
            except Exception as e:                      # noqa: BLE001
                # worker 线程绝不能因为任何异常而死 —— 死了之后 TTS 会静默失效，
                # 表现成"云 TTS 配了但从来不生效"，极难排查。
                with self._lock:
                    self._st["errors"] += 1
                    self._st["last_error"] = f"{type(e).__name__}: {e}"
            finally:
                with self._lock:
                    self._inflight.discard(h)

    # —— 真正的合成 ————————————————————————————————————

    def _synth_and_store(self, text: str, *,
                         timeout_s: float | None = None) -> bytes:
        prov = resolve_provider(self.cfg.provider)
        if not prov:
            raise TtsError(f"未知 provider: {self.cfg.provider}", kind="disabled")
        key_val = os.environ.get(self.cfg.api_key_env, "")
        if not key_val:
            with self._lock:
                self._st["errors"] += 1
                self._st["last_error"] = f"nokey: 环境变量 {self.cfg.api_key_env} 未设置"
            raise TtsError(f"环境变量 {self.cfg.api_key_env} 未设置", kind="nokey")

        to = timeout_s if timeout_s is not None else self.cfg.timeout_s
        url = prov["endpoint"](self.cfg)
        body = prov["build"](text, self.cfg)

        t0 = time.monotonic()
        raw = self._post_json(url, key_val, body, to)
        audio_url = prov["parse"](raw)
        chars = prov["chars"](raw)
        data = self._get_bytes(audio_url, to, prov.get("audio_host") or "")
        latency = time.monotonic() - t0
        if not data:
            raise TtsError("音频内容为空", kind="badjson")

        self._store(text, data)
        with self._lock:
            self._roll_day()
            self._st["calls"] += 1
            self._st["chars"] += chars
            self._st["chars_day"] += chars
            self._st["chars_session"] += chars
            self._st["chars_lap"] += chars
            self._st["last_latency_s"] = round(latency, 3)
            self._st["last_error"] = None
        return data

    def _store(self, text: str, data: bytes) -> None:
        p = self.path_for(text)
        if not p:
            return
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            # 先写临时文件再 rename：worker 与 HTTP 读线程并发时，
            # 直接写目标文件会让读到半个 mp3（前端播放直接报解码错）。
            tmp = p + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
        except OSError as e:
            with self._lock:
                self._st["last_error"] = f"落盘失败: {e}"

    def _post_json(self, url: str, key: str, body: dict[str, Any],
                   timeout_s: float) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}",
                     "User-Agent": "gt7-coach/0.1"},
            method="POST")
        try:
            with self._opener.open(req, timeout=timeout_s) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise TtsError(f"HTTP {e.code}", kind="http") from e
        except TimeoutError as e:
            raise TtsError(f"timeout after {timeout_s}s", kind="timeout") from e
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), TimeoutError):
                raise TtsError(f"timeout after {timeout_s}s", kind="timeout") from e
            raise TtsError(f"net: {e.reason}", kind="net") from e
        except json.JSONDecodeError as e:
            raise TtsError(f"bad json: {e}", kind="badjson") from e
        except Exception as e:                          # noqa: BLE001
            raise TtsError(f"{type(e).__name__}: {e}", kind="net") from e

    def _get_bytes(self, url: str, timeout_s: float, host_suffix: str) -> bytes:
        """下载音频。🔴 那个 URL 来自云端响应，不能照单全收 ——
        限定 scheme + 主机后缀，免得被一个畸形响应牵着去访问内网地址。"""
        u = urlparse(url)
        if u.scheme not in ("http", "https"):
            raise TtsError(f"音频 URL scheme 非法: {u.scheme}", kind="badjson")
        if host_suffix and not (u.hostname or "").endswith(host_suffix):
            raise TtsError(f"音频 URL 主机不在白名单: {u.hostname}", kind="badjson")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gt7-coach/0.1"})
            with self._opener.open(req, timeout=timeout_s) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise TtsError(f"下载音频 HTTP {e.code}", kind="http") from e
        except TimeoutError as e:
            raise TtsError(f"下载音频超时 {timeout_s}s", kind="timeout") from e
        except urllib.error.URLError as e:
            raise TtsError(f"下载音频失败: {e.reason}", kind="net") from e

    # —— 可观测 / 生命周期 ————————————————————————————————

    def status(self) -> dict[str, Any]:
        """给 `GET /api/v1/coach/tts` 与 `state.stats["tts"]` 用。"""
        with self._lock:
            st = dict(self._st)
            inflight = len(self._inflight)
        n_files = 0
        if self._cache_dir:
            try:
                for sub in os.scandir(self._cache_dir):
                    if sub.is_dir():
                        n_files += sum(1 for _ in os.scandir(sub.path))
            except OSError:
                pass
        return {
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason if not self.enabled else None,
            "provider": self.cfg.provider,
            "model": self.cfg.resolved_model,
            "voice": self.cfg.resolved_voice,
            "format": self.cfg.audio_format,
            "sample_rate": self.cfg.sample_rate,
            # 🔴 这三个是 `POST /config` 唯一能改的 tts_* 项，必须回显 ——
            #    否则"改了没生效"与"改了生效了"在状态里长得一模一样，
            #    而这正是 rebuild_tts 要解决的失灵（见 engine.rebuild_tts）。
            "max_chars": self.cfg.max_chars,
            "timeout_s": self.cfg.timeout_s,
            # 🔴 只报变量名与 workspace，绝不回显 key 本身
            "api_key_env": self.cfg.api_key_env,
            "has_key": bool(os.environ.get(self.cfg.api_key_env, "")),
            "workspace_id": self.cfg.workspace_id,
            "cache_dir": self._cache_dir,
            "cache_files": n_files,
            "calls": st["calls"],
            "cache_hits": st["cache_hits"],
            "errors": st["errors"],
            "budget_drops": st["budget_drops"],
            "dropped": st["dropped"],
            "chars": st["chars"],
            "chars_today": st["chars_day"],
            "chars_lap": st["chars_lap"],
            "est_cost_yuan": round(
                st["chars"] / 1000.0 * self.cfg.price_yuan_per_kchar, 6),
            "price_yuan_per_kchar": self.cfg.price_yuan_per_kchar,
            "inflight": inflight,
            "queue": self._q.qsize(),
            "last_latency_s": st["last_latency_s"],
            "last_error": st["last_error"],
            "limits": dict(self.cfg.limits or {}),
        }

    def close(self) -> None:
        """停 worker。测试与进程退出时调用。"""
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    # —— 配置热更新 ————————————————————————————————————

    _CARRY = ("calls", "chars", "chars_day", "chars_session", "chars_lap",
              "lap", "day", "errors", "cache_hits", "budget_drops", "dropped")

    def carry_over(self, prev: "TtsEngine") -> None:
        """从**上一台**引擎继承计费/计数状态（`POST /config` 改数值项后重建引擎时用）。

        🔴 为什么必须继承：`self.cfg` 是 `tts_config()` 的快照，改了配置只能整体
           重建。若重建顺手把预算计数清零，用户每改一次设置就白拿一份日/场/圈
           配额 —— 而那是我们**唯一**的成本闸（`chars_per_day` 硬顶 ¥2.00/天）。
           这不是洁癖：面板上滑一下 `tts_max_chars` 就能重置，闸门形同虚设。
        只继承**计费与计数**，不继承队列/在制品 —— 重建本就是换一条新链路。
        """
        with prev._lock:                     # 先快照、再写入，避免同时持两把锁
            snap = {k: prev._st[k] for k in self._CARRY if k in prev._st}
        with self._lock:
            self._st.update(snap)
