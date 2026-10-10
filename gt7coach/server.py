# -*- coding: utf-8 -*-
"""
独立 HTTP 服务 —— 仪表盘靠它跟赛道工程师联通。
================================================

只用标准库 `ThreadingHTTPServer`，和 GT7 Dash 保持一致的做法：
容器里是 `python:3.12-slim`，装不了也**不需要**额外依赖。

接口（全部带 CORS，浏览器可直接调）：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/api/v1/coach/state`   | 完整状态：位置/参考圈/要说的话/最近播报 |
| GET  | `/api/v1/coach/say`     | 只要「还没说过的话」，消费后自动清空（边沿触发用） |
| GET  | `/api/v1/coach/history` | 最近播报历史 |
| GET  | `/api/v1/coach/config`  | 当前配置（闸门/规则阈值） |
| POST | `/api/v1/coach/config`  | 改配置，body 形如 `{"gate": {"max_per_lap": 3}}` |
| GET  | `/api/v1/coach/panel`   | 播报开关面板：各内容分组的开/关状态 + 最近计数 |
| POST | `/api/v1/coach/panel`   | 设置关掉的分组，body 形如 `{"muted": ["tyres","pace"]}` |
| GET  | `/api/v1/coach/health`  | 存活与 tick 计数 |
| GET  | `/api/v1/coach/cloud`   | R2.2 云接入状态：enabled/provider/**当前模型**/是否免费额度内/今日调用/token/估算费用/降级/违规计数 |
| POST | `/api/v1/coach/cloud`   | 写云措辞的**模型名**，body 形如 `{"model": "qwen3.8-flash"}`（写进 cloud.json，立即热加载生效） |
| GET  | `/api/v1/coach/tts`     | R3 云 TTS 状态：enabled/model/voice/缓存命中/队列深度/字符数/估算费用 |
| GET  | `/api/v1/coach/tts/<id>`| 取某句已合成的音频字节（mp3）。`state` 里 `say[].tts_url` 就指向这里 |
| POST | `/api/v1/coach/tts`     | **同步**合成一句并返回音频字节，body `{"text": "..."}`。慢（0.5~2 s），只给调试/显式合成用 |
| GET  | `/`                     | 自带的极简验证页（不依赖仪表盘就能试） |

🔴 `/state` 是**电平**（每次都返回当前该说的），`/say` 是**边沿**
   （取走就没了）。两种都要有：
     · 电平适合轮询式前端——它不关心有没有漏，看当前状态就行
     · 边沿适合"播完就得忘掉"的播报场景，不用前端自己去重
   只给一种，就会逼消费方写一个和这里重复的去重逻辑。

🔴 R3 的 TTS 是**两步**、且 **A 档永远不参与**：
   `say[].tts_url` 在音频**已就绪**时才非空（合成在后台线程做，要 0.5~2 s）。
   拿到 null 不等于出错 —— 消费方要么短暂等待再取一次，要么直接用浏览器
   `speechSynthesis` 顶上。A 档（出界/打滑/刹车点/换挡）永远是 null。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import cloud
from .contract import CoachState
from .engine import CoachConfig, CoachEngine
from .rules import SLIP_PRESETS
from .version import version

DEFAULT_PORT = 8788  # 紧挨着仪表盘的 8787，别抢

# 音频扩展名 → Content-Type。浏览器拿不到正确的 type 会拒绝播放（尤其 Safari）。
_AUDIO_CTYPE = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/ogg",
    "pcm": "audio/L16",
}


class CoachService:
    """把引擎包成「后台线程持续 tick + 对外给最新状态」。"""

    def __init__(self, engine: CoachEngine, interval_s: float | None = None):
        self.engine = engine
        self.interval = interval_s or engine.cfg.poll_interval_s
        self._lock = threading.Lock()
        self._state: CoachState = CoachState()
        self._pending: list[dict] = []      # /say 的边沿队列
        self._ticks = 0
        self._started = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # —— 生命周期 ————————————————————————————————————

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="gt7coach-tick")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                st = self.engine.tick()
            except Exception as e:              # noqa: BLE001
                # tick 里任何异常都不能把服务打死 —— 顶多这一次没数据
                st = CoachState(connected=False,
                                stats={"tick_error": f"{type(e).__name__}: {e}"})
            with self._lock:
                self._state = st
                self._ticks += 1
                for u in st.say:
                    self._pending.append(u.to_dict())
                del self._pending[50:]
            time.sleep(max(0.0, self.interval - (time.monotonic() - t0)))

    # —— 读取 ————————————————————————————————————————

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state.to_dict()

    def take_pending(self) -> list[dict]:
        with self._lock:
            out = list(self._pending)
            self._pending.clear()
            return out

    def health(self) -> dict[str, Any]:
        with self._lock:
            st = self._state
            return {
                "ok": True,
                "ticks": self._ticks,
                "uptime_s": round(time.time() - self._started, 1),
                "connected": st.connected,
                "ref_ready": st.ref_ready,
                "ref_source": st.stats.get("ref_source"),
                "source_error": st.stats.get("source_error"),
                # 场次发现的状态要放进 health：排障时第一个要回答的问题是
                # 「是网络断了、还是仪表盘在忙、还是真的没在录」。
                # 藏在 state 里的话，用 curl 看 health 的人只会看到
                # connected=false 然后开始瞎猜。
                "sess_state": st.stats.get("sess_state"),
                "sess_error": st.stats.get("sess_error"),
                "ref_state": st.stats.get("ref_state"),
                "ref_error": st.stats.get("ref_error"),
            }

    def config(self) -> dict[str, Any]:
        return {
            "coach": asdict(self.engine.cfg),
            "rules": asdict(self.engine.rules.cfg),
            "gate": asdict(self.engine.gate.cfg),
            # #J：打滑三档预设的各档基线，给 UI 渲染下拉 + 滑块初始值
            "slip_presets": dict(SLIP_PRESETS),
            # #H：云措辞服务商预设（label/base_url/model/api_key_env），
            #     给 UI 渲染「一键填入」下拉。与打滑预设同一个套路：
            #     真值永远在服务端，前端只是渲染器。
            "cloud_presets": {k: dict(v)
                              for k, v in cloud.PROVIDER_PRESETS.items()},
        }

    def cloud_status(self) -> dict[str, Any]:
        """R2.2 云接入状态：enabled / provider / 今日调用 / token /
        估算费用 / 降级状态 / 违规计数。**外加当前模型名与是否在免费额度内**。"""
        return self.engine.narrator.status()

    # —— 云措辞的「OpenAI 兼容三框」写入（#H）—————————————————
    #
    # 🔴 红线不变：**明文 api_key 绝不进配置文件**。这里能写的只有
    #    `api_key_env`（key 所在的**环境变量名**）——key 本体永远待在
    #    环境变量里。所以 UI 那个"api_key 框"填的是变量名，不是 key。
    #    base_url / provider 是普通配置值，落盘没问题。
    CLOUD_WRITABLE = ("model", "enabled", "base_url", "api_key_env",
                      "provider")

    # api_key_env 必须长得像环境变量名（字母/下划线开头，后接字母数字下划线）。
    # 用户把 key 明文粘进这个框是最常见的错法 —— 拦下来并说清楚，比静默
    # 存进文件（违反红线）或存进去后运行时查不到变量（以为设好了）都好。
    _ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def set_cloud(self, body: dict[str, Any]) -> dict[str, Any]:
        """把用户填的云配置写进 cloud.json（另一个入口是直接编辑这个文件）。

        为什么写**文件**而不是只存内存：cloud.json 是唯一真值源，Narrator
        靠 mtime 热加载。只改内存的话重启就丢，用户会以为"我明明设过"。

        校验白名单式（与 update_config 同规矩）：不认识的键直接报错，
        不静默忽略 —— 用户把 `model` 拼成 `modal` 却以为设上了，是最坏的结果。
        """
        path = self.engine.narrator.config_path
        if not path:
            raise ValueError(
                "未配置 cloud.json 路径 —— 启动时加 `--cloud <路径>`，"
                "否则云措辞整条链路是禁用态，写了也不会生效")
        bad = [k for k in body if k not in self.CLOUD_WRITABLE]
        if bad:
            raise ValueError(
                f"不支持的字段 {bad}（只允许 {list(self.CLOUD_WRITABLE)}；"
                f"明文 api_key 永远不收 —— key 请放进环境变量，这里只填变量名）")
        # —— 字符串字段校验（#H 三框）——
        for k in ("model", "base_url", "api_key_env", "provider"):
            if k in body and not isinstance(body[k], str):
                raise ValueError(f"{k} 必须是字符串")
        model = str(body.get("model") or "").strip() if "model" in body else None
        if model is not None and len(model) > 128:
            raise ValueError("model 名过长（>128 字符）")
        base_url = str(body.get("base_url") or "").strip()
        if base_url and not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url 必须以 http:// 或 https:// 开头")
        if len(base_url) > 256:
            raise ValueError("base_url 过长（>256 字符）")
        api_key_env = str(body.get("api_key_env") or "").strip()
        if api_key_env and not self._ENV_NAME_RE.match(api_key_env):
            raise ValueError(
                "api_key_env 必须是合法的环境变量名（字母/下划线开头），"
                "不要把 key 本体粘进来 —— 明文 key 绝不落盘")
        provider = str(body.get("provider") or "").strip()
        if len(provider) > 64:
            raise ValueError("provider 名过长（>64 字符）")

        # 读旧文件：不存在 / 坏了都当空对象，**不因此拒绝写入**
        # （用户第一次用就是没有这个文件）。
        try:
            with open(path, "r", encoding="utf-8") as f:
                cur = json.load(f)
            if not isinstance(cur, dict):
                cur = {}
        except (OSError, json.JSONDecodeError):
            cur = {}

        # 🔴 只覆盖**本次请求带来的**字段：只改 api_key_env 不该顺手把
        #    已设的 model 清掉（#H 之前 model 是无条件覆盖的，单框时代
        #    无所谓，三框时代就是 bug）。
        if model is not None:
            cur["model"] = model
        if "base_url" in body:
            cur["base_url"] = base_url
        if "api_key_env" in body:
            cur["api_key_env"] = api_key_env
        if "provider" in body:
            cur["provider"] = provider
        if "enabled" in body:
            cur["enabled"] = bool(body["enabled"])
        elif model or base_url:
            # 填了模型/端点 = 想用云。不顺手打开的话，用户会看到
            # "填了却没反应"，而真值（enabled=false）藏在文件里，界面上看不见。
            cur["enabled"] = True

        # 原子写（tmp + replace）：别让热加载读到半个文件
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

        self.engine.narrator.reload()      # 立刻生效，不等下一次 render
        return {"ok": True, "cloud": self.cloud_status()}


    # —— R3 云 TTS（只给 B 档句子）———————————————————————

    def tts_status(self) -> dict[str, Any]:
        return self.engine.tts.status()

    def tts_synthesize(self, body: dict[str, Any]) -> bytes:
        """**同步**合成一句，返回音频字节。

        ⚠️ 阻塞 0.5~2 s（缓存命中则毫秒级）。跑在 HTTP 请求线程上，
           不影响 10 Hz 的 tick 主循环 —— 但**别**把它接到会被 tick 调用的地方。

        失败抛 TtsError（路由层转 503），不返回半截数据 ——
        前端拿到非 200 就该回落浏览器 TTS。
        """
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("body 需要非空的 text 字段")
        to = body.get("timeout_s")
        kw = {"timeout_s": float(to)} if isinstance(to, (int, float)) else {}
        return self.engine.tts.synthesize_now(text, **kw)

    def tts_audio(self, handle: str) -> bytes | None:
        """按 `state` 里给的 handle（`<hash>.<ext>`）读音频。没有 → None。"""
        return self.engine.tts.read_cached(handle)

    # —— 播报面板（用户自选"什么播报、什么不播报"）———————————————

    def panel(self) -> dict[str, Any]:
        """当前面板快照：每个内容分组的开关状态 + 最近播报计数。
        #G：每组附带 subs（细分开关的当前值，来自 RuleConfig）。"""
        from . import panel as panel_mod
        muted = getattr(self.engine.gate.cfg, "muted", ()) or ()
        spoken = self.state().get("spoken") or []
        recent = [str(h.get("key", "")) for h in spoken]
        return {"api_version": 1,
                **panel_mod.panel_state(muted, recent, self.engine.rules.cfg)}

    def set_panel(self, body: dict[str, Any]) -> dict[str, Any]:
        """设置被关掉的分组。`body = {"muted": ["tyres", "pace"]}`。

        校验失败抛 ValueError（路由层转 400）—— 宁可报错，也不要静默把
        开关设成一个不存在的分组（那样用户会以为关了、其实没关）。
        """
        from . import panel as panel_mod
        muted = panel_mod.normalize_muted(body.get("muted"))
        self.engine.gate.cfg.muted = muted
        return {"ok": True, **self.panel()}

    def update_config(self, body: dict[str, Any]) -> dict[str, Any]:
        """改闸门/规则阈值。白名单式——
        只允许改**已经存在**的字段，且类型要对得上，否则整条请求拒绝。
        （防止前端一个笔误就把阈值改成字符串，之后比较运算静默全 False。）
        """
        applied: dict[str, list[str]] = {}
        # 🔴 #J：打滑预设 `slip_preset` 是字符串字段，不能走下面的数值白名单，
        #    单独摘出来处理——选预设时把 `slip_threshold` 设回该档基线，
        #    除非调用方同一条请求里还显式给了 `slip_threshold`（那是微调，优先）。
        rules_patch = body.get("rules")
        slip_preset = None
        if isinstance(rules_patch, dict) and "slip_preset" in rules_patch:
            slip_preset = rules_patch.pop("slip_preset")
        for section, obj in (("coach", self.engine.cfg),
                             ("rules", self.engine.rules.cfg),
                             ("gate", self.engine.gate.cfg)):
            patch = body.get(section)
            if not isinstance(patch, dict):
                continue
            names = []
            for k, v in patch.items():
                if not hasattr(obj, k):
                    raise ValueError(f"{section}.{k} 不是可配置项")
                cur = getattr(obj, k)
                if isinstance(cur, bool):
                    # #G：布尔开关（*_on / lap_advice）可改，但类型要真对 ——
                    #    v=0/1 这类"顺手用数字"一律拒收，防止前端把开关写成
                    #    字符串 "false"（真值判断恒 True）而看起来像关了。
                    if not isinstance(v, bool):
                        raise ValueError(f"{section}.{k} 需要布尔值 true/false")
                    setattr(obj, k, v)
                elif isinstance(cur, (int, float)):
                    setattr(obj, k, type(cur)(v))
                else:
                    raise ValueError(f"{section}.{k} 不是数值项，不支持修改")
                names.append(k)
            if names:
                applied[section] = names

        # 🔴 #J：选了打滑预设 → 记下标签，并把阈值设回该档基线
        #    （除非本次同时显式微调了阈值，那句微调优先）。
        if slip_preset is not None:
            if slip_preset not in SLIP_PRESETS:
                raise ValueError(
                    f"rules.slip_preset 必须是 {sorted(SLIP_PRESETS)} 之一")
            self.engine.rules.cfg.slip_preset = slip_preset
            if "slip_threshold" not in applied.get("rules", ()):
                self.engine.rules.cfg.slip_threshold = SLIP_PRESETS[slip_preset]
            applied.setdefault("rules", []).append("slip_preset")

        # 🔴 改到 `tts_*` 必须**重建** TtsEngine 才生效 —— 它在构造时把
        #    `tts_config()` 的快照收进 `self.cfg`，之后不回看 CoachConfig。
        #    不重建就是"存进 cfg 但不生效"：读回配置像是改了、行为一点没变。
        #    见 Engine.rebuild_tts 的说明（含缓存与配额如何保住）。
        tts_rebuilt = False
        if any(k.startswith("tts_") for k in applied.get("coach", ())):
            self.engine.rebuild_tts()
            tts_rebuilt = True

        out: dict[str, Any] = {"ok": True, "applied": applied,
                               "config": self.config()}
        if tts_rebuilt:
            out["tts_rebuilt"] = True
            out["tts"] = self.engine.tts.status()
        return out


class _Handler(BaseHTTPRequestHandler):
    # 从版本模块取，避免发行时漏改这一处（曾经硬编码成 0.1 而漂移）。
    # 用 `version` 子模块而不是 `from . import __version__`：后者会在
    # 包 `__init__` 执行到一半时回环导入本模块，属于隐性循环。
    server_version = "gt7-coach/" + version
    protocol_version = "HTTP/1.1"

    # 日志交给调用方（默认太吵，10Hz 轮询会刷屏）
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
        if getattr(self.server, "verbose", False):          # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    # —— 工具 ——————————————————————————————————————————

    def _send(self, obj: Any, code: int = 200,
              ctype: str = "application/json; charset=utf-8") -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _svc(self) -> CoachService:
        return self.server.svc          # type: ignore[attr-defined]

    def _send_bytes(self, data: bytes, ctype: str,
                    code: int = 200, cache: str = "no-store") -> None:
        """发二进制（音频）。JSON 那条路走 `_send`，别混用。"""
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:       # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:           # noqa: N802
        u = urlparse(self.path)
        p = u.path.rstrip("/") or "/"
        try:
            if p == "/":
                body = DEMO_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif p == "/api/v1/coach/state":
                self._send(self._svc().state())
            elif p == "/api/v1/coach/say":
                self._send({"api_version": 1,
                            "say": self._svc().take_pending()})
            elif p == "/api/v1/coach/history":
                q = parse_qs(u.query)
                n = int(q.get("n", ["20"])[0] or 20)
                st = self._svc().state()
                self._send({"api_version": 1, "spoken": st["spoken"][:n]})
            elif p == "/api/v1/coach/config":
                self._send(self._svc().config())
            elif p == "/api/v1/coach/panel":
                self._send(self._svc().panel())
            elif p == "/api/v1/coach/health":
                self._send(self._svc().health())
            elif p == "/api/v1/coach/cloud":
                self._send(self._svc().cloud_status())
            elif p == "/api/v1/coach/tts":
                self._send(self._svc().tts_status())
            elif p.startswith("/api/v1/coach/tts/"):
                handle = p[len("/api/v1/coach/tts/"):]
                data = self._svc().tts_audio(handle)
                if data is None:
                    self._send({"error": "audio not found", "handle": handle}, 404)
                else:
                    ext = handle.rpartition(".")[2].lower()
                    # 内容由 hash 唯一决定，永远不会变 → 可以放长缓存
                    self._send_bytes(data, _AUDIO_CTYPE.get(
                        ext, "application/octet-stream"),
                        cache="public, max-age=86400")
            else:
                self._send({"error": "not found", "path": p}, 404)
        except Exception as e:              # noqa: BLE001
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:          # noqa: N802
        u = urlparse(self.path)
        p = u.path.rstrip("/")
        if p not in ("/api/v1/coach/config", "/api/v1/coach/panel",
                     "/api/v1/coach/tts", "/api/v1/coach/cloud"):
            self._send({"error": "not found", "path": p}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            if not isinstance(body, dict):
                raise ValueError("body 必须是 JSON 对象")
            if p == "/api/v1/coach/panel":
                self._send(self._svc().set_panel(body))
            elif p == "/api/v1/coach/cloud":
                self._send(self._svc().set_cloud(body))
            elif p == "/api/v1/coach/tts":
                from .tts import TtsError
                try:
                    data = self._svc().tts_synthesize(body)
                except TtsError as e:
                    # 云不可用不是 500 —— 这是**预期内**的降级，前端据此回落浏览器 TTS
                    self._send({"error": str(e), "kind": e.kind}, 503)
                    return
                ext = self._svc().engine.cfg.tts_format
                self._send_bytes(data, _AUDIO_CTYPE.get(
                    ext, "application/octet-stream"))
            else:
                self._send(self._svc().update_config(body))
        except Exception as e:              # noqa: BLE001
            self._send({"error": f"{type(e).__name__}: {e}"}, 400)


def make_server(service: CoachService, host: str = "0.0.0.0",
                port: int = DEFAULT_PORT, verbose: bool = False
                ) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), _Handler)
    srv.svc = service                 # type: ignore[attr-defined]
    srv.verbose = verbose             # type: ignore[attr-defined]
    return srv


DEMO_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>GT7 赛道工程师 —— 自检页</title>
<style>
 body{background:#0b0f14;color:#dbe4ee;font:14px/1.6 system-ui,"Segoe UI",
      "Microsoft YaHei",sans-serif;margin:0;padding:24px}
 h1{font-size:16px;margin:0 0 4px;letter-spacing:.5px}
 .sub{color:#7b8798;font-size:12px;margin-bottom:18px}
 .row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:18px}
 .cell{background:#121924;border:1px solid #1e2937;border-radius:10px;
       padding:10px 14px;min-width:110px}
 .cell b{display:block;font-size:20px;font-weight:600;color:#eaf2fb}
 .cell span{font-size:11px;color:#7b8798}
 .card{background:#121924;border:1px solid #1e2937;border-radius:12px;
       padding:14px 16px;margin-bottom:14px}
 #say{font-size:22px;font-weight:600;color:#5ee6a8;min-height:32px}
 #hist div{font-size:13px;color:#9aa7b6;padding:3px 0;
           border-bottom:1px dashed #1e2937}
 .tag{display:inline-block;font-size:10px;padding:1px 6px;border-radius:6px;
      background:#1b2735;color:#8fa3ba;margin-right:8px}
 button{background:#1b2735;color:#cfe0f2;border:1px solid #2b3a4d;
        border-radius:8px;padding:7px 14px;cursor:pointer;font-size:13px}
 button.on{background:#14532d;border-color:#1e7a45;color:#7ff0b0}
 select,input[type=text]{background:#0e1520;color:#dbe4ee;
        border:1px solid #2b3a4d;border-radius:7px;padding:6px 9px;
        font-size:13px;font-family:inherit}
 input[type=range]{width:180px;vertical-align:middle;accent-color:#1e7a45}
 .ctl{display:flex;align-items:center;gap:10px;flex-wrap:wrap;
      margin:6px 0;font-size:13px}
 .ctl label{color:#7b8798;font-size:12px;min-width:88px}
 .prow{display:flex;align-items:flex-start;gap:8px;padding:5px 0;
       font-size:13px;cursor:pointer}
 .psub{display:flex;align-items:center;gap:7px;padding:2px 0 2px 21px;
       font-size:12px;cursor:pointer;color:#9aa7b6}
 .psub.off label{color:#5b6675;text-decoration:line-through;opacity:.7}
 .psub input{margin:0}
 .warn{color:#fbbf24}
 .okmsg{color:#5ee6a8}
 .err{color:#f87171}
</style></head><body>
<h1>GT7 赛道工程师 · 自检页</h1>
<div class="sub">直接问 /api/v1/coach/state，不经过仪表盘。</div>

<div class="row" id="cells"></div>
<div class="card"><div class="sub">这一 tick 要说的话</div><div id="say">—</div></div>
<div class="card">
  <button id="voice">语音播报：关</button>
  <div class="sub" style="margin:10px 0 0">
    B 档句子优先用云 TTS 音频；音频没就绪（或云不可用）时自动回落浏览器
    本地 TTS（Web Speech API）。A 档永远走浏览器，延迟 0。</div>
  <div class="sub" id="ttsmsg" style="margin:6px 0 0">云 TTS：加载中…</div>
</div>
<div class="card">
  <div class="sub" style="margin-bottom:10px">
    播报开关 —— 关掉哪一类，就不再播那一类（勾选 = 播报）；
    每组下面是<strong>逐规则</strong>细分开关（关掉 = 连评估都不跑）</div>
  <div id="panel"></div>
  <div class="sub" id="panelmsg" style="margin-top:10px">加载中…</div>
</div>
<div class="card">
  <div class="sub" style="margin-bottom:8px">打滑灵敏度 —— 三档预设 + 滑块微调</div>
  <div class="ctl">
    <label for="slipPreset">三档预设</label>
    <select id="slipPreset">
      <option value="strict">严格（街道）</option>
      <option value="standard">标准（赛道日）</option>
      <option value="lenient">宽容（漂移/拉力/泥地）</option>
    </select>
  </div>
  <div class="ctl">
    <label for="slipSlider">滑移率阈值</label>
    <input type="range" id="slipSlider" min="0.08" max="0.30" step="0.005">
    <b id="slipVal" style="min-width:38px">—</b>
  </div>
  <div class="sub" id="slipMsg">滑移率超过阈值才报「打滑」。选预设会把阈值设回该档基线，滑块可在档内微调。</div>
</div>
<div class="card">
  <div class="sub" style="margin-bottom:8px">云措辞（OpenAI 兼容）—— 🔴 key 本体只放环境变量，这里填<strong>变量名</strong>，明文 key 绝不落盘</div>
  <div class="ctl">
    <label for="cloudPreset">服务商预设</label>
    <select id="cloudPreset"><option value="">（不切换，仅查看）</option></select>
    <span class="sub">选一个会把下面三框填成该家的默认值，按保存才生效</span>
  </div>
  <div class="ctl"><label for="cloudBaseUrl">base_url</label>
    <input type="text" id="cloudBaseUrl" size="34"
           placeholder="留空 = 用厂商默认端点"></div>
  <div class="ctl"><label for="cloudKeyEnv">key 变量名</label>
    <input type="text" id="cloudKeyEnv" size="34"
           placeholder="如 GT7_COACH_LLM_KEY"></div>
  <div class="ctl"><label for="cloudModel">模型名</label>
    <input type="text" id="cloudModel" size="34"
           placeholder="留空 = 用该家免费默认模型"></div>
  <div class="ctl">
    <button id="cloudSave">保存云措辞</button>
    <button id="cloudToggle">停用</button>
    <span class="sub" id="cloudMsg"></span>
  </div>
</div>
<div class="card"><div class="sub">最近播报</div><div id="hist"></div></div>

<script>
var speakOn = false, lastKey = "";
document.getElementById("voice").onclick = function(){
  speakOn = !speakOn;
  this.textContent = "语音播报：" + (speakOn ? "开" : "关");
  this.className = speakOn ? "on" : "";
  if (speakOn) say("语音已开启");
};
// 🔴 P0（出界/打滑/刹车晚了）要能**打断**正在念的闲话。
//    浏览器 TTS 默认是排队制：一句 delta 会把随后的"出界"堵在它后面，
//    等念完黄花菜都凉了。真赛车无线电是抢麦，不是排队。
function say(t, prio){
  if(!speakOn || !window.speechSynthesis || !t) return;
  if(prio === 0 && speechSynthesis.speaking) speechSynthesis.cancel();
  var u = new SpeechSynthesisUtterance(t);
  u.lang = "zh-CN"; u.rate = 1.15;
  speechSynthesis.speak(u);
}
// —— R3 云 TTS 播放 ——
// 合成在服务端后台线程做，要 0.5~2 s。所以刚出现的那句话 tts_url 往往是
// null —— 那不是错误，是"还没做好"。策略：**等一小会儿**，等到了就播音频，
// 超时就用浏览器 TTS 顶上。绝不允许因为音频没就绪而整句不播。
var TTS_WAIT_MS = 900;
var ttsOn = false, waitKey = "", waitAt = 0, waitU = null;
function browserSay(u){ say(u.speech || u.text, u.priority); }
function playAudio(u){
  try{
    var a = new Audio(u.tts_url);
    var p = a.play();
    // 浏览器自动播放策略可能拒绝（虽有用户点击解锁语音，仍可能被拦）
    if (p && p.then) p.then(null, function(){ browserSay(u); });
    return true;
  }catch(e){ return false; }
}
function utterKey(u, lap){ return u.key + "|" + u.text + "|" + lap; }
var P_NORMAL = 2;   // 与 contract.py 的 P_NORMAL 对齐：priority < 2 即 A 档
function onSay(u, lap){
  if (!speakOn) return;
  // 🔴 A 档（priority < P_NORMAL）**立即念、永不等待** ——「出界了」晚 0.9 s
  //    就完全失去意义了。它们本来也不会进云 TTS，tts_url 恒为空。
  if (u.priority < P_NORMAL || !ttsOn){ browserSay(u); return; }
  if (u.tts_url){ playAudio(u); return; }   // 音频已就绪 → 直接播
  // 云开着、但这句的音频还没合成好 → 等一小会儿，超时回落浏览器 TTS
  waitKey = utterKey(u, lap); waitAt = Date.now(); waitU = u;
}
// 电平接口：每次拿到当前该说的；用 key 去重，避免同一条重复念
function poll(){
  fetch("/api/v1/coach/state").then(function(r){return r.json();})
  .then(function(d){
    var s = d.stats || {};
    ttsOn = !!(s.tts && s.tts.enabled);
    document.getElementById("cells").innerHTML = [
      ["连接", d.connected ? "在线" : "断开"],
      ["参考圈", d.ref_ready ? (s.ref_source === "profile" ? "Dash 60Hz" : "自攒 10Hz") : "未就绪"],
      ["本圈距离", d.s_m == null ? "—" : Math.round(d.s_m) + " m"],
      ["Delta", d.delta_s == null ? "—" : (d.delta_s > 0 ? "+" : "") + d.delta_s.toFixed(2) + " s"],
      ["下一刹车点", d.next_brake_s == null ? "—" :
         Math.round(d.next_brake_m) + " m / " + d.next_brake_s.toFixed(1) + " s"],
      ["圈", d.lap]
    ].map(function(c){
      return '<div class="cell"><b>'+c[1]+'</b><span>'+c[0]+'</span></div>';
    }).join("");
    var u = (d.say && d.say.length) ? d.say[0] : null;
    document.getElementById("say").textContent = u ? u.text : "—";
    if (u){
      var k = utterKey(u, d.lap);
      if (k !== lastKey){ lastKey = k; onSay(u, d.lap); }
      else if (waitKey === k){
        if (u.tts_url){ waitKey = ""; if (speakOn) playAudio(u); }   // 音频到了
        else if (Date.now() - waitAt > TTS_WAIT_MS && waitU){        // 等超了
          waitKey = ""; if (speakOn) browserSay(waitU);
        }
      }
    }
    document.getElementById("hist").innerHTML = (d.spoken || []).slice(0,10)
      .map(function(h){
        return '<div><span class="tag">'+(h.key||"")+'</span>'+
               h.text+'</div>';
      }).join("");
    renderTts(s.tts);
  })
  // 🔴 双参数 then：单参数 .then(render).catch(hide) 会把 render 里的异常
  //    当成"取不到数"而整页静默停更，页面正常、内容不更新、console 无线索。
  .then(null, function(e){
    document.getElementById("say").textContent = "取数失败：" + e;
  });
}
// —— 云 TTS 状态条 ——
function renderTts(t){
  var el = document.getElementById("ttsmsg");
  if (!t){ el.textContent = "云 TTS：未上报"; return; }
  if (!t.enabled){
    el.textContent = "云 TTS：关（" + (t.disabled_reason || "未配置") + "）"
      + " —— 全程浏览器本地 TTS";
    return;
  }
  el.textContent = "云 TTS：" + t.model + " / " + t.voice
    + " · 合成 " + t.calls + " 句 · 缓存命中 " + t.cache_hits
    + " · 队列 " + (t.queue + t.inflight)
    + " · " + t.chars + " 字符 ≈ ¥" + (t.est_cost_yuan || 0).toFixed(4)
    + (t.errors ? " · 错误 " + t.errors : "");
}
// —— 播报开关面板：勾选=播报，取消=静音该分组 ——
// #G：每组下面还有**逐规则细分开关**（subs），管「这条规则算不算」——
//     与分组静音（出口拦截）是两层，走的接口也不同：分组 POST /panel，
//     细分 POST /config 的 rules 节（布尔白名单）。
function renderPanel(groups){
  document.getElementById("panel").innerHTML = groups.map(function(g){
    var tag = g.advice ? ' <span class="tag">'+g.advice+'</span>' : '';
    var subs = (g.subs || []).map(function(s){
      return '<div class="psub'+(s.on?'':' off')+'" data-sub="'+s.id+'">'+
        '<input type="checkbox" '+(s.on?'checked':'')+' id="sub_'+s.id+'">'+
        '<label for="sub_'+s.id+'">'+s.label+'</label></div>';
    }).join("");
    return '<div><label class="prow">'+
      '<input type="checkbox" data-gid="'+g.id+'" '+(g.muted?'':'checked')+
      ' style="margin-top:3px">'+
      '<span><b>'+g.label+'</b>'+tag+
      '<br><span class="sub">'+g.desc+'</span></span></label>'+subs+'</div>';
  }).join("");
  Array.prototype.forEach.call(
    document.querySelectorAll("#panel input[data-gid]"), function(cb){
      cb.onchange = function(){
        var muted = Array.prototype.map.call(
          document.querySelectorAll("#panel input[data-gid]"),
          function(c){ return c.checked ? null : c.getAttribute("data-gid"); }
        ).filter(function(x){ return x; });
        fetch("/api/v1/coach/panel", {method:"POST",
          headers:{"Content-Type":"application/json"},
          body: JSON.stringify({muted: muted})})
        .then(function(r){ return r.json(); })
        .then(function(d){
          if (d.groups){
            renderPanel(d.groups);
            document.getElementById("panelmsg").textContent =
              "已保存：" + (d.muted.length ? ("静音 "+d.muted.join("、"))
                                           : "全部开启");
          } else {
            document.getElementById("panelmsg").textContent =
              "保存失败：" + (d.error || "未知错误");
          }
        })
        .then(null, function(e){
          document.getElementById("panelmsg").textContent = "保存失败："+e;
        });
      };
    });
  // 细分开关：POST /config {rules:{<id>: bool}} —— 服务端布尔白名单校验
  Array.prototype.forEach.call(
    document.querySelectorAll("#panel .psub input"), function(cb){
      cb.onchange = function(){
        var id = cb.parentElement.getAttribute("data-sub"), on = cb.checked;
        fetch("/api/v1/coach/config", {method:"POST",
          headers:{"Content-Type":"application/json"},
          body: JSON.stringify({rules: (function(o){o[id]=on;return o;})({})})})
        .then(function(r){ return r.json(); })
        .then(function(d){
          var msg = document.getElementById("panelmsg");
          if (d.ok){
            document.getElementById("panelmsg").textContent =
              "已保存：「"+id+"」" + (on ? "开" : "关");
            msg.className = "sub okmsg";
          } else {
            document.getElementById("panelmsg").textContent =
              "保存失败：" + (d.error || "未知错误");
            msg.className = "sub err";
            loadPanel();                       // 回读真值，别让 UI 骗人
          }
        })
        .then(null, function(e){
          document.getElementById("panelmsg").textContent = "保存失败："+e;
        });
      };
    });
}
function loadPanel(){
  fetch("/api/v1/coach/panel").then(function(r){return r.json();})
  .then(function(d){
    if (d.groups){ renderPanel(d.groups);
      var msg = document.getElementById("panelmsg");
      msg.className = "sub";
      msg.textContent =
        d.muted.length ? ("当前静音：" + d.muted.join("、")) : "全部开启"; }
  })
  .then(null, function(e){
    document.getElementById("panelmsg").textContent = "加载失败："+e;
  });
}

// —— 打滑三档（#J）：预设只是标签，真值是 slip_threshold ——
var SLIP_LABEL = {strict:"严格（街道）", standard:"标准（赛道日）",
                  lenient:"宽容（漂移/拉力/泥地）"};
function renderSlip(cfg){
  var sel = document.getElementById("slipPreset"),
      sl = document.getElementById("slipSlider"),
      val = document.getElementById("slipVal");
  if (!sel || !sl || !cfg.rules) return;
  var r = cfg.rules;
  // 🔴 不覆盖正在操作的控件（与仪表盘同一个坑：无条件回填会冲掉正拖的滑块）
  if (document.activeElement !== sel) sel.value = r.slip_preset || "standard";
  var thr = (typeof r.slip_threshold === "number") ? r.slip_threshold
                                                   : parseFloat(sl.value);
  if (document.activeElement !== sl){
    sl.value = thr; val.textContent = thr.toFixed(2);
  }
}
function saveRules(patch, okText){
  var msg = document.getElementById("slipMsg");
  return fetch("/api/v1/coach/config", {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify(patch)})
  .then(function(r){ return r.json(); })
  .then(function(d){
    if (d.ok){ renderSlip(d.config);
      msg.textContent = okText; msg.className = "sub okmsg"; }
    else { msg.textContent = "保存失败：" + (d.error || "未知错误");
           msg.className = "sub err"; }
  })
  .then(null, function(e){
    msg.textContent = "保存失败：" + e; msg.className = "sub err";
  });
}
document.getElementById("slipPreset").onchange = function(){
  saveRules({rules:{slip_preset:this.value}},
    "已切换为「" + (SLIP_LABEL[this.value] || this.value) +
    "」档，阈值回到该档基线。");
};
document.getElementById("slipSlider").onchange = function(){
  saveRules({rules:{slip_threshold: parseFloat(this.value)}},
    "阈值已微调为 " + parseFloat(this.value).toFixed(2) + "。");
};

// —— 云措辞三框（#H）：base_url / key 变量名 / 模型名 ——
var cloudPresets = {};
function renderCloudStatus(d){
  var msg = document.getElementById("cloudMsg"),
      btn = document.getElementById("cloudToggle");
  if (!d){ msg.textContent = "云措辞：未上报"; return; }
  btn.textContent = d.enabled ? "停用" : "启用";
  var warn = d.model_warning ? " ⚠️" + d.model_warning : "";
  msg.textContent = (d.enabled ? "已启用" : "已停用")
    + " · " + (d.model || "（默认模型）")
    + (d.base_url ? " · " + d.base_url : "")
    + (d.api_key_env ? " · key=" + d.api_key_env : "") + warn;
  msg.className = "sub" + (d.model_warning ? " warn" : "");
}
function loadCloudStatus(){
  fetch("/api/v1/coach/cloud").then(function(r){return r.json();})
  .then(renderCloudStatus)
  .then(null, function(){});
}
document.getElementById("cloudSave").onclick = function(){
  var msg = document.getElementById("cloudMsg");
  var body = {
    model: document.getElementById("cloudModel").value.trim(),
    base_url: document.getElementById("cloudBaseUrl").value.trim(),
    api_key_env: document.getElementById("cloudKeyEnv").value.trim()
  };
  msg.className = "sub"; msg.textContent = "保存中…";
  fetch("/api/v1/coach/cloud", {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify(body)})
  .then(function(r){ return r.json(); })
  .then(function(d){
    if (d.ok){ msg.className = "sub okmsg";
      msg.textContent = "已保存（填了模型/端点会自动启用）。";
      renderCloudStatus(d.cloud); }
    else { msg.className = "sub err";
      msg.textContent = "保存失败：" + (d.error || "未知错误"); }
  })
  .then(null, function(e){
    msg.className = "sub err"; msg.textContent = "保存失败：" + e;
  });
};
document.getElementById("cloudToggle").onclick = function(){
  var msg = document.getElementById("cloudMsg");
  var to = this.textContent === "停用" ? false : true;
  msg.className = "sub"; msg.textContent = "保存中…";
  fetch("/api/v1/coach/cloud", {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({enabled: to})})
  .then(function(r){ return r.json(); })
  .then(function(d){
    if (d.ok){ renderCloudStatus(d.cloud);
      msg.className = "sub okmsg";
      msg.textContent = to ? "已启用。" : "已停用（回到纯本地模板）。"; }
    else { msg.className = "sub err";
      msg.textContent = "保存失败：" + (d.error || "未知错误"); }
  })
  .then(null, function(e){
    msg.className = "sub err"; msg.textContent = "保存失败：" + e;
  });
};
document.getElementById("cloudPreset").onchange = function(){
  var k = this.value, p = cloudPresets[k];
  if (!p) return;                       // 「不切换」选项
  // 只填框不保存 —— 用户看完三框再决定（与仪表盘同一交互）
  document.getElementById("cloudBaseUrl").value = p.base_url || "";
  document.getElementById("cloudKeyEnv").value = p.api_key_env || "";
  document.getElementById("cloudModel").value = p.model || "";
};
function loadConfig(){
  fetch("/api/v1/coach/config").then(function(r){return r.json();})
  .then(function(cfg){
    renderSlip(cfg);
    // 服务商预设下拉只填一次（之后不重写，避免冲掉正展开的选项）
    if (cfg.cloud_presets){
      cloudPresets = cfg.cloud_presets;
      var sel = document.getElementById("cloudPreset");
      if (sel.options.length <= 1){
        Object.keys(cloudPresets).forEach(function(k){
          var o = document.createElement("option");
          o.value = k;
          o.textContent = cloudPresets[k].label || k;
          sel.appendChild(o);
        });
      }
    }
  })
  .then(null, function(){});
}
loadPanel();
loadConfig();
loadCloudStatus();
poll(); setInterval(poll, 200);
setInterval(loadConfig, 3000);       // 回填打滑阈值（不碰正在操作的控件）
setInterval(loadCloudStatus, 3000);  // 云状态（调用数/费用/警告）
</script></body></html>
"""
