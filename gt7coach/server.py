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
| GET  | `/api/v1/coach/health`  | 存活与 tick 计数 |
| GET  | `/`                     | 自带的极简验证页（不依赖仪表盘就能试） |

🔴 `/state` 是**电平**（每次都返回当前该说的），`/say` 是**边沿**
   （取走就没了）。两种都要有：
     · 电平适合轮询式前端——它不关心有没有漏，看当前状态就行
     · 边沿适合"播完就得忘掉"的播报场景，不用前端自己去重
   只给一种，就会逼消费方写一个和这里重复的去重逻辑。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .contract import CoachState
from .engine import CoachConfig, CoachEngine

DEFAULT_PORT = 8788  # 紧挨着仪表盘的 8787，别抢


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
        }

    def update_config(self, body: dict[str, Any]) -> dict[str, Any]:
        """改闸门/规则阈值。白名单式——
        只允许改**已经存在**的字段，且类型要对得上，否则整条请求拒绝。
        （防止前端一个笔误就把阈值改成字符串，之后比较运算静默全 False。）
        """
        applied: dict[str, list[str]] = {}
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
                if isinstance(cur, bool) or not isinstance(cur, (int, float)):
                    raise ValueError(f"{section}.{k} 不是数值项，不支持修改")
                setattr(obj, k, type(cur)(v))
                names.append(k)
            if names:
                applied[section] = names
        return {"ok": True, "applied": applied, "config": self.config()}


class _Handler(BaseHTTPRequestHandler):
    server_version = "gt7-coach/0.1"
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
            elif p == "/api/v1/coach/health":
                self._send(self._svc().health())
            else:
                self._send({"error": "not found", "path": p}, 404)
        except Exception as e:              # noqa: BLE001
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:          # noqa: N802
        u = urlparse(self.path)
        p = u.path.rstrip("/")
        if p != "/api/v1/coach/config":
            self._send({"error": "not found", "path": p}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            if not isinstance(body, dict):
                raise ValueError("body 必须是 JSON 对象")
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
</style></head><body>
<h1>GT7 赛道工程师 · 自检页</h1>
<div class="sub">直接问 /api/v1/coach/state，不经过仪表盘。</div>

<div class="row" id="cells"></div>
<div class="card"><div class="sub">这一 tick 要说的话</div><div id="say">—</div></div>
<div class="card">
  <button id="voice">语音播报：关</button>
  <div class="sub" style="margin:10px 0 0">
    用的是浏览器本地 TTS（Web Speech API），零成本、零网络。</div>
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
// 电平接口：每次拿到当前该说的；用 key 去重，避免同一条重复念
function poll(){
  fetch("/api/v1/coach/state").then(function(r){return r.json();})
  .then(function(d){
    var s = d.stats || {};
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
    var txt = (d.say && d.say.length) ? d.say[0].text : "—";
    document.getElementById("say").textContent = txt;
    if (d.say && d.say.length) {
      var k = d.say[0].key + "|" + d.say[0].text + "|" + d.lap;
      if (k !== lastKey) { lastKey = k; say(d.say[0].text, d.say[0].priority); }
    }
    document.getElementById("hist").innerHTML = (d.spoken || []).slice(0,10)
      .map(function(h){
        return '<div><span class="tag">'+(h.key||"")+'</span>'+
               h.text+'</div>';
      }).join("");
  })
  // 🔴 双参数 then：单参数 .then(render).catch(hide) 会把 render 里的异常
  //    当成"取不到数"而整页静默停更，页面正常、内容不更新、console 无线索。
  .then(null, function(e){
    document.getElementById("say").textContent = "取数失败：" + e;
  });
}
poll(); setInterval(poll, 200);
</script></body></html>
"""
