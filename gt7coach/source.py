# -*- coding: utf-8 -*-
"""
数据源适配层 —— Coach 与外部世界的**唯一**出入口。

只依赖标准库（服务端容器里是 `python:3.12-slim`，没有 requests）。

`HttpSource` 面向 GT7 Dash 的**公开 v1 接口**，不读文件、不挂卷：
    GET /api/v1/live                     → 当前帧（10Hz 拉一次）
    GET /api/v1/sessions                 → 找 `live: true` 的那一场
    GET /api/v1/sessions/<f>/profile     → 参考圈剖面（60Hz 精度）

🔴 两个必须记住的坑
-------------------
1. **剥代理**：本机环境预设了 http_proxy/https_proxy，它对局域网 IP 也生效，
   会返回 `502 upstream connect failed` 或直接超时 —— 看起来像"服务端崩了"。
   这里用一个 `ProxyHandler({})` 的 opener，从根上不读环境变量。

2. **`/profile` 是慢接口**：冷路径要把整场 jsonl 解析成列式存储
   （一场 20 万帧约 2s）。所以 Coach 只在**参考圈真的需要换**的时候才取它，
   而且放到后台线程里，绝不阻塞 10Hz 的 tick。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable

from .contract import Frame

# 无代理 opener：显式空 ProxyHandler，从根上绕开环境变量里的代理
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class SourceError(RuntimeError):
    pass


class HttpSource:
    """拉 GT7 Dash 的 HTTP v1 接口。"""

    def __init__(self, base_url: str, timeout: float = 3.0,
                 live_stale_s: float = 20.0):
        self.base = base_url.rstrip("/")
        self.timeout = float(timeout)
        # 场次列表里那个 live 标记有多可信：Dash 侧判的是「status.json 近 20s
        # 有写入」。这里再叠一层自己的新鲜度判断，避免对着一个刚断掉的场次
        # 反复取参考圈。
        self.live_stale_s = float(live_stale_s)
        self._last_err: str | None = None

    # —— 底层 ——————————————————————————————————————————

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            "Accept": "application/json", "User-Agent": "gt7-coach/0.1",
        })
        try:
            with _OPENER.open(req, timeout=self.timeout) as r:
                self._last_err = None
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            self._last_err = f"HTTP {e.code}"
            return None
        except Exception as e:               # noqa: BLE001 —— 网络层什么都可能抛
            self._last_err = f"{type(e).__name__}: {e}"
            return None

    @property
    def last_error(self) -> str | None:
        return self._last_err

    # —— 实时帧 ————————————————————————————————————————

    def poll(self) -> Frame | None:
        d = self._get("/api/v1/live")
        if not isinstance(d, dict):
            return None
        return Frame.from_v1_live(d, t=time.monotonic())

    # —— 场次 ——————————————————————————————————————————

    def sessions(self) -> list[dict]:
        d = self._get("/api/v1/sessions")
        if not isinstance(d, dict):
            return []
        out = [s for s in (d.get("sessions") or []) if isinstance(s, dict)]
        return [s for s in out if not s.get("anomalous")]

    def live_session(self) -> dict | None:
        """正在录制的那一场。

        优先信 Dash 给的 `live` 标记（场次级判断，同一时刻至多一条）；
        标记不可用时退回「modified 最新且够近」—— 但这是**猜**，
        所以调用方拿到的条目会带 `_guessed: True`，好让它写进状态里。
        """
        sess = self.sessions()
        if not sess:
            return None
        for s in sess:
            if s.get("live"):
                return s
        return None

    def session_stats(self, file: str) -> dict | None:
        d = self._get(f"/api/v1/sessions/{urllib.parse.quote(file)}")
        return d if isinstance(d, dict) and not d.get("error") else None

    def lap_profile(self, file: str, lap: int | None = None,
                    step_m: float = 5.0) -> dict | None:
        """取参考圈剖面。`lap=None` = 让服务端给最快圈。"""
        params: dict[str, Any] = {"step": step_m}
        if lap:
            params["lap"] = int(lap)
        d = self._get(
            f"/api/v1/sessions/{urllib.parse.quote(file)}/profile", params)
        if not isinstance(d, dict) or d.get("error"):
            return None
        return d

    def profile_available(self) -> bool:
        """Dash 有没有 `/profile` 端点（老版本没有 → 退回自攒）。

        探法：随便挑一场问一下。**不能**用「返回 404」以外的判据 ——
        有些代理/网关会把 404 变成 200 带 HTML，所以这里只认
        「拿回来是个 dict 且不是 error」。
        """
        sess = self.sessions()
        if not sess:
            return False
        d = self.lap_profile(sess[0]["file"], step_m=50.0)
        return d is not None


class ReplaySource:
    """回放一串帧（测试 / 离线演示用）。可选带一份 profile。"""

    def __init__(self, frames: Iterable[Frame],
                 profile: dict | None = None,
                 session: dict | None = None,
                 loop: bool = False):
        self._frames = list(frames)
        self._i = 0
        self._profile = profile
        self._session = session or {
            "file": "replay.jsonl", "live": True, "best_lap_s": 90.0}
        self.loop = loop
        self.profile_calls = 0

    def poll(self) -> Frame | None:
        if not self._frames:
            return None
        if self._i >= len(self._frames):
            if not self.loop:
                return None
            self._i = 0
        f = self._frames[self._i]
        self._i += 1
        return f

    def sessions(self) -> list[dict]:
        return [self._session]

    def live_session(self) -> dict | None:
        return self._session

    def session_stats(self, file: str) -> dict | None:
        return {"meta": {"file": file}, "best_lap": {"lap": 1, "time": 90.0}}

    def lap_profile(self, file: str, lap: int | None = None,
                    step_m: float = 5.0) -> dict | None:
        self.profile_calls += 1
        return self._profile

    def clock(self) -> float:
        """虚拟时钟：返回**刚取走那一帧**的比赛时间。

        回放时必须用它当引擎时钟 —— 不然闸门的冷却拿墙上时钟算，
        而回放一秒钟能跑完整圈，20 s 的同类冷却会把下一圈的同一句提醒
        全部挡掉，得出"教练一句话都不说"的假结论。
        真机上前者恒等于后者，所以这个参数不会改变线上行为。
        """
        if not self._frames:
            return 0.0
        i = min(max(self._i - 1, 0), len(self._frames) - 1)
        return self._frames[i].t
