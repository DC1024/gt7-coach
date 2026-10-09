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
from pathlib import Path
from typing import Any, Iterable

from .contract import Frame, car_verdict, norm_u16
from .refindex import RefLap

# 无代理 opener：显式空 ProxyHandler，从根上绕开环境变量里的代理
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class SourceError(RuntimeError):
    pass


class HttpSource:
    """拉 GT7 Dash 的 HTTP v1 接口。"""

    def __init__(self, base_url: str, timeout: float = 3.0,
                 slow_timeout: float = 40.0, live_stale_s: float = 20.0):
        self.base = base_url.rstrip("/")
        self.timeout = float(timeout)
        # 🔴 慢接口要单独给宽限：`/api/v1/sessions` 冷态要**流式扫每个场次文件**
        #    才能算出场次列表里的「最快圈」（实测服务器上 5 场里有个 111 MB 的，
        #    冷态 11 秒、容器重启后又是冷态）。它走的都是后台线程，
        #    等 40 秒没有代价；用 3 秒的通用超时反而永远发现不了场次。
        self.slow_timeout = float(slow_timeout)
        # 场次列表里那个 live 标记有多可信：Dash 侧判的是「status.json 近 20s
        # 有写入」。这里再叠一层自己的新鲜度判断，避免对着一个刚断掉的场次
        # 反复取参考圈。
        self.live_stale_s = float(live_stale_s)
        self._last_err: str | None = None

    # —— 底层 ——————————————————————————————————————————

    def _get(self, path: str, params: dict[str, Any] | None = None,
             slow: bool = False) -> Any:
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            "Accept": "application/json", "User-Agent": "gt7-coach/0.1",
        })
        try:
            with _OPENER.open(req, timeout=self.slow_timeout if slow
                              else self.timeout) as r:
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
        # 实时帧必须快：10Hz 的 tick 在等它。超时就当这一帧没来。
        d = self._get("/api/v1/live")
        if not isinstance(d, dict):
            return None
        return Frame.from_v1_live(d, t=time.monotonic())

    # —— 场次 ——————————————————————————————————————————

    def sessions(self) -> list[dict]:
        d = self._get("/api/v1/sessions", slow=True)
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

    def faster_sessions(self, best_s: float, exclude: str = "",
                        car_name: str = "", same_car: bool = True,
                        limit: int = 3, car_code: int = 0) -> list[dict]:
        """比 `best_s` 更快的**其它场次**，按最快圈升序，最多 `limit` 条。

        用于跨场次参考圈（拿你自己的历史最好成绩当标杆，而不是本场最好）。

        🔴 默认只挑**同一辆车**的场次：换了更快的车之后，那个历史成绩根本
           够不着，拿去当参考只会让人一路看着 +8 秒（见 `refindex.RefLap.pace_ok`
           里"几何面 / 速度面"那笔账）。

        🔴 判据**优先用 `car_code`（数字车型码）**，两边都有值就直接比数字；
           拿不到才退回比 `car_name`。原因：车型名要过一道 `cars.csv` 查表，
           表没命中时本场和候选场都拿到空串 → 判不出差别 → 等于静默跨车采用。
           数字相等就是同一辆车，没有这道中间环节。
           （`car_code` 也要等 Dash 的场次列表带上它 —— 见仪表盘 `_first_car_code`。）
        """
        code = norm_u16(car_code)
        out: list[dict] = []
        for s in self.sessions():
            f = s.get("file")
            b = s.get("best_lap_s")
            if not f or f == exclude or not isinstance(b, (int, float)):
                continue
            if b <= 0 or b >= best_s * 0.999:
                continue
            if same_car and car_verdict(code, car_name,
                                        s.get("car_code"),
                                        s.get("car_name")) == "cross_car":
                continue
            out.append(s)
        out.sort(key=lambda x: x["best_lap_s"])
        return out[:max(1, limit)]

    def lap_profile(self, file: str, lap: int | None = None,
                    step_m: float = 5.0) -> dict | None:
        """取参考圈剖面。`lap=None` = 让服务端给最快圈。

        同样走 slow 超时：冷路径要把整场 jsonl 解析成列式存储
        （一场 20 万帧约 2s；服务器上有 111 MB 的场次），
        而它只在后台线程里被调用，等待没有代价。
        """
        params: dict[str, Any] = {"step": step_m}
        if lap:
            params["lap"] = int(lap)
        d = self._get(
            f"/api/v1/sessions/{urllib.parse.quote(file)}/profile", params,
            slow=True)
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
        # 跨场次参考圈的候选（测试用）；真要回放历史时也可以填
        self.history_candidates: list[dict] = []
        self.profiles: dict[str, dict] = {}   # file -> profile（多候选时用）

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

    def faster_sessions(self, best_s: float, exclude: str = "",
                        car_name: str = "", same_car: bool = True,
                        limit: int = 3, car_code: int = 0) -> list[dict]:
        """回放模式下由测试直接指定候选（`history_candidates`）。

        同车过滤与 `HttpSource` 走**同一个** `car_verdict`，这样"测试里能
        被挑中的候选"与"真机上会被挑中的候选"是一回事 —— 否则测试过了、
        真机上照样跨车采用。
        """
        code = norm_u16(car_code)
        out = [c for c in self.history_candidates
               if c.get("file") != exclude
               and isinstance(c.get("best_lap_s"), (int, float))
               and 0 < c["best_lap_s"] < best_s * 0.999
               and (not same_car
                    or car_verdict(code, car_name,
                                   c.get("car_code"),
                                   c.get("car_name")) != "cross_car")]
        out.sort(key=lambda x: x["best_lap_s"])
        return out[:max(1, limit)]

    def lap_profile(self, file: str, lap: int | None = None,
                    step_m: float = 5.0) -> dict | None:
        self.profile_calls += 1
        # 多候选场景（跨场次参考圈）按文件名取，缺省回落单一 profile
        if file in self.profiles:
            return self.profiles[file]
        if self.profiles and self._profile is None:
            return None
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


class FileSource:
    """读 jsonl 场次文件当数据源 —— **完全离线**，连 GT7 Dash 都不需要。

    用途是**调参**：跑一遍历史场次，看教练在什么位置会说什么话。
    阈值（`brake_late_m` / `apex_slow_kph` / 冷却时间…）以前改一次就得
    上方向盘试一次，而"感觉不对"这种反馈既慢又没法对比。有了它，
    改一个数跑一遍就能看到会说的话怎么变 —— 而且是可 diff 的。

    只依赖文件，所以也能当"没有服务端时的回放器"。
    """

    def __init__(self, path: str, *, lap: int | None = None,
                 ref_lap: int | None = None, step_m: float = 5.0,
                 max_frames: int = 0):
        self.path = Path(path)
        self.step_m = step_m
        self.header, self._all = _read_session(self.path)
        # `lap` 只回放这一天圈；`ref_lap` 指定参考圈（缺省 = 本文件最快圈）
        self._frames = ([f for f in self._all if f.lap == lap] if lap
                        else list(self._all))
        if max_frames:
            self._frames = self._frames[:max_frames]
        self._i = 0
        self._ref_lap = ref_lap
        self._ref: RefLap | None = None
        self.profile_calls = 0
        self._session = {
            "file": self.path.name, "live": True,
            "best_lap_s": self.best_lap_s,
            "car_name": str(self.header.get("car") or ""),
            # 离线回放也要带车型码：跨场次参考圈的"同车判据"优先用它。
            # 老场次文件头里没有 → 退回取**实际帧里**出现最多的那个。
            "car_code": _dominant_car_code(self._all),
        }

    # —— 元信息 ————————————————————————————————————————

    @property
    def best_lap_s(self) -> float | None:
        best = None
        for lap, (t0, t1) in _lap_spans(self._all).items():
            dur = t1 - t0
            if dur >= 20.0 and (best is None or dur < best):
                best = dur
        return round(best, 3) if best else None

    @property
    def lap_spans(self) -> dict[int, tuple[float, float]]:
        return _lap_spans(self._all)

    def frames_of_lap(self, lap: int) -> list[Frame]:
        return [f for f in self._all if f.lap == lap]

    # —— Source 协议 ————————————————————————————————————

    def poll(self) -> Frame | None:
        if self._i >= len(self._frames):
            return None
        f = self._frames[self._i]
        self._i += 1
        return f

    def clock(self) -> float:
        """虚拟时钟 = 帧自带的墙钟。用墙上时钟当闸门时钟会让 20s 冷却
        在一次几秒跑完的回放里永远生效，得出"教练一句话都不说"的假结论。"""
        if not self._frames:
            return 0.0
        return self._frames[min(max(self._i - 1, 0), len(self._frames) - 1)].t

    def sessions(self) -> list[dict]:
        return [self._session]

    def live_session(self) -> dict | None:
        return self._session

    def faster_sessions(self, *_a, **_k) -> list[dict]:
        """离线只有一场，没有"跨场次更快"可言。"""
        return []

    def _best_lap_no(self) -> int | None:
        best, best_dur = None, None
        for lap, (t0, t1) in _lap_spans(self._all).items():
            dur = t1 - t0
            if dur >= 20.0 and (best_dur is None or dur < best_dur):
                best, best_dur = lap, dur
        return best

    def lap_profile(self, file: str, lap: int | None = None,
                    step_m: float = 5.0) -> dict | None:
        """本地从帧建参考圈剖面 —— 不连服务端。

        顺带把「拿不到 Dash `/profile` 时自攒参考圈」那条降级路径也跑通了：
        离线回放走的正是同一条代码路径。
        """
        self.profile_calls += 1
        want = lap if lap else (self._ref_lap or self._best_lap_no())
        if not want:
            return None
        fs = self.frames_of_lap(int(want))
        if len(fs) < 30:
            return None
        try:
            self._ref = RefLap.from_frames(fs, lap=int(want), step_m=step_m)
        except ValueError:
            return None
        return self._ref.to_profile()

    @property
    def last_error(self) -> str | None:
        return None


# u16 字段归一统一走契约层的 `norm_u16`（Dash 的 `_U16_FIELDS` 同一口径）：
# 菜单态会把 lap / num_cars / quali_pos 全写成 65535，`int()` 转出来是个
# 完全合法的"第 65535 名"，不归一就会漏进播报。
_u16 = norm_u16


def _lap_spans(frames: list[Frame]) -> dict[int, tuple[float, float]]:
    """每圈的首末帧时刻（**用帧自己的 t**，不是圈内计时）。

    注意与 `lap_time_s` 的区别：那个是圈内相对时间，用来喂规则；
    这里是绝对跨度，用来判"这圈是不是跑满了一整圈"（≥20s，同 Dash 口径）。
    """
    out: dict[int, tuple[float, float]] = {}
    for f in frames:
        if f.lap <= 0:
            continue
        t0, t1 = out.get(f.lap, (f.t, f.t))
        out[f.lap] = (min(t0, f.t), max(t1, f.t))
    return out


def _dominant_car_code(frames: list[Frame]) -> int:
    """整场出现次数最多的非 0 车型码（一场通常同一辆车）。

    离线回放假托场次头里没有 `car_code`，但**每一帧**里都有 —— 所以取众数，
    比"读第一帧"稳：首帧常在菜单态（car_code=0）。
    """
    from collections import Counter
    cnt: Counter[int] = Counter()
    for f in frames:
        if f.car_code > 0:
            cnt[f.car_code] += 1
    return cnt.most_common(1)[0][0] if cnt else 0


def _read_session(path: Path) -> tuple[dict, list[Frame]]:
    """流式读一个场次 jsonl。返回 (header, frames)。

    🔴 两条必须遵守的规矩（都是踩过的）：
      1. **最后一行可能没有换行符** —— 正在录制时那是写了一半的行，
         `json.loads` 会抛异常把整场读成 0 帧（Dash 侧实测 4183 → 0）。
         没换行符就直接丢。
      2. `t` 是**绝对墙钟**，所以圈内用时必须自己用「本圈首帧」做差 ——
         不能拿 `t` 当圈内计时直接喂规则。
    """
    frames: list[Frame] = []
    header: dict = {}
    lap_start: dict[int, float] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.endswith("\n"):
                break                       # 半截行，丢掉
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if i == 0:
                header = d
                continue
            lap = d.get("lap") or 0
            try:
                lap = int(lap)
            except (TypeError, ValueError):
                lap = 0
            if lap < 0 or lap >= 0xFFFF:
                lap = 0
            t = float(d.get("t") or 0.0)
            if lap > 0 and lap not in lap_start:
                lap_start[lap] = t
            g = d.get("g_force") or [0.0, 0.0, 0.0]
            tt = d.get("tyre_temp") or []
            wr = d.get("wheel_rads") or []
            mx = d.get("max_alert_rpm") or 0.0
            frames.append(Frame(
                t=t,
                lap_time_s=max(0.0, t - lap_start.get(lap, t)),
                last_lap_ms=(float(d["last_lap_ms"])
                             if isinstance(d.get("last_lap_ms"), (int, float))
                             else None),
                speed_kph=float(d.get("speed_kph") or 0.0),
                rpm=float(d.get("rpm") or 0.0),
                max_rpm=float(mx) if isinstance(mx, (int, float)) else 0.0,
                gear=int(d.get("gear") or 0),
                throttle=float(d.get("throttle") or 0.0),
                brake=float(d.get("brake") or 0.0),
                lap=lap,
                x=float(d.get("car_x") or 0.0),
                y=float(d.get("car_y") or 0.0),
                z=float(d.get("car_z") or 0.0),
                glon=float(g[0]) if len(g) > 0 else 0.0,
                glat=float(g[1]) if len(g) > 1 else 0.0,
                tyre_temp=tuple(float(v) for v in tt) if isinstance(tt, list) else (),
                wheel_rads=tuple(float(v) for v in wr) if isinstance(wr, list) else (),
                fuel_pct=float(d.get("gas_level") or 0.0),
                fuel_capacity_l=float(d.get("gas_capacity") or 0.0),
                powertrain=str(d.get("powertrain") or ""),
                # 🔴 当前名次取 `quali_pos`（0x84）：比赛进行中它随排名实时变，
                #    与 Dash `/live` 的 `race.grid_position` 同一个源。
                #    `position` 是格式 A 的另一字段，与实时名次不是一回事。
                position=_u16(d.get("quali_pos")),
                num_cars=_u16(d.get("num_cars")),
                laps_in_race=_u16(d.get("laps_in_race")),
                car_code=_u16(d.get("car_code")),
                connected=True,
            ))
    return header, frames
