# -*- coding: utf-8 -*-
"""
参考圈索引 —— 把「一圈」变成能按**赛道位置**查询的表。
========================================================

两个来源，同一个结构：

`RefLap.from_profile(d)`   ← GT7 Dash 的 `GET /api/v1/sessions/<f>/profile`
                              60Hz 全量数据算出来的（首选）
`RefLap.from_frames(frames)` ← 自己从实时帧攒（降级路径，见下）

🔴 距离轴是**几何弧长**（相邻点弦长累积），不是速度积分。
   速度积分一圈漂移 60~300 m，拿它做「本圈 1200 m vs 参考圈 1200 m」对齐，
   实际赛道上能差十几米 —— 实时刹车点预告会直接指错位置。

为什么保留自己攒图的降级路径
---------------------------
Dash 那边 `/profile` 是后加的。Coach 要能对着**任何** GT7 Dash 跑起来，
所以拿不到 profile 时退回「自己拿实时帧攒」—— 代价是：
  · 只能从第 2 圈开始有参考（第 1 圈纯记录）
  · 实时侧只有 10Hz 轮询，点数比 60Hz 的 profile 少
  · 关键点只有刹车区和对少数几个弯心，没有完整峰谷分析
`RefLap.source` 会如实写 `"profile"` 还是 `"self"`，`warnings` 里也会说，
**不允许静默降级**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

G = 9.80665

# 单帧位移超过这个数（>3600 km/h）判为坐标跳变（换圈/重生），不计入弧长
_ARC_GAP_M = 100.0


def arc_lengths(xs: list[float], zs: list[float]) -> tuple[list[float], bool]:
    """相邻点弦长累积的几何弧长（米）。返回 (弧长, 是否真的用了几何)。

    🔴 只依赖坐标，与速度无关 —— 这是它相对「速度积分」的唯一也是全部优势：
       速度积分一圈漂移 60~300 m，拿它做跨圈/跨场对齐会错十几米。
       坐标缺失时返回 ([0.0]*n, False)，调用方**必须**把降级暴露出去。

    提成模块级函数是因为 lapstats（每圈分段用时）也要用同一份实现 ——
    两处各写一遍，口径分叉了也不会有人发现。
    """
    n = len(xs)
    if n == 0 or len(zs) != n:
        return [], False
    if not all(isinstance(v, (int, float)) for v in xs + zs):
        return [0.0] * n, False
    if (max(abs(v) for v in xs) + max(abs(v) for v in zs)) <= 1e-6:
        return [0.0] * n, False
    out = [0.0]
    acc = 0.0
    px, pz = xs[0], zs[0]
    for i in range(1, n):
        cx, cz = xs[i], zs[i]
        d = math.hypot(cx - px, cz - pz)
        if d < _ARC_GAP_M:
            acc += d
        px, pz = cx, cz
        out.append(acc)
    return out, True


def interp_at(dists: list[float], values: list[float], d: float) -> float:
    """在**单调不减**的 dists 上按 d 线性插值取 values。两端夹住不外推。"""
    if not dists:
        return 0.0
    if d <= dists[0]:
        return values[0]
    if d >= dists[-1]:
        return values[-1]
    lo, hi = 0, len(dists) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if dists[mid] <= d:
            lo = mid
        else:
            hi = mid
    d0, d1 = dists[lo], dists[hi]
    if d1 <= d0:
        return values[hi]
    k = (d - d0) / (d1 - d0)
    return values[lo] + (values[hi] - values[lo]) * k


# 弯心半径：κ = G_lat·g / v²，|κ| 小于这个值就当直线（不给半径）
_KAPPA_EPS = 5e-4


@dataclass
class RefLap:
    """一圈的按位置索引剖面。所有数组 `grid_m` 一一对齐。"""

    lap: int = 0
    # "profile" 本场最快圈（Dash 60Hz）| "self" 自攒 | "history" 跨场次历史最快
    source: str = "profile"
    grid_m: list[float] = field(default_factory=list)
    speed_kph: list[float] = field(default_factory=list)
    throttle: list[float] = field(default_factory=list)
    brake: list[float] = field(default_factory=list)
    t_rel_s: list[float] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    zs: list[float] = field(default_factory=list)
    length_m: float = 0.0
    lap_time_s: float = 0.0
    brake_in: list[dict] = field(default_factory=list)
    apex: list[dict] = field(default_factory=list)
    throttle_on: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # —— 构造 ——————————————————————————————————————————

    @staticmethod
    def from_profile(d: dict[str, Any], require_geometry: bool = True,
                     source: str = "profile") -> "RefLap":
        """解析 GT7 Dash `/profile` 的响应。

        `require_geometry=True`（缺省）时，**没有坐标的剖面直接拒绝** ——
        没有折线就没法做实时最近点定位，硬用积分距离定位会错十几米。
        宁可退回自己攒图（那个至少有本轮实测坐标）。
        """
        if not d or d.get("error"):
            raise ValueError(f"profile 不可用: {(d or {}).get('error', '空响应')}")
        xs = list((d.get("pt") or {}).get("x") or [])
        zs = list((d.get("pt") or {}).get("z") or [])
        grid = list(d.get("grid_m") or [])
        if not grid:
            raise ValueError("profile 缺 grid_m")
        if require_geometry and (not xs or not zs or len(xs) != len(grid)):
            raise ValueError("profile 没有可用的坐标折线（无法实时定位）")

        mk = d.get("markers") or {}
        ref = RefLap(
            lap=int(d.get("lap") or 0),
            source=source,
            grid_m=[float(v) for v in grid],
            speed_kph=[float(v) for v in (d.get("speed_kph") or [])],
            throttle=[float(v) for v in (d.get("throttle") or [])],
            brake=[float(v) for v in (d.get("brake") or [])],
            t_rel_s=[float(v) for v in (d.get("t_rel_s") or [])],
            xs=[float(v) for v in xs],
            zs=[float(v) for v in zs],
            length_m=float(d.get("length_m") or 0.0),
            lap_time_s=float(d.get("lap_time_s") or 0.0),
            brake_in=list(mk.get("brake_in") or []),
            apex=list(mk.get("apex") or []),
            throttle_on=list(mk.get("throttle_on") or []),
            warnings=[str(w) for w in (d.get("warnings") or [])],
        )
        ref._sanity()
        return ref

    @staticmethod
    def from_frames(frames: Iterable[Any], lap: int = 0,
                    step_m: float = 5.0) -> "RefLap":
        """降级路径：自己拿实时帧攒一圈。

        `frames` 是同一圈的 `Frame` 序列（要按时间升序）。
        坐标缺失时直接抛 —— 没有坐标就做不了几何定位，这个降级路径
        **本身**也依赖坐标，不存在"再降一级用积分"的说法（那会给出错的
        刹车点预告，比不给更糟）。
        """
        fs = [f for f in frames if f.coords_ok]
        if len(fs) < 30:
            raise ValueError(f"可用帧太少（{len(fs)}），无法自攒参考圈")

        s, _geo_ok = arc_lengths([f.x for f in fs], [f.z for f in fs])
        total = s[-1] if s else 0.0
        if total < 200.0:
            raise ValueError(f"圈长不足 200 m（{total:.1f}）")

        step = max(1.0, float(step_m))
        if total / step > 3000:
            step = total / 3000
        grid: list[float] = []
        d = 0.0
        while d < total:
            grid.append(round(d, 1))
            d += step

        def interp(vals: list[float], q: float) -> float:
            return interp_at(s, vals, q)

        def col(attr: str) -> list[float]:
            raw = [float(getattr(f, attr) or 0.0) for f in fs]
            return [round(interp(raw, q), 3) for q in grid]

        ref = RefLap(
            lap=lap, source="self",
            grid_m=grid,
            speed_kph=col("speed_kph"),
            throttle=col("throttle"),
            brake=col("brake"),
            t_rel_s=[round(interp(
                [f.lap_time_s - fs[0].lap_time_s for f in fs], q), 3)
                for q in grid],
            xs=[round(interp([f.x for f in fs], q), 2) for q in grid],
            zs=[round(interp([f.z for f in fs], q), 2) for q in grid],
            length_m=round(total, 1),
            lap_time_s=round(fs[-1].lap_time_s - fs[0].lap_time_s, 3),
            warnings=["参考圈是 Coach 自己用实时帧（10Hz）攒的，"
                      "精度低于 GT7 Dash /profile 的 60Hz 口径"],
        )
        ref._sanity()
        ref._derive_markers_simple()
        return ref

    def _sanity(self) -> None:
        n = len(self.grid_m)
        for name in ("speed_kph", "throttle", "brake", "t_rel_s"):
            col = getattr(self, name)
            if len(col) != n:
                raise ValueError(f"profile 通道 {name} 长度 {len(col)} != 网格 {n}")
        if not self.xs or len(self.xs) != n:
            raise ValueError("profile 坐标折线与网格不对齐")
        if self.grid_m != sorted(self.grid_m):
            raise ValueError("profile 网格不单调")

    # —— 查询 ——————————————————————————————————————————

    def _at(self, col: list[float], s: float) -> float | None:
        """在距离 s 处线性插值取列值。"""
        if not col or not self.grid_m:
            return None
        return interp_at(self.grid_m, col, s)

    def t_at_s(self, s: float) -> float | None:
        """参考圈跑到距离 s 用了多少秒（圈首为 0）。"""
        return self._at(self.t_rel_s, s)

    def v_at_s(self, s: float) -> float | None:
        return self._at(self.speed_kph, s)

    def next_brake(self, s: float) -> dict | None:
        """在 s 之后（含回绕）最近的一个刹车入点。

        回绕是必须的：跑到圈末时「下一个刹车点」就是起点前面那个，
        否则最后一整段路都没提示。
        """
        if not self.brake_in:
            return None
        for z in sorted(self.brake_in, key=lambda x: x.get("s_in_m", 0.0)):
            if z.get("s_in_m", 0.0) > s:
                return z
        return sorted(self.brake_in, key=lambda x: x.get("s_in_m", 0.0))[0]

    def next_apex(self, s: float) -> dict | None:
        if not self.apex:
            return None
        for a in sorted(self.apex, key=lambda x: x.get("s_m", 0.0)):
            if a.get("s_m", 0.0) > s:
                return a
        return sorted(self.apex, key=lambda x: x.get("s_m", 0.0))[0]

    def nearest(self, x: float, z: float) -> tuple[int, float, float]:
        """车辆坐标 → 参考圈上的下标 / 距离(s) / 到参考线的横向距离(米)。

        🔴 用**暴力全扫**而不是「上一帧索引 ±N 的单调窗口」：
           本方法的网格最多 3000 点，10Hz 下全扫 ~0.2ms，完全不是瓶颈；
           而单调窗口要多维护一个 hint，还必须在起跑线重置，
           一旦漏了就整圈定位错误 —— 拿可靠性换不需要的省时，不值。
        """
        best_i, best_d2 = 0, float("inf")
        for i in range(len(self.grid_m)):
            dx = x - self.xs[i]
            dz = z - self.zs[i]
            d2 = dx * dx + dz * dz
            if d2 < best_d2:
                best_d2, best_i = d2, i
        return best_i, self.grid_m[best_i], math.sqrt(best_d2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lap": self.lap, "source": self.source,
            "length_m": round(self.length_m, 1),
            "lap_time_s": round(self.lap_time_s, 3),
            "points": len(self.grid_m),
            "step_m": round(self.grid_m[1] - self.grid_m[0], 2)
            if len(self.grid_m) > 1 else None,
            "brake_in": len(self.brake_in), "apex": len(self.apex),
            "warnings": self.warnings,
        }

    def idx_at(self, s: float) -> int:
        """距离 s 对应网格下标（夹在范围内）。"""
        g = self.grid_m
        if not g:
            return 0
        import bisect
        i = bisect.bisect_left(g, s)
        return min(max(i, 0), len(g) - 1)

    # —— 圈界消歧 ——————————————————————————————————————

    def resolve_wrap(self, s: float, lap_time_s: float) -> float:
        """消掉「起点 ≡ 终点」这个闭环歧义。

        🔴 折线是闭环，起跑线那个点上 `s=0` 与 `s=L` 是同一个物理位置，
           最近点搜索返回哪个只取决于几厘米的坐标差。而游戏的圈计时器与
           位置**不同步**：过线那一帧位置已经进新圈（s≈0），圈号/圈计时
           却还停在上圈的 69.70 s。两边一相减就是 `69.70 - 0 = +69.70 s`，
           一个看起来像"丢了一整圈"的假 delta。

        用圈计时器来消歧：小 s 配大圈时 = 上一圈冲线那一帧，算作 `L`；
        大 s 配小圈时 = 刚过线，压到 0。
        """
        if self.length_m <= 0:
            return s
        wrap_m = min(200.0, self.length_m * 0.05)
        # 阈值取圈速的 15%（且不小于 5 s）：70 s 的圈 → 10.5 s。
        # 用比例而不是固定值，短赛道（20 s 一圈）也不会误判。
        thr = max(5.0, self.lap_time_s * 0.15) if self.lap_time_s else 10.0
        if s < wrap_m and lap_time_s > thr:
            return self.length_m          # 上一圈的最后一帧，落在终点线上
        if s > self.length_m - wrap_m and lap_time_s < thr:
            return 0.0                    # 刚过线，算新圈起点
        return s

    def apex_of_zone(self, zone: dict) -> dict | None:
        """某个刹车区之后的第一个弯心。用于把「刹车晚了」限制在入弯阶段。"""
        s_in = zone.get("s_in_m")
        if s_in is None or not self.apex:
            return None
        cands = [a for a in sorted(self.apex, key=lambda x: x.get("s_m", 0.0))
                 if a.get("s_m", 0.0) >= s_in]
        return cands[0] if cands else None

    # —— 降级路径的关键点（简单版）———————————————————————

    def _derive_markers_simple(self) -> None:
        """只在自攒路径用：阈值刹车区 + 局部最低速当弯心。

        不做 Dash 那套 prominence 峰谷分析 —— 那是为了给「直线段/减速段」
        分色用的；规则只关心「哪里开始刹」「哪里最慢」，简单版够用。
        """
        g, brk, spd, thr = self.grid_m, self.brake, self.speed_kph, self.throttle
        if len(g) < 3:
            return
        pitch = max(g[1] - g[0], 1e-6)
        win_apex = max(2, int(200.0 / pitch))     # 弯心搜索窗：刹车区后 200 m
        win_thr = max(2, int(250.0 / pitch))      # 给油搜索窗：弯心后 250 m
        # 刹车区
        i = 0
        while i < len(g):
            if brk[i] < 0.2:
                i += 1
                continue
            j = i
            while j + 1 < len(g) and brk[j + 1] >= 0.2:
                j += 1
            self.brake_in.append({
                "s_in_m": round(g[i], 1), "s_out_m": round(g[j], 1),
                "speed_in_kph": round(spd[i], 1),
                "peak_brake": round(max(brk[i:j + 1]), 3),
                "v_min_kph": round(min(spd[i:j + 1]), 1),
            })
            i = j + 1
        # 弯心：刹车区之后的局部最低速（阈值 3 km/h 防抖）
        for z in self.brake_in:
            lo = self.idx_at(z["s_out_m"])
            hi = min(len(g) - 1, lo + win_apex)
            if hi <= lo:
                continue
            seg = spd[lo:hi + 1]
            vmin = min(seg)
            k = lo + seg.index(vmin)
            if vmin >= z["speed_in_kph"] - 3.0:
                continue          # 减速幅度太小，不算弯
            self.apex.append({
                "s_m": round(g[k], 1), "speed_kph": round(vmin, 1),
                "turn": None, "radius_m": None,
            })
            for m in range(k, min(len(g), k + win_thr)):
                if thr[m] >= 0.5:
                    self.throttle_on.append({
                        "s_m": round(g[m], 1), "speed_kph": round(spd[m], 1),
                        "after_apex_m": round(g[m] - g[k], 1),
                    })
                    break

    def shape_distance(self, other: "RefLap") -> float | None:
        """两条折线的**形状差异**（米）—— 中位最近点距离。

        用来判断"这条参考圈是不是同一条赛道的"：同一赛道的两条走线
        （哪怕快慢差几秒）中位距离只有几米；换一条赛道就是几十上百米。
        反例（反向布局、不同线路变体）长度可能一样，但形状差得远 —— 所以
        必须比**形状**而不是比圈长。

        取中位数而不是均值/最大值：走线在个别弯里差十几米很正常（那是水平差异，
        不是赛道不同），而均值会被这些点拉高，最大值更是完全没有判别力。

        返回 None = 两边有一方没有折线（无法判断，调用方应当保守处理）。
        """
        if not self.xs or not other.xs:
            return None
        ds: list[float] = []
        for x, z in zip(self.xs, self.zs):
            best = float("inf")
            for ox, oz in zip(other.xs, other.zs):
                d2 = (x - ox) ** 2 + (z - oz) ** 2
                if d2 < best:
                    best = d2
            ds.append(math.sqrt(best))
        if not ds:
            return None
        ds.sort()
        n = len(ds)
        return ds[n // 2] if n % 2 else (ds[n // 2 - 1] + ds[n // 2]) / 2.0

    def to_profile(self) -> dict[str, Any]:
        """导出成 GT7 Dash `/profile` 的形状。

        两个用途：
          · 离线回放（`FileSource`）—— 不连服务端也能有参考圈
          · 让 `from_profile` / `to_profile` 形成**可测的往返**：
            序列化出来的东西能不能被自己解析回去，是能写断言的
        """
        n = len(self.grid_m)
        step = round(self.grid_m[1] - self.grid_m[0], 2) if n > 1 else 0.0
        return {
            "lap": self.lap,
            "frames": 0,
            "length_m": self.length_m,
            "length_by_speed_m": self.length_m,
            "length_drift_pct": None,
            "geometry_used": True,
            "lap_time_s": self.lap_time_s,
            "step_m": step,
            "grid_m": list(self.grid_m),
            "speed_kph": list(self.speed_kph),
            "throttle": list(self.throttle),
            "brake": list(self.brake),
            "t_rel_s": list(self.t_rel_s),
            # 几何弧长口径下这两个通道本地没算（规则也不用），给空列表
            "glat": [], "glon": [],
            "pt": {"x": list(self.xs), "z": list(self.zs)},
            "markers": {"brake_in": list(self.brake_in),
                        "apex": list(self.apex),
                        "throttle_on": list(self.throttle_on),
                        "peak": [], "valley": []},
            "warnings": list(self.warnings),
        }
