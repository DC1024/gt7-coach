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

from .contract import car_verdict, norm_u16

G = 9.80665

# 两份参考圈的圈速差超过这个比例 → 判「性能不可比」，速度面禁用。
# 10% 大约是"快车 vs 慢车在同一个弯差 15~20 km/h"的量级：到这个份上，
# delta 量的已经不是"你慢了多少"，而是"车慢了多少"。
PACE_TOL = 0.10

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


def _meta_in_progress(meta: dict[str, Any], lap: int) -> bool:
    """Dash 的响应有没有明说「返回给你的这一圈还在跑」。

    两种写法都认：
      · `in_progress: true`                  —— 布尔，直说（新 Dash 这么发）；
      · `in_progress_lap: N` 且 N == 返回的那一圈 —— 等价说法，
        也方便手工拼响应 / 用 JSON 手测（不用回去数布尔）。
    两个都没有（老版 Dash）→ False。**那时靠 `engine._ref_block_reason` 里
    「圈号撞上当前圈 + 同一个场次文件」那条兜底判据**，不是不管了。
    """
    v = meta.get("in_progress")
    if v is not None:
        return bool(v)
    il = meta.get("in_progress_lap")
    if il is None or lap <= 0:
        return False
    try:
        return int(il) == int(lap)
    except (TypeError, ValueError):
        return False


@dataclass
class RefLap:
    """一圈的按位置索引剖面。所有数组 `grid_m` 一一对齐。"""

    lap: int = 0
    # "profile" 本场最快圈（Dash 60Hz）| "self" 自攒 | "history" 跨场次历史最快
    source: str = "profile"
    # 🔴 这份参考圈**来自哪个场次文件**。没有它就无法区分两种完全不同的情况：
    #      · 这份 ref 是本场跑完的一圈（lap < 当前圈）→ 可以放心用；
    #      · 这份 ref 就是**本场正在跑的那一圈**（lap == 当前圈，几何只覆盖
    #        半条赛道）→ 拿它定位会大面积失配，必须拒绝。
    #    跨场次的（缓存 / history）与 `lap` 号跨场不可比，所以只能靠文件名分辨。
    #    `from_frames` 自攒的那份填 ""：它按构造就是本场已跑完的圈，天然可信。
    session_file: str = ""
    # 🔴 这份剖面**还在跑**。Dash 在「本场只有一圈」时会把正在跑的那一圈回给我们
    #    （它自己的注释：`usable = trimmed or usable`，"不能把唯一的一圈也排掉"），
    #    于是"手上有一个 ref 对象"根本不等价于"有可用的参考圈"：
    #    那半圈的折线只覆盖半条赛道、圈长还随车前进**一直变长**（实测一场里
    #    5491 m 一路涨到 7493 m）。拿它做最近点定位 → 横向距离算成几十米 →
    #    一句"出界了"；拿它预告刹车点 → 指到别的弯。而这一切都发生在刚开局、
    #    车手最需要听清的时候。判据必须落在这里，见 `engine._ref_block_reason`。
    in_progress: bool = False
    # 🔴 这份参考圈是**哪辆车**跑出来的。跨场次采用历史圈时，它是判断
    #    「速度面能不能用」的唯一依据（见 `pace_ok`）。
    #    0 / "" = 未知（老版 Dash、或 `cars.csv` 没命中）。
    car_code: int = 0
    car_name: str = ""
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
                     source: str | None = None) -> "RefLap":
        """解析 GT7 Dash `/profile` 的响应。

        `require_geometry=True`（缺省）时，**没有坐标的剖面直接拒绝** ——
        没有折线就没法做实时最近点定位，硬用积分距离定位会错十几米。
        宁可退回自己攒图（那个至少有本轮实测坐标）。

        `source=None` 时优先读 `meta.source`（`to_profile` 往返会带上它，
        所以经本地缓存转一圈回来的参考圈仍能记住自己是 history 还是 profile）；
        都没有才落到 `"profile"`。

        🔴 同时会把 `meta` 里的"这一圈还在跑"读进 `in_progress`。**这里不拒绝它**
           （调用方要能看见"取到了但还不能用"），拒绝在 `engine._fetch_ref` 里做。
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
        _meta = d.get("meta") or {}
        lap = int(d.get("lap") or 0)
        ref = RefLap(
            lap=lap,
            source=source or str(_meta.get("source") or "") or "profile",
            session_file=str(_meta.get("file") or ""),
            in_progress=_meta_in_progress(_meta, lap),
            # 经本地缓存往返一圈回来后，车型身份全靠 `meta` 带回来
            car_code=norm_u16(_meta.get("car_code")),
            car_name=str(_meta.get("car_name") or ""),
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

    # —— 参考圈的「两面」：几何面 vs 速度面 ————————————————————
    #
    # 🔴 一份参考圈其实有**两种**完全不同的用法，混在一起就会失真：
    #
    #   · **几何面**：赛车线的形状、刹车点 / 弯心在赛道上的**位置**。
    #     换车后依然成立 —— T1 的刹车点还在那个地方，顶多差几米。
    #   · **速度面**：参考圈在那里跑多快、用了多少秒。
    #     换车后**完全不成立** —— 拿慢车的最快圈去量快车，delta 会是一个
    #     恒定的 +8 秒，零信息量；弯心速度、刹车入点速度同理。
    #
    # 规则层因此按"用哪一面"分成两组（判断在 `rules.py`，判据在这里）：
    #   · 只用几何面 → 跨车照常工作：`off_track`（横向偏差）、
    #     `throttle_late`（过弯心后 40~120 m 还没给油，位置判据）；
    #   · 用速度面 → 跨车必须闭嘴：`brake_warn`、`brake_late`、`apex_slow`、
    #     `delta`、`projected_lap`，以及 `lap_summary` 的 `vs_ref_s`。
    #   十四规则里有七条依赖速度面 —— 这就是"跨车参考会严重失真"的量级。
    #
    # 为什么不给速度面加"差异补偿"（比如按圈速比缩放参考速度）：
    #   车的快慢不是均匀分布的 —— 大直道尽头差 30 km/h，发夹弯里只差 3 km/h。
    #   一个统一的缩放系数会同时在直道上低估、在弯里高估，比不给更糟。

    def car_match(self, other: "RefLap") -> str:
        """我与 `other` 是同一辆车跑的吗 → "same_car" | "cross_car" | "unknown"。

        判据统一走契约层的 `car_verdict`（先比数字车型码，再比名字），
        与 `source.faster_sessions` 挑候选时用同一把尺 —— 两处口径分叉
        会出现"挑的时候说是同车、采用的时候说不是"这种自相矛盾。
        """
        return car_verdict(self.car_code, self.car_name,
                           other.car_code, other.car_name)

    def pace_ratio(self, other: "RefLap") -> float | None:
        """我相对 `other` 快多少倍（>1 = 我更快）。None = 算不出来。

        用**圈速比**而不是中位速度比：圈速是单一标量，两边直接可比；
        中位速度还要再假设两条折线的采样密度一致，多引入一处误差。
        """
        if self.lap_time_s <= 0 or other.lap_time_s <= 0:
            return None
        return other.lap_time_s / self.lap_time_s

    def pace_ok(self, other: "RefLap", tol: float = PACE_TOL) -> bool:
        """我的**速度面**能拿去量 `other`（本场这辆车）吗？

        `other` 是本场自己跑出来的基准圈，所以这个问题等价于
        "我（可能是跨场次的历史圈）与当前这辆车的性能可比吗"。

        · **同车** → 可以。这正是 `history_best` 存在的意义：今天跑得烂，
          教练也该拿你的历史最好当标杆，而不是拿"今天最烂的一圈"夸你。
          ⚠️ 已知盲点：GT7 的 BoP 会按赛事调动力/车重，同一个 car_code
             在不同赛事里性能可以差出一截。协议里没有 BoP 数据，我们
             **无法分辨**"BoP 变了"和"你今天状态不好"。这是可接受的代价 ——
             反过来（把同车的历史圈也禁掉）会毁掉这个功能本身。
        · **不同车 / 判不出来** → 看圈速差：超过 `tol` 就别用。
        · **圈速算不出来** → 不给用（不知道就是不知道）。
        """
        if self.car_match(other) == "same_car":
            return True
        r = self.pace_ratio(other)
        if r is None:
            return False
        return (1.0 - tol) <= r <= (1.0 + tol)

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

    def track_fingerprint(self, n_points: int = 64, decimals: int = 2) -> str:
        """稳定的赛道形状指纹（同赛道跨场次一致，不同赛道不同）。

        用几何弧长把折线重采样到固定点数 → 去质心 → 按首末点弦长归一化 →
        粗量化 → 哈希。只依赖折线**形状**，与圈速、跑的方向无关
        （反向布局会得到不同指纹，符合预期：不同线路变体本就该区分）。

        用途：给参考圈做本地缓存的 key（见 `refcache`）。同一个赛道你跑一百次，
        指纹都不变；换条赛道立刻不同。重采样保证不同采样密度的 profile 也能
        对齐到同一个指纹。
        """
        import hashlib
        if len(self.xs) < 4:
            return ""
        arc, _ = arc_lengths(self.xs, self.zs)
        total = arc[-1]
        if total <= 0:
            return ""
        pts: list[tuple[float, float]] = []
        for i in range(n_points):
            d = total * i / (n_points - 1)
            pts.append((interp_at(arc, self.xs, d),
                        interp_at(arc, self.zs, d)))
        # 去质心
        cx = sum(p[0] for p in pts) / n_points
        cz = sum(p[1] for p in pts) / n_points
        pts = [(p[0] - cx, p[1] - cz) for p in pts]
        # 按圈长归一化（用首末点弦长当尺度，闭环也 nonzero）
        norm = math.hypot(pts[0][0] - pts[-1][0],
                          pts[0][1] - pts[-1][1]) or 1.0
        blob: list[str] = []
        fmt = f"{{:.{decimals}f}}"
        for x, z in pts:
            blob.append(fmt.format(x / norm))
            blob.append(fmt.format(z / norm))
        return hashlib.sha1("|".join(blob).encode("utf-8")).hexdigest()[:16]

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
            # 🔴 带上"来自哪一场、是哪一类、跑完了没有"，否则经本地缓存往返
            #    一圈回来这些都会丢成空/False —— 而它们正是"这份参考圈能不能
            #    现在用"的全部判据（见 `engine._ref_block_reason`）。
            "meta": {"file": self.session_file, "source": self.source,
                     "in_progress": self.in_progress,
                     # 🔴 车型身份必须跟着走：缓存里那份参考圈是哪辆车跑的，
                     #    决定了它的速度面能不能拿去量当前这辆车。
                     "car_code": self.car_code,
                     "car_name": self.car_name},
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
