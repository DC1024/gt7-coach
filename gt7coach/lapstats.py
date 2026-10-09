# -*- coding: utf-8 -*-
"""
本地圈统计 —— 分段用时与油耗，**全部从实时帧自己算**。
============================================================

为什么不打 GT7 Dash 的 `/sectors` 和 `/pitstops`：

| | 调 Dash 接口 | 自己算 |
|---|---|---|
| 直播场次上的代价 | 每次都要**重新解析整场 jsonl**（`_load_frames` 按 mtime/size 缓存，而直播文件一直在变）→ 一场 20 万帧要 2s+，且随着比赛进行越来越贵 | 0（帧本来就在手上） |
| 频率 | 每圈一次 | 0 次网络 |
| 精度 | 60Hz | 10Hz（实时帧的采样率） |

「分段用时」这个量对采样的敏感度很低 —— 段边界是**按绝对距离**等分的，
10Hz 下每帧走 5~8 m，边界处的线性插值误差量级只有几十毫秒，
而它要回答的问题是「哪个段慢了 0.4 秒」。用 60Hz 换这点精度不值得每次
重解析整场文件。

🔴 段边界必须按**绝对距离**等分，不能按各圈自己的圈长等分。
   GT7 Dash 的 `/sectors` 是按各圈自己的圈长等分的（同一场里圈长能差 ±0.3%，
   段边界于是错开十几米）。教练这边做**跨圈比较**，边界必须钉在同一段路上 ——
   而几何弧长轴天然满足这一点（这也正是 `/profile` 用几何弧长而不是速度积分的原因）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .refindex import RefLap, arc_lengths, interp_at

# 圈长与参考圈相差超过这个比例就判为残缺圈，不给分段（同 Dash 的 dist_tol）
LEN_TOL = 0.03
# 首帧圈内用时超过这个值 = 我们是从半圈开始记的，这一圈不算
FRESH_LAP_S = 0.5


def sector_times(s: list[float], t: list[float], length_m: float,
                 n_sectors: int) -> list[float] | None:
    """按**绝对距离**把 [0, length_m] 等分成 n 段，返回各段用时（秒）。

    `s` 单调不减的几何弧长，`t` 与之对应的圈内时间。插值取边界时刻再作差，
    所以段用时的和恒等于整圈用时（不会有累积误差）。

    跑不满整圈（`s[-1]` 明显小于 `length_m`）返回 None —— 那种情况下
    最后一段会被算成 0，看着像"这段飞快"，比不给更糟。
    """
    if n_sectors < 2 or length_m <= 0 or len(s) < 4:
        return None
    if len(t) != len(s) or s[-1] < length_m * (1.0 - LEN_TOL):
        return None
    out: list[float] = []
    prev = interp_at(s, t, 0.0)
    for k in range(1, n_sectors + 1):
        cur = interp_at(s, t, length_m * k / n_sectors)
        out.append(cur - prev)
        prev = cur
    return out


def ref_sector_times(ref: RefLap, n_sectors: int) -> list[float] | None:
    """参考圈的各段用时（用同一个绝对距离分法，保证可比）。"""
    if ref is None or not ref.grid_m or not ref.t_rel_s:
        return None
    return sector_times(ref.grid_m, ref.t_rel_s, ref.length_m, n_sectors)


@dataclass
class FuelTracker:
    """每圈油耗的滚动中位数。

    用**中位数**而不是上一圈：进站前后、遇到慢车、跑了个暖胎圈都会让单圈油耗
    偏得离谱，而"还能跑几圈"这个数字要稳到能据以决策。
    """

    window: int = 5
    values: list[float] = field(default_factory=list)
    mark: float | None = None      # 上一圈起点时的剩余量

    def start_lap(self, level: float) -> None:
        """在圈首调用，记下起点剩余量。"""
        self.mark = level

    def end_lap(self, level: float) -> float | None:
        """在圈尾调用：**推进标记**并返回本圈消耗；数据不可信时返回 None。

        **不记统计** —— 记不记由调用方在这圈被判定有效之后调 `record()`。
        两段式是必需的：圈统计会因为"中途接入 / 残缺圈"被整体拒绝，而那一圈
        仍然**真实消耗了油**。如果标记不推进，下一个有效圈量到的就是**两圈**的
        消耗 —— 8 变 16，而且刚好落在合法区间（0.05~50）里，
        **没有任何护栏会挡住它**。这类"错得刚好合法"的 bug 最难查。

        三个不可信的来源都要挡住：
          · 没记过圈首（中途接入）
          · 剩余量**上升** → 中途加过油/换过胎（或者读到菜单态的重置值）
          · 变化量离谱（>50 或 <0.1）→ 多半是菜单/结算画面把读数刷成别的
        """
        if self.mark is None:
            return None
        used = self.mark - level
        self.mark = level          # ← 无论可不可信都要推进，这是关键
        if not (0.05 <= used <= 50.0):
            return None
        return used

    def record(self, used: float) -> None:
        """把一圈的消耗记进统计（调用方已确认这圈有效）。"""
        self.values.append(used)
        del self.values[:-self.window]

    @property
    def per_lap(self) -> float | None:
        if not self.values:
            return None
        xs = sorted(self.values)
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0

    def laps_left(self, level: float) -> float | None:
        p = self.per_lap
        if not p or p <= 0 or level <= 0:
            return None
        return level / p

    def to_dict(self) -> dict[str, Any]:
        return {"per_lap": round(self.per_lap, 3) if self.per_lap else None,
                "samples": len(self.values)}


@dataclass
class LapResult:
    """刚跑完那一圈的本地统计。"""

    lap: int
    length_m: float
    lap_time_s: float
    sectors: list[float] = field(default_factory=list)
    fuel_used: float | None = None
    raw_frames: int = 0
    ok: bool = False
    why: str = ""
    # 🔴 内部数组：`corner_losses` 要用**同一份**弧长/时间算每弯损失。
    #    重算一遍弧长虽然便宜，但两处实现迟早分叉（本仓库已经吃过这种亏）。
    #    repr/compare 关掉、to_dict 不输出 —— 它们是内部中间产物。
    s_arr: list[float] = field(default_factory=list, repr=False, compare=False)
    t_arr: list[float] = field(default_factory=list, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lap": self.lap, "ok": self.ok, "why": self.why,
            "length_m": round(self.length_m, 1),
            "lap_time_s": round(self.lap_time_s, 3),
            "sectors": [round(x, 3) for x in self.sectors],
            "fuel_used": round(self.fuel_used, 3) if self.fuel_used else None,
        }


def lap_result(frames: Iterable[Any], *, lap: int, n_sectors: int,
               expected_len_m: float | None, fuel: FuelTracker | None = None
               ) -> LapResult:
    """把一圈的实时帧结算成 `LapResult`。

    `frames` 必须是**同一圈**、按时间升序的 `Frame`（实时侧 10Hz 那种）。
    `expected_len_m` = 参考圈圈长；给了就要求本圈与它相差 ≤3%（残缺圈不给分段）。
    """
    fs = [f for f in frames]
    res = LapResult(lap=lap, length_m=0.0, lap_time_s=0.0,
                    raw_frames=len(fs))
    if len(fs) < 4:
        res.why = "帧太少"
        return res

    # 先结算油量（推进标记），但**暂时不记进统计** —— 这一圈可能被下面的
    # 检查判为无效。标记必须推进，否则无效圈会把它的消耗累到下一圈头上。
    pending_fuel = fuel.end_lap(fs[-1].fuel_pct) if fuel is not None else None

    # 只有「从圈首开始记」的这一圈才算得准：中途接入时弧长起点不是起跑线，
    # 分段边界全错。圈首那一帧的圈内用时应当接近 0。
    if (fs[0].lap_time_s or 0.0) > FRESH_LAP_S:
        res.why = "中途接入，本圈不完整"
        return res

    usable = [f for f in fs if f.coords_ok]
    if len(usable) < 4:
        res.why = "本圈无坐标"
        return res

    s, geo_ok = arc_lengths([f.x for f in usable], [f.z for f in usable])
    if not geo_ok:
        res.why = "坐标无效"
        return res
    t = [f.lap_time_s for f in usable]
    length = s[-1]
    res.length_m = length
    res.lap_time_s = t[-1] - t[0]
    res.s_arr, res.t_arr = s, t

    if expected_len_m:
        if abs(length - expected_len_m) / expected_len_m > LEN_TOL:
            res.why = (f"圈长 {length:.0f}m 与参考 {expected_len_m:.0f}m "
                       f"差得太多（残缺圈）")
            return res
        # 用**参考圈圈长**当分段基准，跨圈才可比
        base_len = expected_len_m
    else:
        base_len = length

    sec = sector_times(s, t, base_len, n_sectors)
    if sec is None:
        res.why = "跑不满整圈，分段不可比"
        return res
    res.sectors = sec
    # 🔴 整圈用时取**各段之和**，而不是 `t[-1] - t[0]`。
    #    两者会差 0.05~0.2s：分段只覆盖到公共基准圈长 `base_len`，而本圈自己的
    #    弧长末端可能比它多出几米（10Hz 采样下最后一帧落在冲线后一点）。
    #    这个差是**口径差**不是误差，但用户看到"三段加起来 ≠ 整圈"只会认为
    #    算错了。统一到公共基准之后，几个数互相自洽。
    #    权威圈速另有其人：`lap_summary` 用的是游戏上报的 `last_lap_ms`。
    res.lap_time_s = float(sum(sec))

    if fuel is not None and pending_fuel is not None:
        fuel.record(pending_fuel)      # 确认有效了才记进统计
        res.fuel_used = pending_fuel
    res.ok = True
    return res


@dataclass
class SectorTracker:
    """本场各段的最好值 → 理论最快圈（**本地版**，不调 Dash 的 `/sectors`）。

    两个用途：
      · `best_each_s`  —— 「你这一段相对自己最好水平差多少」，用于圈后点评
      · `theory_best_s` —— 各段最好值之和 = 理论最快圈，`gain_s` = 还能再快多少

    与 Dash 那版的关键差别：**不设可信圈门槛**。Dash 要过滤异常圈/残圈
    （它面对的是整场文件，可能包含冲出赛道、进站、跑残的圈），而教练这边
    每圈都先过 `lap_result` 的圈长与「跑满整圈」检查，进来的都是干净圈。
    代价是第 1 圈的理论值就等于实际值（没空间），从第 2 圈起才有意义 ——
    所以 `samples` 里不足 2 的段会被规则跳过，不报"慢 0.00"。
    """

    n_sectors: int = 3
    best: list[float | None] = field(default_factory=list)
    samples: list[int] = field(default_factory=list)
    lap_totals: list[float] = field(default_factory=list)

    def add(self, sectors: list[float], lap_time_s: float) -> None:
        if len(sectors) != self.n_sectors:
            return
        if len(self.best) != self.n_sectors:
            self.best = [None] * self.n_sectors
            self.samples = [0] * self.n_sectors
        for i, v in enumerate(sectors):
            cur = self.best[i]
            if cur is None or v < cur:
                self.best[i] = v
            self.samples[i] += 1
        self.lap_totals.append(lap_time_s)
        del self.lap_totals[:-30]

    def theory_s(self) -> float | None:
        """各段最好值之和。任一段还没有样本时返回 None（不给半成品）。"""
        if len(self.best) != self.n_sectors:
            return None
        if any(v is None for v in self.best):
            return None
        return float(sum(v for v in self.best if v is not None))

    def best_actual_s(self) -> float | None:
        return min(self.lap_totals) if self.lap_totals else None

    def gain_s(self) -> float | None:
        t, b = self.theory_s(), self.best_actual_s()
        if t is None or b is None:
            return None
        # theory > actual 只可能是采样噪声（本地版各段最好值必来自真实圈），
        # 不报负数，避免"潜在 -0.02"这种自己打自己脸的输出
        return max(0.0, round(b - t, 3))

    def to_dict(self) -> dict[str, Any] | None:
        t, g = self.theory_s(), self.gain_s()
        if t is None:
            return None
        return {
            "n_sectors": self.n_sectors,
            "best_each_s": [round(v, 3) for v in self.best if v is not None],
            "samples": list(self.samples),
            "theory_best_s": round(t, 3),
            "best_actual_s": (round(self.best_actual_s(), 3)
                              if self.best_actual_s() else None),
            "gain_s": g,
            "laps": len(self.lap_totals),
        }


# ===========================================================================
# 每弯累积失误（R1.6）—— 唯一的真缺口
# ===========================================================================
#
# 教练能主动说 13 件事，但「哪个弯**反复**亏」它从来不说 —— 因为那是跨圈的
# 统计，不是单圈事件。而它恰恰是最值钱的一条：单圈说"S2 慢 0.4"可能是偶发，
# 连续三圈都在同一个弯慢，就是**习惯**，值得专门提醒。
#
# 🔴 为什么这是本地能做、而且只有本地能做的：
#    每圈的帧都在手上（弧长 + 圈内时间），参考圈的 t_rel 也在手上 ——
#    两边一插值作差就是"这个弯相对参考亏了多少"。零网络、零模型。

# 弯窗口：弯心前后的覆盖范围（米）。太窄会漏掉入弯前的减速差，太宽会把
# 相邻两个弯并成一个 —— 300m 在赛道尺度上是 8~12 秒的车程，够包容
# 「入弯前的减速差 + 出弯后的加速差」，又不至于把相邻弯吃进来。
CORNER_BEFORE_M = 150.0
CORNER_AFTER_M = 150.0


def corner_windows(ref: RefLap, *, before_m: float = CORNER_BEFORE_M,
                   after_m: float = CORNER_AFTER_M) -> list[dict]:
    """从参考圈的弯心标记生成「弯窗口」`[{s1, s2, label}, ...]`（按 s 升序）。

    label 用 T1/T2/… 按赛道顺序编号 —— 玩家脑子里记的是"第几个弯"，
    不是弧长米数。

    🔴 重叠的窗口要合并：两个弯心离得近（连续 S 弯）时，各自 ±150m 会重叠，
       不合并就会把同一段路算两遍，损失被重复计入。
    🔴 跨起跑线的窗口**不做回绕**，直接夹到 [0, L] —— 第一个弯在起跑线前
       的那部分会被截掉。这是已知的近似（记在 warnings 里），比做回绕的
       复杂度划算：截掉的是弯心前 150m 里的一小段，不是整个弯。
    """
    apexes = sorted((a for a in (ref.apex or [])
                     if a.get("s_m") is not None), key=lambda a: a["s_m"])
    if not apexes or ref.length_m <= 0:
        return []
    raw: list[list[float]] = []
    for a in apexes:
        s1 = max(0.0, float(a["s_m"]) - before_m)
        s2 = min(ref.length_m, float(a["s_m"]) + after_m)
        if s2 - s1 < 20.0:            # 太窄的窗口（贴着起跑线被夹没的）不要
            continue
        if raw and s1 <= raw[-1][1]:
            raw[-1][1] = max(raw[-1][1], s2)
        else:
            raw.append([s1, s2])
    return [{"s1": round(w[0], 1), "s2": round(w[1], 1),
             "label": f"T{i + 1}"} for i, w in enumerate(raw)]


def corner_losses(s: list[float], t: list[float], windows: list[dict],
                  ref: RefLap) -> dict[str, float]:
    """本圈在每个弯窗口里相对参考圈亏了多少秒。

    返回 `{label: 损失秒}`，只包含**窗口完整落在本圈内**的弯 ——
    窗口超出本圈弧长时插值会被夹住，那段"损失"是假的，宁可不算。
    """
    out: dict[str, float] = {}
    if not windows or not s or s[-1] <= 0:
        return out
    for w in windows:
        if w["s2"] > s[-1]:
            continue                   # 本圈没跑到这个弯的出弯处
        lt2 = interp_at(s, t, w["s2"])
        lt1 = interp_at(s, t, w["s1"])
        rt2 = ref.t_at_s(w["s2"])
        rt1 = ref.t_at_s(w["s1"])
        if rt1 is None or rt2 is None:
            continue
        out[w["label"]] = (lt2 - lt1) - (rt2 - rt1)
    return out


@dataclass
class CornerTracker:
    """跨圈累积每个弯的损失，找出「反复亏」的那个弯。

    🔴 窗口一旦生成就**钉死**：参考圈换了（跑出更快的圈）弯心可能微移，
       如果跟着换，跨圈的损失就不可比了 —— 与 `_sector_len_m` 同一个道理。
    """

    min_laps: int = 3                 # 至少这么多圈才开始下结论
    min_loss_s: float = 0.30          # 平均亏这么多才值一条播报
    window: int = 5                   # 取最近 N 圈的均值（老圈的失误会淡出）
    windows: list[dict] = field(default_factory=list)
    losses: dict[str, list[float]] = field(default_factory=dict)

    def setup(self, ref: RefLap) -> bool:
        """用参考圈的弯心建窗口；已建过就返回 False（不重建）。"""
        if self.windows:
            return False
        w = corner_windows(ref)
        self.windows = w
        return bool(w)

    def add(self, per: dict[str, float]) -> None:
        for k, v in per.items():
            xs = self.losses.setdefault(k, [])
            xs.append(v)
            del xs[:-self.window]

    def habit(self) -> dict[str, Any] | None:
        """「反复亏」的弯：样本 ≥min_laps、**中位损失** ≥min_loss_s 里最严重的。

        🔴 用**中位数**而不是均值 —— 和 `FuelTracker` 用中位数是同一个理由：
           均值对单圈离群值不鲁棒。实测：最近三圈损失 [0, 0, 3.0] 的均值是 1.0，
           会触发"这个弯你反复亏"的播报；但那三圈里有两圈是 0 —— 那是被慢车
           挡了一圈的偶发，不是习惯。中位数是 0，正确地不触发。
        """
        best: dict[str, Any] | None = None
        for label, xs in self.losses.items():
            if len(xs) < self.min_laps:
                continue
            srt = sorted(xs)
            n = len(srt)
            med = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2.0
            if med < self.min_loss_s:
                continue
            if best is None or med > best["median_loss_s"]:
                best = {"label": label, "median_loss_s": round(med, 3),
                        "metric": round(med, 3), "laps": n,
                        "recent": [round(x, 2) for x in xs]}
        return best

    def to_dict(self) -> dict[str, Any]:
        return {"corners": len(self.windows),
                "tracked": sorted(self.losses),
                "min_laps": self.min_laps}
