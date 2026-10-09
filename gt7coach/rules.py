# -*- coding: utf-8 -*-
"""
规则引擎 —— 把「现在这一帧」翻译成「该说什么」。
====================================================

🔴 全部判断都在这里本地做完，**不调用任何云服务**。理由：
   一次云调用 0.5~2s，250 km/h 时 2 秒 = 139 米 ——
   "T1 刹车点提前 10 米"晚 139 米就是废话。云只能负责事后措辞。

十条规则，分三档优先级（数字越小越急）：

| key                     | 档 | 触发 |
|-------------------------|---|------|
| `off_track`             | P0 | 距参考线横向 > 18 m 持续 0.4 s |
| `slip_front/rear/all`   | P0 | 滑移率 > 0.15 持续 0.25 s |
| `brake_late@<刹车区>`   | P0 | 已过入点 > 8 m 还没踩刹车（**只到弯心为止**） |
| `brake_warn@<刹车区>`   | P1 | 距下一个刹车入点 < 1.5 s |
| `shift`                 | P1 | rpm ≥ 换挡灯上限持续 0.5 s |
| `apex_slow@<弯心>`      | P2 | 弯中速度比参考低 > 5 km/h |
| `throttle_late@<弯心>`  | P2 | 过弯心后 40~120 m 油门仍 < 0.3 持续 0.5 s |
| `tyre_hot` / `tyre_cold`| P2 | 任一胎温 > 110 / < 60 °C 持续 2 s |
| `delta`                 | P3 | 本圈 delta 首次跨过 ±0.2 s |
| `projected_lap`         | P3 | 跑过本圈 35% 后，预测最终圈速（差 >0.3s 才报）|
| `lap_summary`           | P3 | 每圈结束时报上一圈成绩 |
| `sector_loss`           | P2 | 圈后：指出相对「各段最好值」最慢的那一段（差 >0.15s）|
| `fuel_range`            | P2 | 圈后：按本场油耗中位数算还能跑几圈（≤3 圈才报）|

🔴 `brake_late` 是 P0 而不是 P1，两个理由：
   ① 它是「现在就得动作」的话，和出界/打滑同一性质；
   ② 它和 `brake_warn` 只相隔 1~2 秒，如果同档就会被**跨类冷却**吃掉 ——
      而"预警刚说完、紧接着告诉你刹晚了"是最该连着说的一对。

`@<刹车区>` / `@<弯心>` 这种**带位置标识的 key** 是故意的：
闸门按 key 做「每圈只报一次」，于是天然得到「同一个弯只提醒一次」，
而且跨圈重复抑制也是**按弯**统计的（"这个弯你老是刹晚"）。

滑移率用的轮胎半径是**在线自标定**的：只在自由滚动帧上累计
`R = Σ(v·ω) / Σ(ω²)`。**前轴与后轴分开标**——项目里实测它们是
0.3391 / 0.3435，差 1.3%。虽然对 15% 的告警阈值来说这点偏差不至于误判，
但分开标定的成本只是两个累加器，而合成一个会把这 1.3% 直接变成
滑移率的固定偏置（阈值附近就会开始误报）。

（这段注释曾经写着"用一个半径足够"，与代码不符 —— 文档和实现打架时
以代码为准，然后**把文档改对**，不然下一个人会照着注释去"简化"。）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:      # 只为注解，避免 rules↔lapstats 循环 import
    from .lapstats import LapResult

from . import phrases
from .contract import (Frame, P_CRITICAL, P_HIGH, P_LOW, P_NORMAL,
                       Utterance)
from .refindex import RefLap

# 🔴 措辞（怎么说）已经搬到 `phrases.py`，判断（什么时候说）留在这里。
#    `fmt_lap_time` 从 phrases 转出来只为兼容既有调用方（__init__/cli/tests）——
#    新代码请直接用 `phrases.fmt_lap_time`。
fmt_lap_time = phrases.fmt_lap_time

G = 9.80665


@dataclass
class RuleConfig:
    # 出界
    # 🔴 判据是「距**赛车线**多少米」而不是「距赛道边缘多少米」——
    #    `/profile` 给的是跑得最快的那条线，不是赛道中线，协议里也没有赛道边界。
    #    所以这条阈值必然是近似的，取值要**宁可放过不可错杀**：
    #    误报一次"出界"就把教练的可信度毁了。
    off_track_m: float = 18.0          # 直线段的基准阈值
    # 弯道里大家走的线差异本来就大（切弯内侧 / 出弯放开外侧），
    # 所以按横向 G 线性放宽：|glat| 达到 off_track_g_ref 时放宽到 1+slack 倍。
    off_track_corner_slack: float = 0.8
    off_track_g_ref: float = 1.2
    off_track_hold_s: float = 0.40
    min_speed_kph: float = 20.0

    # 打滑
    slip_threshold: float = 0.15
    slip_hold_s: float = 0.25
    tyre_radius_default: float = 0.34
    free_glat_max: float = 0.15      # 自由滚动的判定窗
    free_glon_max: float = 0.15
    free_thr_lo: float = 0.15
    free_thr_hi: float = 0.85
    free_brake_max: float = 0.05
    calib_min_samples: int = 150

    # 刹车
    brake_warn_s: float = 1.5
    brake_warn_kph: float = 3.0      # 比参考快多少才额外提醒
    brake_late_m: float = 8.0
    brake_late_hold_s: float = 0.15

    # 弯中
    apex_slow_kph: float = 5.0
    apex_hold_s: float = 0.10
    throttle_late_from_m: float = 40.0
    throttle_late_to_m: float = 120.0
    throttle_late_hold_s: float = 0.50
    throttle_late_on: float = 0.30

    # 换挡 / 胎温 / delta
    shift_hold_s: float = 0.50
    tyre_hot_c: float = 110.0
    tyre_cold_c: float = 60.0
    tyre_hold_s: float = 2.0
    delta_threshold_s: float = 0.20

    # —— 本地统计类（全部离线可算，零网络）——
    sectors_n: int = 3                # 分段数（与 Dash 的 /sectors 口径独立）
    # 预测圈速：太早报没意义（delta 还在抖），太小报是噪声
    projected_after_frac: float = 0.35   # 跑完这个比例的本圈才报
    projected_min_delta_s: float = 0.30
    # 圈后分段：低于这个差距就不值一提
    sector_loss_min_s: float = 0.15
    # 续航提醒：剩多少圈的时候说
    fuel_warn_laps: float = 3.0
    # —— 每弯累积失误（R1.6）——
    corner_min_laps: int = 3        # 同一个弯至少这么多圈才下结论
    corner_min_loss_s: float = 0.30  # 平均亏这么多才值一条播报
    # 同一个弯两次提醒之间隔几圈。每圈都念同一句就成了唠叨；
    # 隔 3 圈 = 提醒之后给你 3 圈去改，改不好再提。
    corner_repeat_laps: int = 3


@dataclass
class Ctx:
    """一次评估的全部输入。"""

    f: Frame
    prev: Frame | None = None
    ref: RefLap | None = None
    s: float | None = None            # 本圈沿参考圈的已跑距离（米）
    lateral_m: float | None = None    # 到参考线的横向距离（米）
    dt: float = 0.0                   # 距上一 tick 的秒数
    st: dict[str, Any] = field(default_factory=dict)
    # —— 本地统计（引擎算好放进来，规则只读）——
    lap: "LapResult | None" = None    # 刚跑完那一圈的本地统计
    theory: dict[str, Any] | None = None   # 本场各段最好值 → 理论最快圈
    fuel: dict[str, Any] | None = None     # 每圈油耗 / 还能跑几圈
    corners: dict[str, Any] | None = None  # 每弯累积（含 habit：反复亏的那个弯）


class RuleSet:
    """无状态规则 + 有状态累积器（累积器都放在 ctx.st 里）。"""

    def __init__(self, cfg: RuleConfig | None = None):
        self.cfg = cfg or RuleConfig()

    @staticmethod
    def fresh_state() -> dict[str, Any]:
        return {
            "hold": {},          # key -> 已持续秒数
            "radius": {"fn": 0.0, "fd": 0.0, "rn": 0.0, "rd": 0.0,
                       "n": 0, "front": None, "rear": None},
            "lap": None,
        }

    def roll_lap(self, st: dict, lap: int) -> None:
        """圈变化时清掉「持续计时」，但**保留**半径标定。"""
        if st.get("lap") != lap:
            st["lap"] = lap
            st["hold"] = {}

    # —— 工具 ——————————————————————————————————————————

    def _hold(self, st: dict, key: str, cond: bool, dt: float) -> float:
        """条件成立就累加时间，否则清零。返回已持续的秒数。"""
        if cond:
            st["hold"][key] = st["hold"].get(key, 0.0) + dt
        else:
            st["hold"][key] = 0.0
        return st["hold"][key]

    # —— 主入口 ————————————————————————————————————————

    def evaluate(self, c: Ctx) -> list[Utterance]:
        out: list[Utterance] = []
        for fn in (self._off_track, self._wheel_slip, self._brake_warn,
                   self._brake_late, self._apex_slow, self._throttle_late,
                   self._shift, self._tyre_temp, self._delta,
                   self._lap_summary, self._projected_lap,
                   self._sector_loss, self._fuel_range, self._next_focus):
            u = fn(c)
            if u is not None:
                out.append(u)
        return out

    # —— 1. 出界 ———————————————————————————————————————

    def _off_track(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        if c.lateral_m is None or c.f.speed_kph < cfg.min_speed_kph:
            self._hold(c.st, "off", False, c.dt)
            return None
        # 弯中放宽：赛车线在弯里切内侧，走别的线偏离十几米很正常
        slack = min(abs(c.f.glat) / max(cfg.off_track_g_ref, 1e-6), 1.0)
        thr = cfg.off_track_m * (1.0 + cfg.off_track_corner_slack * slack)
        if self._hold(c.st, "off", c.lateral_m > thr,
                      c.dt) < cfg.off_track_hold_s:
            return None
        return Utterance(
            key="off_track", text="出界了，回到赛道", priority=P_CRITICAL,
            ttl_s=2.5, short="出界",
            evidence={"lateral_m": round(c.lateral_m, 1),
                      "threshold_m": round(thr, 1),
                      "glat": round(c.f.glat, 2)})

    # —— 2. 打滑 ———————————————————————————————————————

    def _wheel_slip(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        f = c.f
        if len(f.wheel_rads) < 4 or f.speed_ms < 3.0:
            self._hold(c.st, "slip", False, c.dt)
            return None

        rad = c.st["radius"]
        # 自由滚动帧才用来标定半径：横向/纵向 G 都小、油门在中间、没踩刹车
        if (abs(f.glat) < cfg.free_glat_max and abs(f.glon) < cfg.free_glon_max
                and cfg.free_thr_lo < f.throttle < cfg.free_thr_hi
                and f.brake < cfg.free_brake_max):
            v = f.speed_ms
            fl, fr, rl, rr = f.wheel_rads[:4]
            rad["fn"] += v * (fl + fr) / 2.0
            rad["fd"] += ((fl * fl + fr * fr) / 2.0)
            rad["rn"] += v * (rl + rr) / 2.0
            rad["rd"] += ((rl * rl + rr * rr) / 2.0)
            rad["n"] += 1
        if rad["fd"] > 0 and rad["n"] >= cfg.calib_min_samples:
            rad["front"] = rad["fn"] / rad["fd"]
            rad["rear"] = rad["rn"] / rad["rd"]
        rf = rad["front"] or cfg.tyre_radius_default
        rr = rad["rear"] or cfg.tyre_radius_default

        v = max(f.speed_ms, 1.0)
        fl, fr, rl, rr_w = f.wheel_rads[:4]
        sf = (((fl + fr) / 2.0) * rf - v) / v
        sr = (((rl + rr_w) / 2.0) * rr - v) / v

        both = abs(sf) > cfg.slip_threshold and abs(sr) > cfg.slip_threshold
        front = abs(sf) > cfg.slip_threshold
        rear = abs(sr) > cfg.slip_threshold
        bad = both or front or rear
        if self._hold(c.st, "slip", bad, c.dt) < cfg.slip_hold_s:
            return None
        # 抱死（四轮同时 -1.0 左右）是真事，不是哨兵 —— 别去"修"它
        if both:
            key, txt, short = "slip_all", "四轮打滑", "打滑"
        elif front:
            key, txt, short = "slip_front", "前轮打滑", "打滑"
        else:
            key, txt, short = "slip_rear", "后轮打滑", "打滑"
        return Utterance(key=key, text=txt, priority=P_CRITICAL, ttl_s=2.0,
                         short=short,
                         evidence={"slip_front": round(sf, 3),
                                   "slip_rear": round(sr, 3),
                                   "speed_kph": round(f.speed_kph, 1)})

    # —— 3. 刹车区预告 ————————————————————————————————

    def _brake_warn(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        if c.ref is None or c.s is None or c.f.speed_kph < cfg.min_speed_kph:
            return None
        z = c.ref.next_brake(c.s)
        if not z:
            return None
        d = z["s_in_m"] - c.s
        if d < 0:
            d += c.ref.length_m          # 回绕：圈末的下一个刹车点是起点那个
        t_go = d / max(c.f.speed_ms, 1.0)
        if not (0.0 < t_go <= cfg.brake_warn_s):
            return None

        ev: dict[str, Any] = {
            "s_in_m": z["s_in_m"], "t_go_s": round(t_go, 2),
            "v_min_kph": z.get("v_min_kph"),
        }
        v_ref = c.ref.v_at_s(c.s)
        over = None
        if v_ref is not None and c.f.speed_kph > v_ref + cfg.brake_warn_kph:
            over = c.f.speed_kph - v_ref
            ev["v_ref_kph"] = round(v_ref, 1)
            # 🔴 `over` 必须进 evidence —— 它是句子「快 57」里的**唯一数字**。
            #    之前只算了不存，于是真车回放被数字白名单抓出「编造 57」：
            #    白名单的规则是"句子里每个数都要能在 facts 里找到"，
            #    而 `v_ref_kph` 与 `speed_kph` 都不等于 57（那是两者的差）。
            #    这正好证明白名单不是形式主义 —— 它抓的是**真漏**。
            ev["over_kph"] = round(over, 1)
        # 🔴 文案只有两种，都 ≤8 字 —— 这是被 `ttl_s` 逼出来的硬约束：
        #    A 档的 ttl 由**信息有效期**决定（这条消息一进刹车区就没用了），
        #    而不是由"我愿意等它念完"决定。原来那句
        #    「1.4 秒后重刹区，参考最低 90」＝17 字≈3.8 s 语音 > ttl 1.2 s，
        #    意味着它**根本来不及念完**就过期 —— 实测才发现（见
        #    tests/test_rules.py::TestTtlFitsSpeech）。
        #    `v_min_kph`（参考最低速）是分析信息不是行动信息，移进 evidence。
        txt = (f"快 {over:.0f}，准备重刹" if over is not None
               else f"{t_go:.1f} 秒后重刹")
        # 🔴 ttl 由 `phrases.ttl_for` 统一算（**别再手写常数**）：
        #    `floor_s = t_go + 0.6` 是"信息有效期"—— 进刹车区后 0.6 s 内
        #    还算"预告"，再晚就该 `brake_late` 接手说"你晚了"；
        #    再叠加"这句话本身的语音时长"，取大者。
        #    原来写死 1.2 s 是**错的**：`brake_warn_s=1.5` ⇒ t_go 最大 1.5
        #    ⇒ ①只给到 2.1 s，而「1.0 秒后重刹」8 字就需 1.78 s，
        #    在 t_go≤1.0（最常见的接近窗口）①只剩 1.2 s ⇒ 整段静默丢消息。
        ttl = phrases.ttl_for(txt, P_HIGH, floor_s=t_go + 0.6)
        return Utterance(
            key=f"brake_warn@{int(z['s_in_m'] // 10) * 10}",
            text=txt,
            priority=P_HIGH, ttl_s=ttl,
            short="准备刹车", evidence=ev)

    # —— 4. 刹车点晚了 ————————————————————————————————

    def _brake_late(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        if c.ref is None or c.s is None:
            return None
        # 找**刚过去**的那个刹车入点
        zones = [z for z in c.ref.brake_in
                 if 0 <= c.s - z["s_in_m"] <= 200.0]
        if not zones:
            self._hold(c.st, "blate", False, c.dt)
            return None
        z = max(zones, key=lambda x: x["s_in_m"])

        # 🔴 只在**入弯阶段**才算"晚"。一旦过了这个刹车区对应的弯心，
        #    车本来就在加速出弯，"没踩刹车而且速度高"完全正常 ——
        #    不加这道界限会在出弯直道上报"刹车晚了 155 米"。
        apex = c.ref.apex_of_zone(z)
        if apex is not None and c.s > apex["s_m"]:
            self._hold(c.st, "blate", False, c.dt)
            return None

        over = c.s - z["s_in_m"]
        # 已经减速到位（速度掉到入点速度的 92% 以下）就不算晚
        not_braking = (c.f.brake < 0.10
                       and c.f.speed_kph > z.get("speed_in_kph", 0) * 0.92)
        # 🔴 不要再加「throttle 小才算晚」这类条件：全油门冲进刹车区
        #    恰恰是最该报的场景。防出弯误报靠上面 apex 那道界限就够了。
        bad = over > cfg.brake_late_m and not_braking
        tag = f"brake_late@{int(z['s_in_m'] // 10) * 10}"
        if self._hold(c.st, "blate", bad, c.dt) < cfg.brake_late_hold_s:
            return None
        txt = f"刹车晚了 {over:.0f} 米"
        # ttl 2.5 s：这条消息的有效期是"到你减速到位为止"，不是 1.5 s。
        # 原值 1.5 s 装不下这句话本身（9~10 字 ≈ 2.0~2.2 s 语音）——
        # 同 brake_warn，由 TestTtlFitsSpeech 抓出来。
        # 用 ttl_for（floor_s = 信息有效期）统一算，别手写 speech_s 兜底。
        ttl = phrases.ttl_for(txt, P_CRITICAL, floor_s=2.5)
        return Utterance(
            key=tag, text=txt, priority=P_CRITICAL,
            ttl_s=ttl, short=f"晚 {over:.0f}",
            evidence={"over_m": round(over, 1), "metric": round(over, 1),
                      "s_in_m": z["s_in_m"]})

    # —— 5. 弯心慢了 ——————————————————————————————————

    def _apex_slow(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        if c.ref is None or c.s is None:
            return None
        near = [a for a in c.ref.apex
                if 0 <= c.s - a["s_m"] <= 60.0]
        if not near:
            self._hold(c.st, "apex", False, c.dt)
            return None
        a = max(near, key=lambda x: x["s_m"])
        gap = a["speed_kph"] - c.f.speed_kph
        bad = gap > cfg.apex_slow_kph
        if self._hold(c.st, "apex", bad, c.dt) < cfg.apex_hold_s:
            return None
        return Utterance(
            key=f"apex_slow@{int(a['s_m'] // 10) * 10}",
            text=f"弯心慢了 {gap:.0f}", priority=P_NORMAL,
            ttl_s=phrases.ttl_for(f"弯心慢了 {gap:.0f}", P_NORMAL, "apex_slow"),
            short=f"慢 {gap:.0f}",
            evidence={"gap_kph": round(gap, 1), "metric": round(gap, 1),
                      "s_m": a["s_m"], "ref_kph": a["speed_kph"]})

    # —— 6. 给油晚了 ——————————————————————————————————

    def _throttle_late(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        if c.ref is None or c.s is None:
            return None
        zone = [a for a in c.ref.apex
                if cfg.throttle_late_from_m
                <= c.s - a["s_m"] <= cfg.throttle_late_to_m]
        if not zone:
            self._hold(c.st, "thrlate", False, c.dt)
            return None
        a = max(zone, key=lambda x: x["s_m"])
        bad = c.f.throttle < cfg.throttle_late_on
        if self._hold(c.st, "thrlate", bad, c.dt) < cfg.throttle_late_hold_s:
            return None
        return Utterance(
            key=f"throttle_late@{int(a['s_m'] // 10) * 10}",
            text="给油晚了", priority=P_NORMAL,
            ttl_s=phrases.ttl_for("给油晚了", P_NORMAL, "throttle_late"),
            short="给油",
            evidence={"s_m": a["s_m"],
                      "throttle": round(c.f.throttle, 2)})

    # —— 7. 换挡 ———————————————————————————————————————

    def _shift(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        f = c.f
        need = f.max_rpm > 0 and f.rpm >= f.max_rpm
        if self._hold(c.st, "shift", need, c.dt) < cfg.shift_hold_s:
            return None
        return Utterance(key="shift", text="换挡", priority=P_HIGH, ttl_s=1.0,
                         short="换挡",
                         evidence={"rpm": round(f.rpm), "gear": f.gear})

    # —— 8. 胎温 ———————————————————————————————————————

    def _tyre_temp(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        tt = [t for t in c.f.tyre_temp if t > 0]
        if len(tt) < 4:
            self._hold(c.st, "tyre", False, c.dt)
            return None
        hot = max(tt) > cfg.tyre_hot_c
        cold = max(tt) < cfg.tyre_cold_c
        if self._hold(c.st, "tyre", hot or cold, c.dt) < cfg.tyre_hold_s:
            return None
        names = ["左前", "右前", "左后", "右后"]
        if hot:
            i = tt.index(max(tt))
            txt = f"{names[i]}胎过热 {tt[i]:.0f}"
            return Utterance(key="tyre_hot", text=txt,
                             priority=P_NORMAL,
                             ttl_s=phrases.ttl_for(txt, P_NORMAL, "tyre_temp"),
                             short="胎温",
                             evidence={"tyre_temp_c": [round(x, 1) for x in tt]})
        return Utterance(key="tyre_cold", text="轮胎太凉，抓地不够",
                         priority=P_NORMAL,
                         ttl_s=phrases.ttl_for("轮胎太凉，抓地不够", P_NORMAL,
                                               "tyre_temp"),
                         short="胎温",
                         evidence={"tyre_temp_c": [round(x, 1) for x in tt]})

    # —— 9. delta ——————————————————————————————————————

    def _delta(self, c: Ctx) -> Utterance | None:
        if c.ref is None or c.s is None or c.f.lap_time_s <= 0:
            return None
        t_ref = c.ref.t_at_s(c.s)
        if t_ref is None:
            return None
        delta = c.f.lap_time_s - t_ref
        if abs(delta) < self.cfg.delta_threshold_s:
            return None
        sign = "+" if delta > 0 else "-"
        txt = f"{delta:+.2f}"
        return Utterance(key="delta", text=txt, priority=P_LOW,
                         ttl_s=phrases.ttl_for(txt, P_LOW, "delta"),
                         short=f"{sign}{abs(delta):.1f}",
                         evidence={"delta_s": round(delta, 3),
                                   "s_m": round(c.s, 1)})

    # —— 10. 圈后小结 ——————————————————————————————————

    def _lap_summary(self, c: Ctx) -> Utterance | None:
        ms = c.f.last_lap_ms
        if not ms or ms <= 0:
            return None
        sec = ms / 1000.0
        ev: dict[str, Any] = {"last_lap_ms": round(ms, 1),
                              "lap_time_s": round(sec, 3)}
        if c.ref is not None and c.ref.lap_time_s > 0:
            ev["vs_ref_s"] = round(sec - c.ref.lap_time_s, 3)
            ev["ref_lap_time_s"] = round(c.ref.lap_time_s, 3)
        txt = phrases.render("lap_summary", ev)
        return Utterance(key="lap_summary", text=txt, priority=P_LOW,
                         ttl_s=phrases.ttl_for(txt, P_LOW, "lap_summary"),
                         short=txt, evidence=ev)

    # —— 11. 预测圈速 ————————————————————————————————————
    #
    # 比赛无线电里最常被问的一句。原理很朴素：你现在相对参考圈差多少，
    # 保持下去最终就会差多少 → 预计圈速 = 参考圈圈速 + 当前 delta。
    # 之所以成立，是因为 delta 与参考圈的距离轴是同一个（几何弧长）。

    def _projected_lap(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        if c.ref is None or c.s is None or c.f.lap_time_s <= 0:
            return None
        # 跑得太早时 delta 还在抖（起步、暖胎、第一弯的噪声），报出来是误导
        if c.ref.length_m <= 0:
            return None
        if c.s < c.ref.length_m * cfg.projected_after_frac:
            return None
        t_ref = c.ref.t_at_s(c.s)
        if t_ref is None:
            return None
        delta = c.f.lap_time_s - t_ref
        if abs(delta) < cfg.projected_min_delta_s:
            return None
        projected = c.ref.lap_time_s + delta
        if not (10.0 < projected < 3600.0):
            return None
        ev: dict[str, Any] = {"projected_s": round(projected, 3),
                              "ref_lap_time_s": round(c.ref.lap_time_s, 3),
                              "delta_s": round(delta, 3),
                              "s_m": round(c.s, 1)}
        # 🔴 这里**故意不做"比最好圈快/慢多少"的比较**（试过，撤回了）：
        #    按参考圈口径算，它就是 `delta` 本身（已单独播报，纯重复）；
        #    按自己最好圈口径算，会和同一圈里 `lap_summary` 的参考圈口径打架。
        #    详细账记在 `phrases.projected_lap` 的文档里。
        # 🔴 ttl 2.0 s 与句子长度是绑在一起的：R2.1 定为「预计 1:11.010」
        #    （11 字 ≈ 2.4 s 语音）。要加长这句，ttl 必须同时加长 ——
        #    tests/test_rules.py::TestTtlFitsSpeech 守着这个等式。
        txt = phrases.render("projected_lap", ev)
        return Utterance(key="projected_lap", text=txt,
                         priority=P_LOW,
                         ttl_s=phrases.ttl_for(txt, P_LOW, "projected_lap"),
                         short=f"{projected:.1f}", evidence=ev)

    # —— 12. 圈后分段 ————————————————————————————————————
    #
    # 「你哪一段最慢」比「你这圈慢 0.4」有用得多 —— 前者能直接指导下一圈。
    # 比较基准取**本场各段的最好值**（自己跟自己比），不是参考圈：
    #  - 参考圈可能是别人的/历史的，拿它逐段比会让人以为处处都慢
    #  - 各段最好值就是"你已经做到过的最好水平"，差距纯粹是执行问题

    def _sector_loss(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        if not c.lap or not c.lap.ok or not c.lap.sectors:
            return None
        if not c.theory:
            return None
        best = c.theory.get("best_each_s") or []
        counts = c.theory.get("samples") or []
        if len(best) != len(c.lap.sectors):
            return None
        losses: list[tuple[float, int]] = []
        for i, got in enumerate(c.lap.sectors):
            # 该段样本不足 2 个时"最好值"就是本圈自己，差值恒为 0，报它没意义
            if i >= len(counts) or counts[i] < 2:
                continue
            losses.append((got - best[i], i))
        if not losses:
            return None
        loss, idx = max(losses)
        if loss < cfg.sector_loss_min_s:
            return None
        ev: dict[str, Any] = {
            "sector": idx + 1, "loss_s": round(loss, 3),
            "metric": round(loss, 3), "lap": c.lap.lap,
            "sectors": [round(x, 3) for x in c.lap.sectors],
            "best_each_s": [round(x, 3) for x in best],
        }
        gain = c.theory.get("gain_s")
        if gain and gain >= cfg.sector_loss_min_s:
            ev["gain_s"] = round(gain, 3)
        txt = phrases.render("sector_loss", ev)
        return Utterance(key="sector_loss", text=txt, priority=P_NORMAL,
                         ttl_s=phrases.ttl_for(txt, P_NORMAL, "sector_loss"),
                         short=f"S{idx + 1} 慢 {loss:.1f}",
                         evidence=ev)

    # —— 13. 续航 ————————————————————————————————————————

    def _fuel_range(self, c: Ctx) -> Utterance | None:
        if not c.fuel:
            return None
        left = c.fuel.get("laps_left")
        if left is None or left > self.cfg.fuel_warn_laps:
            return None
        # 电车是"电量还够"，油车是"油还够" —— 说错一次就没人信了
        unit = "电量" if (c.f.powertrain or "") == "electric" else "油"
        ev: dict[str, Any] = {"unit": unit, "laps_left": round(left, 2),
                              "per_lap": c.fuel.get("per_lap"),
                              "level": c.fuel.get("level"),
                              "powertrain": c.f.powertrain or None}
        txt = phrases.render("fuel_range", ev)
        return Utterance(
            key="fuel_range", text=txt,
            priority=P_NORMAL,
            ttl_s=phrases.ttl_for(txt, P_NORMAL, "fuel_range"),
            short=f"{unit}够 {left:.1f} 圈",
            evidence=ev)

    # —— 14. 主动建议（R2.4）：哪个弯反复亏 ————————————————————
    #
    # 这是整条链路里**唯一需要跨圈累积**的规则，也是教练和"仪表盘"最本质的
    # 区别：仪表盘显示的是**现在**，教练能告诉你的是**你的习惯**。
    #
    # 🔴 为什么单独算每弯而不复用 sector_loss（按段）：
    #    一段里可能有 2~3 个弯，"S2 慢 0.4"没法回答"T3 到底该怎么改"；
    #    而按弯（弯心 ±150m）算出来的损失才能直接对应到"哪个弯"。
    #
    # 🔴 为什么文本是"指出"而不是"开药方"：
    #    "刹车早一点"这种处方需要知道失误的模式（晚刹？入弯太快？），
    #    本地只能给出"这个弯你反复亏多少"这个事实。开药方交给 R2 的云润色 ——
    #    那是它擅长且唯一该做的事。

    def _next_focus(self, c: Ctx) -> Utterance | None:
        if not c.corners:
            return None
        h = c.corners.get("habit")
        if not h:
            return None
        # 🔴 兜第二道闸：CornerTracker 已经按 min_laps 挡过一次，但这里不能信
        #    传入值 —— 将来 narrate/云侧也可能组装这个 dict。样本不足的"习惯"
        #    比没有习惯更糟（它会让玩家以为自己真的有一个改不掉的毛病）。
        if h.get("laps", 0) < self.cfg.corner_min_laps:
            return None
        if c.lap is None:
            return None
        # 🔴 同一个弯要隔几圈才提醒第二次：每圈都念同一句就成了唠叨，
        #    而唠叨会让人开始忽略教练 —— 比不说还糟。
        seen = c.st.setdefault("habit_last_lap", {})
        last = seen.get(h["label"], -999)
        if c.lap.lap - last < self.cfg.corner_repeat_laps:
            return None
        seen[h["label"]] = c.lap.lap
        txt = phrases.render("next_focus", h)
        return Utterance(
            key=f"corner_habit@{h['label']}",
            text=txt,
            priority=P_NORMAL,
            ttl_s=phrases.ttl_for(txt, P_NORMAL, "next_focus"),
            short=f"重点 {h['label']}",
            evidence=h)
