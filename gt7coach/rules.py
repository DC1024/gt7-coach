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
| `lap_summary`           | P3 | 每圈结束时报上一圈成绩 |

🔴 `brake_late` 是 P0 而不是 P1，两个理由：
   ① 它是「现在就得动作」的话，和出界/打滑同一性质；
   ② 它和 `brake_warn` 只相隔 1~2 秒，如果同档就会被**跨类冷却**吃掉 ——
      而"预警刚说完、紧接着告诉你刹晚了"是最该连着说的一对。

`@<刹车区>` / `@<弯心>` 这种**带位置标识的 key** 是故意的：
闸门按 key 做「每圈只报一次」，于是天然得到「同一个弯只提醒一次」，
而且跨圈重复抑制也是**按弯**统计的（"这个弯你老是刹晚"）。

滑移率用的轮胎半径是**在线自标定**的：只在自由滚动帧上累计
`R = Σ(v·ω) / Σ(ω²)`。项目里实测前/后轴半径是 0.3391 / 0.3435 而不是同一个值，
但两者只差 1.3%，而滑移告警阈值是 15% —— 用一个半径足够，别为这个加复杂度。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .contract import (Frame, P_CRITICAL, P_HIGH, P_LOW, P_NORMAL,
                       Utterance)
from .refindex import RefLap

G = 9.80665


@dataclass
class RuleConfig:
    # 出界
    off_track_m: float = 18.0
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


def fmt_lap_time(seconds: float | None) -> str:
    """92.412 → "1:32.412"。"""
    if not seconds or seconds <= 0:
        return "-"
    m = int(seconds // 60)
    return f"{m}:{seconds - m * 60:06.3f}"


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
                   self._lap_summary):
            u = fn(c)
            if u is not None:
                out.append(u)
        return out

    # —— 1. 出界 ———————————————————————————————————————

    def _off_track(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        ok = (c.lateral_m is not None
              and c.f.speed_kph >= cfg.min_speed_kph
              and c.lateral_m > cfg.off_track_m)
        if self._hold(c.st, "off", ok, c.dt) < cfg.off_track_hold_s:
            return None
        return Utterance(
            key="off_track", text="出界了，回到赛道", priority=P_CRITICAL,
            ttl_s=2.5, short="出界",
            evidence={"lateral_m": round(c.lateral_m or 0.0, 1)})

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
        extra = ""
        if v_ref is not None and c.f.speed_kph > v_ref + cfg.brake_warn_kph:
            extra = f"，比参考快 {c.f.speed_kph - v_ref:.0f}"
            ev["v_ref_kph"] = round(v_ref, 1)
        vmin = z.get("v_min_kph")
        vmin_txt = f"，参考最低 {vmin:.0f}" if vmin else ""
        return Utterance(
            key=f"brake_warn@{int(z['s_in_m'] // 10) * 10}",
            text=f"{t_go:.1f} 秒后重刹区{vmin_txt}{extra}",
            priority=P_HIGH, ttl_s=1.2, short="准备刹车", evidence=ev)

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
        return Utterance(
            key=tag, text=f"刹车晚了 {over:.0f} 米", priority=P_CRITICAL,
            ttl_s=1.5, short=f"晚 {over:.0f}",
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
            text=f"弯心慢了 {gap:.0f}", priority=P_NORMAL, ttl_s=1.5,
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
            text="给油晚了", priority=P_NORMAL, ttl_s=1.5, short="给油",
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
            return Utterance(key="tyre_hot", text=f"{names[i]}胎过热 {tt[i]:.0f}",
                             priority=P_NORMAL, ttl_s=4.0, short="胎温",
                             evidence={"tyre_temp_c": [round(x, 1) for x in tt]})
        return Utterance(key="tyre_cold", text="轮胎太凉，抓地不够",
                         priority=P_NORMAL, ttl_s=4.0, short="胎温",
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
        return Utterance(key="delta", text=f"{delta:+.2f}", priority=P_LOW,
                         ttl_s=1.5, short=f"{sign}{abs(delta):.1f}",
                         evidence={"delta_s": round(delta, 3),
                                   "s_m": round(c.s, 1)})

    # —— 10. 圈后小结 ——————————————————————————————————

    def _lap_summary(self, c: Ctx) -> Utterance | None:
        ms = c.f.last_lap_ms
        if not ms or ms <= 0:
            return None
        sec = ms / 1000.0
        txt = fmt_lap_time(sec)
        ev: dict[str, Any] = {"last_lap_ms": round(ms, 1)}
        if c.ref is not None and c.ref.lap_time_s > 0:
            d = sec - c.ref.lap_time_s
            ev["vs_ref_s"] = round(d, 3)
            if abs(d) >= 0.05:
                txt += f"，比参考{'慢' if d > 0 else '快'} {abs(d):.2f}"
        return Utterance(key="lap_summary", text=txt, priority=P_LOW,
                         ttl_s=5.0, short=txt, evidence=ev)
