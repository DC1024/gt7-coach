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
| `off_track`             | P0 | 距参考线横向 > 18 m **且** 轮胎打滑（滑移率 > 0.10）同时持续 0.4 s（#I 双重确认，减少走线图误差误报）|
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
| `next_focus`            | P2 | 圈后：同一个弯连续 N 圈反复亏 → 提醒（隔 3 圈才提第二次）|
| `lap_advice`            | P2 | **R2.4**：把上面四条事实合并成**一句**圈后综合建议（每圈一次）|
| `position`              | P3 | **R3.1**：名次变了（追回 / 掉了 N 位）|
| `encourage`             | P3 | **R3.1**：后半区 → 每 N 圈随机鼓励一次 |
| `leader@take`           | P3 | **R3.1**：刚拿到 P1 |
| `leader@hold`           | P3 | **R3.1**：持续领跑时隔 N 圈提醒"保持住" |
| `race_finish`           | P3 | **R3.2**：最后一圈冲线后报最终名次（此次比赛第X位）|

> `lap_advice`（R2.4）：默认 `RuleConfig.lap_advice=True` 时，圈后**只发这一条**
> 合并句（`lap_summary`/`sector_loss`/`fuel_range`/`next_focus` 的**判断**照跑、
> 事实照收，只是不再各自单独播报）；设 `False` 退回旧的四条各自单说。

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
from .contract import (MAX_PLAUSIBLE_CARS, Frame, P_CRITICAL, P_HIGH,
                       P_LOW, P_NORMAL, Utterance)
from .narrate import Narrator
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
    # 🔴 双重确认闸门（#I）：横向偏离参考线**且**轮胎确实在打滑才判"出界"。
    #    原因：赛车线只是条"建议线"，车手走另一条线（仍轧在柏油上）时
    #    横向偏离照样很大，纯按距离判会误杀。而草皮/砂石上的**真实出界**
    #    必然伴随轮胎打滑（抓地力骤降），所以拿打滑当第二重闸门——
    #    两者同时满足才报，显著减少走线图误差带来的误报。
    #    False 退回旧行为（只看横向距离），保留给"不要这层过滤"的用户。
    off_track_require_slip: bool = True
    # 第二重闸门用的滑移率门限。比 slip_threshold(0.15) 低一点：这里只要
    # "有可见打滑"就够确认出界，不必到打滑规则那条更严格的播报线。
    off_track_slip_min: float = 0.10

    # 打滑
    # 🔴 #J：打滑阈值三档预设（严格/标准/宽容）+ 用户微调。
    #    `slip_preset` 只是个标签（给 UI 做下拉 + 记忆用），**真正参与判据的
    #    永远是 `slip_threshold`**。选预设时 server 会把 `slip_threshold` 设回
    #    该档基线，之后用户在滑块上微调改的就是 `slip_threshold` 本身。
    #      strict   街道：任何打滑都是坏事 → 低门限
    #      standard 赛道日：默认
    #      lenient  漂移/拉力/泥地：本来就在故意滑 → 高门限，别老报
    slip_preset: str = "standard"
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
    # 🔴 原名 throttle_late_on —— #G 引入逐规则开关后 `_on` 后缀让给了
    #    布尔开关，阈值改名 `_min`（"油门低于这个值算没给油"），语义也更准。
    throttle_late_min: float = 0.30

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

    # —— 名次 / 情绪向（R3.1）——
    # 🔴 这些规则**不依赖参考圈**，所以跨车型/暖胎期之外它们照常工作。
    #    风险不在"算错"，而在"太吵" —— 名次每圈都可能变，鼓励更是纯情绪。
    #    所以下面每一条都有自己的**最小间隔**，宁可少说。
    # 少于这么多辆车就不报名次变化：两人对跑时"你追回 1 位"毫无意义。
    position_min_cars: int = 3
    # 鼓励只在**后半区**发，且参赛车数至少这么多：3 车赛的第 2 名不算"后半区"。
    encourage_min_cars: int = 6
    # 两次鼓励之间至少隔几圈。天天被鼓励的人会开始怀疑自己是不是很差。
    encourage_every_laps: int = 3
    # 拿到 P1 才算"领跑"：一个人跑时间赛时不说"你是 P1"。
    leader_min_cars: int = 2
    # 持续领跑时，隔几圈提醒一次"保持住"。
    leader_hold_every_laps: int = 5

    # —— R2.4 圈后综合建议 ——
    # True（默认）：把 lap_summary / sector_loss / fuel_range / next_focus 四条
    #   事实合并成**一句** `lap_advice`（每圈一条，云润色只调用一次）。
    # False：退回旧行为 —— 四条各自单说（保留给逐条调试 / A-B 对比用）。
    lap_advice: bool = True

    # —— #G 播报细分开关 ————————————————————————————————————
    #
    # 🔴 面板（panel.py）的分组开关是「整类静音」（说完不说的出口闸）；
    #    这里是更细的**逐规则**开关（规则根本不评估，evidence 也不产出）。
    #    两层各管各的：分组开关管"出口拦不拦"，这里的开关管"算不算"。
    #    全部默认 True = 现行为不变；id 与 panel.py GROUPS[].subs 一一对应，
    #    UI 直接拿字段名当开关 id。
    # 安全组（brake_late 与 brake_warn 相隔 1~2 秒本就该连着说，共用一个开关）
    off_track_on: bool = True
    slip_on: bool = True
    brake_on: bool = True
    shift_on: bool = True
    # 驾驶组
    apex_slow_on: bool = True
    throttle_late_on: bool = True
    # 轮胎组（一条规则产两种播报，热/凉分开开关，见 _tyre_temp）
    tyre_hot_on: bool = True
    tyre_cold_on: bool = True
    # 圈速组
    delta_on: bool = True
    projected_on: bool = True
    # 圈后组（lap_advice=True 合并句模式下，子开关决定各事实**进不进合并句**；
    #         lap_advice=False 各自单说模式下，子开关就是各自的总开关）
    lap_summary_on: bool = True
    sector_loss_on: bool = True
    fuel_range_on: bool = True
    next_focus_on: bool = True
    # 名次 / 情绪组
    position_on: bool = True
    encourage_on: bool = True
    leader_on: bool = True
    race_finish_on: bool = True


# 🔴 #J：打滑三档预设 → 各档基线阈值（见 RuleConfig.slip_preset 说明）。
#    选预设时 server 把 `slip_threshold` 设回对应基线，之后滑块微调只动
#    `slip_threshold` 本身。UI 用这份表渲染下拉 + 各档默认值。
SLIP_PRESETS: dict[str, float] = {
    "strict": 0.08,    # 街道：任何打滑都该报
    "standard": 0.15,  # 赛道日：默认
    "lenient": 0.30,   # 漂移/拉力/泥地：故意滑，高门限别老报
}


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
    # 🔴 参考圈的**速度面**与本场这辆车可比吗？（几何面永远可比）
    #
    #    False 只会出现在一种情况：手上这份参考圈是跨车型采用的历史圈，
    #    且两车圈速差超过容差（默认 10%）。此时**依赖速度的规则必须全部闭嘴** ——
    #    拿慢车的最快圈去量快车，delta 是一个恒定的 +8 秒，零信息量；
    #    弯心速度、刹车入点速度同理，都是"车慢"而不是"你慢"。
    #
    #    而 **几何面**（赛车线形状、刹车点与弯心的**位置**）跨车依然成立：
    #    T1 的刹车点还在那个地方，顶多差几米。所以 `off_track`（横向偏差）
    #    与 `throttle_late`（过弯心后 40~120 m 还没给油，纯位置判据）
    #    **不受这个标志影响**，照常工作。
    #
    #    判据在 `refindex.RefLap.pace_ok`，由引擎在"采用"那一刻算好传进来。
    #    默认 True —— 不传就是"可比"，与加这个标志之前的行为一致。
    ref_pace_ok: bool = True
    # 本场**还没跑完第一圈**（暖胎期）→ 一个字都别乱说。
    #
    # 🔴 依赖参考圈的规则不用自己判这个：引擎在暖胎期会把 `ref`/`s` 收回去，
    #    而每一条依赖它的检查开头都写着 `if c.ref is None or c.s is None: return None`，
    #    于是"把参考圈收回去"就等于"自动只播报不依赖参考圈的信息"。
    #    这个标志是留给**另一类坑**的：「上一圈成绩」（`f.last_lap_ms`）是游戏
    #    给的"最后一次冲线"值，**重开比赛后它还停在上一轮** —— 不按住它，
    #    新的一局刚发车，教练先把上一局的圈速念一遍。
    warmup: bool = False


class RuleSet:
    """无状态规则 + 有状态累积器（累积器都放在 ctx.st 里）。"""

    def __init__(self, cfg: RuleConfig | None = None,
                 narrator: "Narrator | None" = None):
        self.cfg = cfg or RuleConfig()
        # 措辞层：规则只产出 facts，最终句子交给 narrator 渲染。
        # 不传 narrator 时退化为「禁用态 narrator」→ 直接出本地模板，
        # 判断逻辑与输出都与 R2.1 完全一致（既有测试零改动）。
        self.narrator = narrator or Narrator()

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
        # 🔴 #G：逐规则开关在**调用点**过滤 —— 关掉的规则根本不评估，
        #    evidence 也不产出（与面板的"出口静音"是两层，见 RuleConfig 注释）。
        #    ⚠️ 关掉再打开的边角：_position_now 等的状态位只在真播出后推进
        #       （on_spoken），关掉期间不推进，重开时第一次名次比较可能
        #       报一个累计变化 —— 罕见操作，宁可这样也不给它偷偷记状态。
        for fn, flag in ((self._off_track, self.cfg.off_track_on),
                         (self._wheel_slip, self.cfg.slip_on),
                         (self._brake_warn, self.cfg.brake_on),
                         (self._brake_late, self.cfg.brake_on),
                         (self._apex_slow, self.cfg.apex_slow_on),
                         (self._throttle_late, self.cfg.throttle_late_on),
                         (self._shift, self.cfg.shift_on),
                         # 胎温一条规则产两种播报（过热/太凉），热/凉各自的
                         # 开关在方法内部处理 —— 这里恒放行，保持原有顺序。
                         (self._tyre_temp, True),
                         (self._delta, self.cfg.delta_on),
                         (self._projected_lap, self.cfg.projected_on)):
            if not flag:
                continue
            u = fn(c)
            if u is not None:
                out.append(u)
        # —— 圈后播报（R2.4）——
        # 默认把四条事实合并成**一句** `lap_advice`（每圈一条；云润色因此
        # 每圈最多花一次）。`lap_advice=False` 时退回旧的四条各自单说。
        # #G：四个子开关在两种模式下都生效 —— 合并句里决定事实进不进，
        #     单说模式里就是各自的总开关。
        if self.cfg.lap_advice:
            u = self._lap_debrief(c)
            if u is not None:
                out.append(u)
        else:
            for fn, flag in ((self._lap_summary, self.cfg.lap_summary_on),
                             (self._sector_loss, self.cfg.sector_loss_on),
                             (self._fuel_range, self.cfg.fuel_range_on),
                             (self._next_focus, self.cfg.next_focus_on)):
                if not flag:
                    continue
                u = fn(c)
                if u is not None:
                    out.append(u)
        # —— 名次 / 情绪向（R3.1）——
        # 🔴 独立于上面的圈后合并句：情绪不该去挤成绩/习惯那些硬信息的
        #    位置（合并句有长度预算，加鼓励就会把主体挤掉），而且要能
        #    在面板上单独关掉（分组 `mood`）—— 有人就是不想被鼓励。
        for fn, flag in ((self._position_now, self.cfg.position_on),
                         (self._encourage, self.cfg.encourage_on),
                         (self._leader, self.cfg.leader_on),
                         (self._race_finish, self.cfg.race_finish_on)):
            if not flag:
                continue
            u = fn(c)
            if u is not None:
                out.append(u)
        return out

    def on_spoken(self, spoken: list[Utterance], st: dict) -> None:
        """闸门放行后由引擎回调：**只有真念出口了**才推进规则状态位。

        🔴 为什么规则不能当场自己记账：规则产出的只是**候选**，能不能出
           闸门由闸门说了算（每 tick 只放 `max_per_tick` 条，且按优先级
           升序取）。以前 `_position_now` 一发现名次变了就把 `pos_last`
           更新成新值再返回候选，于是这条候选只在那一个 tick 里存在 ——
           被闸门 `break` 跳过就永久消失。而名次变化恰恰总发生在刹车点 /
           弯中（超车那一刻），那时驾驶指导必然同时在排队，P_LOW 必然被
           跳过 → 实测一场 16 车 6 圈的 Spa：23 条候选播出 **0** 条，
           功能等于不存在。

           改成"播出了才记账"后，没播出的下一 tick 会再提一次，直到闸门
           放行。等待期间名次若继续变化，`moved` 自动累计成净变化，所以
           不会念出过时的数字。

        🔴 为什么用 key 前缀而不是精确匹配：`leader@take` / `leader@hold`
           是两个 key 但共享同一份状态位。
        """
        for u in spoken:
            ev = u.evidence or {}
            base = u.key.split("@", 1)[0]
            if base == "position":
                st["pos_last"] = int(ev.get("position") or 0)
            elif base == "encourage":
                st["encourage_lap"] = int(ev.get("lap") or 0)
            elif base == "leader":
                st["leader_was"] = True
                st["leader_hold_lap"] = int(ev.get("lap") or 0)
            elif base == "race_finish":
                st["finish_done_laps"] = int(ev.get("laps_in_race") or 0)

    # —— 1. 出界 ———————————————————————————————————————

    def _slip_rates(self, c: Ctx) -> tuple[float, float] | None:
        """当前帧的滑移率 (sf, sr)；条件不足返回 None。

        🔴 顺带在自由滚动帧上**在线标定**轮胎半径——累积器在
            `c.st["radius"]`，被 `_wheel_slip` 与 `_off_track` 共用。
            把标定和滑移率计算并到一处，两条规则都不必各写一遍，
            也不会出现"两处半径不一致"。
        """
        cfg = self.cfg
        f = c.f
        if len(f.wheel_rads) < 4 or f.speed_ms < 3.0:
            return None
        rad = c.st["radius"]
        # 自由滚动帧才用来标定半径：横向/纵向 G 都小、油门在中间、没踩刹车
        if (abs(f.glat) < cfg.free_glat_max and abs(f.glon) < cfg.free_glon_max
                and cfg.free_thr_lo < f.throttle < cfg.free_thr_hi
                and f.brake < cfg.free_brake_max):
            v = f.speed_ms
            fl, fr, rl, rr_w = f.wheel_rads[:4]
            rad["fn"] += v * (fl + fr) / 2.0
            rad["fd"] += ((fl * fl + fr * fr) / 2.0)
            rad["rn"] += v * (rl + rr_w) / 2.0
            rad["rd"] += ((rl * rl + rr_w * rr_w) / 2.0)
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
        return sf, sr

    def _off_track(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        if c.lateral_m is None or c.f.speed_kph < cfg.min_speed_kph:
            self._hold(c.st, "off", False, c.dt)
            return None
        # 弯中放宽：赛车线在弯里切内侧，走别的线偏离十几米很正常
        slack = min(abs(c.f.glat) / max(cfg.off_track_g_ref, 1e-6), 1.0)
        thr = cfg.off_track_m * (1.0 + cfg.off_track_corner_slack * slack)
        off = c.lateral_m > thr

        # 🔴 双重确认（#I）：横向偏离参考线 **且** 轮胎确实在打滑才判"出界"。
        #    /profile 给的是最快赛车线不是赛道边界，纯按横向距离判会误杀
        #    "走了另一条线但仍在柏油上"的车，一次误报就毁掉可信度。
        #    草皮/砂石上的真实出界必然伴随轮胎打滑（抓地力骤降），所以拿
        #    打滑当第二重闸门，两者同时满足才报。算不出滑移率（缺轮速/
        #    还没标定出来）时**宁可放过**——不报比乱报强。
        slip = True
        slip_detail: tuple[float, float] | None = None
        if cfg.off_track_require_slip:
            res = self._slip_rates(c)
            if res is None:
                slip = False
            else:
                sf, sr = res
                slip = (abs(sf) > cfg.off_track_slip_min
                        or abs(sr) > cfg.off_track_slip_min)
                slip_detail = (round(sf, 3), round(sr, 3))

        trig = off and slip
        if self._hold(c.st, "off", trig, c.dt) < cfg.off_track_hold_s:
            return None
        ev: dict[str, Any] = {
            "lateral_m": round(c.lateral_m, 1),
            "threshold_m": round(thr, 1),
            "glat": round(c.f.glat, 2),
        }
        if slip_detail is not None:
            ev["slip_front"] = slip_detail[0]
            ev["slip_rear"] = slip_detail[1]
        return Utterance(
            key="off_track", text="出界了，回到赛道", priority=P_CRITICAL,
            ttl_s=2.5, short="出界", evidence=ev)

    # —— 2. 打滑 ———————————————————————————————————————

    def _wheel_slip(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        f = c.f
        res = self._slip_rates(c)
        if res is None:
            self._hold(c.st, "slip", False, c.dt)
            return None
        sf, sr = res

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
        # 🔴 跨车型时**整条**规则都停，不只是"快 N"那半句。理由不是保守，
        #    是**方向性**：`history_best` 只会挑比本场最快圈**更快**的场次
        #    （`b < best_s * 0.999`），所以跨车采用时参考车总是更快的那辆 ——
        #    它的刹车点比你这辆车该刹车的位置**更晚**。按它预告，
        #    "1.4 秒后重刹"会说到你早就该减速之后才响，是**反向的**误导。
        if not c.ref_pace_ok:
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
        # 🔴 跨车型静默：判据是 `speed_kph > z["speed_in_kph"] * 0.92`，
        #    而 `speed_in_kph` 是**参考车**入弯时的速度 —— 慢车的入弯速度
        #    比你这辆车能带的低，于是"没踩刹车而且速度高"会被判成"你晚了"，
        #    可你只是车更快。
        if not c.ref_pace_ok:
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
        # 🔴 跨车型静默：这条是**纯速度差**（参考车弯心速度 − 你的速度）。
        #    车慢 20 km/h 会让它恒为真，和"你弯心慢了"完全是两回事。
        if not c.ref_pace_ok:
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
        bad = c.f.throttle < cfg.throttle_late_min
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
            # #G：过热开关单独关 → 不报（凉的那半本来也不会同时触发）
            if not cfg.tyre_hot_on:
                return None
            i = tt.index(max(tt))
            txt = f"{names[i]}胎过热 {tt[i]:.0f}"
            return Utterance(key="tyre_hot", text=txt,
                             priority=P_NORMAL,
                             ttl_s=phrases.ttl_for(txt, P_NORMAL, "tyre_temp"),
                             short="胎温",
                             evidence={"tyre_temp_c": [round(x, 1) for x in tt]})
        # #G：太凉开关单独关 → 不报
        if not cfg.tyre_cold_on:
            return None
        # 🔴 #A 复审：工程师不能只说"太凉"，要告诉车手**怎么升温** ——
        #    GT7 里冷胎的两大解法：直线上轻拖刹车（刹车盘热量喂给胎）、
        #    走线上多左右摆动（摩擦生热）。再精简过：用户反馈一句为宜。
        txt = "轮胎太凉，轻拖刹车多摆走线，升温再推"
        return Utterance(key="tyre_cold", text=txt,
                         priority=P_NORMAL,
                         ttl_s=phrases.ttl_for(txt, P_NORMAL, "tyre_temp"),
                         short="胎温",
                         evidence={"tyre_temp_c": [round(x, 1) for x in tt]})

    # —— 9. delta ——————————————————————————————————————

    def _delta(self, c: Ctx) -> Utterance | None:
        if c.ref is None or c.s is None or c.f.lap_time_s <= 0:
            return None
        # 🔴 跨车型静默 —— 这是**最要紧**的一条。delta 是教练的主输出，
        #    跨车时它会退化成一个恒定的偏移量（"+8.00"），每天每圈都这么报，
        #    既不随时间变化、也不随你开得好坏变化 —— 零信息量，而且会让人
        #    误以为"我今天一直慢 8 秒"，进而去改一个根本不存在的毛病。
        if not c.ref_pace_ok:
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
        # 🔴 暖胎期**不能**报"上一圈成绩"：`last_lap_ms` 是游戏给的"最后一次
        #    冲线"值，重开比赛后它还停在上一轮 —— 报出来就是"新的一局刚发车，
        #    教练先念上一局的圈速"。本场没跑完一圈时，这个数一定不是本场的。
        if c.warmup:
            return None
        ms = c.f.last_lap_ms
        if not ms or ms <= 0:
            return None
        sec = ms / 1000.0
        ev: dict[str, Any] = {"last_lap_ms": round(ms, 1),
                              "lap_time_s": round(sec, 3)}
        # 🔴 跨车型时**只丢 `vs_ref_s`**，圈速本身照报 —— 圈速是游戏给的
        #    实测值，与参考圈无关，换了车也照样成立。而"比参考圈快/慢多少"
        #    是两辆车之间的比较，跨车没有意义。
        if c.ref_pace_ok and c.ref is not None and c.ref.lap_time_s > 0:
            vs = sec - c.ref.lap_time_s
            # 防御：参考圈圈速若被污染/错用，差值会跳到整圈量级
            # （如参考圈 1:00、当前圈 1:41 → 41 秒），这种"快/慢 40 秒"
            # 只会摧毁可信度，宁可不报差值也不报离谱数字。
            if abs(vs) <= max(30.0, c.ref.lap_time_s * 0.25):
                ev["vs_ref_s"] = round(vs, 3)
            ev["ref_lap_time_s"] = round(c.ref.lap_time_s, 3)
        txt = self.narrator.render("lap_summary", ev)
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
        # 🔴 跨车型静默：预测圈速 = 参考圈速 + 当前 delta，两项都建立在
        #    "参考车和你的车一样快"这个前提上。前提不成立，这个数就是编的。
        if not c.ref_pace_ok:
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
        txt = self.narrator.render("projected_lap", ev)
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
        txt = self.narrator.render("sector_loss", ev)
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
        # —— 接上「还剩几圈到终点」——
        # 只知道"油够跑 2.4 圈"是半个答案：够不够跑完**这一局**才是
        # 车手真正要据以决策的数。`laps_to_go` 为 None（总圈数未知）时
        # 保持旧说法，不猜。
        to_go = c.f.laps_to_go
        if to_go is not None:
            ev["laps_to_go"] = int(to_go)
            # 🔴 派生余量必须进 evidence：`fuel_range` 模板里"余/差 N 圈"
            #    的 N 是这一句里唯一的数字，不在 facts 里就会被数字白名单
            #    判成"编的数字"（与 `brake_warn` 的 `over_kph` 同一个坑）。
            ev["margin_laps"] = round(float(left) - float(to_go), 2)
        txt = self.narrator.render("fuel_range", ev)
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
        txt = self.narrator.render("next_focus", h)
        return Utterance(
            key=f"corner_habit@{h['label']}",
            text=txt,
            priority=P_NORMAL,
            ttl_s=phrases.ttl_for(txt, P_NORMAL, "next_focus"),
            short=f"重点 {h['label']}",
            evidence=h)

    # —— 15. 圈后综合建议（R2.4）————————————————————————————
    #
    # 把上面四条（成绩 / 最慢段 / 续航 / 习惯）的事实**合并成一句**播报。
    #
    # 🔴 判断逻辑一行没搬：`_lap_summary` / `_sector_loss` / `_fuel_range` /
    #    `_next_focus` 仍是各自负责"什么时候该说、事实是什么"的地方，本方法
    #    只是把它们**已经产出的** `evidence` 收上来、按优先级拼成一句。
    #    这样"哪个弯算习惯""油够不够"这些判断不会在这里被复制成第二份
    #    （复制 = 迟早分叉）。合并只做一件事：把四条缩短成一条。
    #
    # 🔴 为什么是"合并"而不是"再加一条"：方案 §4 的验收口径是「每圈恰 1 条
    #    建议」。四条各说各的再加一条综合 = 每圈 5 句，既超出 gate 的每圈额度，
    #    也让云润色每圈要打 4~5 次。合并后每圈恰好一次云调用（`limits.per_lap`）。

    def _lap_debrief(self, c: Ctx) -> Utterance | None:
        facts: dict[str, Any] = {}

        # 🔴 #G：四个子开关在这里生效 —— 关掉的那条事实**不进合并句**。
        #    注意 _next_focus 带"隔几圈提醒"的状态副作用：关掉期间状态机
        #    停走，重开后从头计圈 —— 关开关的人不在乎这个。
        ls = self._lap_summary(c) if self.cfg.lap_summary_on else None
        if ls is not None:
            for k in ("lap_time_s", "vs_ref_s", "ref_lap_time_s"):
                if k in ls.evidence:
                    facts[k] = ls.evidence[k]

        fu = (self._fuel_range(c)             # 续航：只在"警告区"（≤fuel_warn_laps）才有
              if self.cfg.fuel_range_on else None)
        if fu is not None:
            # 🔴 连 `laps_to_go` / `margin_laps` 一起透传：圈后综合句要能说
            #    "够不够到终点"，而余量是派生值，必须由这里带进 facts，
            #    否则云句引用它就会被数字白名单判成"编的"。
            for k in ("unit", "laps_left", "laps_to_go",
                      "margin_laps"):
                if k in fu.evidence:
                    facts[k] = fu.evidence[k]

        se = (self._sector_loss(c)            # 最慢段
              if self.cfg.sector_loss_on else None)
        if se is not None:
            facts["sector"] = se.evidence.get("sector")
            facts["loss_s"] = se.evidence.get("loss_s")

        nf = (self._next_focus(c)             # 习惯弯（含"隔几圈提醒一次"的副作用）
              if self.cfg.next_focus_on else None)
        if nf is not None:
            h = nf.evidence
            facts["focus_label"] = h.get("label")
            facts["focus_laps"] = h.get("laps")
            facts["focus_loss_s"] = h.get("median_loss_s")

        # 四条都空 = 这一圈没什么可说的（如第 1 圈残圈、且无油量/习惯）→ 静默
        if not any(k in facts for k in
                   ("lap_time_s", "laps_left", "sector", "focus_label")):
            return None

        txt = self.narrator.render("lap_advice", facts)
        if not txt:
            return None
        return Utterance(
            key="lap_advice", text=txt, priority=P_NORMAL,
            ttl_s=phrases.ttl_for(txt, P_NORMAL, "lap_advice"),
            short=txt.split("，")[0],
            evidence=facts)

    # —— 16~18. 名次与情绪向（R3.1）—————————————————————————
    #
    # 这三条是**唯一不依赖参考圈**的播报，也是唯一带情绪的一类。数据来自
    # `car.race.*`（当前名次 / 参赛车数），协议里本来就有 —— 只是此前
    # `contract.Frame.from_v1_live` 从没读它（见 #21）。
    #
    # 🔴 这一类的风险不在"算错"而在**太吵**，所以立两条纪律：
    #
    #   ① **随机必须可复现**。鼓励是"随机挑一条"，可教练是**确定性系统** ——
    #      `FileSource` 回放是核心调试手段（同一份 jsonl 必须产出同一串播报），
    #      挑词一旦吃全局 `random` 状态，回放就不可复现、测试也无从断言。
    #      所以挑哪条由 `(lap, position, num_cars)` 播种，见 `phrases._pick`。
    #
    #   ② **永不抢占驾驶指导**。三条一律 P_LOW（最低档）：名次变化再激动，
    #      也不该把"刹车晚了"挤下去。闸门排序时它们永远排最后，
    #      且受 `max_per_lap` 约束（一圈里挤不进去就不说）。
    #
    # ⚠️ 三条都**不走云润色**（不在 `phrases.RENDERERS` 里）：本地模板已经
    #    够口语，而云会把"别急"扩写成「注意补油」这类 facts 里没有的处方 ——
    #    `invented_advice` 能拦，但拦住的代价是白花一次钱和 0.5~2 s 延迟。

    def _race_ok(self, c: Ctx, min_cars: int) -> bool:
        """名次数据可用吗 —— 三条规则**共用**的这一道闸。

        🔴 为什么要单独提出来：`position` / `num_cars` 都是 u16，菜单态、时间赛、
           练习赛里 GT7 不给值（已被 `norm_u16` 归一成 0），此时**必须闭嘴**；
           刷成垃圾值（如 31847）时也不能照念。三种情况合到一处，
           免得将来新加第四条规则时漏判一种。
        """
        f = c.f
        if f.position <= 0 or f.num_cars <= 0:
            return False
        if f.num_cars > MAX_PLAUSIBLE_CARS:
            return False
        # 名次不可能超过参赛车数 —— 出现了就是数据自相矛盾，别照着念
        if f.position > f.num_cars:
            return False
        return f.num_cars >= min_cars

    def _position_now(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        f = c.f
        # 🔴 暖胎期静默：发车那一团里名次每秒都在跳，这时候报"追回 2 位"
        #    播的是随机噪声 —— 随机噪声比沉默更伤信任。
        if c.warmup:
            return None
        if not self._race_ok(c, cfg.position_min_cars):
            return None
        last = c.st.get("pos_last")
        if last is None:
            # 首见只记不播：没有"从哪来"，就谈不上"追回 / 掉了"
            c.st["pos_last"] = int(f.position)
            return None
        if int(last) == int(f.position):
            return None
        # 🔴 这里**故意不更新** `pos_last` —— 更新推迟到这句话真正播出
        #    之后（见 `RuleSet.on_spoken`）。
        #
        #    为什么：名次变化总是发生在超车那一刻，而那一刻正是刹车点 /
        #    弯中，驾驶指导必然同时在排队。闸门每 tick 只放一条
        #    （`max_per_tick=1`）且按优先级升序取，P_LOW 会被 `break`
        #    直接跳过。以前规则当场就推进了状态，于是这条话只活了一个
        #    tick 就永久消失 —— 实测一场 16 车 6 圈的 Spa：23 条候选播出
        #    0 条。不推进状态 = 下一 tick 再提一次，直到闸门放行；等待
        #    期间名次若继续变，`moved` 会自动累计成净变化。
        #
        # 🔴 `moved` 是派生值，必须写进 evidence：它是这一句里唯一的数字，
        #    不在 facts 里就会被数字白名单判成"编的"（与 `brake_warn`
        #    的 `over_kph` 同一个坑）。
        moved = int(last) - int(f.position)     # >0 = 前进了几位
        ev: dict[str, Any] = {"position": int(f.position), "moved": moved,
                              "num_cars": int(f.num_cars),
                              "lap": int(f.lap)}
        txt = phrases.position_now(ev)
        return Utterance(key="position", text=txt, priority=P_LOW,
                         ttl_s=phrases.ttl_for(txt, P_LOW, "position"),
                         short=f"P{int(f.position)}", evidence=ev)

    def _encourage(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        f = c.f
        # 只在**圈后**说：鼓励不该插在弯里。这也顺带保证了暖胎期静默 ——
        # `c.lap` 要等本场跑完一圈才有值。
        if c.warmup or c.lap is None:
            return None
        if not self._race_ok(c, cfg.encourage_min_cars):
            return None
        # 后半区判据写成 `2*pos > num_cars` 而不是 `pos > num_cars/2`：
        # 都是整数比较，但这样不引入浮点、也不用纠结整除往哪取整
        # （20 车第 10 名：20 > 20 为假 → 不算后半区；第 11 名：22 > 20 ✓）。
        if int(f.position) * 2 <= int(f.num_cars):
            return None
        # 隔几圈才鼓励一次：天天被鼓励的人会开始怀疑自己是不是很差。
        last = c.st.get("encourage_lap", -999)
        if c.lap.lap - last < cfg.encourage_every_laps:
            return None
        # 🔴 同样**不在这里记账** —— 等 `on_spoken` 确认播出后再记。
        #    否则一次竞争失败就把这次机会吃掉了，要再等 3 圈。
        ev: dict[str, Any] = {"position": int(f.position),
                              "num_cars": int(f.num_cars),
                              "lap": int(c.lap.lap)}
        txt = phrases.encourage(ev)
        return Utterance(key="encourage", text=txt, priority=P_LOW,
                         ttl_s=phrases.ttl_for(txt, P_LOW, "encourage"),
                         short="稳住", evidence=ev)

    def _leader(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        f = c.f
        if c.warmup or c.lap is None:
            return None
        if int(f.position) != 1:
            # 🔴 掉出 P1 就把"已经在领跑"的记忆清掉。不清的话，重新夺回
            #    P1 那一刻只会说"保持住"—— 而那一刻玩家想听的是"拿回来了"。
            c.st["leader_was"] = False
            return None
        if not self._race_ok(c, cfg.leader_min_cars):
            return None
        if c.st.get("leader_was"):
            # 已经领跑了 → 隔几圈提醒一次"保持住"，别每圈念
            last = c.st.get("leader_hold_lap", -999)
            if c.lap.lap - last < cfg.leader_hold_every_laps:
                return None
            mode = "hold"
        else:
            mode = "take"
        # 🔴 同样推迟到 `on_spoken`：只有真念出口了才算"已经领跑"。
        ev: dict[str, Any] = {"position": 1, "mode": mode,
                              "num_cars": int(f.num_cars),
                              "lap": int(c.lap.lap)}
        txt = phrases.leader(ev)
        return Utterance(key=f"leader@{mode}", text=txt, priority=P_LOW,
                         ttl_s=phrases.ttl_for(txt, P_LOW, "leader"),
                         short="P1", evidence=ev)

    # —— 20. 冲线名次（R3.2）————————————————————————————
    #
    # 固定圈数比赛跑完最后一圈、冲过终点的那一刻，报最终名次。
    #
    # 🔴 触发用「已完成的圈数 ≥ 本局总圈数」：`c.lap.lap` 是**刚跑完的那一圈**
    #    的编号（第 5 圈完成 → c.lap.lap == 5），而 `f.lap` 在最后一圈全程
    #    都 == laps_in_race（第五圈里它一直是 5）。若只看 `f.lap` 会从最后一圈
    #    一开始就误报；看「完成了几圈」才精确卡在冲线那一下。
    #
    # 🔴 只报一次：用 `st["finish_done_laps"]` 记这一局已报过的 laps_in_race；
    #    新一局（已完成圈数 < 总圈数）自动复位，下一局再报。
    def _race_finish(self, c: Ctx) -> Utterance | None:
        cfg = self.cfg
        f = c.f
        if c.warmup or f.laps_in_race <= 0:
            return None
        if c.lap is None or c.lap.lap < f.laps_in_race:
            # 还没跑完最后一圈（含新一局刚开始）→ 清掉上一局标记，准备重报
            c.st.pop("finish_done_laps", None)
            return None
        if not self._race_ok(c, cfg.position_min_cars):
            return None
        if c.st.get("finish_done_laps") == f.laps_in_race:
            return None  # 这一局已经报过
        ev: dict[str, Any] = {"position": int(f.position),
                              "num_cars": int(f.num_cars),
                              "laps_in_race": int(f.laps_in_race)}
        txt = phrases.race_finish(ev)
        return Utterance(key="race_finish", text=txt, priority=P_LOW,
                         ttl_s=phrases.ttl_for(txt, P_LOW, "race_finish"),
                         short=f"P{int(f.position)}", evidence=ev)
