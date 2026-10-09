# -*- coding: utf-8 -*-
"""
对外契约层 —— 两个仓库之间**唯一**的耦合面。
=================================================

GT7 Dash（仪表盘）与 GT7 Coach（赛道工程师）是两个独立仓库，
靠 HTTP 上的这份结构通信。这里定义的一帧、一句话、一份状态，
就是全部约定。

🔴 改动规则（与 GT7 Dash 的 API v1 同一套纪律）
--------------------------------------------
字段名与单位**只加不改**：
  - 新增字段 → 随便加，老消费方不受影响
  - 改名 / 改单位 / 删字段 → 必须升 `COACH_API_VERSION`

为什么要单独一个文件：
    把「怎么解析 Dash 的 JSON」和「引擎内部用什么结构」分开。
    Dash 改字段名时只动 `Frame.from_v1_live()` 一处，
    引擎、规则、测试全都不用碰。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# 契约版本。破坏性改动必须 +1。
COACH_API_VERSION = 1

# 优先级：数字越小越紧急。闸门按它排序，紧急的可以抢占冷却。
P_CRITICAL = 0   # 出界 / 打滑 —— 可以打断一切
P_HIGH = 1       # 刹车点 / 换挡
P_NORMAL = 2     # 弯心速度 / 给油时机 / 胎温
P_LOW = 3        # delta 播报 / 圈后总结


def has_coords(x: float, y: float, z: float) -> bool:
    """
    坐标是否有效。

    GT7 有些场次不开坐标（has_coords=false），三个分量会恒为 0。
    单点落在原点几乎不可能，但整圈都是 0 就一定是没坐标 ——
    调用方应当在**整圈**层面判断，这里只做单帧的廉价排除。
    """
    return abs(x) + abs(y) + abs(z) > 1e-6


@dataclass(frozen=True)
class Frame:
    """
    一帧遥测的规范视图（与数据源无关）。

    🔴 时间有两个，别混：
        `t`         —— 单调墙上时钟（秒）。用于冷却、闸门、间隔判断。
        `lap_time_s`—— 本圈已用时（游戏给的）。用于与参考圈算 delta。
    """

    t: float = 0.0                  # 单调时钟
    lap_time_s: float = 0.0         # 本圈已用时（圈首 ≈ 0）
    last_lap_ms: float | None = None   # 上一圈圈速（毫秒）；None = 还没跑完一圈
    speed_kph: float = 0.0
    rpm: float = 0.0
    max_rpm: float = 0.0            # 换挡灯上限（0 = 该帧没给）
    gear: int = 0
    throttle: float = 0.0           # 0~1
    brake: float = 0.0              # 0~1
    lap: int = 0                    # 已归一（菜单态 0xFFFF → 0）
    x: float = 0.0
    y: float = 0.0                  # 高度
    z: float = 0.0
    glat: float = 0.0               # 横向 G（正 = 左转，已验证）
    glon: float = 0.0               # 纵向 G
    tyre_temp: tuple[float, ...] = ()
    wheel_rads: tuple[float, ...] = ()   # 四轮角速度 rad/s（顺序 FL,FR,RL,RR）
    # 剩余能量：油车是百分比 0~100，纯电车是剩余电量 kWh（两者的语义都是
    # 「还剩多少」。要算「还能跑几圈」必须配合 powertrain 判断口径）。
    fuel_pct: float = 0.0
    fuel_capacity_l: float = 0.0
    powertrain: str = ""             # "fuel" | "electric"
    connected: bool = True

    @property
    def speed_ms(self) -> float:
        return self.speed_kph / 3.6

    @property
    def coords_ok(self) -> bool:
        return has_coords(self.x, self.y, self.z)

    @property
    def g_mag(self) -> float:
        return (self.glat * self.glat + self.glon * self.glon) ** 0.5

    # —— 适配器：GT7 Dash `GET /api/v1/live` → Frame ——
    @staticmethod
    def from_v1_live(d: dict[str, Any], t: float) -> "Frame":
        """
        解析 GT7 Dash 的 `/api/v1/live`。

        Dash 那边字段缺失时（比如没连上 PS5）一律给 0 / 空，
        这里不做抛错 —— 引擎要靠「坐标全 0」自己判断能不能建参考圈。
        """
        car = d.get("car") or {}
        # 🔴 `timing` 是**顶层**字段，不在 car 里。写成 car.get("current_lap_time_s")
        #    会静默拿到 0 —— 而 0 是个完全合法的值（圈首），看不出错了。
        tm = d.get("timing") or {}
        pos = car.get("position_m") or {}
        g = car.get("g_force") or {}
        shift = car.get("shift_alert") or {}

        def _num(v, default=0.0):
            return float(v) if isinstance(v, (int, float)) else default

        lap = tm.get("current_lap", car.get("current_lap", 0))
        if not isinstance(lap, int) or lap < 0 or lap >= 0xFFFF:
            lap = 0

        tt = car.get("tyre_temp_c") or []
        # 旧键 wheel_rev_per_s 名字是错的（单位是 rad/s 不是转/秒），
        # Dash 保留了它但标废弃。两个键取值相同，优先用新键。
        wr = car.get("wheel_rad_per_s") or car.get("wheel_rev_per_s") or []

        return Frame(
            t=t,
            lap_time_s=_num(tm.get("current_lap_time_s")),
            last_lap_ms=(float(tm["last_lap_ms"])
                         if isinstance(tm.get("last_lap_ms"), (int, float))
                         else None),
            speed_kph=_num(car.get("speed_kph")),
            rpm=_num(car.get("rpm")),
            max_rpm=_num(shift.get("max_rpm")),
            gear=int(car.get("gear") or 0),
            throttle=_num(car.get("throttle")),
            brake=_num(car.get("brake")),
            lap=lap,
            x=_num(pos.get("x")),
            y=_num(pos.get("y")),
            z=_num(pos.get("z")),
            glat=_num(g.get("lateral")),
            glon=_num(g.get("longitudinal")),
            tyre_temp=tuple(float(v) for v in tt) if isinstance(tt, list) else (),
            wheel_rads=tuple(float(v) for v in wr) if isinstance(wr, list) else (),
            fuel_pct=_num(car.get("fuel_pct")),
            fuel_capacity_l=_num(car.get("fuel_capacity_l")),
            powertrain=str(car.get("powertrain") or ""),
            connected=bool(d.get("connected")),
        )


@dataclass
class Utterance:
    """
    一句「该说的话」。

    `key`   —— 规则 id。冷却与「每圈只报一次」都按它算，必须稳定。
    `short` —— 弯中短句（≤4 字）。G 大时只说这个，不念长句。
    `ttl_s` —— 有效期。过了还没说得出口（一直在大 G 里）就作废，
               免得攒到出弯再播一条 3 秒前的旧消息。
    """

    key: str
    text: str
    priority: int = P_NORMAL
    ttl_s: float = 4.0
    short: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.short:
            self.short = self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "text": self.text,
            "short": self.short,
            "priority": self.priority,
            "ttl_s": round(self.ttl_s, 2),
            "evidence": self.evidence,
        }


@dataclass
class CoachState:
    """
    `/api/v1/coach/state` 的响应体 —— 仪表盘消费的就是这个。
    """

    api_version: int = COACH_API_VERSION
    connected: bool = False
    # 参考圈是否就绪。False 时下面的 s/delta/next_brake 全是 None，
    # 而且规则只保留不依赖参考圈的那几条（出界/换挡/胎温/打滑）。
    ref_ready: bool = False
    ref_lap: int | None = None
    ref_len_m: float | None = None
    lap: int = 0
    # 本圈已跑距离（沿参考圈折线的弧长，米）
    s_m: float | None = None
    # 相对参考圈的时间差（秒）。负 = 比参考圈快。
    delta_s: float | None = None
    # 参考圈在当前位置的速度（km/h）
    ref_speed_kph: float | None = None
    # 距下一个刹车入点：米 / 秒
    next_brake_m: float | None = None
    next_brake_s: float | None = None
    # —— 本地统计（全部零网络，自己从实时帧算）——
    projected_lap_s: float | None = None   # 按当前 delta 预测的最终圈速
    last_lap: dict[str, Any] | None = None  # 刚跑完那圈的 {lap, lap_time_s, sectors}
    theory_best_s: float | None = None     # 本场各段最好值之和
    potential_gain_s: float | None = None  # 实际最快 − 理论最快
    fuel_per_lap: float | None = None      # 每圈消耗（油 % / 电 kWh）
    fuel_laps_left: float | None = None
    # 这一 tick 决定要说的话（已过闸门）。通常 0 或 1 条。
    say: list[Utterance] = field(default_factory=list)
    # 最近播报（供 UI 显示历史，最多 20 条，新的在前）
    spoken: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["say"] = [u.to_dict() for u in self.say]
        return d
