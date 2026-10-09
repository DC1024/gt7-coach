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

# 一局比赛总圈数的**可信上限**。超过就判"不知道总圈数"，而不是照着报。
#
# 🔴 为什么要有这道闸：`laps_in_race`（packet A 的 totalLaps，int16）在
#    菜单态、时间赛、练习赛里会被刷成哨兵值（65535）或残留值。u16 归一
#    只能挡掉 65535 —— 挡不住"刷成了 30000"这种。没有上界的话，
#    续航播报会念出「油差 29997 圈」，而那一句**每个数字都在 facts 里**，
#    白名单反而是通的。这类"数字合法、语义荒唐"的错最容易被漏掉。
#    GT7 里一局超过 500 圈是不存在的，取 500 足够宽松。
MAX_PLAUSIBLE_LAPS = 500

# 参赛车数的**可信上限**。超过就判"不知道名次"，而不是照着报。
#
# 🔴 与上面同一类问题，但多一层：名次是**要念出口**的。`num_cars` 一旦被刷成
#    垃圾值，播报会变成「还在 P31847，别急」—— 数字本身合法、白名单拦不住，
#    而这句话一旦念出来，玩家对教练的信任就没了。
#    GT7 正赛发车位最多 20 个，取 40 是给自定义/特殊赛事留的余量。
MAX_PLAUSIBLE_CARS = 40

# 优先级：数字越小越紧急。闸门按它排序，紧急的可以抢占冷却。
P_CRITICAL = 0   # 出界 / 打滑 —— 可以打断一切
P_HIGH = 1       # 刹车点 / 换挡
P_NORMAL = 2     # 弯心速度 / 给油时机 / 胎温
P_LOW = 3        # delta 播报 / 圈后总结


def norm_u16(v: Any) -> int:
    """u16 遥测字段归一：哨兵 65535、负数、非数字一律归 **0**（0 = 未知）。

    🔴 三个地方都要同一个口径，所以提成这一个函数：
        `Frame.from_v1_live`（解析 `/live`）、`source.FileSource`（解析 jsonl）、
        `refindex.RefLap`（解析缓存里的车型码）。各写一份的话，
        "第 65535 名"只在其中一处被挡住 —— 而从哪个口子漏进来都一样糟。

    ⚠️ 0 一律表示"**未知**"，不是"第 0 名 / 0 辆车"。这些字段在菜单态、
        时间赛、练习赛里不会给值，而 0 恰好是个完全合法的整数 ——
        判"有没有"必须用 `> 0`，别用 `is not None`。
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return 0
    i = int(v)
    return i if 0 < i < 0xFFFF else 0


def car_verdict(a_code: Any, a_name: Any,
                b_code: Any, b_name: Any) -> str:
    """这两者（两个场次 / 两份参考圈）是同一辆车吗？

    返回 `"same_car"` | `"cross_car"` | `"unknown"`。

    🔴 优先比 **car_code（数字车型码）**，两边都有值就直接比数字；拿不到才
       退回比 `car_name`。原因不是"数字更快"，而是**字符串这条路径有个洞**：
       车型名要过一道 `cars.csv` 查表，表没命中时本场和候选场都拿到空串 →
       判不出差别 → 于是静默跨车采用，连条日志都没有。
       数字相等就是同一辆车，没有这道中间环节。
       （`car_code` 要 Dash 的场次列表带上它才有，见仪表盘 `_first_car_code`；
        没有时退化成按名字比 —— 比"完全不判"强，但那道洞还在。）

    ⚠️ `"unknown"` 是**三态里的第三态**，不是"当成同车"。判不出就老实说判不出，
       由调用方按"性能窗口"兜底（见 `refindex.RefLap.pace_ok`）。
    """
    ca = norm_u16(a_code)
    cb = norm_u16(b_code)
    if ca and cb:
        return "same_car" if ca == cb else "cross_car"
    na = str(a_name or "").strip()
    nb = str(b_name or "").strip()
    if na and nb:
        return "same_car" if na == nb else "cross_car"
    return "unknown"


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
    # —— 比赛信息（Dash 的 `car.race.*`）——
    # 🔴 0 一律表示"**未知**"，不是"第 0 名 / 0 辆车"：菜单态、时间赛、
    #    练习赛里这些字段不会给值，而 0 又恰好是个合法整数 ——
    #    判"有没有"必须用 `> 0`，别用 `is not None`。
    position: int = 0                # 当前名次（1-based；比赛进行中随排名变）
    num_cars: int = 0                # 参赛车数
    laps_in_race: int = 0            # 本局总圈数（0 = 不限圈/未知）
    car_code: int = 0                # 车型码（数字，比车型名字符串可靠）
    connected: bool = True

    @property
    def speed_ms(self) -> float:
        return self.speed_kph / 3.6

    @property
    def laps_to_go(self) -> int | None:
        """到终点还剩几圈（**含当前这一圈**）；总圈数未知时为 None。

        🔴 含当前圈是与"油够跑几圈"对齐的唯一口径：
           10 圈赛跑在第 3 圈上，还得跑 3~10 共 8 圈，
           而 `laps_in_race - lap` 只会给 7 —— 少算一圈，
           恰好是"油够 7.5 圈但差 0.5 圈"这种最要命的判断上出错。

        ⚠️ None 在这里有两层意思（**菜单态** 与 **总圈数未知/离谱**），
           调用方不需要区分 —— 两种情况下都该"别提终点"。
        """
        if self.lap <= 0:
            return None
        if not (0 < self.laps_in_race <= MAX_PLAUSIBLE_LAPS):
            return None
        return max(0, self.laps_in_race - self.lap + 1)

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
        race = car.get("race") or {}

        def _num(v, default=0.0):
            return float(v) if isinstance(v, (int, float)) else default

        # `_int` 用契约层的统一口径（见 `norm_u16`）：Dash 会把 65535 这类
        # u16 哨兵归一（见其 `_U16_FIELDS`），但老版本或代理可能漏了 ——
        # 这里再兜一道，免得"第 65535 名"漏进播报。
        _int = norm_u16

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
            # 🔴 `race.grid_position` 名字带 grid 但**不是**发车位：比赛进行中
            #    它随排名实时变，真正的发车位是 `grid_start`。取错会得到
            #    "整场名次不动"的假象。
            position=_int(race.get("grid_position")),
            num_cars=_int(race.get("num_cars")),
            laps_in_race=_int(tm.get("laps_in_race")),
            car_code=_int(car.get("car_code")),
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
    # 语音专用串：把 `text` 里的阿拉伯数字逐位中文化（54→五四，更贴近真实
    # 无线电播报）。屏幕显示仍用 `text`（保留原样数字便于扫读）；旧版教练/
    # 未开启时该字段为 None，消费方应退回 `text`。见 `phrases.spell_digits`。
    speech: str | None = None
    # R3 云 TTS：这句的合成音频地址（相对路径，如 /api/v1/coach/tts/<hash>.mp3）。
    # 🔴 None = **还没有**，不是错误 —— 云合成要 0.5~2 s 的后台时间，消费方
    #    应当先用浏览器 `speechSynthesis` 顶上（或短暂等待后再取），
    #    绝不要因为它是 None 就不播。A 档（出界/打滑/刹车点/换挡）永远为 None。
    tts_url: str | None = None

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
            "speech": self.speech,
            "tts_url": self.tts_url,
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
