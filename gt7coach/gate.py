# -*- coding: utf-8 -*-
"""
闸门 —— 决定「这条话到底要不要说出口」。
==========================================

一句话概括原则：**不打扰 优先于 多说话**。一个话痨副驾比没有副驾更糟。

四道闸（全部在 `Gate.filter` 里，顺序即优先级）：

1. **同类冷却**：同一个 key 至少隔 `cooldown_same_s` 才允许再说
2. **跨类冷却**：任意两条话之间至少隔 `cooldown_any_s`（P_CRITICAL 可抢）
3. **每圈上限**：一圈最多 `max_per_lap` 句（P_CRITICAL 单独计数）
4. **只报新信息**：同一个 key 每圈只报一次 —— "刹车晚了 12 米"连续三圈
   一模一样地播报，等于没有信息量，只会让人想关掉

外加一条**弯中禁言**：|G| 大时只念 `short`（≤ 4 字），长句憋着不说 ——
因为玩家那时手上没空听句子，只来得及处理一个词。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contract import P_CRITICAL, Utterance


@dataclass
class GateConfig:
    cooldown_same_s: float = 20.0     # 同类最小间隔
    cooldown_any_s: float = 6.0       # 跨类最小间隔
    # 更紧急的话允许插队的间隔。见 Gate.filter 里的说明 ——
    # 「刹车区预告」和「你刹晚了」相隔 1~2 秒，用 6 秒的跨类冷却会把后者
    # 整个吃掉，而那正是最该说的一句。
    cooldown_urgent_s: float = 2.0
    max_per_lap: int = 6              # 每圈非紧急播报上限
    max_critical_per_lap: int = 12    # 紧急播报单独计数，不占上面那个额度
    g_loud: float = 0.8               # |G| 超过它 → 只念 short
    max_per_tick: int = 1             # 一次 tick 最多说几句（默认 1）

    def enabled(self, u: Utterance) -> bool:  # pragma: no cover - 便于将来扩展
        return True


class Gate:
    """有状态的闸门。跨 tick 保留冷却与计数，状态本身是普通 dict。"""

    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()

    # —— 状态初始化 ————————————————————————————————————

    @staticmethod
    def fresh_state() -> dict:
        return {
            "last_any": -1e9,
            "last_any_prio": 99,
            "last_by_key": {},
            "lap": None,
            "spoken_lap": 0,
            "spoken_critical_lap": 0,
            "keys_lap": set(),
            "dropped": 0,
        }

    def _roll_lap(self, st: dict, lap: int) -> None:
        if st.get("lap") != lap:
            st["lap"] = lap
            st["spoken_lap"] = 0
            st["spoken_critical_lap"] = 0
            st["keys_lap"] = set()

    # —— 主逻辑 ————————————————————————————————————————

    def filter(self, cands: list[Utterance], now: float, lap: int,
               g_mag: float, st: dict) -> list[Utterance]:
        """从候选里挑出真正要说的话（已按优先级排序、已应用弯中禁言）。"""
        self._roll_lap(st, lap)
        cfg = self.cfg
        out: list[Utterance] = []
        # 排序：优先级升序（0 最急）；同级按 key 稳定排序，
        # 避免 dict/集合迭代顺序让"这次说哪句"变得不可复现。
        for u in sorted(cands, key=lambda x: (x.priority, x.key)):
            if len(out) >= cfg.max_per_tick:
                break
            critical = u.priority <= P_CRITICAL
            if not critical:
                if u.key in st["keys_lap"]:
                    st["dropped"] += 1
                    continue
                if st["spoken_lap"] >= cfg.max_per_lap:
                    st["dropped"] += 1
                    continue
            else:
                if st["spoken_critical_lap"] >= cfg.max_critical_per_lap:
                    st["dropped"] += 1
                    continue
            if now - st["last_by_key"].get(u.key, -1e9) < cfg.cooldown_same_s:
                st["dropped"] += 1
                continue
            # 跨类冷却。三档：
            #   · 紧急（P0）：完全不受跨类冷却约束，随时插队
            #   · 比刚说过的那句更紧急：只用 cooldown_urgent_s 的短间隔
            #   · 其余：走完整的 cooldown_any_s
            # 中档是为了解决一个真实翻车：「1.4 秒后重刹区」刚说完，
            # 1.2 秒后「刹车晚了 12 米」被 6 秒跨类冷却吃掉 —— 而那正是
            # 最该说的那一句（预警之后立刻纠错）。优先级本来就是为
            # 「谁能让谁闭嘴」定的，只用它排序、不用它抢占是白定。
            if not critical:
                gap = now - st["last_any"]
                limit = (cfg.cooldown_urgent_s
                         if u.priority < st.get("last_any_prio", 99)
                         else cfg.cooldown_any_s)
                if gap < limit:
                    st["dropped"] += 1
                    continue

            # 弯中禁言：换成短句。短句为空就整条丢掉（宁可不说，也别念长句）
            if g_mag >= cfg.g_loud:
                if not u.short or u.short == u.text:
                    st["dropped"] += 1
                    continue
                u = Utterance(key=u.key, text=u.short, priority=u.priority,
                              ttl_s=u.ttl_s, short=u.short,
                              evidence=u.evidence)

            out.append(u)
            st["last_any"] = now
            st["last_any_prio"] = u.priority
            st["last_by_key"][u.key] = now
            st["keys_lap"].add(u.key)
            if critical:
                st["spoken_critical_lap"] += 1
            else:
                st["spoken_lap"] += 1
        return out
