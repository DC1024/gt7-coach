# -*- coding: utf-8 -*-
"""
闸门 —— 决定「这条话到底要不要说出口」。
==========================================

一句话概括原则：**不打扰 优先于 多说话**。一个话痨副驾比没有副驾更糟。

四道闸（全部在 `Gate.filter` 里，顺序即优先级）：

1. **同类冷却**：同一个 key 至少隔 `cooldown_same_s` 才允许再说
2. **跨类冷却**：任意两条话之间至少隔 `cooldown_any_s`（P_CRITICAL 可抢）
3. **每圈上限**：一圈最多 `max_per_lap` 句（P_CRITICAL 单独计数；
   `own_quota_groups` 里的情绪向分组另走 `max_per_lap_low` 的小额度）
4. **只报新信息**：同一个 key 每圈只报一次 —— "刹车晚了 12 米"连续三圈
   一模一样地播报，等于没有信息量，只会让人想关掉

外加一条**弯中禁言**：|G| 大时只念 `short`（≤ 4 字），长句憋着不说 ——
因为玩家那时手上没空听句子，只来得及处理一个词。

再加一道**面板开关**（用户自选）：`GateConfig.muted` 里的内容分组整类拦下，
对应"面板上关掉某个开关"。它在四道闸**之前**生效（先过滤用户不想听的，
再在剩下的里做冷却/限量竞争），单独计进 `st["muted"]`。分组口径见 `panel.py`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contract import P_CRITICAL, Utterance
from .panel import group_of_key


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
    # 🔴 情绪向播报（名次 / 鼓励 / 领跑）单独走一个**更小**的配额，
    #    不占上面那条 6 条/圈的额度。
    #    理由：这类话按设计定级 P_LOW（永不抢占驾驶指导），如果和刹车点/
    #    弯心那些硬信息共用 6 条额度，就会被挤到一粒不剩 —— 实测一场
    #    16 车 6 圈的 Spa：23 条候选播出 0 条。给它自己的小额度，
    #    既保证说得出，又保证不会反过来把成绩/习惯那些挤掉。
    max_per_lap_low: int = 2
    # 哪些面板分组走上面那个 `max_per_lap_low`（分组口径见 panel.GROUPS）。
    # 不在名单里的一律走常规配额 —— 将来新加情绪类分组时在这里登记即可。
    own_quota_groups: tuple[str, ...] = ("mood",)
    g_loud: float = 0.8               # |G| 超过它 → 只念 short
    max_per_tick: int = 1             # 一次 tick 最多说几句（默认 1）
    # 面板上被用户**关掉**的内容分组（见 panel.GROUPS）。
    # 空 = 全开（默认，与加这个功能之前完全一致）。
    muted: tuple[str, ...] = ()

    def enabled(self, u: Utterance) -> bool:
        """这条该不该放行 —— 面板开关的落点。

        关掉一个分组 = 整类不出闸门；**没被任何分组认领的 key 永远放行**
        （见 panel 模块头：宁可多播，也不要被一个没登记的开关悄悄吞掉）。
        """
        return not self.muted or group_of_key(u.key) not in self.muted


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
            "spoken_low_lap": 0,
            "keys_lap": set(),
            "dropped": 0,      # 轮到了但被冷却/配额判掉
            "skipped": 0,      # 排队没轮上（max_per_tick 用尽）
            "muted": 0,        # 被面板开关关掉（与"闸门竞争掉"分开计数）
        }

    def _roll_lap(self, st: dict, lap: int) -> None:
        if st.get("lap") != lap:
            st["lap"] = lap
            st["spoken_lap"] = 0
            st["spoken_critical_lap"] = 0
            st["spoken_low_lap"] = 0
            st["keys_lap"] = set()

    # —— 主逻辑 ————————————————————————————————————————

    def filter(self, cands: list[Utterance], now: float, lap: int,
               g_mag: float, st: dict) -> list[Utterance]:
        """从候选里挑出真正要说的话（已按优先级排序、已应用弯中禁言）。"""
        self._roll_lap(st, lap)
        cfg = self.cfg
        # —— 面板开关：被用户关掉的分组，整类在出口拦下 ——
        # 放在最前面（排序/冷却之前）：这些不是"竞争失败"，而是"用户不想听"，
        # 单独计进 st["muted"]，别和 dropped 混在一起（排障时两者含义不同）。
        if cfg.muted:
            kept: list[Utterance] = []
            for u in cands:
                if cfg.enabled(u):
                    kept.append(u)
                else:
                    st["muted"] += 1
            cands = kept
        out: list[Utterance] = []
        # 排序：优先级升序（0 最急）；同级按 key 稳定排序，
        # 避免 dict/集合迭代顺序让"这次说哪句"变得不可复现。
        for u in sorted(cands, key=lambda x: (x.priority, x.key)):
            if len(out) >= cfg.max_per_tick:
                # 🔴 排队没轮上**也要计数**。以前这一支只 `break` 不加计数，
                #    于是排障时看到「闸门丢掉 0 条」就误判成"闸门什么都没拦"
                #    —— 实际上被跳过的候选一条都没少。名次播报那次正是被
                #    这个 0 骗过去的：23 条候选全被跳过，报告却写着 0。
                st["skipped"] += 1
                continue
            critical = u.priority <= P_CRITICAL
            # 情绪向分组走自己的小额度，其余非紧急走常规额度。
            # 只是**配额分账**，不改变排序 —— P_LOW 依然排在最后，
            # 依然要等跨类冷却，所以不会在刹车点抢麦。
            low = (not critical
                   and group_of_key(u.key) in cfg.own_quota_groups)
            if not critical:
                if u.key in st["keys_lap"]:
                    st["dropped"] += 1
                    continue
                if low:
                    if st["spoken_low_lap"] >= cfg.max_per_lap_low:
                        st["dropped"] += 1
                        continue
                elif st["spoken_lap"] >= cfg.max_per_lap:
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
            elif low:
                st["spoken_low_lap"] += 1
            else:
                st["spoken_lap"] += 1
        return out
