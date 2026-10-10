# -*- coding: utf-8 -*-
"""
播报开关分组 —— 「面板」上那排开关的**唯一口径**。
====================================================

用户要能自由选择「什么内容播报、什么不播报」。这里把规则产出的各种 key
归成几个**开车的人看得懂**的分组（安全 / 驾驶 / 轮胎 / 圈速 / 圈后建议），
面板上一组一个开关；闸门 `Gate.filter` 在出口处按 `GateConfig.muted`
把被关掉的分组整类拦下。

🔴 分组用**前缀匹配**：规则 key 大多带位置后缀（`brake_late@<刹车区>`、
   `apex_slow@<弯心>`、`corner_habit@T1`），按 key 全等匹配会漏掉它们。
   前缀表就是唯一事实源 —— 加新规则时在这里认领分组，忘了认领的 key
   默认**永远可播**（宁可多播一条，也不要被一个没登记的开关悄悄吞掉）。

🔴 分组只做**「说完不说」**，不改变任何判断/数值：被关掉只是这一条不出闸门，
   规则照常评估、evidence 照常进 state（诊断时仍能看到"本来会说什么"）。
"""

from __future__ import annotations

from typing import Any

# 顺序即面板上的显示顺序。`advice` 是给 UI 的提示（安全那组不建议关）。
#
# 🔴 #G：每个分组下的 `subs` 是**逐规则细分开关**（id 直接用 RuleConfig 的
#    字段名，server 据此读写 `engine.rules.cfg`）。分组开关管「整类出口静音」，
#    子开关管「这条规则算不算」—— 两层各管各的（见 rules.py RuleConfig 注释）。
#    子开关列表只做**展示口径**：label 给 UI、id 给读写映射；改子开关的
#    POST 走 /api/v1/coach/config 的 rules 节（布尔白名单），不走 panel 接口。
GROUPS: list[dict[str, Any]] = [
    {
        "id": "safety", "label": "安全告警", "advice": "不建议关闭",
        "desc": "出界 / 打滑 / 刹车晚了 / 刹车区预告 / 换挡",
        "prefixes": ("off_track", "slip_", "brake_late", "brake_warn",
                     "shift"),
        "subs": [
            {"id": "off_track_on", "label": "出界"},
            {"id": "slip_on", "label": "打滑"},
            {"id": "brake_on", "label": "刹车（预警+晚了）"},
            {"id": "shift_on", "label": "换挡"},
        ],
    },
    {
        "id": "driving", "label": "驾驶指导",
        "desc": "弯心速度偏慢 / 出弯给油太晚",
        "prefixes": ("apex_slow", "throttle_late"),
        "subs": [
            {"id": "apex_slow_on", "label": "弯心偏慢"},
            {"id": "throttle_late_on", "label": "出弯给油晚"},
        ],
    },
    {
        "id": "tyres", "label": "轮胎温度",
        "desc": "胎温过高 / 过低",
        "prefixes": ("tyre_",),
        "subs": [
            {"id": "tyre_hot_on", "label": "胎温过高"},
            {"id": "tyre_cold_on", "label": "轮胎太凉"},
        ],
    },
    {
        "id": "pace", "label": "圈速提示",
        "desc": "实时 delta / 预测圈速",
        "prefixes": ("delta", "projected_lap"),
        "subs": [
            {"id": "delta_on", "label": "实时 delta"},
            {"id": "projected_on", "label": "预测圈速"},
        ],
    },
    {
        "id": "debrief", "label": "圈后综合建议",
        "desc": "成绩 / 最慢段 / 续航 / 习惯（R2.4 合并句）",
        "prefixes": ("lap_advice", "lap_summary", "sector_loss",
                     "fuel_range", "next_focus", "corner_habit"),
        "subs": [
            {"id": "lap_advice", "label": "合并成一句"},
            {"id": "lap_summary_on", "label": "上一圈成绩"},
            {"id": "sector_loss_on", "label": "最慢分段"},
            {"id": "fuel_range_on", "label": "续航提醒"},
            {"id": "next_focus_on", "label": "习惯弯提醒"},
        ],
    },
    {
        # 🔴 单独一组而不是并进 `debrief`：有人**只想关掉鼓励**，不想连
        #    "这圈 1:32.412、T3 连续 3 圈慢 0.42" 一起关掉。情绪类和事实类
        #    不是一回事，合并了就等于没得选。
        "id": "mood", "label": "名次与情绪",
        "desc": "名次变化 / 后半区鼓励 / 领跑提醒（R3.1）",
        "prefixes": ("position", "encourage", "leader", "race_finish"),
        "subs": [
            {"id": "position_on", "label": "名次变化"},
            {"id": "encourage_on", "label": "鼓励"},
            {"id": "leader_on", "label": "领跑提醒"},
            {"id": "race_finish_on", "label": "冲线名次"},
        ],
    },
]


def known_ids() -> set[str]:
    """所有合法分组 id（`GateConfig.muted` / API 的校验口径）。"""
    return {g["id"] for g in GROUPS}


def known_sub_ids() -> set[str]:
    """所有合法子开关 id（= RuleConfig 字段名，#G）。"""
    out: set[str] = set()
    for g in GROUPS:
        for s in g.get("subs", ()):      # type: ignore[union-attr]
            out.add(s["id"])
    return out


def get_sub(cfg: Any, sub_id: str) -> bool:
    """从 RuleConfig 读一个子开关的当前值；拿不到就当 True（默认全开）。"""
    v = getattr(cfg, sub_id, True)
    return bool(v)


def group_of_key(key: str) -> str | None:
    """规则 key → 分组 id。没认领的 key 返回 None（= 永远可播、不可静音）。"""
    if not key:
        return None
    for g in GROUPS:
        if key.startswith(tuple(g["prefixes"])):
            return g["id"]
    return None


def normalize_muted(items: Any) -> tuple[str, ...]:
    """把外部传入的「静音分组」规整成合法 tuple。

    - 只保留已知 id，去重、排序（保证幂等与可比较）；
    - 非 list/tuple → 抛 ValueError（宁可 400，也不要静默把开关设错）。
    """
    if items is None:
        return ()
    if not isinstance(items, (list, tuple)):
        raise ValueError("muted 必须是字符串数组")
    ids = known_ids()
    bad = [x for x in items if not isinstance(x, str) or x not in ids]
    if bad:
        raise ValueError(f"未知分组 {bad}（可用：{sorted(ids)}）")
    return tuple(sorted(set(items)))


def panel_state(muted: Any, recent_keys: list[str] | None = None,
                rules_cfg: Any = None
                ) -> dict[str, Any]:
    """给面板/接口用的快照：每个分组的当前开关 + 最近播报计数。

    #G：`rules_cfg`（RuleConfig）传入时，每组附带 `subs`（细分开关的
    当前值）；不传（老调用方）时 subs 里的 on 一律 True —— 与"默认全开"
    语义一致，老调用方零感知。
    """
    mset = set(muted or ())
    counts: dict[str, int] = {}
    for k in (recent_keys or []):
        gid = group_of_key(k)
        if gid:
            counts[gid] = counts.get(gid, 0) + 1
    return {
        "groups": [
            {"id": g["id"], "label": g["label"], "desc": g["desc"],
             "advice": g.get("advice"), "muted": g["id"] in mset,
             "recent": counts.get(g["id"], 0),
             "subs": [{"id": s["id"], "label": s["label"],
                       "on": get_sub(rules_cfg, s["id"])}
                      for s in g.get("subs", ())]}
            for g in GROUPS
        ],
        "muted": sorted(mset),
    }
