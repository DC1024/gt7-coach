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
GROUPS: list[dict[str, Any]] = [
    {
        "id": "safety", "label": "安全告警", "advice": "不建议关闭",
        "desc": "出界 / 打滑 / 刹车晚了 / 刹车区预告 / 换挡",
        "prefixes": ("off_track", "slip_", "brake_late", "brake_warn",
                     "shift"),
    },
    {
        "id": "driving", "label": "驾驶指导",
        "desc": "弯心速度偏慢 / 出弯给油太晚",
        "prefixes": ("apex_slow", "throttle_late"),
    },
    {
        "id": "tyres", "label": "轮胎温度",
        "desc": "胎温过高 / 过低",
        "prefixes": ("tyre_",),
    },
    {
        "id": "pace", "label": "圈速提示",
        "desc": "实时 delta / 预测圈速",
        "prefixes": ("delta", "projected_lap"),
    },
    {
        "id": "debrief", "label": "圈后综合建议",
        "desc": "成绩 / 最慢段 / 续航 / 习惯（R2.4 合并句）",
        "prefixes": ("lap_advice", "lap_summary", "sector_loss",
                     "fuel_range", "next_focus", "corner_habit"),
    },
]


def known_ids() -> set[str]:
    """所有合法分组 id（`GateConfig.muted` / API 的校验口径）。"""
    return {g["id"] for g in GROUPS}


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


def panel_state(muted: Any, recent_keys: list[str] | None = None
                ) -> dict[str, Any]:
    """给面板/接口用的快照：每个分组的当前开关 + 最近播报计数。"""
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
             "recent": counts.get(g["id"], 0)}
            for g in GROUPS
        ],
        "muted": sorted(mset),
    }
