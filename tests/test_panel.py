# -*- coding: utf-8 -*-
"""播报开关面板 —— 分组口径与闸门静音。

面板让用户自选"什么内容播报、什么不播报"。守两件事：
  1. key → 分组的映射（含带位置后缀的 key，如 brake_late@400）；
  2. 静音只影响"出不出闸门"，且**没被认领的 key 永远放行**。
"""
from __future__ import annotations

import pytest

from gt7coach import panel
from gt7coach.contract import P_CRITICAL, P_NORMAL, Utterance
from gt7coach.gate import Gate, GateConfig


def _u(key: str, prio: int = P_NORMAL) -> Utterance:
    return Utterance(key=key, text="测试", priority=prio, ttl_s=1.0,
                     short="测", evidence={})


def _run_muted(muted, cands, *, g_mag=0.0):
    """用「冷却全关」的闸门跑一次 filter，隔离出静音行为。"""
    cfg = GateConfig(muted=tuple(muted), cooldown_same_s=0.0,
                     cooldown_any_s=0.0, max_per_tick=50, max_per_lap=50)
    gate = Gate(cfg)
    st = Gate.fresh_state()
    out = gate.filter(cands, now=100.0, lap=1, g_mag=g_mag, st=st)
    return out, st


class TestGrouping:
    @pytest.mark.parametrize("key,gid", [
        ("off_track", "safety"),
        ("slip_front", "safety"),
        ("slip_all", "safety"),
        ("brake_late@400", "safety"),      # 带刹车区后缀
        ("brake_warn@400", "safety"),
        ("shift", "safety"),
        ("apex_slow@480", "driving"),
        ("throttle_late@480", "driving"),
        ("tyre_hot", "tyres"),
        ("tyre_cold", "tyres"),
        ("delta", "pace"),
        ("projected_lap", "pace"),
        ("lap_advice", "debrief"),
        ("corner_habit@T1", "debrief"),    # 旧行为（lap_advice=False）也认领
        ("lap_summary", "debrief"),        # 兼容旧键
    ])
    def test_key_maps_to_group(self, key, gid):
        assert panel.group_of_key(key) == gid

    def test_unknown_key_is_unclaimed(self):
        """没认领的 key → None（永远可播，不会被哪个开关误吞）。"""
        assert panel.group_of_key("brand_new_rule") is None
        assert panel.group_of_key("") is None

    def test_group_ids_stable(self):
        assert panel.known_ids() == {"safety", "driving", "tyres", "pace",
                                     "debrief"}

    def test_every_group_has_label_and_desc(self):
        for g in panel.GROUPS:
            assert g["id"] and g["label"] and g["desc"] and g["prefixes"]


class TestNormalize:
    def test_dedup_and_sort(self):
        assert panel.normalize_muted(["tyres", "pace", "tyres"]) == \
            ("pace", "tyres")

    def test_none_is_empty(self):
        assert panel.normalize_muted(None) == ()

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            panel.normalize_muted(["tyres", "nope"])

    def test_rejects_non_list(self):
        with pytest.raises(ValueError):
            panel.normalize_muted("tyres")

    def test_rejects_non_str_item(self):
        with pytest.raises(ValueError):
            panel.normalize_muted([123])


class TestPanelState:
    def test_shape(self):
        st = panel.panel_state(("tyres",), ["tyre_hot", "delta", "delta"])
        ids = [g["id"] for g in st["groups"]]
        assert ids == ["safety", "driving", "tyres", "pace", "debrief"]
        assert st["muted"] == ["tyres"]
        by = {g["id"]: g for g in st["groups"]}
        assert by["tyres"]["muted"] is True and by["pace"]["muted"] is False
        # 最近计数：delta 说了 2 次 → pace=2
        assert by["pace"]["recent"] == 2
        assert by["tyres"]["recent"] == 1


class TestGateMute:
    def test_mute_drops_whole_group(self):
        out, st = _run_muted(["tyres"],
                             [_u("tyre_hot"), _u("delta"), _u("tyre_cold")])
        assert [u.key for u in out] == ["delta"]
        assert st["muted"] == 2

    def test_default_all_enabled(self):
        """不设 muted（默认）→ 与加这个功能之前完全一致（全部放行）。"""
        out, st = _run_muted([], [_u("tyre_hot"), _u("delta")])
        assert {u.key for u in out} == {"tyre_hot", "delta"}
        assert st["muted"] == 0

    def test_mutes_critical_too(self):
        """用户关了安全组，P0 也不再播 —— 这是用户的明确选择（UI 有提示）。"""
        out, _ = _run_muted(["safety"],
                            [_u("off_track", P_CRITICAL), _u("delta")])
        assert [u.key for u in out] == ["delta"]

    def test_unknown_key_never_muted(self):
        out, _ = _run_muted(sorted(panel.known_ids()),
                            [_u("brand_new_rule"), _u("tyre_hot")])
        assert [u.key for u in out] == ["brand_new_rule"]

    def test_multiple_groups_muted(self):
        out, st = _run_muted(["tyres", "pace"],
                             [_u("tyre_hot"), _u("delta"),
                              _u("projected_lap"), _u("shift")])
        assert [u.key for u in out] == ["shift"]
        assert st["muted"] == 3
