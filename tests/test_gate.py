# -*- coding: utf-8 -*-
"""闸门测试 —— 「不打扰」这条原则的具体实现，全在这里守着。"""
from __future__ import annotations

from gt7coach.contract import P_CRITICAL, P_HIGH, P_LOW, P_NORMAL, Utterance
from gt7coach.gate import Gate, GateConfig


def U(key, priority=P_NORMAL, text=None, short=None) -> Utterance:
    return Utterance(key=key, text=text or f"话-{key}", priority=priority,
                     short=short if short is not None else (text or f"话-{key}"))


class TestPerLap:
    def test_same_key_once_per_lap(self):
        g = Gate()
        st = Gate.fresh_state()
        first = g.filter([U("k1")], now=100.0, lap=1, g_mag=0.0, st=st)
        assert len(first) == 1
        again = g.filter([U("k1")], now=200.0, lap=1, g_mag=0.0, st=st)
        assert again == [], "同一圈同一个 key 只报一次"

    def test_new_lap_resets(self):
        g = Gate()
        st = Gate.fresh_state()
        g.filter([U("k1")], now=100.0, lap=1, g_mag=0.0, st=st)
        # 下一圈（约 90 s 后）
        got = g.filter([U("k1")], now=190.0, lap=2, g_mag=0.0, st=st)
        assert len(got) == 1, "换圈后应当可以重新报"

    def test_same_key_cooldown_spans_laps(self):
        """同类冷却**跨圈生效**，不受换圈影响。

        这是故意的：换圈重置的是「每圈额度」和「本圈报过哪些 key」，
        而不是「多久没说过这句话」。否则在圈界附近两句话能贴在一起说。
        """
        g = Gate(GateConfig(cooldown_same_s=20.0))
        st = Gate.fresh_state()
        g.filter([U("off_track")], now=100.0, lap=1, g_mag=0.0, st=st)
        got = g.filter([U("off_track")], now=105.0, lap=2, g_mag=0.0, st=st)
        assert got == [], "5 秒内同一个 key 不该因为换圈就重说"

    def test_max_per_lap(self):
        g = Gate(GateConfig(max_per_lap=2, cooldown_any_s=0.0,
                            cooldown_same_s=0.0, max_per_tick=1))
        st = Gate.fresh_state()
        spoken = 0
        for i in range(6):
            for k in (f"a{i}", f"b{i}"):
                spoken += len(g.filter([U(k)], now=100.0 + i, lap=1,
                                       g_mag=0.0, st=st))
        assert spoken == 2, spoken

    def test_critical_does_not_consume_normal_quota(self):
        """出界/打滑这类紧急提示不能被普通播报的额度挤掉。"""
        g = Gate(GateConfig(max_per_lap=1, cooldown_any_s=0.0,
                            cooldown_same_s=0.0))
        st = Gate.fresh_state()
        g.filter([U("normal1")], now=1.0, lap=1, g_mag=0.0, st=st)
        got = g.filter([U("off_track", P_CRITICAL)], now=2.0, lap=1,
                       g_mag=0.0, st=st)
        assert got and got[0].key == "off_track"


class TestCooldown:
    def test_cross_key_cooldown(self):
        g = Gate(GateConfig(cooldown_any_s=6.0, cooldown_same_s=0.0))
        st = Gate.fresh_state()
        assert g.filter([U("a")], now=100.0, lap=1, g_mag=0.0, st=st)
        assert g.filter([U("b")], now=103.0, lap=1, g_mag=0.0, st=st) == []
        assert g.filter([U("b")], now=107.0, lap=1, g_mag=0.0, st=st)

    def test_critical_bypasses_any_cooldown(self):
        g = Gate(GateConfig(cooldown_any_s=6.0))
        st = Gate.fresh_state()
        g.filter([U("a")], now=100.0, lap=1, g_mag=0.0, st=st)
        got = g.filter([U("slip_rear", P_CRITICAL)], now=101.0, lap=1,
                       g_mag=0.0, st=st)
        assert got, "紧急提示要能插队"

    def test_same_key_cooldown_critical_still_holds(self):
        """紧急提示能插队，但自己也别刷屏（连续打滑不要每秒念一次）。"""
        g = Gate(GateConfig(cooldown_any_s=0.0, cooldown_same_s=5.0))
        st = Gate.fresh_state()
        assert g.filter([U("slip_rear", P_CRITICAL)], now=10.0, lap=1,
                        g_mag=0.0, st=st)
        assert g.filter([U("slip_rear", P_CRITICAL)], now=12.0, lap=1,
                        g_mag=0.0, st=st) == []
        # 但 keys_lap 对 critical 不设限，所以换个 key（前轮）应当还能说
        assert g.filter([U("slip_front", P_CRITICAL)], now=12.0, lap=1,
                        g_mag=0.0, st=st)

    def test_per_tick_cap(self):
        g = Gate(GateConfig(cooldown_any_s=0.0, cooldown_same_s=0.0,
                            max_per_tick=1))
        st = Gate.fresh_state()
        got = g.filter([U("a"), U("b")], now=1.0, lap=1, g_mag=0.0, st=st)
        assert len(got) == 1


class TestCornerSilence:
    def test_long_sentence_becomes_short_in_corner(self):
        g = Gate(GateConfig(cooldown_any_s=0.0, cooldown_same_s=0.0))
        st = Gate.fresh_state()
        u = U("k", text="刹车晚了 12 米", short="晚 12")
        got = g.filter([u], now=1.0, lap=1, g_mag=1.2, st=st)
        assert got and got[0].text == "晚 12", "大 G 时只念短句"

    def test_straight_keeps_long_sentence(self):
        g = Gate(GateConfig(cooldown_any_s=0.0, cooldown_same_s=0.0))
        st = Gate.fresh_state()
        u = U("k", text="刹车晚了 12 米", short="晚 12")
        got = g.filter([u], now=1.0, lap=1, g_mag=0.3, st=st)
        assert got[0].text == "刹车晚了 12 米"

    def test_no_short_variant_dropped_in_corner(self):
        """没写短句的候选在大 G 时**丢掉**，不是念长句 ——
        玩家在弯里只来得及处理一个词，念一整句等于没说还添乱。"""
        g = Gate(GateConfig(cooldown_any_s=0.0, cooldown_same_s=0.0))
        st = Gate.fresh_state()
        u = Utterance(key="k", text="这是一句很长的提示")
        assert g.filter([u], now=1.0, lap=1, g_mag=1.5, st=st) == []


class TestOrdering:
    def test_priority_decides_who_speaks_first(self):
        g = Gate(GateConfig(cooldown_any_s=0.0, cooldown_same_s=0.0,
                            max_per_tick=1))
        st = Gate.fresh_state()
        got = g.filter([U("low", P_LOW), U("high", P_HIGH),
                        U("mid", P_NORMAL)], now=1.0, lap=1, g_mag=0.0, st=st)
        assert got[0].key == "high"

    def test_deterministic_on_tie(self):
        g = Gate(GateConfig(cooldown_any_s=0.0, cooldown_same_s=0.0,
                            max_per_tick=1))
        st = Gate.fresh_state()
        got = g.filter([U("zz"), U("aa"), U("mm")], now=1.0, lap=1,
                       g_mag=0.0, st=st)
        assert got[0].key == "aa", "同优先级按 key 稳定排序，结果必须可复现"

    def test_dropped_counter(self):
        g = Gate(GateConfig(cooldown_any_s=999.0))
        st = Gate.fresh_state()
        g.filter([U("a")], now=1.0, lap=1, g_mag=0.0, st=st)
        g.filter([U("b")], now=2.0, lap=1, g_mag=0.0, st=st)
        assert st["dropped"] == 1
