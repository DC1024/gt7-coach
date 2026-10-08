# -*- coding: utf-8 -*-
"""
引擎 —— 把数据源、参考圈、规则、闸门串成一个 `tick()`。
=========================================================

一 tick 干这些事：

    拉一帧 → 圈变化了就结算上一圈 → 定位「我在赛道哪」→ 跑规则 → 过闸门
    → 返回 CoachState（含这一 tick 决定要说的话）

🔴 参考圈必须**异步取**。`/profile` 冷路径要解析整场 jsonl（一场 20 万帧约 2s），
   同步取的话这 2 秒里 tick 完全停摆 —— 而"正在冲线前"恰恰是最不能停的时候。
   所以 `RefProvider` 用后台线程，tick 永远只读它当下的结果，从不等待。

🔴 但要**允许**参考圈落后：取 profile 期间用上一份（或自攒的）继续干活，
   总比不干活好。`CoachState.ref_source` 会写着用的是哪一份。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .contract import CoachState, Frame, Utterance
from .gate import Gate, GateConfig
from .refindex import RefLap
from .rules import Ctx, RuleConfig, RuleSet
from .source import HttpSource, ReplaySource


@dataclass
class CoachConfig:
    poll_interval_s: float = 0.10      # 与仪表盘一致的 10Hz
    profile_step_m: float = 5.0        # 参考圈网格步长
    self_ref_min_frames: int = 30
    lap_buffer_max: int = 3000         # 自攒参考圈用的帧缓冲上限
    spoken_history: int = 20
    profile_retry_s: float = 20.0      # 取 profile 失败后多久重试
    max_lateral_m: float = 400.0       # 横向距离超过它判为定位失败
    # 🔴 场次列表的轮询间隔。绝**不能**每 tick（10Hz）问一次 —— 那个接口要
    #    遍历目录、stat 每个文件、查车型表，10Hz 打上去是自己给自己造负载。
    sess_poll_boot_s: float = 2.0      # 还没拿到参考圈时：勤问
    sess_poll_idle_s: float = 15.0     # 已有参考圈时：偶尔问一次换没换场


class RefProvider:
    """后台取参考圈剖面。线程只写、tick 只读。"""

    def __init__(self, source: Any, step_m: float = 5.0,
                 retry_s: float = 20.0):
        self.src = source
        self.step_m = step_m
        self.retry_s = retry_s
        self._lock = threading.Lock()
        self._ref: RefLap | None = None
        self._key: tuple | None = None
        self._state = "idle"           # idle | loading | ready | failed | unsupported
        self._err: str | None = None
        self._last_try = 0.0
        self._profile_ok: bool | None = None

    @staticmethod
    def key_for(sess: dict) -> tuple:
        """参考圈的身份：换了场次、或最快圈被刷新，就要重取。"""
        return (sess.get("file"), sess.get("best_lap_s"))

    def request(self, sess: dict) -> None:
        """有需要就在后台起一次取数；重复调用是安全的（幂等）。"""
        if not sess or not sess.get("file"):
            return
        key = self.key_for(sess)
        now = time.monotonic()
        with self._lock:
            if self._state == "unsupported":
                return
            if key == self._key and self._state in ("loading", "ready"):
                return
            if (self._state == "failed" and key == self._key
                    and now - self._last_try < self.retry_s):
                return                       # 失败了别死循环重试，等一会儿
            self._key = key
            self._state = "loading"
            self._last_try = now
        threading.Thread(target=self._load, args=(sess, key),
                         daemon=True, name="gt7coach-ref").start()

    def _load(self, sess: dict, key: tuple) -> None:
        try:
            d = self.src.lap_profile(sess["file"], None, self.step_m)
            if not d:
                with self._lock:
                    # 拿不到就记失败，但**保留**旧的 ref 继续用
                    self._state = "failed"
                    self._err = (f"/profile 不可用"
                                 f"（{getattr(self.src, 'last_error', None)}）")
                return
            ref = RefLap.from_profile(d)
            with self._lock:
                self._ref = ref
                self._state = "ready"
                self._err = None
                self._profile_ok = True
        except Exception as e:               # noqa: BLE001 —— 解析/网络都可能炸
            with self._lock:
                self._state = "failed"
                self._err = f"{type(e).__name__}: {e}"

    def get(self) -> tuple[RefLap | None, str, str | None]:
        with self._lock:
            return self._ref, self._state, self._err

    def clear(self) -> None:
        with self._lock:
            self._ref = None
            self._key = None
            self._state = "idle"
            self._err = None


class CoachEngine:
    """赛道工程师主循环（不含 IO 与播报，纯逻辑 + 状态）。"""

    def __init__(self, source: Any, cfg: CoachConfig | None = None,
                 rule_cfg: RuleConfig | None = None,
                 gate_cfg: GateConfig | None = None,
                 clock: Callable[[], float] | None = None):
        self.src = source
        self.cfg = cfg or CoachConfig()
        self.rules = RuleSet(rule_cfg)
        self.gate = Gate(gate_cfg)
        # 时钟可注入：回放用比赛时钟，真机用墙上时钟（两者本就相等）
        self._clock = clock or time.monotonic
        self.refs = RefProvider(source, step_m=self.cfg.profile_step_m,
                                retry_s=self.cfg.profile_retry_s)

        self._st_rules = RuleSet.fresh_state()
        self._st_gate = Gate.fresh_state()

        self._lap: int | None = None
        self._lap_buf: list[Frame] = []
        self._self_ref: RefLap | None = None
        self._last_t: float | None = None
        self._spoken: list[dict[str, Any]] = []
        self._ticks = 0
        self._ref_sess_key: tuple | None = None
        self._last_sess_poll = -1e9

    # —— 对外 ——————————————————————————————————————————

    def tick(self) -> CoachState:
        f = self.src.poll()
        now = self._clock()
        self._ticks += 1
        if f is None:
            err = getattr(self.src, "last_error", None)
            return CoachState(
                connected=False,
                ref_ready=self._current_ref() is not None,
                lap=self._lap or 0,
                stats={"ticks": self._ticks,
                       "source_error": err or "no frame"})
        return self._tick_frame(f, now)

    def run(self, on_state: Callable[[CoachState], None] | None = None,
            interval_s: float | None = None,
            should_stop: Callable[[], bool] | None = None) -> None:
        """阻塞式主循环（CLI 用）。"""
        iv = interval_s or self.cfg.poll_interval_s
        while True:
            if should_stop and should_stop():
                return
            st = self.tick()
            if on_state:
                on_state(st)
            time.sleep(iv)

    # —— 内部 ——————————————————————————————————————————

    def _current_ref(self) -> RefLap | None:
        ref, _state, _err = self.refs.get()
        return ref or self._self_ref

    def _tick_frame(self, f: Frame, now: float) -> CoachState:
        dt = self.cfg.poll_interval_s if self._last_t is None else max(
            min(f.t - self._last_t, 1.0), 1e-3)
        self._last_t = f.t

        # —— 圈变化：结算上一圈（自攒参考圈），并在新圈上重置闸门 ——
        if self._lap is None:
            self._lap = f.lap
        elif f.lap != self._lap:
            self._finalize_lap(self._lap)
            self._lap = f.lap
            self._lap_buf = []
        self.rules.roll_lap(self._st_rules, f.lap)

        self._lap_buf.append(f)
        if len(self._lap_buf) > self.cfg.lap_buffer_max:
            del self._lap_buf[:len(self._lap_buf) - self.cfg.lap_buffer_max]

        # —— 参考圈：找当前场次，需要就后台取 ——
        self._maybe_request_ref()

        ref = self._current_ref()
        s: float | None = None
        lateral_m: float | None = None
        if ref is not None and f.coords_ok:
            _i, s, lateral_m = ref.nearest(f.x, f.z)
            # 闭环折线在起跑线上是同一个点，用圈计时器消歧（见 resolve_wrap）
            s = ref.resolve_wrap(s, f.lap_time_s)
            if lateral_m > self.cfg.max_lateral_m:
                # 定位明显失败（不在本赛道 / 坐标跳变）→ 不给基于位置的建议
                s = None
                lateral_m = None

        prev = self._lap_buf[-2] if len(self._lap_buf) >= 2 else None
        ctx = Ctx(f=f, prev=prev, ref=ref if s is not None else None, s=s,
                  lateral_m=lateral_m, dt=dt, st=self._st_rules)
        cands = self.rules.evaluate(ctx)
        say = self.gate.filter(cands, now=now, lap=f.lap,
                               g_mag=f.g_mag, st=self._st_gate)
        self._record_spoken(say, now, f)

        delta = None
        ref_v = None
        next_m = None
        next_s = None
        if ref is not None and s is not None:
            t_ref = ref.t_at_s(s)
            if t_ref is not None and f.lap_time_s > 0:
                d = f.lap_time_s - t_ref
                # 兜底闸：定位或时间轴一旦不一致，delta 会变成"整圈"量级。
                # 宁可不给这个数，也不能给一个离谱的数（会被当成真丢了一圈）。
                if abs(d) <= max(30.0, ref.lap_time_s * 0.5):
                    delta = round(d, 3)
            ref_v = ref.v_at_s(s)
            z = ref.next_brake(s)
            if z:
                d = z["s_in_m"] - s
                if d < 0:
                    d += ref.length_m
                next_m = round(d, 1)
                next_s = round(d / max(f.speed_ms, 1.0), 2)

        _ref, ref_state, ref_err = self.refs.get()
        return CoachState(
            connected=f.connected,
            ref_ready=ref is not None,
            ref_lap=ref.lap if ref else None,
            ref_len_m=round(ref.length_m, 1) if ref else None,
            lap=f.lap,
            s_m=round(s, 1) if s is not None else None,
            delta_s=delta,
            ref_speed_kph=round(ref_v, 1) if ref_v is not None else None,
            next_brake_m=next_m,
            next_brake_s=next_s,
            say=say,
            spoken=list(self._spoken),
            stats={
                "ticks": self._ticks,
                "speed_kph": round(f.speed_kph, 1),
                "ref_source": ref.source if ref else None,
                "ref_state": ref_state,
                "ref_error": ref_err,
                "lateral_m": round(lateral_m, 1) if lateral_m is not None else None,
                "lap_frames": len(self._lap_buf),
                "dropped_by_gate": self._st_gate.get("dropped", 0),
                "source_error": getattr(self.src, "last_error", None),
                "wheel_radius_m": {
                    "front": round(self._st_rules["radius"]["front"], 4)
                    if self._st_rules["radius"]["front"] else None,
                    "rear": round(self._st_rules["radius"]["rear"], 4)
                    if self._st_rules["radius"]["rear"] else None,
                    "samples": self._st_rules["radius"]["n"],
                },
            },
        )

    def _maybe_request_ref(self) -> None:
        # 🔴 按节流问场次列表，不是每 tick 都问（理由见 CoachConfig）。
        #    还没参考圈时勤问（尽快能用），有了之后偶尔问一次（换个场要能发现）。
        have_ref = self._current_ref() is not None
        gap = (self.cfg.sess_poll_idle_s if have_ref
               else self.cfg.sess_poll_boot_s)
        now = time.monotonic()
        if now - self._last_sess_poll < gap:
            return
        self._last_sess_poll = now

        sess = self.src.live_session()
        if not sess or not sess.get("file"):
            return
        key = RefProvider.key_for(sess)
        if key != self._ref_sess_key:
            # 换场次 = 换赛道：自攒的旧参考圈必须作废，否则会拿上一条赛道的
            # 折线去做「最近点定位」，结果是**静默**地把车定位到错误的位置。
            self._ref_sess_key = key
            self._self_ref = None
            self._lap_buf = []
        self.refs.request(sess)

    def _finalize_lap(self, lap: int) -> None:
        """一圈跑完：用它更新自攒参考圈（profile 拿不到时的降级路径）。

        只在**更快**时才替换 —— 参考圈的定义是「本圈跑到最好的那一圈」。
        """
        buf = [x for x in self._lap_buf if x.coords_ok]
        if len(buf) < self.cfg.self_ref_min_frames:
            return
        try:
            cand = RefLap.from_frames(buf, lap=lap,
                                      step_m=self.cfg.profile_step_m)
        except ValueError:
            return
        cur = self._self_ref
        if cur is None or cand.lap_time_s < cur.lap_time_s:
            self._self_ref = cand

    def _record_spoken(self, say: list[Utterance], now: float,
                       f: Frame) -> None:
        for u in say:
            self._spoken.insert(0, {
                "t": round(now, 3),
                "lap": f.lap,
                "key": u.key,
                "text": u.text,
                "priority": u.priority,
                "evidence": u.evidence,
            })
        del self._spoken[self.cfg.spoken_history:]
