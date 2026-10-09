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
from . import phrases
from .refcache import RefCache
from .gate import Gate, GateConfig
from .contract import P_NORMAL, CoachState, Frame, Utterance
from .lapstats import (CornerTracker, FuelTracker, LapResult,
                       SectorTracker, corner_loss_split, corner_losses,
                       lap_result)
from .narrate import Narrator
from .refindex import RefLap
from .rules import Ctx, RuleConfig, RuleSet
from .source import HttpSource, ReplaySource
from .tts import TtsConfig, TtsEngine


@dataclass
class CoachConfig:
    poll_interval_s: float = 0.10      # 与仪表盘一致的 10Hz
    profile_step_m: float = 5.0        # 参考圈网格步长
    self_ref_min_frames: int = 30
    # 自攒参考圈的最低门槛。跑满一整圈至少要 20 s（与 Dash 的 clean_laps
    # 同一口径）；低于它多半是维修区/菜单/出场圈的碎片。
    min_lap_s: float = 20.0
    # 🔴 帧缓冲按**时长**而不是帧数裁剪。
    #    帧率是不固定的：实时侧 10Hz、jsonl 文件 60Hz。旧参数是 3000 **帧**，
    #    在 10Hz 下够 300 秒，在 60Hz 下只有 50 秒 —— 连一圈都装不下，
    #    于是每圈都被判成"中途接入"，分段/自攒参考圈全部失效。
    #    （这个 bug 是离线回放工具第一次跑真实场次时抓到的：实时侧永远
    #     看不到，因为实时就是 10Hz。）
    lap_buffer_s: float = 900.0        # 一圈最长按 15 分钟算，足够任何赛道
    lap_buffer_hard: int = 60000       # 硬上限：防菜单里长期不动时无限增长
    spoken_history: int = 20
    profile_retry_s: float = 20.0      # 取 profile 失败后多久重试
    max_lateral_m: float = 400.0       # 横向距离超过它判为定位失败
    # —— 参考圈策略 ——
    # session_best：用本场最快圈（最保守）
    # history_best：在本场最快圈之外，再找**你自己跨场次的历史最快圈**，
    #               并用几何形状比对确认是同一条赛道才采用。
    #   开一场慢的，教练就该拿你的历史最好当标杆 —— 否则它只会拿"你今天
    #   最烂的那一圈"来夸你。失败时自动退回 session_best，不会没参考可用。
    ref_policy: str = "history_best"
    history_max_tries: int = 3          # 最多试几个候选（每个要一次 /profile）
    history_shape_tol_m: float = 60.0   # 形状中位距离超过它就判为不同赛道
    # 自攒参考圈最早从第几圈开始建。默认 2 = **跳过第 1 圈（出场 / 暖胎圈）**：
    # 第 1 圈通常慢且不具代表性，拿它当参考会让第 2~3 圈的刹车点 / 弯心速度全建在
    # 慢圈上 → 用户感知的"前几圈瞎播报"。时间赛里第 1 圈就是飞行圈时改成 1 即可。
    self_ref_min_lap: int = 2
    # —— 参考圈本地缓存（根治「前几圈瞎播报」的第二条杠杆）——
    # 把跑过的最好一圈按「赛道指纹 + 车型」存盘，下次同赛道直接复用，
    # 省掉 history_best 去服务端翻历史的冷启动延迟。设为 None → 禁用（纯内存）。
    ref_cache_dir: str | None = None
    # 🔴 场次列表的轮询间隔。绝**不能**每 tick（10Hz）问一次 —— 那个接口要
    #    遍历目录、stat 每个文件、查车型表，10Hz 打上去是自己给自己造负载。
    sess_poll_boot_s: float = 2.0      # 还没拿到参考圈时：勤问
    sess_poll_idle_s: float = 15.0     # 已有参考圈时：偶尔问一次换没换场
    # —— R2.2 云接入 ——
    # cloud.json 的挂载路径（容器内 /opt/gt7-coach/data/cloud.json）。
    # 设为 None → narrator 全程禁用态（纯本地模板，零网络），降级到 R2.1 行为。
    cloud_path: str | None = None
    # —— R3 云 TTS（只给 **B 档**句子；A 档永远走浏览器 TTS）——
    # 默认**全禁**：不配就是纯浏览器 `speechSynthesis`（= R2.2 的行为）。
    # 要开启得凑齐三样：tts_enabled + tts_workspace_id + 可写的 tts_cache_dir；
    # 缺任何一样都静默降级 —— 云挂了不该把教练搞哑。
    # key 复用 R2.2 那一把（百炼的 key 是业务空间级的，语音与语言模型不分开授权）。
    tts_enabled: bool = False
    tts_provider: str = "bailian"
    tts_workspace_id: str = ""               # 百炼业务空间 ID（ws- 开头），非密钥
    tts_api_key_env: str = "GT7_COACH_LLM_KEY"   # 🔴 只存变量名
    tts_model: str = ""        # 空 → provider 默认（cosyvoice-v3-flash）
    tts_voice: str = ""        # 空 → provider 默认（longanyang / 龙安洋）
    tts_format: str = "mp3"
    tts_sample_rate: int = 24000
    tts_timeout_s: float = 8.0     # B 档不抢麦，可比 chat 的 2s 宽松
    tts_cache_dir: str | None = None
    tts_max_chars: int = 60        # 单句最长字符（防呆：超长句既不合理也烧钱）

    def tts_config(self) -> TtsConfig:
        """把散在 CoachConfig 上的 tts_* 收成一个 TtsConfig。

        用扁平字段而不是嵌套 dataclass，是为了让 `POST /config` 的白名单
        照常工作（它只认数值字段，字符串/布尔一律拒绝 —— 嵌套对象会把
        那个校验绕过去，一次笔误就能把整块配置换成字符串）。
        """
        return TtsConfig(
            enabled=self.tts_enabled,
            provider=self.tts_provider,
            workspace_id=self.tts_workspace_id,
            api_key_env=self.tts_api_key_env,
            model=self.tts_model,
            voice=self.tts_voice,
            audio_format=self.tts_format,
            sample_rate=self.tts_sample_rate,
            timeout_s=self.tts_timeout_s,
            max_chars=self.tts_max_chars,
            cache_dir=self.tts_cache_dir,
        )


class RefProvider:
    """后台取「当前场次」与参考圈剖面。线程只写、tick 只读。

    🔴 为什么连「找场次」也要后台做：
       `/api/v1/sessions` 的冷态要把每个场次文件流式扫一遍才能给出列表里的
       「最快圈」（实测服务器上 5 场里有个 111 MB 的 → **11 秒**），而容器
       每次重启都会回到冷态。早先这一句是同步写在 tick 里的，后果是
       每个 tick 卡 3~11 秒、10Hz 直接掉到 0.3Hz —— 仪表盘慢一下，
       教练就整个哑掉。**慢的依赖绝不能在 tick 线程上等。**
    """

    def __init__(self, source: Any, step_m: float = 5.0,
                 retry_s: float = 20.0,
                 boot_gap_s: float = 2.0, idle_gap_s: float = 15.0,
                 policy: str = "history_best", history_max_tries: int = 3,
                 history_shape_tol_m: float = 60.0,
                 history_same_car: bool = True,
                 cache: "RefCache | None" = None):
        self.src = source
        self.step_m = step_m
        self.retry_s = retry_s
        self.policy = policy
        self.history_max_tries = max(0, int(history_max_tries))
        self.history_shape_tol_m = float(history_shape_tol_m)
        self.history_same_car = bool(history_same_car)
        # 参考圈本地缓存：把跑过的最好一圈按「赛道指纹 + 车型」存盘，下次同赛道
        # 直接复用，省掉 history_best 去服务端翻历史的冷启动延迟。None = 禁用。
        self._cache = cache
        # 还没拿到参考圈时勤问（尽快能用）；有了之后偶尔问一次（换场要能发现）
        self.boot_gap_s = boot_gap_s
        self.idle_gap_s = idle_gap_s
        self._lock = threading.Lock()
        self._ref: RefLap | None = None
        self._key: tuple | None = None
        self._state = "idle"           # idle | loading | ready | failed | unsupported
        self._err: str | None = None
        self._last_try = 0.0
        # —— 历史参考圈（跨场次）的尝试结果，供排障 ——
        self._hist: dict[str, Any] = {"candidates": [], "rejected": [],
                                      "adopted": None}
        # 本轮是否采用了本地缓存参考圈（先于服务端历史搜索命中）
        self._cached = False
        # —— 场次发现（独立于参考圈）——
        self._sess: dict | None = None
        self._sess_state = "idle"      # idle | loading | ok | empty | failed
        self._sess_err: str | None = None
        self._last_sess_poll = -1e9

    def key_for(self, sess: dict) -> tuple:
        """参考圈的身份：换场次 / 最快圈刷新 / **改了策略**，都要重取。"""
        return (sess.get("file"), sess.get("best_lap_s"), self.policy)

    # —— 场次发现 ————————————————————————————————————

    def tick_session(self, now: float) -> dict | None:
        """由 tick 调用的**非阻塞**入口：到点了就叫醒后台去问，立刻返回已知值。

        绝不在调用线程上发 HTTP。上一次还没回来就不重复起线程
        （否则慢服务端会让线程越堆越多）。
        """
        with self._lock:
            gap = self.idle_gap_s if self._ref is not None else self.boot_gap_s
            due = (self._sess_state != "loading"
                   and now - self._last_sess_poll >= gap)
            if due:
                self._last_sess_poll = now
                self._sess_state = "loading"
            sess = self._sess
        if due:
            threading.Thread(target=self._discover, daemon=True,
                             name="gt7coach-sess").start()
        return sess

    def _discover(self) -> None:
        try:
            sess = self.src.live_session()
            with self._lock:
                self._sess = sess
                self._sess_state = "ok" if sess else "empty"
                self._sess_err = None
        except Exception as e:               # noqa: BLE001
            with self._lock:
                self._sess_state = "failed"
                self._sess_err = f"{type(e).__name__}: {e}"

    def session(self) -> dict | None:
        with self._lock:
            return self._sess

    def session_status(self) -> tuple[str, str | None]:
        with self._lock:
            return self._sess_state, self._sess_err

    # —— 参考圈 ————————————————————————————————————
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

    def _fetch_ref(self, file: str, source: str) -> RefLap | None:
        d = self.src.lap_profile(file, None, self.step_m)
        if not d:
            return None
        try:
            return RefLap.from_profile(d, source=source)
        except ValueError:
            return None

    def _load(self, sess: dict, key: tuple) -> None:
        """后台一次搞完：先让教练**马上**有参考圈，再悄悄找更好的。

        顺序很关键（否则本地缓存这条杠杆等于摆设）：

          1. 取本场剖面 `base`。
          2. **先查本地缓存**（用 `base` 的几何算指纹，找同赛道更快的）。
             命中就**立刻发布**它 —— 教练从第一帧起就有靠谱参考，
             不必等服务端历史搜索（那要额外几次 `/profile`，慢好几秒）。
          3. 没命中/不更快才发布 `base`。
          4. 把本次参考圈存盘（供以后复用）。注意 **存盘在"读缓存"之后**，
             否则上一轮存进去的 `base` 会把磁盘上的更快缓存覆盖掉，
             读缓存永远读到自己 → 缓存永远不生效。
          5. 再找服务端历史更快的（跨场次）：可能比缓存更好，找到就升级。

        缓存采用的安全底线与 history_best 同一把尺：必须与 `base` 做
        `shape_distance` 比对，形状吻合（< history_shape_tol_m）才采用，
        否则换条赛道误用旧参考圈 = 静默定位错误。
        """
        try:
            base = self._fetch_ref(sess["file"], "profile")
            if base is None:
                with self._lock:
                    # 拿不到就记失败，但**保留**旧的 ref 继续用
                    self._state = "failed"
                    self._err = (f"/profile 不可用"
                                 f"（{getattr(self.src, 'last_error', None)}）")
                return
            car = sess.get("car_name", "") or ""
            fp = base.track_fingerprint()
            # ② 先查本地缓存：同赛道、更快 → 立刻发布（毫秒级，无需等服务端）
            cached = None
            used_cache = False
            self._cached = False
            if self._cache is not None:
                cached = self._cache.load(car, fp)
                if (cached is not None and cached.lap_time_s > 0
                        and cached.lap_time_s < base.lap_time_s):
                    sd = cached.shape_distance(base)
                    if sd is not None and sd <= self.history_shape_tol_m:
                        with self._lock:
                            self._ref = cached
                            self._state = "ready"
                            self._err = None
                            self._cached = True
                        used_cache = True
            # ① 没命中缓存才发布本场剖面：教练立刻有参考圈可用
            if not used_cache:
                with self._lock:
                    self._ref = base
                    self._state = "ready"
                    self._err = None
            # ④ 存盘（在"读缓存"之后，避免把磁盘上的更快缓存覆盖掉）
            if self._cache is not None:
                self._cache.save(base, car)
                if cached is not None:
                    self._cache.save(cached, car)
            # ⑤ 再找服务端历史更快的（跨场次）：可能比缓存更好
            better = self._find_history_ref(sess, base)
            adopted = better if better is not None else (cached if used_cache else base)
            if better is not None and self._cache is not None:
                self._cache.save(better, car)
            # ⑥ 落定最终参考圈（历史/缓存更快就升级，否则保持已发布的）
            with self._lock:
                self._ref = adopted
                self._state = "ready"
                self._err = None
        except Exception as e:               # noqa: BLE001 —— 解析/网络都可能炸
            with self._lock:
                self._state = "failed"
                self._err = f"{type(e).__name__}: {e}"

    def _find_history_ref(self, sess: dict, base: RefLap) -> RefLap | None:
        """本场之外更快的历史圈 —— 用**几何形状**确认是同一条赛道才采用。

        为什么用形状而不是圈长/赛道名：
          · 圈长几乎一样的两条不同赛道（长度相同的不同布局）会被圈长骗过；
          · 赛道识别（`track_id`）要靠服务端逐个算指纹，实测 10 个场次里
            只有 2 个已识别，靠它等于大部分时候用不上。
        而两条折线的中位最近点距离，同赛道只有几米、不同赛道几十上百米，
        判别力干净（合成数据实测：同 0.00m / 不同 200m）。
        """
        best = sess.get("best_lap_s")
        if self.policy != "history_best" or self.history_max_tries == 0:
            return None
        if not isinstance(best, (int, float)) or best <= 0:
            return None
        try:
            cands = self.src.faster_sessions(
                float(best), exclude=sess.get("file", ""),
                car_name=sess.get("car_name", ""),
                same_car=self.history_same_car,
                limit=self.history_max_tries)
        except Exception as e:               # noqa: BLE001
            with self._lock:
                self._hist = {"candidates": [], "rejected": [],
                              "adopted": None, "error": f"{type(e).__name__}: {e}"}
            return None
        rejected: list[dict] = []
        adopted: dict | None = None
        found: RefLap | None = None
        for c in cands:
            r = self._fetch_ref(c["file"], "history")
            if r is None:
                rejected.append({"file": c["file"], "why": "profile 取不到"})
                continue
            dist = r.shape_distance(base)
            if dist is None or dist > self.history_shape_tol_m:
                # 换过赛道 / 反向布局 / 不同线路变体 —— 圈长可能一样，
                # 但形状差得远，拿它当参考会让定位整体失准
                rejected.append({"file": c["file"],
                                 "shape_m": None if dist is None else round(dist, 1),
                                 "why": "形状不像同一条赛道"})
                continue
            found = r
            adopted = {"file": c["file"], "best_lap_s": c["best_lap_s"],
                       "shape_m": round(dist, 1)}
            break
        with self._lock:
            self._hist = {"candidates": [c["file"] for c in cands],
                          "rejected": rejected, "adopted": adopted}
        return found

    def history_status(self) -> dict[str, Any]:
        with self._lock:
            return {**self._hist, "cached_adopted": self._cached}

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
        # R2.2：措辞编排层。cloud_path 为 None → 禁用态（纯本地模板）。
        self.narrator = Narrator(self.cfg.cloud_path)
        # R3：云 TTS（只给 B 档句）。禁用态下 request() 恒返回 None，
        # 前端回落浏览器 TTS —— 与"完全没配"表现一致。
        self.tts = TtsEngine(self.cfg.tts_config())
        self.rules = RuleSet(rule_cfg, self.narrator)
        self.gate = Gate(gate_cfg)
        # 时钟可注入：回放用比赛时钟，真机用墙上时钟（两者本就相等）
        self._clock = clock or time.monotonic
        self.refs = RefProvider(source, step_m=self.cfg.profile_step_m,
                                retry_s=self.cfg.profile_retry_s,
                                boot_gap_s=self.cfg.sess_poll_boot_s,
                                idle_gap_s=self.cfg.sess_poll_idle_s,
                                policy=self.cfg.ref_policy,
                                history_max_tries=self.cfg.history_max_tries,
                                history_shape_tol_m=self.cfg.history_shape_tol_m,
                                cache=RefCache(self.cfg.ref_cache_dir))

        self._st_rules = RuleSet.fresh_state()
        self._st_gate = Gate.fresh_state()

        self._lap: int | None = None
        self._lap_buf: list[Frame] = []
        self._self_ref: RefLap | None = None
        # —— 本地圈统计（分段 / 油耗）——
        self._sectors = SectorTracker(self.rules.cfg.sectors_n)
        self._fuel = FuelTracker()
        self._corners = CornerTracker(min_laps=self.rules.cfg.corner_min_laps,
                                      min_loss_s=self.rules.cfg.corner_min_loss_s)
        self._corners = CornerTracker(min_laps=self.rules.cfg.corner_min_laps,
                                      min_loss_s=self.rules.cfg.corner_min_loss_s)
        self._prev_lap: LapResult | None = None
        # 🔴 分段基准圈长必须**跨圈稳定**：段边界按绝对距离等分，基准一变，
        #    同一段路在不同圈被切成不同范围，跨圈比段落就没有意义了。
        #    所以第一圈跑完整后就把它的圈长钉死当基准，之后不再跟随参考圈变化。
        self._sector_len_m: float | None = None
        self._last_t: float | None = None
        self._spoken: list[dict[str, Any]] = []
        self._ticks = 0
        self._ref_sess_key: tuple | None = None

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
                       "source_error": err or "no frame",
                       "cloud": self.narrator.status(),
                       "tts": self.tts.status()})
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
            self._fuel.start_lap(f.fuel_pct)
        elif f.lap != self._lap:
            # 先结算旧圈（里面会用缓冲末帧的剩余量算本圈油耗），再开新圈
            self._finalize_lap(self._lap)
            self._lap = f.lap
            self._lap_buf = []
            self._fuel.start_lap(f.fuel_pct)
        self.rules.roll_lap(self._st_rules, f.lap)
        # R2.2：每圈预算计数（圈变化时才重置；幂等，可每 tick 调）
        self.narrator.note_lap(f.lap)
        # R3：TTS 的预算与 LLM 分开计（两件事、两笔钱、两个额度）
        self.tts.note_lap(f.lap)

        self._lap_buf.append(f)
        self._trim_lap_buffer(f)

        # —— 参考圈：场次发现与剖面取数全在后台线程，这里只读缓存 ——
        self._maybe_request_ref(now)

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
                  lateral_m=lateral_m, dt=dt, st=self._st_rules,
                  lap=self._prev_lap, theory=self._sectors.to_dict(),
                  fuel=self._fuel_snapshot(), corners=self._corner_snapshot())
        cands = self.rules.evaluate(ctx)
        say = self.gate.filter(cands, now=now, lap=f.lap,
                               g_mag=f.g_mag, st=self._st_gate)
        # 语音专用串：把 `text` 里的阿拉伯数字逐位中文化（54→五四），
        # 屏幕显示仍用原样 `text`。幂等且字符数等价，不影响 ttl 预算。
        # 放在闸门之后：此时 `u.text` 已是最终要念的那句（含弯中禁言的短句替换）。
        for u in say:
            u.speech = phrases.spell_digits(u.text)
            # R3：B 档句子进云 TTS 队列。🔴 这里**只丢任务、绝不等待** ——
            # `request()` 是 O(1) 的内存操作（缓存命中则顺带给回 URL），
            # 真正的合成在 tts 的后台线程里跑。把合成写进这一行，10Hz 主循环
            # 会当场掉到 0.5Hz，整个教练连带仪表盘一起废掉。
            # A 档（P_CRITICAL / P_HIGH）在这里就被 `>= P_NORMAL` 挡在外面。
            if u.priority >= P_NORMAL:
                u.tts_url = self.tts.request(u.speech)
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
        sess_state, sess_err = self.refs.session_status()
        # 预测圈速：保持当前 delta 跑完，最终就是这个时间。
        projected = None
        if ref is not None and delta is not None and ref.lap_time_s > 0:
            cand = ref.lap_time_s + delta
            if 10.0 < cand < 3600.0:
                projected = round(cand, 3)
        theory = self._sectors.to_dict()
        fuel = self._fuel_snapshot()
        # 刚跑完那圈：只给展示需要的三项，别把整个 LapResult 塞进 state
        round_lap = None
        if self._prev_lap is not None:
            round_lap = {"lap": self._prev_lap.lap, "ok": self._prev_lap.ok,
                         "why": self._prev_lap.why,
                         "lap_time_s": round(self._prev_lap.lap_time_s, 3),
                         "sectors": [round(x, 3)
                                     for x in self._prev_lap.sectors],
                         "fuel_used": (round(self._prev_lap.fuel_used, 3)
                                       if self._prev_lap.fuel_used else None)}
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
            projected_lap_s=projected,
            last_lap=round_lap,
            theory_best_s=theory.get("theory_best_s") if theory else None,
            potential_gain_s=theory.get("gain_s") if theory else None,
            fuel_per_lap=fuel.get("per_lap") if fuel else None,
            fuel_laps_left=fuel.get("laps_left") if fuel else None,
            say=say,
            spoken=list(self._spoken),
            stats={
                "ticks": self._ticks,
                "speed_kph": round(f.speed_kph, 1),
                "ref_source": ref.source if ref else None,
                "ref_state": ref_state,
                "ref_history": self.refs.history_status(),
                "ref_error": ref_err,
                # 场次发现单独暴露：它可能比参考圈更早失败，
                # 而且失败原因往往是"仪表盘冷态在算最快圈"这种慢，
                # 不区分开就分不清是网络断了还是对方在忙。
                "sess_state": sess_state,
                "sess_error": sess_err,
                "lateral_m": round(lateral_m, 1) if lateral_m is not None else None,
                "lap_frames": len(self._lap_buf),
                "dropped_by_gate": self._st_gate.get("dropped", 0),
                # 被面板开关关掉的分组条数（与"闸门竞争掉"分开，见 gate.py）
                "muted_by_gate": self._st_gate.get("muted", 0),
                "sector_len_m": (round(self._sector_len_m, 1)
                                 if self._sector_len_m else None),
                "lap_samples": len(self._sectors.lap_totals),
                "fuel_samples": len(self._fuel.values),
                "corners": self._corners.to_dict(),
                "corner_habit": self._corners.habit(),
                "corners": self._corners.to_dict(),
                "corner_habit": self._corners.habit(),
                "source_error": getattr(self.src, "last_error", None),
                # R2.2 云状态：degraded 等 Observability 进 state，仪表盘可显示
                "cloud": self.narrator.status(),
                # R3 云 TTS 状态：enabled / 缓存命中 / 队列深度 / 费用。
                # 前端靠 `enabled` 决定"要不要等 tts_url 一会儿再决定用哪条嗓子"。
                "tts": self.tts.status(),
                "wheel_radius_m": {
                    "front": round(self._st_rules["radius"]["front"], 4)
                    if self._st_rules["radius"]["front"] else None,
                    "rear": round(self._st_rules["radius"]["rear"], 4)
                    if self._st_rules["radius"]["rear"] else None,
                    "samples": self._st_rules["radius"]["n"],
                },
            },
        )

    def _trim_lap_buffer(self, newest: Frame) -> None:
        """按**时长**裁剪本圈缓冲（帧率不同，固定帧数是错的）。

        缓冲必须能装下**一整圈**：分段用时、油耗、自攒参考圈都要用全圈的帧。
        实时侧 10Hz 时一圈约 900 帧，60Hz 的 jsonl 是 5500 帧 —— 差 6 倍，
        用一个固定帧数不可能同时对。
        """
        buf = self._lap_buf
        span = newest.lap_time_s - buf[0].lap_time_s
        if span <= self.cfg.lap_buffer_s and len(buf) <= self.cfg.lap_buffer_hard:
            return
        keep_limit = self.cfg.lap_buffer_hard - 1000
        cut = 0
        while cut + 2 < len(buf) and (
                newest.lap_time_s - buf[cut].lap_time_s > self.cfg.lap_buffer_s
                or len(buf) - cut > keep_limit):
            cut += 1
        if cut:
            del buf[:cut]

    def _maybe_request_ref(self, now: float) -> None:
        """参考圈：拿后台已发现的场次去要剖面。**这里不发任何 HTTP。**"""
        sess = self.refs.tick_session(now)
        if not sess or not sess.get("file"):
            return
        key = self.refs.key_for(sess)
        if key != self._ref_sess_key:
            # 🔴 「场次文件变了」和「同一场里最快圈刷新了」必须分开处理。
            #    参考圈的 key 包含 best_lap_s（刷新了要重取剖面），但**只有
            #    真的换了场次文件**才该作废自攒参考圈与帧缓冲。
            #    曾经两者混在一起，后果是：直播场次里每跑完一圈 best_lap_s 就变，
            #    于是每圈跑到 ~15s（下次轮询）时帧缓冲被清空 ——
            #    分段统计永远是 0 圈、自攒参考圈永远建不起来，
            #    而表面上一切正常（指令照回、状态照出）。
            file_changed = (self._ref_sess_key is None
                            or key[0] != self._ref_sess_key[0])
            self._ref_sess_key = key
            if file_changed:
                # 换场次 = 换赛道：自攒的旧参考圈必须作废，否则会拿上一条赛道的
                # 折线去做「最近点定位」，结果是**静默**地把车定位到错误的位置。
                self._self_ref = None
                self._lap_buf = []
                self._sector_len_m = None      # 换赛道 → 分段基准也要重来
                self._sectors = SectorTracker(self.rules.cfg.sectors_n)
                self._corners = CornerTracker(          # 弯窗口也跟着换赛道重来
                    min_laps=self.rules.cfg.corner_min_laps,
                    min_loss_s=self.rules.cfg.corner_min_loss_s)
                self._corners = CornerTracker(          # 弯窗口也跟着换赛道重来
                    min_laps=self.rules.cfg.corner_min_laps,
                    min_loss_s=self.rules.cfg.corner_min_loss_s)
            self.refs.request(sess)

    def _local_lap_stats(self, lap: int) -> None:
        """一圈跑完：本地算分段用时与油耗。

        **不调任何接口**。Dash 有 `/sectors` 和 `/pitstops`，但直播场次上调它们
        每次都要重解析整场 jsonl（`_load_frames` 按 mtime/size 缓存，而直播文件
        一直在变），一场 20 万帧要 2s+ 且随比赛进行越来越贵 —— 而这些数
        从手上的实时帧就能算，精度足够回答"哪段慢了 0.4 秒"。
        """
        res = lap_result(self._lap_buf, lap=lap,
                         n_sectors=self.rules.cfg.sectors_n,
                         expected_len_m=(self._sector_len_m
                                         or (self._current_ref().length_m
                                             if self._current_ref() else None)),
                         fuel=self._fuel)
        self._prev_lap = res
        if not res.ok or res.lap_time_s < self.cfg.min_lap_s:
            return
        if self._sector_len_m is None:
            self._sector_len_m = res.length_m
        self._sectors.add(res.sectors, res.lap_time_s)
        # —— 每弯累积（R1.6）：用同一份弧长/时间算每个弯相对参考亏多少 ——
        #
        # 🔴 这段**只能出现一次**。它曾经因为生成脚本跑了两遍而重复，
        #    后果是每圈的损失被记两次、`laps` 翻倍（"连续 3 圈"变成"连续 6 圈"），
        #    而当时的测试只断言了"有没有播报"、没断言样本数，所以全绿放行。
        #    现在 tests/test_engine.py::test_lap_counted_once_per_lap 锁死这一点。
        ref = self._current_ref()
        if ref is not None and res.s_arr:
            if not self._corners.windows:
                self._corners.setup(ref)     # 窗口一生成一次，跨圈钉死
            if self._corners.windows:
                w = self._corners.windows
                self._corners.add(corner_losses(res.s_arr, res.t_arr, w, ref))
                # 同一批窗口的"损失主要出在刹车区吗" —— 给 next_focus 的证据
                self._corners.add_shares(
                    corner_loss_split(res.s_arr, res.t_arr, w, ref))

    def _corner_snapshot(self) -> dict[str, Any] | None:
        habit = self._corners.habit()
        if habit is None and not self._corners.losses:
            return None
        return {"habit": habit,
                "corners": len(self._corners.windows),
                "tracked": sorted(self._corners.losses),
                "min_laps": self._corners.min_laps}

    def _fuel_snapshot(self) -> dict[str, Any] | None:
        per = self._fuel.per_lap
        if per is None:
            return None
        level = self._lap_buf[-1].fuel_pct if self._lap_buf else 0.0
        return {"per_lap": round(per, 3), "level": round(level, 2),
                "laps_left": (round(self._fuel.laps_left(level), 2)
                              if self._fuel.laps_left(level) is not None
                              else None),
                "samples": len(self._fuel.values)}

    def _finalize_lap(self, lap: int) -> None:
        """一圈跑完：先本地结算，再用它更新自攒参考圈（降级路径）。

        只在**更快**时才替换参考圈 —— 参考圈的定义是「跑到最好的那一圈」。
        """
        # 🔴 `lap <= 0` 是菜单/维修区/出场状态（GT7 用 0 表示"还没进入计时圈"）。
        #    这种圈啥都不算（分段、油耗、每弯损失、自攒参考圈全跳过）。
        if lap <= 0:
            return
        # 🔴 本圈本地统计照常进行：分段用时、油耗、每弯损失累积都依赖全圈帧，
        #    第 1 圈的习惯该记还得记（next_focus 那套跨圈累积不能断）。
        self._local_lap_stats(lap)
        buf = [x for x in self._lap_buf if x.coords_ok]
        if len(buf) < self.cfg.self_ref_min_frames:
            return
        try:
            cand = RefLap.from_frames(buf, lap=lap,
                                      step_m=self.cfg.profile_step_m)
        except ValueError:
            return
        if cand.lap_time_s < self.cfg.min_lap_s:
            return                    # 残圈（出场/被切断），不配当参考
        # 🔴 `lap < self_ref_min_lap` 的圈**只跳过自攒参考圈**，不影响上面的统计：
        #    第 1 圈通常是出场/暖胎圈，慢且不具代表性，拿它当参考会让第 2~3 圈的
        #    「刹车点 / 弯心速度」全建在慢圈上 → 用户感知的"前几圈瞎播报"。
        #    默认 `self_ref_min_lap = 2` 跳过第 1 圈（时间赛第 1 圈就是飞行圈时
        #    改成 1 即可）。自攒参考圈最早从第 2 圈跑完才开始建。
        if lap < self.cfg.self_ref_min_lap:
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
