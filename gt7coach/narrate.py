# -*- coding: utf-8 -*-
"""
措辞编排层（R2.2）——“模板 → 云润色 → 数字白名单 → 失败回落模板”。
========================================================================

这是 `phrases.py`（本地模板）与云 LLM 之间的那一层。规则层（`rules.py`）
只把 `facts` 交给这里，拿到一句「人话」就走，完全不知道背后有没有云。

🔴 四条硬规则（与方案 §0 一致）：

  1. **数字只能来自本地 facts**。云句必须过**四道闸**（都在 `phrases.py`，
     任何一道不过 → 丢弃云句、回落模板、计数）：
       ① `invented_numbers` —— 数从哪来（编数字）
       ② `missing_mandatory` —— 少说了主体（facts 有圈速而句子没说）
       ③ `invented_advice`   —— 没授权的动作指令（"注意补油"）
       ④ `misattributed`     —— 数归谁（把 vs_ref 的 0.42 安到"第二段"头上）
     这是**代码层拦截**，prompt 约束不住 LLM 的这四类毛病。
  2. **失败永远回落模板，绝不静默**。云不可用体现为 `degraded` 状态位 +
     继续用本地句子，玩家不该感知到云的存在。
  3. **预算闸**。per_lap / per_session / per_day 超了直接走模板（如实标注"自律非强制"）。
  4. **不新增依赖**。cloud.py 已全是 urllib。

可靠性（原计划 R2.3「云接入的第二天」必须补上的部分，这里一并落地，
因为掉网 robustness 是「回落模板」能成立的前提）：
  · 幂等缓存：同 (prompt版本 + key + facts) 直接回缓存
  · 熔断退避：连续 3 次失败 → 冷却 60s（指数到 300s 封顶）→ 半开放 1 探测
  · 主备切换：主家熔断时顺位顶上，切换记进日志（switches 计数）
  · 可观测：degraded 状态位、num_violations、tokens、估算费用、延迟全进 status()
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from . import cloud, phrases, prompts


# —— 配置 ————————————————————————————————————————————————

@dataclass
class CloudConfig:
    """cloud.json 的对应结构。明文 key 绝不进文件，只写**环境变量名**。"""
    enabled: bool = False
    provider: str = cloud.DEFAULT_PROVIDER
    base_url: str = ""                       # 空 → 用预设厂商的 base_url
    model: str = ""                          # 空 → 用预设厂商的默认 model
    api_key_env: str = "GT7_COACH_LLM_KEY"   # 🔴 只存变量名
    timeout_s: float = 2.0
    fallbacks: list[str] = field(default_factory=list)
    # 预算闸。per_lap 默认 5＝R2.2 阶段允许每个 B 档 key 每圈各润色一次；
    # R2.4 的「每圈一句话综合建议」会另行把有效云调用压到 ~1 次/圈。
    limits: dict[str, int] = field(default_factory=lambda: {
        "per_lap": 5, "per_session": 30, "per_day": 300})
    # 允许走云的 key；空 → 全部 phrases.RENDERERS（即所有 B 档键）。
    keys: list[str] = field(default_factory=list)
    # 计价口径（元 / 百万 token）。默认值 = 百炼 qwen-flash 华北2（北京）官方价
    # （2026-10 查证：输入 0.15、输出 1.5 元/百万 token）。
    # 🔴 换厂商/换模型**必须同步改这两项**，否则 /api/v1/coach/cloud 报的费用是错的。
    #    前车之鉴：这里曾把方案 §5 里 TTS 的「1.4 元/万字符」当成 token 单价硬编码，
    #    实测高估约 600 倍（一次 300 token 的调用被报成 ¥0.042，真实 ¥0.00007）。
    price_in_yuan_per_mtok: float = 0.15
    price_out_yuan_per_mtok: float = 1.5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CloudConfig":
        """从宽 dict 安全构造：只取已知字段，类型不对就用默认，绝不抛。"""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean: dict[str, Any] = {}
        for k in known:
            if k in d:
                v = d[k]
                if k == "enabled":
                    clean[k] = bool(v)
                elif k in ("timeout_s", "price_in_yuan_per_mtok",
                           "price_out_yuan_per_mtok"):
                    clean[k] = float(v)
                elif k in ("fallbacks", "keys"):
                    clean[k] = list(v) if isinstance(v, (list, tuple)) else []
                elif k == "limits":
                    clean[k] = {kk: int(vv) for kk, vv in (v or {}).items()
                                if kk in ("per_lap", "per_session", "per_day")}
                else:
                    clean[k] = v
        return cls(**clean)


# 熔断参数（与方案 §3 一致）
_CB_BASE_S = 60.0          # 首次冷却 60s
_CB_MAX_S = 300.0          # 封顶 300s
_CB_TRIP = 3               # 连续失败几次触发
_CACHE_MAX = 256           # 幂等缓存条目上限
_TTL_CAP_NOTE = None       # 占位（保持与 phrases 的预算口径一致，这里不另立）


class Narrator:
    """把「facts + key」变成一句播报文本，可选经云润色。"""

    def __init__(self, config_path: str | None = None,
                 cfg: CloudConfig | None = None,
                 clock: Any = time.monotonic):
        self._path = config_path
        self._clock = clock
        self._lock = threading.Lock()
        # 配置：优先显式传入的 cfg，其次尝试从文件加载，都没有 → 禁用态
        self._cfg = cfg or (self._load_file(force=True) if config_path else None) \
            or CloudConfig()
        self._mtime: float | None = None
        if config_path and os.path.exists(config_path):
            try:
                self._mtime = os.stat(config_path).st_mtime
            except OSError:
                self._mtime = None
        self._st: dict[str, Any] = self._fresh_state()
        # 幂等缓存：cache_key -> 润色后的文本
        self._cache: dict[str, str] = {}

    # —— 配置热加载 ————————————————————————————————————

    def _load_file(self, force: bool = False) -> CloudConfig | None:
        if not self._path or not os.path.exists(self._path):
            return None
        try:
            m = os.stat(self._path).st_mtime
        except OSError:
            return None
        if not force and m == self._mtime:
            return None
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self._mtime = m
            return CloudConfig.from_dict(d)
        except (OSError, json.JSONDecodeError):
            return None

    def _reload_if_changed(self) -> CloudConfig:
        """每次调用前 stat 一次 cloud.json，mtime 变了就重读（微秒级开销）。"""
        fresh = self._load_file()
        if fresh is not None:
            self._cfg = fresh
        return self._cfg

    @property
    def config_path(self) -> str | None:
        """cloud.json 的路径（None = 启动时没配 --cloud → 全程禁用态）。

        给写接口用：模型名要落到**唯一真值源**上，不能只存内存（重启就丢，
        用户会觉得"我明明设过"）。
        """
        return self._path

    def reload(self) -> bool:
        """强制重读 cloud.json。写完文件后立刻见效，不用等下一次 render。

        返回是否读到了一份合法配置。与 `_reload_if_changed` 的区别：
        那个靠 mtime 比大小，而**写完文件立刻读**时 mtime 可能没变
        （同秒内写入 + 文件系统 mtime 精度只有 1s 的场景），所以这里 force。
        """
        if not self._path:
            return False
        fresh = self._load_file(force=True)
        if fresh is None:
            return False
        with self._lock:
            self._cfg = fresh
        return True

    # —— 预算 / 熔断 状态 ——————————————————————————————————

    @staticmethod
    def _fresh_state() -> dict[str, Any]:
        return {
            "calls": 0, "errors": 0, "num_violations": 0, "tokens": 0,
            # 计费要分输入/输出两档单价，所以分开记（tokens = 两者之和，保持兼容）
            "tokens_prompt": 0, "tokens_completion": 0,
            "no_key": 0, "cache_hits": 0, "budget_drops": 0, "switches": 0,
            # 白名单查不出的三种毛病，各自单独计数（见 phrases 的"第二~四道闸"）
            "drop_missing": 0,      # 丢了主体（facts 里有圈速，云句里没说）
            "drop_advice": 0,       # 编了建议（facts 没授权的进站/补油类指令）
            "drop_misattr": 0,      # 张冠李戴（把某个实体的值安到别的实体头上）
            "last_guard": None,     # 最近一次被闸掉的具体内容，便于排查
            "last_switch_to": None, "last_latency": None,
            "degraded": False,
            "day": date.today().isoformat(), "calls_day": 0,
            "calls_session": 0,
            "lap": None, "calls_lap": 0,
            "consec_fail": 0, "cooldown_until": 0.0,
        }

    def note_lap(self, lap: int) -> None:
        """圈变化时由引擎调用：重置每圈预算计数。"""
        with self._lock:
            if self._st["lap"] != lap:
                self._st["lap"] = lap
                self._st["calls_lap"] = 0

    def _budget_ok(self, cfg: CloudConfig) -> bool:
        """检查并扣减预算。任一上限触顶 → 返回 False（调用方回落模板）。"""
        st = self._st
        today = date.today().isoformat()
        if st["day"] != today:           # 跨天：重置日预算
            st["day"] = today
            st["calls_day"] = 0
        lim = cfg.limits
        if st["calls_day"] >= lim.get("per_day", 10 ** 9):
            return False
        if st["calls_session"] >= lim.get("per_session", 10 ** 9):
            return False
        if st["calls_lap"] >= lim.get("per_lap", 10 ** 9):
            return False
        st["calls_day"] += 1
        st["calls_session"] += 1
        st["calls_lap"] += 1
        return True

    def _in_cooldown(self) -> bool:
        return self._clock() < self._st["cooldown_until"]

    def _on_success(self) -> None:
        st = self._st
        st["consec_fail"] = 0
        st["cooldown_until"] = 0.0
        st["degraded"] = False

    def _on_failure(self) -> None:
        st = self._st
        st["consec_fail"] += 1
        st["degraded"] = True
        if st["consec_fail"] >= _CB_TRIP:
            back = min(_CB_BASE_S * (2 ** (st["consec_fail"] - _CB_TRIP)), _CB_MAX_S)
            st["cooldown_until"] = self._clock() + back

    # —— 主入口 ————————————————————————————————————————

    def render(self, key: str, facts: dict[str, Any]) -> str:
        """给定 key + facts，返回一句播报文本（本地模板或云润色）。

        调用方（rules.py）对云的存在毫不知情：无论成功/失败/禁用，
        这里都返回一个**合法、数字都在 facts 里**的句子。
        """
        cfg = self._reload_if_changed()

        # ① 禁用 / 该 key 不在云端候选集 → 直接本地模板（零网络）
        if not cfg.enabled or key not in self._cloudable(cfg):
            return phrases.render(key, facts)

        # ② 熔断冷却中 → 回落模板（不消耗预算、不打外呼）
        if self._in_cooldown():
            with self._lock:
                self._st["budget_drops"] += 1   # 复用作「被闸掉的请求」计数
            return phrases.render(key, facts)

        # ③ 预算闸
        if not self._budget_ok(cfg):
            with self._lock:
                self._st["budget_drops"] += 1
            return phrases.render(key, facts)

        # ④ 幂等缓存命中 → 直接回（不重复花钱）
        ck = self._cache_key(key, facts)
        with self._lock:
            cached = self._cache.get(ck)
        if cached is not None:
            with self._lock:
                self._st["cache_hits"] += 1
            return cached

        # ⑤ 真去云上润色；任何失败都回落模板
        polished = self._polish(cfg, key, facts)
        if polished is None:
            return phrases.render(key, facts)
        with self._lock:
            if len(self._cache) >= _CACHE_MAX:
                self._cache.pop(next(iter(self._cache)))   # 简单 FIFO
            self._cache[ck] = polished
        return polished

    def _cloudable(self, cfg: CloudConfig) -> set[str]:
        if cfg.keys:
            return set(cfg.keys)
        return set(phrases.RENDERERS.keys())

    @staticmethod
    def _cache_key(key: str, facts: dict[str, Any]) -> str:
        blob = prompts.PROMPT_VERSION + "|" + key + "|" + json.dumps(
            facts, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _polish(self, cfg: CloudConfig, key: str,
                facts: dict[str, Any]) -> str | None:
        """尝试云润色。返回润色文本或 None（None=回落模板）。"""
        key_val = os.environ.get(cfg.api_key_env, "")
        if not key_val:
            with self._lock:
                self._st["no_key"] += 1
            return None

        messages = prompts.messages_for(key, facts)
        providers = [cfg.provider] + list(cfg.fallbacks)
        last_err_kind = "error"
        for i, prov in enumerate(providers):
            p = cloud.resolve_provider(prov)
            if i == 0:
                # 🔴 主家：**用户在 cloud.json 里显式填的 model/base_url 优先**。
                #    这是"让用户自己选模型"的唯一通道。原先是 `p.get("model")
                #    or cfg.model`，后果是用户在 cloud.json 把 model 改成 kimi-k3
                #    却**完全无效** —— 预设厂商永远用自己的 model，用户以为换了、
                #    实际还在用默认那个（实测抓出来的）。之所以长期没暴露：
                #    测试全用未知厂商 provider="test"，那条路本来就只能落到
                #    cfg.model，正好绕开了这个分支。
                base_url = cfg.base_url or p.get("base_url")
                model = cfg.model or p.get("model")
            else:
                # 🔴 备用厂商：必须用**它自己**的预设，不能沿用主家的 ——
                #    顺序写反会让主家熔断后备用仍打主家的 URL（实测抓出来的）。
                base_url = p.get("base_url") or cfg.base_url
                model = p.get("model") or cfg.model
            if not base_url or not model:
                # 未知厂商又没在 cloud.json 显式给 base_url/model → 跳过
                continue
            try:
                reply = cloud.chat(base_url, model, key_val, messages,
                                   timeout_s=cfg.timeout_s)
            except cloud.CloudError as e:
                last_err_kind = e.kind
                # 主备切换：记一次切换，试下一个厂商
                if i < len(providers) - 1:
                    with self._lock:
                        self._st["switches"] += 1
                        self._st["last_switch_to"] = providers[i + 1]
                continue

            # 成功拿到回复
            with self._lock:
                self._st["calls"] += 1
                self._st["tokens"] += reply.usage.prompt + reply.usage.completion
                self._st["tokens_prompt"] += reply.usage.prompt
                self._st["tokens_completion"] += reply.usage.completion
                self._st["last_latency"] = reply.latency_s

            # 🔴 数字白名单断言：句子里的每个数都要能在 facts 里找到
            viol = phrases.invented_numbers(reply.content, facts)
            if viol:
                with self._lock:
                    self._st["num_violations"] += 1
                    self._st["errors"] += 1
                self._on_failure()
                return None                     # 丢弃云句，回落模板

            # 🔴 第二~四道闸。白名单只问"句中的数字是不是来自 facts"，问不出这三件事：
            #    · 少说了一个数（丢主体）—— 实测云句两次都丢了圈速
            #    · 编了一句没有数字的指令（"注意补油"）—— 白名单对它完全无感
            #    · 张冠李戴（把 vs_ref 的 0.42 安到"第二段"头上）——
            #      两个数都来自 facts，白名单照样放行
            #    三条都在这里丢弃云句、回落模板。见 phrases 各同名函数的注释。
            missing = phrases.missing_mandatory(reply.content, facts)
            advice = phrases.invented_advice(reply.content, facts)
            misattr = phrases.misattributed(reply.content, facts)
            if missing or advice or misattr:
                with self._lock:
                    self._st["errors"] += 1
                    if missing:
                        self._st["drop_missing"] += 1
                    if advice:
                        self._st["drop_advice"] += 1
                    if misattr:
                        self._st["drop_misattr"] += 1
                    self._st["last_guard"] = {"text": reply.content,
                                              "missing": missing,
                                              "advice": advice,
                                              "misattr": misattr}
                # ⚠️ 与"编数字"同一条处理路径：也记一次失败。
                #    代价是连着 3 次内容问题会触发熔断冷却，把云播报整体停 60s
                #    —— 这是**有意**的（模型连续不守规矩就别再花钱了），
                #    代价与收益都从 /api/v1/coach/cloud 的 fallback_ratio 看得见。
                self._on_failure()
                return None                     # 丢弃云句，回落模板

            self._on_success()
            return reply.content

        # 所有厂商都失败
        with self._lock:
            self._st["errors"] += 1
            if last_err_kind in ("nokey",):
                pass
        self._on_failure()
        return None

    # —— 可观测 ————————————————————————————————————————

    def model_info(self) -> dict[str, Any]:
        """当前**实际会发出去**的模型名 + 它是否在免费名单里。

        🔴 为什么要单独抽出来：**"配置里填的"和"真发出去的"可能不是一回事** ——
           主家用 `cfg.model`（用户填的），备用家各自用自己的预设（见 _polish）。
           界面/状态要显示的是**真发出去的那个**，否则用户改完看不到变化，
           会以为"改了没生效"——那正是这次修掉的那个 bug 的表现。
        """
        cfg = self._cfg
        if cfg is None:
            return {"model": "", "provider": None, "from_user": False,
                    "is_free": False, "free": None, "warning": None,
                    "api_key_env": "", "has_key": False}
        preset = cloud.resolve_provider(cfg.provider).get("model") or ""
        model = cfg.model or preset
        return {
            "model": model,
            "provider": cfg.provider,
            "from_user": bool(cfg.model),       # True=用户填的，False=厂商预设
            "is_free": cloud.is_free_model(model),
            "free": cloud.free_info(model),
            "warning": cloud.model_warning(model, provider=cfg.provider),
            # 🔴 只报**变量名**与"有没有设"，绝不回显 key 本身
            "api_key_env": cfg.api_key_env,
            "has_key": bool(os.environ.get(cfg.api_key_env, "")),
        }

    def status(self) -> dict[str, Any]:
        """给 GET /api/v1/coach/cloud 用：enabled / provider / 今日调用 /
        token / 估算费用 / 降级状态 / 违规计数 + 其它运维指标。"""
        with self._lock:
            st = dict(self._st)
            cfg = self._cfg
            mi = self.model_info()
        # 估算费用：输入/输出**分档计价**（云 API 就是两档单价），单价取自
        # cfg.price_*_yuan_per_mtok，随 cloud.json 可改。usage 缺失时 narration
        # 层已按字符估算并标 estimated，这里照算 —— 量级正确即可，不当账用。
        if cfg:
            est_yuan = round(
                st["tokens_prompt"] / 1e6 * cfg.price_in_yuan_per_mtok
                + st["tokens_completion"] / 1e6 * cfg.price_out_yuan_per_mtok, 6)
        else:
            est_yuan = 0.0
        return {
            "enabled": bool(cfg and cfg.enabled),
            "provider": cfg.provider if cfg else None,
            "fallbacks": list(cfg.fallbacks) if cfg else [],
            "degraded": st["degraded"] or self._in_cooldown(),
            "calls": st["calls"],
            "calls_today": st["calls_day"],
            "cache_hits": st["cache_hits"],
            "budget_drops": st["budget_drops"],
            "errors": st["errors"],
            "num_violations": st["num_violations"],
            # —— 四种"云句拿到了但不敢用"的原因，分开计 ——
            #    fallback_ratio = 被丢掉的云句 / 拿到的云句。这是判断
            #    "提示词调好没有""闸门是不是太严"的唯一量化口径：
            #    接近 0 = 云句基本都能用；接近 1 = 云在空烧钱，不如关掉。
            "drop_missing": st["drop_missing"],
            "drop_advice": st["drop_advice"],
            "drop_misattr": st["drop_misattr"],
            "cloud_rejects": (st["num_violations"] + st["drop_missing"]
                              + st["drop_advice"] + st["drop_misattr"]),
            "fallback_ratio": (round((st["num_violations"] + st["drop_missing"]
                                      + st["drop_advice"] + st["drop_misattr"])
                                     / st["calls"], 4)
                               if st["calls"] else 0.0),
            "last_guard": st["last_guard"],
            "no_key": st["no_key"],
            "switches": st["switches"],
            "last_switch_to": st["last_switch_to"],
            "tokens": st["tokens"],
            "tokens_prompt": st["tokens_prompt"],
            "tokens_completion": st["tokens_completion"],
            "est_cost_yuan": est_yuan,
            "price_yuan_per_mtok": (
                {"in": cfg.price_in_yuan_per_mtok,
                 "out": cfg.price_out_yuan_per_mtok} if cfg else {}),
            "last_latency_s": st["last_latency"],
            "limits": dict(cfg.limits) if cfg else {},
            "prompt_version": prompts.PROMPT_VERSION,
            # —— 模型（"用户填了什么 / 真发出去的是哪个 / 是否在免费额度内"）——
            #    model_warning 非空 = 这个模型不在免费名单里或快到期了。
            #    🔴 只提示、**不拦截**（用户明确要求：非免费模型只警告不拦）。
            "model": mi["model"],
            "model_from_user": mi["from_user"],
            "model_is_free": mi["is_free"],
            "model_free": mi["free"],
            "model_warning": mi["warning"],
            # 只报变量名与"有没有设"，绝不回显 key
            "api_key_env": mi["api_key_env"],
            "has_key": mi["has_key"],
        }
