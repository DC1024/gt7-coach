# -*- coding: utf-8 -*-
"""
云端 LLM 客户端 —— R2.2 的「嗓子」与「措辞」通道。
==================================================

🔴 设计红线（与全仓库一致）：

  1. **只用标准库 urllib**。容器是 `python:3.12-slim`，没有 `requests`，
     也不引入任何第三方依赖。
  2. **显式剥代理**。urllib 默认会读 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量，
     而本机常挂着代理（公司网/科学上网），会让 127.0.0.1 的请求也绕去代理
     然后超时。这是本仓库踩过的坑，必须用 `ProxyHandler({})` 关掉。
  3. **请求只发 5 个字段**，`stream=false`。云端只做「事后措辞」，
     不要把整帧遥测、历史、prompt 之外的东西塞出去。
  4. **失败必须抛 `CloudError`**。narrate 层据此回落模板，玩家不该感知云。
  5. **超时是硬约束**。一次云调用 0.5~2s，250km/h 时 2s=139m，
     `timeout_s` 默认 2.0，超过就当失败 —— 宁可说模板句，也不能卡住主循环。

预设厂商：实测「Bearer 直接过」的 11 家 OpenAI 兼容端点。
  · 零一万物（410，已停服）→ 不进
  · 讯飞（需 HMAC 签名）→ 不支持
  · ollama（本地，非 SaaS）→ 不接
  · "OpenAI 官方域名"不通是预期行为（格式兼容 ≠ 域名可达），写进 UI 提示
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class CloudError(Exception):
    """云调用失败的统一异常。narrate 据此回落模板。"""

    def __init__(self, msg: str, *, kind: str = "error"):
        super().__init__(msg)
        self.kind = kind          # "http" | "timeout" | "net" | "nokey" | "badjson"


@dataclass
class Usage:
    prompt: int = 0
    completion: int = 0
    estimated: bool = False       # usage 缺失时按字符估算，标 true


@dataclass
class CloudReply:
    content: str
    usage: Usage
    latency_s: float


# —— 11 家预设厂商（base_url + 默认 model）—————————————————————————
#
# 字段含义：
#   base_url  —— `/chat/completions` 之前的基地址（chat() 会自动补后缀）
#   model     —— 该厂商的默认模型名（cloud.json 里可覆盖）
#
# 这些是「纯 Bearer」端点：把 `Authorization: Bearer <key>` 直接打过去就过。
# 需要额外签名（讯飞 HMAC）或本地进程（ollama）的不在此列。
PROVIDERS: dict[str, dict[str, str]] = {
    "dashscope": {           # 百炼（默认）
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # 🔴 默认模型必须是**免费额度名单里**的（见下面的 FREE_MODELS）。
        #    原先这里是 `qwen-flash` —— 它**不在**免费名单里，用户只要填了
        #    key 就会静默按量计费。教练是"玩家自己电脑上的小工具"，
        #    不该在用户不知情的时候花钱。
        #    qwen3.8-flash 是名单里 runway 最长的 flash 档（见 FREE_MODELS.until）。
        #    用户可在 cloud.json / 仪表盘卡片的输入框里改成任意模型名；
        #    改成非免费模型**不拦**，但状态里会一直红字提示可能计费。
        "model": "qwen3.8-flash",
    },
    "ark": {                 # 方舟（火山）
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "ep-2025xxxxxxxx",   # 用户需替换为自己的 EndpointId
    },
    "siliconflow": {         # 硅基流动
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "zhipu": {               # 智谱
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
    "moonshot": {            # 月之暗面
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
    },
    "stepfun": {             # 阶跃
        "base_url": "https://api.stepfun.com/v1",
        "model": "step-1-flash",
    },
    "qianfan": {             # 千帆（百度）
        "base_url": "https://qianfan.baidubce.com/v2",
        "model": "ernie-4.0-8k-latest",
    },
    "hunyuan": {             # 混元（腾讯）
        "base_url": "https://api.hunyuan.cloud.tencent.com/v1",
        "model": "hunyuan-lite",
    },
    "sensetime": {           # 商汤
        "base_url": "https://api.sensenova.cn/v1",
        "model": "SenseChat-Turbo",
    },
    "minimax": {             # MiniMax
        "base_url": "https://api.minimax.chat/v1",
        "model": "abab6.5s-chat",
    },
}

DEFAULT_PROVIDER = "dashscope"


# —— #H：给仪表盘 UI 用的「服务商预设」（一键填入三框）—————————————
#
#    与上面 PROVIDERS 的分工：PROVIDERS 是给**代码**用的端点表（resolve_provider），
#    这一份是给**人**看的（含中文 label + 建议的 key 环境变量名），UI 拿它
#    渲染下拉，选中后把 base_url / model / api_key_env 三个框填上。
#
#    🔴 api_key_env 存的是**环境变量名**，明文 key 绝不落盘 —— 全仓库红线，
#       #H 也不例外。用户把 key 放进环境变量，配置文件里只有变量名。
#
#    openai / ollama 不在 PROVIDERS 端点表里没关系：预设填的是 cloud.json 的
#    base_url / model，`narrate._polish` 里用户显式值本来就优先于厂商预设。
#    ollama 本地不校验 key，但 narrate 侧「无 key 即回落模板」，所以 UI 提示：
#    随便设一个非空环境变量（如 GT7_COACH_LLM_KEY=local）当占位即可。
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai":    {"label": "OpenAI 官方",
                  "base_url": "https://api.openai.com/v1",
                  "model": "gpt-4o-mini",
                  "api_key_env": "OPENAI_API_KEY"},
    "deepseek":  {"label": "DeepSeek 深度求索",
                  "base_url": "https://api.deepseek.com/v1",
                  "model": "deepseek-chat",
                  "api_key_env": "DEEPSEEK_API_KEY"},
    "dashscope": {"label": "阿里百炼（默认）",
                  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                  "model": "qwen3.8-flash",
                  "api_key_env": "DASHSCOPE_API_KEY"},
    "zhipu":     {"label": "智谱 AI",
                  "base_url": "https://open.bigmodel.cn/api/paas/v4",
                  "model": "glm-4-flash",
                  "api_key_env": "ZHIPU_API_KEY"},
    "moonshot":  {"label": "月之暗面 Kimi",
                  "base_url": "https://api.moonshot.cn/v1",
                  "model": "moonshot-v1-8k",
                  "api_key_env": "MOONSHOT_API_KEY"},
    "ollama":    {"label": "本地 Ollama",
                  "base_url": "http://127.0.0.1:11434/v1",
                  "model": "qwen2.5:7b",
                  "api_key_env": "GT7_COACH_LLM_KEY"},
}


# —— 百炼「免费额度」模型名单 ——————————————————————————————————
#
# 🔴 为什么把这份名单写进代码：
#    百炼的免费额度是**按模型**发的，超出额度就按量计费。`PROVIDERS` 里的
#    默认模型曾经是 `qwen-flash`（不在免费名单里）→ 用户填了 key 就会
#    **静默扣费**。教练的定位是玩家自己电脑上的小工具，不该悄悄花钱。
#    所以：① 默认值挑名单里的；② 用户改成名单外的模型**只警告不拦**
#    （用户明确要求），状态里 `free_info()` 会一直给出提示。
#
# 数据来源：用户 2026-10-09 从百炼控制台「免费额度」页导出的表格。
#   ⚠️ 额度与到期日**随活动变化，以控制台为准** —— 这里只用于提醒，
#      不参与任何拦截判断，过期了也只是提示不再准确，不会让功能失效。
#   ⚠️ 只收录本项目**真会用到**的两类：大语言模型（云措辞）与语音合成
#      （云 TTS）。控制台里还有视觉/多模态/向量/ASR/sambert 音色等
#      几十项，与本项目无关，不收。
#
# 字段：model -> {kind 类型, quota 额度说明, until 到期日 or None=每月重置}
FREE_MODELS: dict[str, dict[str, Any]] = {
    # —— 大语言模型：云措辞（narrate）走这一类 ——
    "qwen3.8-flash":        {"kind": "大语言模型", "quota": "1M tokens", "until": "2026-11-25"},
    "qwen3.7-flash":        {"kind": "大语言模型", "quota": "1M tokens", "until": "2026-10-23"},
    "qwen3.7-flash-2026-07-15": {"kind": "大语言模型", "quota": "1M tokens", "until": "2026-10-23"},
    "qwen3.8-27b":          {"kind": "大语言模型", "quota": "1M tokens", "until": "2026-11-17"},
    "qwen3.8-max":          {"kind": "大语言模型", "quota": "1M tokens", "until": "2026-11-01"},
    "qwen3.8-max-0902":     {"kind": "大语言模型", "quota": "1M tokens", "until": "2026-12-01"},
    "qwen3.8-2.4t-a95b":    {"kind": "大语言模型", "quota": "1M tokens", "until": "2026-11-12"},
    "deepseek-v4.1-flash":  {"kind": "大语言模型", "quota": "1M tokens", "until": "2026-12-13"},
    "deepseek-v4-flash-0731": {"kind": "大语言模型", "quota": "1M tokens", "until": "2026-10-31"},
    "deepseek-v4-pro-0813": {"kind": "大语言模型", "quota": "1M tokens", "until": "2026-11-13"},
    "glm-5.3":              {"kind": "大语言模型", "quota": "1M tokens", "until": "2026-11-23"},
    "kimi-k3":              {"kind": "大语言模型", "quota": "1M tokens", "until": "2026-11-17"},
    # —— 语音合成：云 TTS（tts.py）走这一类 ——
    #    ⚠️ 当前 TTS 默认模型 `cosyvoice-v3-flash` **不在**名单里（名单里只有
    #       v1）。v1 不支持 `longanyang` 这类 v2/v3 音色，换过去要连音色一起改，
    #       没实测过不敢当默认值 —— 所以 TTS 这边**只警告、不换默认值**。
    #       真要省，把 tts_model 填成 cosyvoice-v1 并换一个 v1 音色（见 README）。
    "cosyvoice-v1":         {"kind": "语音合成", "quota": "10K 字符/月", "until": None},
    "cosyvoice-clone-v1":   {"kind": "语音合成", "quota": "10K 字符/月", "until": None},
    "qwen-audio-3.1-tts-flash": {"kind": "语音合成", "quota": "1M tokens", "until": "2026-12-21"},
    "qwen-audio-3.1-tts-next":  {"kind": "语音合成", "quota": "1M tokens", "until": "2026-12-21"},
    "qwen-audio-3.0-tts-flash": {"kind": "语音合成", "quota": "10K 字符", "until": "2026-10-12"},
    "qwen-audio-3.0-tts-plus":  {"kind": "语音合成", "quota": "10K 字符", "until": "2026-10-12"},
}


def _days_left(until: str | None) -> int | None:
    """距到期还有几天（None = 每月重置/长期有效，不给天数）。"""
    if not until:
        return None
    try:
        import datetime as _dt
        d = _dt.date.fromisoformat(until)
    except (TypeError, ValueError):
        return None
    return (d - _dt.date.today()).days


def free_info(model: str) -> dict[str, Any] | None:
    """这个模型在不在免费名单里？是 → 返回详情（含剩余天数），否 → None。

    🔴 只用于**提示**，不参与拦截：用户明确要求"非免费模型只警告不拦"。
    """
    if not model:
        return None
    info = FREE_MODELS.get(model)
    if info is None:
        return None
    out = dict(info)
    left = _days_left(info.get("until"))
    out["days_left"] = left
    out["expired"] = left is not None and left < 0
    return out


def is_free_model(model: str) -> bool:
    """是否在免费名单内且**没过期**。"""
    info = free_info(model)
    return bool(info) and not info.get("expired")


def model_warning(model: str, *, provider: str = "") -> str | None:
    """非免费模型 → 一句中文提示；免费/没填 → None。

    措辞刻意不写"会扣钱"这种绝对话（额度/单价随时会变），只说"可能计费"。
    """
    if not model:
        return None
    info = free_info(model)
    if info is None:
        return (f"模型 {model} 不在内置免费名单里，可能按量计费。"
                f"（名单只覆盖百炼免费额度，且以控制台为准）")
    if info.get("expired"):
        return (f"模型 {model} 的免费额度已于 {info['until']} 到期，"
                f"继续使用可能按量计费。")
    left = info.get("days_left")
    if left is not None and left <= 7:
        return (f"模型 {model} 的免费额度 {info['until']} 到期，"
                f"还剩 {left} 天。")
    return None


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    """🔴 显式关掉代理。否则本机代理会把请求（含 127.0.0.1）拐去代理而超时。"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def chat(base_url: str, model: str, key: str, messages: list[dict[str, str]],
         *, timeout_s: float = 2.0, opener: urllib.request.OpenerDirector | None = None
         ) -> CloudReply:
    """一次标准的 OpenAI 兼容 chat 调用。

    Args:
        base_url: 基地址（不含 /chat/completions）。
        model:    模型名。
        key:      API key（Bearer token）。
        messages: [{role, content}, ...]。
        timeout_s: 整体超时（连接+读取）。默认 2.0。
        opener:   测试可注入自定义 opener（如指向假服务器）。

    Raises:
        CloudError: 任何失败（HTTP 非 2xx / 超时 / 网络 / 返回结构异常）。
    """
    if not key:
        raise CloudError("no api key (env not set)", kind="nokey")
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 120,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "gt7-coach/0.1",
        },
        method="POST",
    )

    t0 = time.monotonic()
    try:
        # 超时只覆盖「连接+读取」，不覆盖「建立 opener」。2s 一到立刻当失败。
        with (opener or _no_proxy_opener()).open(req, timeout=timeout_s) as r:
            raw = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise CloudError(f"HTTP {e.code}", kind="http") from e
    except TimeoutError as e:
        raise CloudError(f"timeout after {timeout_s}s", kind="timeout") from e
    except urllib.error.URLError as e:
        # URLError 在超时时会包一个 socket.timeout —— 归并到 timeout
        if isinstance(getattr(e, "reason", None), TimeoutError):
            raise CloudError(f"timeout after {timeout_s}s", kind="timeout") from e
        raise CloudError(f"net: {e.reason}", kind="net") from e
    except Exception as e:  # noqa: BLE001
        raise CloudError(f"{type(e).__name__}: {e}", kind="net") from e
    latency = time.monotonic() - t0

    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise CloudError(f"bad response shape: {e}", kind="badjson") from e
    if not isinstance(content, str) or not content.strip():
        raise CloudError("empty content", kind="badjson")

    u = raw.get("usage") or {}
    if u:
        usage = Usage(prompt=int(u.get("prompt_tokens", 0)),
                      completion=int(u.get("completion_tokens", 0)),
                      estimated=False)
    else:
        # 🔴 usage 缺失：按字符估算并标 estimated（真实 token 通常≈字符数/1.6，
        #   中文约 1.5~2 字/token，这里用保守的「字符数」当 completion 估算）。
        usage = Usage(prompt=0, completion=len(content), estimated=True)

    return CloudReply(content=content.strip(), usage=usage, latency_s=round(latency, 3))


def resolve_provider(name: str) -> dict[str, str]:
    """取预设厂商的 base_url/model；未知厂商返回空（由 cloud.json 显式给）。"""
    return PROVIDERS.get(name, {})
