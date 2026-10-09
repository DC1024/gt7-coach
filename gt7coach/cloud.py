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
        "model": "qwen-flash",
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
