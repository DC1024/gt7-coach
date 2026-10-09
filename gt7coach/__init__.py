# -*- coding: utf-8 -*-
"""GT7 赛道工程师（Race Engineer）—— 实时遥测监控与语音副驾。

设计原则（整个包都服从这三条）：
  1. **判断在本地，措辞在云上**：所有「该不该说、说的是哪个数」都由本地
     确定性代码算（毫秒级、零成本、不会编数字）。云 LLM 只能负责事后措辞，
     且完全可选 —— R1 不接任何云也能用。
  2. **按延迟分档**：硬实时（<300ms）禁止走云。一次云调用 0.5~2s，
     250 km/h 时 2 秒 = 139 米，"刹车点提前 10 米"晚 139 米就是废话。
  3. **不打扰优先于多说话**：冷却 + 每圈上限 + 弯中禁言 + 只报新信息。
"""

from .contract import (COACH_API_VERSION, CoachState, Frame, P_CRITICAL,
                       P_HIGH, P_LOW, P_NORMAL, Utterance, has_coords)
from .engine import CoachConfig, CoachEngine, RefProvider
from .gate import Gate, GateConfig
from .lapstats import (FuelTracker, LapResult, SectorTracker,
                       lap_result, ref_sector_times, sector_times)
from .refindex import RefLap
from .rules import Ctx, RuleConfig, RuleSet, fmt_lap_time
from .source import FileSource, HttpSource, ReplaySource, SourceError

__version__ = "0.1.0"
__all__ = [
    "COACH_API_VERSION", "CoachState", "Frame", "Utterance", "has_coords",
    "P_CRITICAL", "P_HIGH", "P_NORMAL", "P_LOW",
    "CoachEngine", "CoachConfig", "RefProvider",
    "Gate", "GateConfig", "RefLap", "RuleSet", "RuleConfig", "Ctx",
    "fmt_lap_time", "HttpSource", "ReplaySource", "FileSource",
    "SourceError",
    "LapResult", "FuelTracker", "SectorTracker", "lap_result",
    "sector_times", "ref_sector_times",
]
