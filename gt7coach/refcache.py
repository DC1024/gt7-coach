# -*- coding: utf-8 -*-
"""
参考圈本地缓存 —— 把你跑过的**最好一圈**按「赛道形状指纹 + 车型」存盘。
========================================================================

目的：跨场次 / 跨重启复用你的最好参考圈，省掉 `history_best` 去服务端翻历史
（冷态 3~11 秒）的延迟，也让教练在「服务端历史是冷的」时仍有靠谱参考。

🔴 安全底线（与 `history_best` 的 shape_distance 校验同一把尺）：
    缓存**绝不**在没核对赛道形状之前被采用。副驾拿到缓存参考圈后，必须用它与
    「服务端权威几何」做 `shape_distance` 比对，形状对不上就丢弃 ——
    否则换条赛道误用旧参考圈 = 静默定位错误（比没有参考圈更糟）。

设计要点：
  · key = `(car_name, track_fingerprint)`。指纹由 `RefLap.track_fingerprint()`
    从折线形状算出，同赛道跨场次一致、不同赛道不同；不依赖服务端给的 track_id
    （实测 10 个场次只有 2 个有可靠 track_id）。
  · 序列化走 `RefLap.to_profile()` / `from_profile()`（已有的往返通道），
    并在文件里额外存一份指纹，加载时核对，防串改/损坏。
  · 目录建不了就**禁用并静默**，绝不抛 —— 缓存是加速项，不是必需项。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .refindex import RefLap

_META_FP = "__fp__"


class RefCache:
    """把参考圈按 (车型, 赛道指纹) 存到本地目录。"""

    def __init__(self, dir_path: str | None):
        self.dir = Path(dir_path) if dir_path else None
        self._lock = threading.Lock()
        if self.dir:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.dir = None  # 建不了目录就禁用，绝不抛

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def _path(self, car: str, fp: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_"
                       else "_" for c in (car or "unknown"))
        return self.dir / f"ref_{safe}_{fp}.json"

    def save(self, ref: RefLap, car: str) -> None:
        """存下一条参考圈（覆盖同 key 的旧值）。失败静默。"""
        if not self.enabled or not ref.xs:
            return
        fp = ref.track_fingerprint()
        if not fp:
            return
        payload = ref.to_profile()
        payload[_META_FP] = fp
        try:
            with self._lock, self._path(car, fp).open("w", encoding="utf-8") as f:
                json.dump(payload, f)
        except OSError:
            pass

    def load(self, car: str, fp: str) -> RefLap | None:
        """按 (车型, 指纹) 取回参考圈。指纹对不上 / 损坏 / 缺失 → None。"""
        if not self.enabled or not fp:
            return None
        try:
            with self._lock, self._path(car, fp).open("r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(d, dict) or d.get(_META_FP) != fp:
            return None  # 指纹不符：文件被串改 / 损坏 / key 算错了
        try:
            return RefLap.from_profile(d)
        except (ValueError, KeyError):
            return None
