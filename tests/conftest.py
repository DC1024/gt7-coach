# -*- coding: utf-8 -*-
"""共用夹具与工具。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def wait_for(pred, timeout: float = 3.0, interval: float = 0.01) -> bool:
    """等条件成立。

    参考圈是**后台线程**取的，测试必须等它落地，不能 sleep 一个魔法数字
    （机器一慢就 flaky）。返回 False 表示超时，调用方自己决定怎么断言。
    """
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(interval)
    return False


@pytest.fixture(scope="session")
def synth():
    from gt7coach import synth as s
    return s
