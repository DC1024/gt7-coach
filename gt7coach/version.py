# -*- coding: utf-8 -*-
"""版本号的唯一定义处。

单独成模块而不是写在 `__init__.py` 里，是为了让子模块（如 `server.py`
要拼 HTTP `Server:` 头）能直接 `from .version import version`，而不必
从包 `__init__` 回环导入自己所在的包 —— 那会在包初始化到一半时触发
隐性循环。

改版本只改这里一处。`__init__.__version__` 与 `pyproject.toml` 的
`version` 都应与本文件保持一致。

PEP 440 里预发布的规范写法是 `1.0.0b1`；对应的发行标识（git tag /
Release 名）是 `v1.0.0-beta.1`。两者指同一个版本。
"""

version = "1.0.0b1"
__all__ = ["version"]
