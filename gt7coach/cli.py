# -*- coding: utf-8 -*-
"""命令行入口。

    python -m gt7coach serve --dash http://192.168.43.18:8787
    python -m gt7coach demo                    # 合成赛道，不连仪表盘
    python -m gt7coach once --dash ...         # 单次 tick，排障用
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .engine import CoachConfig, CoachEngine
from .server import DEFAULT_PORT, CoachService, make_server
from .source import HttpSource, ReplaySource


def _fix_console() -> None:
    """Windows 控制台默认 GBK：print 里出现 GBK 码表外的字符（⚠/emoji）会
    UnicodeEncodeError 直接崩。这里只影响显示，不动数据。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")     # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def _build(args) -> CoachEngine:
    cfg = CoachConfig(poll_interval_s=args.interval)
    if args.demo:
        from .synth import synth_lap_frames, synth_profile
        frames = synth_lap_frames(laps=3)
        prof = synth_profile()
        src = ReplaySource(frames, profile=prof, loop=True)
        print(f"[demo] 合成赛道 {len(frames)} 帧（10Hz，3 圈），"
              f"圈长 {prof['length_m']:.0f} m，圈速 {prof['lap_time_s']:.2f} s")
        # 用回放自己的比赛时钟，否则冷却按墙上时钟算（回放一秒钟跑完整圈）
        return CoachEngine(src, cfg, clock=src.clock)
    return CoachEngine(HttpSource(args.dash, timeout=args.timeout), cfg)


def main(argv: list[str] | None = None) -> int:
    _fix_console()
    ap = argparse.ArgumentParser(prog="gt7coach",
                                 description="GT7 赛道工程师（实时副驾）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, helptext in (("serve", "起 HTTP 服务，供仪表盘取数"),
                           ("demo", "用合成赛道跑一遍，不连仪表盘"),
                           ("once", "跑一次 tick 并把状态打到 stdout")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--dash", default="http://192.168.43.18:8787",
                       help="GT7 Dash 的基地址")
        p.add_argument("--port", type=int, default=DEFAULT_PORT)
        p.add_argument("--bind", default="0.0.0.0")
        p.add_argument("--interval", type=float, default=0.1,
                       help="tick 间隔秒数，缺省 0.1（10Hz）")
        p.add_argument("--timeout", type=float, default=3.0)
        p.add_argument("--verbose", action="store_true")
        if name == "demo":
            p.set_defaults(demo=True)
        if name == "serve":
            p.add_argument("--ticks", type=int, default=0,
                           help="跑 N 次 tick 后退出（0 = 一直跑）")

    args = ap.parse_args(argv)
    if not getattr(args, "demo", False):
        args.demo = False

    engine = _build(args)

    if args.cmd == "once":
        for _ in range(5):            # 多跑几次让参考圈有机会到位
            st = engine.tick()
            time.sleep(args.interval)
        print(json.dumps(st.to_dict(), ensure_ascii=False, indent=2))
        return 0

    svc = CoachService(engine, interval_s=args.interval)
    srv = make_server(svc, host=args.bind, port=args.port,
                      verbose=args.verbose)
    svc.start()
    print(f"[gt7coach] 赛道工程师已启动  http://{args.bind}:{args.port}/"
          f"  (自检页 /)")
    print(f"[gt7coach] 数据源 {getattr(engine.src, 'base', '合成回放')}"
          f"  tick {args.interval * 1000:.0f} ms")
    try:
        if args.cmd == "serve" and args.ticks:
            for _ in range(args.ticks):
                time.sleep(args.interval)
                st = svc.state()
                for u in st["say"]:
                    print(f"[{u['key']}] {u['text']}")
            return 0
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[gt7coach] 收到中断，退出")
    finally:
        svc.stop()
        srv.server_close()
    return 0
