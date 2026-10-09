# -*- coding: utf-8 -*-
"""命令行入口。

    python -m gt7coach serve --dash http://192.168.43.18:8787
    python -m gt7coach demo                    # 合成赛道，不连仪表盘
    python -m gt7coach once --dash ...         # 单次 tick，排障用
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .engine import CoachConfig, CoachEngine
from .server import DEFAULT_PORT, CoachService, make_server
from .source import FileSource, HttpSource, ReplaySource


def _fix_console() -> None:
    """Windows 控制台默认 GBK：print 里出现 GBK 码表外的字符（⚠/emoji）会
    UnicodeEncodeError 直接崩。这里只影响显示，不动数据。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")     # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def _resolve_ref_cache_dir(args) -> str | None:
    """参考圈本地缓存目录：CLI > 环境变量 > 禁用。

    这是「根治前几圈瞎播报」的第二条杠杆。生产容器里挂一个持久卷
    （如 /opt/gt7-coach/data/refcache）并设好这个目录，下次同赛道就能
    跳过服务端历史搜索的冷启动，教练从第 1 圈起就有靠谱参考。
    """
    d = getattr(args, "ref_cache_dir", None)
    if not d:
        d = os.environ.get("GT7_COACH_REF_CACHE_DIR")
    return d or None


def _resolve_tts(args, *, auto: bool = True) -> dict[str, object]:
    """R3 云 TTS 的开关与参数：CLI > 环境变量 > 禁用。

    与参考圈缓存同构，但多一个「自动开」的语义 —— 凑齐两样就开：
      ① 业务空间 ID：--tts-workspace / GT7_COACH_TTS_WORKSPACE
      ② 本地缓存目录：--tts-cache-dir / GT7_COACH_TTS_CACHE_DIR
    key 不在这里传，复用 R2.2 那一把 GT7_COACH_LLM_KEY（百炼的 key 是业务
    空间级的，语音和语言模型不分开授权）。

    `auto=False` 用于 `replay`：离线调参不该偷偷联网、更不该白花钱。
    那时只有显式 `--tts` 才开（`--tts` 与 `--no-tts` 都是三态：
    不写 = None = 交给这里的自动判断）。

    🔴 这里只解析意图，不保证一定生效 —— 缓存目录是否可写由 TtsEngine
       再查一遍。所以「配了却没声音」时不能只看这里，要看 `disabled_reason`。
    """
    ws = (getattr(args, "tts_workspace", None)
          or os.environ.get("GT7_COACH_TTS_WORKSPACE") or "").strip()
    cache = (getattr(args, "tts_cache_dir", None)
             or os.environ.get("GT7_COACH_TTS_CACHE_DIR"))
    flag = getattr(args, "tts", None)          # True / False / None(没写)
    if flag is None:
        # 🔴 判据是 `ws or cache` 而不是 `ws and cache`。
        #    只给了 workspace（忘了缓存目录）时，如果判成"没打算开"，启动横幅会
        #    说"未开启（cfg.enabled=false）"—— 用户明明想开，却被告知没开，
        #    真正的缺失项（cache_dir）反而看不见。这里放行成 enabled=True，
        #    让引擎去说**准确**的原因（"未配置 cache_dir"）。
        #    引擎那边 enabled 仍为 False → 不起 worker、一次网络都不会有。
        intent = bool(ws or cache)
        enabled = intent if auto else False
    else:
        enabled = bool(flag)
    return {
        "tts_enabled": enabled,
        "tts_workspace_id": ws,
        "tts_cache_dir": cache or None,
        "tts_voice": getattr(args, "tts_voice", None) or "",
        "tts_model": getattr(args, "tts_model", None) or "",
    }


def _print_cloud_state(engine: CoachEngine) -> None:
    """把「云措辞」的落地状态打出来（对应 R2.2，此前**一行都不打**）。

    🔴 为什么必须打：`/api/v1/coach/cloud` 是给程序看的，而启动横幅是给
       **人**看的。用户最需要一眼确认的两件事在这里：
         ① 现在到底用哪个模型（是不是自己填的那个）；
         ② 这个模型在不在免费额度里 —— 不在就是"可能被计费"。
       这两件事原来都看不见，用户只能靠猜。
    """
    st = engine.narrator.status()
    if not st["enabled"]:
        # 只在"碰过这件事"时出声（配了 cloud.json / 环境里有 key），
        # 否则纯本地模板的用户会平白多一行噪音。
        if engine.cfg.cloud_path:
            print(f"[gt7coach] 云措辞未启用（配置在 {engine.cfg.cloud_path}）"
                  " —— 全部走本地模板，零外呼")
        return
    src = "用户填的" if st.get("model_from_user") else "厂商预设"
    line = (f"[gt7coach] 云措辞已启用  {st['provider']}/{st['model']}"
            f"（{src}）  key={'有' if st['has_key'] else '缺'}"
            f"  {st['price_yuan_per_mtok'].get('in', '?')}/"
            f"{st['price_yuan_per_mtok'].get('out', '?')} 元/百万 token")
    print(line)
    if st.get("model_warning"):
        # ⚠ 只警告不拦（按用户要求）。但必须显眼 —— 这条是防"静默扣费"的。
        print(f"[gt7coach] ⚠ {st['model_warning']}")
        print("[gt7coach]   想换模型：编辑 cloud.json 的 model 字段，"
              "或在仪表盘「赛道工程师」卡片里直接填（改完立即生效，不用重启）")


def _print_tts_state(engine: CoachEngine) -> None:
    """把云 TTS 的落地状态打出来。缺一样就静默降级是设计如此，但**必须
    说一声**：否则"配置明明写了却没声音"会变成一场排查噩梦。"""
    st = engine.tts.status()
    if st["enabled"]:
        print(f"[gt7coach] 云 TTS 已启用  {st['provider']}/{st['model']}"
              f"  音色 {st['voice']}  {st['format']}/{st['sample_rate']}Hz"
              f"  缓存已存 {st['cache_files']} 条"
              f"  {st['price_yuan_per_kchar']} 元/千字符"
              f"  key={'有' if st['has_key'] else '缺'}")
    elif engine.cfg.tts_enabled or st["workspace_id"] or st["has_key"]:
        # 只要"碰过"这件事就出声：显式开了、给了 workspace、或环境里有 key。
        # 全都空 = 从没打算用，那就别在启动日志里制造噪音。
        print(f"[gt7coach] 云 TTS 未启用：{st['disabled_reason']}"
              f"（B 档句回落浏览器 TTS，不影响 A 档）")


def _build(args, *, tts_auto: bool = True) -> CoachEngine:
    cfg = CoachConfig(poll_interval_s=args.interval,
                      cloud_path=getattr(args, "cloud", None),
                      ref_cache_dir=_resolve_ref_cache_dir(args),
                      **_resolve_tts(args, auto=tts_auto))
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
                           ("once", "跑一次 tick 并把状态打到 stdout"),
                           ("replay", "拿历史场次 jsonl 离线跑一遍，看会说什么")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--dash", default="http://192.168.43.18:8787",
                       help="GT7 Dash 的基地址")
        p.add_argument("--port", type=int, default=DEFAULT_PORT)
        p.add_argument("--bind", default="0.0.0.0")
        p.add_argument("--interval", type=float, default=0.1,
                       help="tick 间隔秒数，缺省 0.1（10Hz）")
        p.add_argument("--timeout", type=float, default=3.0)
        p.add_argument("--cloud", default=None,
                       help="cloud.json 路径（R2.2 云润色；不填=纯本地模板）")
        p.add_argument("--ref-cache-dir", default=None,
                       help="参考圈本地缓存目录（根治前几圈瞎播报的第二条杠杆）；"
                            "不填则读 GT7_COACH_REF_CACHE_DIR，都没有=禁用")
        # —— R3 云 TTS：给 B 档句合成真人嗓子 ——
        # A 档（刹车晚/打滑/出界）永远走浏览器 TTS，**任何参数都不会改变
        # 这一点**：云合成要 0.5~2s，来不及抢麦。
        p.add_argument("--tts", action="store_true", default=None,
                       help="开启云 TTS（B 档句）。不写=看下面两项是否配齐")
        p.add_argument("--no-tts", action="store_false", dest="tts",
                       help="强制关闭云 TTS（优先级最高，用于临时排查）")
        p.add_argument("--tts-workspace", default=None, metavar="WS_ID",
                       help="百炼业务空间 ID（ws- 开头，不是密钥）；"
                            "不填则读 GT7_COACH_TTS_WORKSPACE")
        p.add_argument("--tts-cache-dir", default=None,
                       help="合成音频缓存目录（必填才会启用 —— 没缓存=每句都"
                            "真花钱）；不填则读 GT7_COACH_TTS_CACHE_DIR")
        p.add_argument("--tts-voice", default=None,
                       help="音色（缺省 longanyang/龙安洋）")
        p.add_argument("--tts-model", default=None,
                       help="模型（缺省 cosyvoice-v3-flash；注意 v3.5 系列"
                            "不支持系统音色）")
        p.add_argument("--verbose", action="store_true")
        if name == "demo":
            p.set_defaults(demo=True)
        if name == "replay":
            p.add_argument("session", help="场次 jsonl 路径")
            p.add_argument("--lap", type=int, default=None,
                           help="只回放这一圈（缺省：整个场次）")
            p.add_argument("--ref-lap", type=int, default=None,
                           help="指定参考圈（缺省：该场次最快圈）")
            p.add_argument("--sectors", type=int, default=None,
                           help="分段数（缺省用配置里的 3）")
            p.add_argument("--conf", action="append", default=[],
                           metavar="KEY=VALUE",
                           help="覆盖阈值，可多次。前缀 rules./gate./coach.，"
                                "缺省 rules. 例：--conf brake_late_m=12")
            p.add_argument("--json", action="store_true",
                           help="以 JSON 输出（便于 diff 两次调参的结果）")
        if name == "serve":
            p.add_argument("--ticks", type=int, default=0,
                           help="跑 N 次 tick 后退出（0 = 一直跑）")

    args = ap.parse_args(argv)
    if not getattr(args, "demo", False):
        args.demo = False
    if args.cmd == "replay":
        return _replay(args)

    engine = _build(args)
    _print_cloud_state(engine)
    _print_tts_state(engine)

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


# —— 离线调参（replay）————————————————————————————————————
#
# 这一段存在的意义：**改阈值不用上车**。
# 以前改一个数（比如"刹车晚了"从 8 米调到 12 米）只能上车试，而"感觉
# 不太对"这种反馈既慢又不能对比。现在拿历史场次跑一遍，会说的话直接
# 打出来 —— 而且是可 diff 的（`--json` 两次输出一对比就知道改了什么）。

def _apply_conf(args: argparse.Namespace, engine: CoachEngine) -> list[str]:
    """`--conf rules.brake_late_m=12` → 改配置。返回改了哪些。"""
    applied: list[str] = []
    for item in args.conf or []:
        if "=" not in item:
            raise SystemExit(f"--conf 要写成 KEY=VALUE，收到：{item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        section, _, name = key.partition(".")
        if not name:
            name, section = section, "rules"
        obj = {"rules": engine.rules.cfg, "gate": engine.gate.cfg,
               "coach": engine.cfg}.get(section)
        if obj is None:
            raise SystemExit(f"未知配置段 {section}（可用 rules/gate/coach）")
        if not hasattr(obj, name):
            raise SystemExit(f"{section}.{name} 不是可配置项")
        cur = getattr(obj, name)
        if isinstance(cur, bool):
            val: object = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            val = int(float(raw))
        elif isinstance(cur, float):
            val = float(raw)
        else:
            val = raw
        setattr(obj, name, val)
        applied.append(f"{section}.{name}={val}")
    return applied


def _replay(args: argparse.Namespace) -> int:
    src = FileSource(args.session, lap=args.lap, ref_lap=args.ref_lap)
    if not src._frames:
        print(f"[replay] {args.session} 里没有可回放的帧", file=sys.stderr)
        return 2
    tts = _resolve_tts(args, auto=False)
    cfg = CoachConfig(poll_interval_s=0.0,
                      cloud_path=getattr(args, "cloud", None),
                      ref_cache_dir=_resolve_ref_cache_dir(args),
                      # auto=False：replay 是离线调参，不联网、不花钱。
                      # 想顺便验云 TTS 就显式加 --tts（下面会警告一次）。
                      **tts)
    eng = CoachEngine(src, cfg, clock=src.clock)
    if eng.tts.enabled:
        print("[replay] ⚠ 云 TTS 已开启：下面会真的调百炼合成"
              f"（按 {eng.cfg.tts_config().price_yuan_per_kchar} 元/千字符计费）。"
              "只想看文本的话去掉 --tts。")
    elif tts["tts_workspace_id"] or tts["tts_cache_dir"]:
        # 配了却没说一句，用户会以为配置被忽略了 —— 明说这是 replay 的有意行为
        print("[replay] 云 TTS 默认关闭（离线调参不联网、不花钱）。"
              "要顺带验合成再加 --tts。")
    if args.sectors:
        eng.rules.cfg.sectors_n = int(args.sectors)
        eng._sectors.n_sectors = int(args.sectors)
    changed = _apply_conf(args, eng)

    laps: dict[int, dict] = {}
    rows: list[dict] = []
    say_by_key: dict[str, int] = {}
    for _ in range(len(src._frames) + 20):
        st = eng.tick()
        if st.last_lap and st.last_lap["lap"] not in laps:
            laps[st.last_lap["lap"]] = st.last_lap
        for u in st.say:
            rows.append({"lap": st.lap, "s_m": st.s_m, "delta_s": st.delta_s,
                         "key": u.key, "text": u.text, "priority": u.priority,
                         "evidence": u.evidence})
            say_by_key[u.key] = say_by_key.get(u.key, 0) + 1
        if not st.connected:
            break

    theory = eng._sectors.to_dict()
    fuel = eng._fuel.to_dict()
    _ref_obj, ref_state, ref_err = eng.refs.get()
    out = {
        "session": str(src.path),
        "laps_available": sorted(src.lap_spans),
        "ref": ({"lap": eng._current_ref().lap,
                 "source": eng._current_ref().source,
                 "lap_time_s": round(eng._current_ref().lap_time_s, 3),
                 "length_m": round(eng._current_ref().length_m, 1)}
                if eng._current_ref() else None),
        "config_overrides": changed,
        "ref_state": ref_state,
        "ref_error": ref_err,
        "laps": [laps[k] for k in sorted(laps)],
        "theory": theory,
        "fuel": fuel,
        "said": rows,
        "said_by_key": say_by_key,
        "dropped_by_gate": eng._st_gate.get("dropped", 0),
        # 排队没轮上（max_per_tick 用尽被 skip）与被冷却/配额判掉是两回事：
        # 前者是"没轮上"，后者是"轮到了但不该说"。分开报，否则
        # 「闸门丢掉 0 条」会让人以为闸门什么都没拦（踩过一次）。
        "skipped_by_gate": eng._st_gate.get("skipped", 0),
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    ref = out["ref"]
    print(f"场次 {src.path.name}")
    print(f"  圈：{out['laps_available']}"
          + (f"  回放第 {args.lap} 圈" if args.lap else "  回放整个场次"))
    if ref:
        print(f"  参考圈：第 {ref['lap']} 圈（{ref['source']}）"
              f"  {ref['lap_time_s']}s  {ref['length_m']}m")
    # 🔴 参考圈取失败时必须显式打出来。它一旦悄悄退回自攒，后面所有定位/
    #    delta/刹车点都是错的，而输出看起来仍然"很正常" —— 实测就是这样：
    #    一个 NameError 被日志吞掉，然后拿菜单帧攒出 390m 的假参考圈。
    if out["ref_state"] != "ready" or not ref or ref["length_m"] < 200.0:
        print(f"  ⚠ 参考圈状态 {out['ref_state']}"
              + (f"：{out['ref_error']}" if out["ref_error"] else "")
              + "（结果不可信，先查这个）")
    if changed:
        print(f"  已覆盖：{', '.join(changed)}")

    print("\n逐圈（本地算，零网络）：")
    for lp in out["laps"]:
        secs = " / ".join(f"{x:.2f}" for x in lp["sectors"]) or lp["why"]
        fuel_txt = (f"  油耗 {lp['fuel_used']:.2f}"
                    if lp.get("fuel_used") else "")
        flag = "" if lp["ok"] else "  ⚠ "
        print(f"  第{lp['lap']:>2}圈 {lp['lap_time_s']:>7.3f}s  "
              f"S {secs}{fuel_txt}{flag}")
    if theory:
        print(f"  理论最快 {theory['theory_best_s']}s"
              f"（实际最快 {theory['best_actual_s']}s，"
              f"潜在 {theory['gain_s']}s，样本 {theory['samples']}）")
    if fuel.get("per_lap"):
        print(f"  每圈油耗 {fuel['per_lap']}（{fuel['samples']} 圈样本）")

    print(f"\n会说的话（{len(rows)} 条）：")
    for r in rows:
        pos = "    —" if r["s_m"] is None else f"{r['s_m']:>5.0f}m"
        dl = "     " if r["delta_s"] is None else f"{r['delta_s']:+5.2f}"
        print(f"  第{r['lap']:>2}圈 s={pos} Δ={dl}  [{r['key']}]  {r['text']}")
    if say_by_key:
        print("\n按 key：")
        for k, n in sorted(say_by_key.items(), key=lambda x: -x[1]):
            print(f"  {n:>3} × {k}")
    print(f"\n闸门丢掉 {out['dropped_by_gate']} 条候选"
          f"（另有 {out['skipped_by_gate']} 条排队没轮上）")
    return 0
