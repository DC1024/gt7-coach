# -*- coding: utf-8 -*-
"""部署后的端到端体检：全部走真实网络，不靠日志猜。

    python tools/verify.py --dash http://192.168.43.18:8787 \\
                           --coach http://192.168.43.18:8788

查五件事：
  1. Coach 活着，且 **tick 速率 = 10Hz**（见下）
  2. `/profile` 能对**真实场次**算出剖面（不是只在合成数据上能跑）
  3. 实时接口的「本圈已用时」口径来源是 lap 还是 session
  4. 仪表盘页面里真的有工程师卡片
  5. Coach 与 Dash 的场次发现状态（区分"网络断了"和"对方在忙"）

🔴 为什么单列 tick 速率：慢的依赖绝不能阻塞 tick。仪表盘的
   `/api/v1/sessions` 冷态要流式扫每个场次文件算最快圈（实测 5 场里有个
   111 MB 的 → 11 秒），而容器每次重启都回到冷态。踩过一次：那一句同步写在
   tick 里，于是 10Hz 掉到 0.3Hz，`/health` 看起来像"服务挂了"，
   其实两边都在正常运行。
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
failures: list[str] = []


def get(url: str, timeout: float = 60.0):
    with OPENER.open(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ck(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'✓' if ok else '✗'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="部署后端到端体检")
    ap.add_argument("--dash", default="http://127.0.0.1:8787")
    ap.add_argument("--coach", default="http://127.0.0.1:8788")
    args = ap.parse_args(argv)

    print("=== 1. Coach 存活与 tick 速率 ===")
    h1 = get(args.coach + "/api/v1/coach/health")
    print(f"    {json.dumps(h1, ensure_ascii=False)}")
    ck("coach ok", h1.get("ok") is True)
    time.sleep(5)
    h2 = get(args.coach + "/api/v1/coach/health")
    rate = (h2.get("ticks", 0) - h1.get("ticks", 0)) / 5.0
    # tick 间隔 100ms → 期望 10/s。容差放到 6 是为了容忍测量误差，
    # 但 0.3/s（旧 bug 的值）一定会被抓到。
    ck(f"tick 速率 ≈10Hz（实测 {rate:.1f}/s）", rate >= 6.0)
    ck("场次发现状态已暴露",
       h2.get("sess_state") in ("idle", "loading", "ok", "empty", "failed"),
       f"sess_state={h2.get('sess_state')} err={h2.get('sess_error')}")
    ck("参考圈状态已暴露", "ref_state" in h2)

    print("\n=== 2. /profile 对真实场次可用 ===")
    t0 = time.time()
    sess = get(args.dash + "/api/v1/sessions", timeout=180)
    print(f"    场次列表 {time.time() - t0:.2f}s，{len(sess['sessions'])} 个")
    hit = None
    for s in sess["sessions"]:
        url = (args.dash + "/api/v1/sessions/"
               + urllib.parse.quote(s["file"]) + "/profile?step=10")
        t0 = time.time()
        try:
            d = get(url, timeout=300)
        except Exception as e:                      # noqa: BLE001
            print(f"    {s['file']}: {type(e).__name__}")
            continue
        if "error" in d:
            print(f"    {s['file']}: error={d['error']}")
            continue
        hit = (s["file"], d, time.time() - t0)
        break
    ck("至少一个场次能算出剖面", hit is not None)
    if hit:
        name, d, el = hit
        print(f"    {name}  {el:.2f}s")
        print(f"      第 {d['lap']} 圈 | 圈长 {d['length_m']}m | "
              f"圈速 {d['lap_time_s']}s | 网格 {len(d['grid_m'])} | "
              f"漂移 {d['length_drift_pct']}%")
        print(f"      刹车区 {len(d['markers']['brake_in'])} | "
              f"弯心 {len(d['markers']['apex'])} | "
              f"给油点 {len(d['markers']['throttle_on'])} | "
              f"折线点 {len(d['pt']['x'])}")
        ck("几何折线非空", len(d["pt"]["x"]) == len(d["grid_m"]))
        ck("通道与网格一一对齐",
           all(len(d[k]) == len(d["grid_m"])
               for k in ("speed_kph", "throttle", "brake", "t_rel_s", "glat")))
        ck("关键点非空", bool(d["markers"]["brake_in"] and d["markers"]["apex"]),
           f"warnings={d['warnings']}")
        ck("无降级警告", not d["warnings"], str(d["warnings"]))

    print("\n=== 3. 实时接口的本圈用时口径 ===")
    lv = get(args.dash + "/api/v1/live", timeout=30)
    tm = lv["timing"]
    src = tm.get("current_lap_time_source")
    print(f"    current_lap={tm['current_lap']} "
          f"current_lap_time_s={tm['current_lap_time_s']} source={src}")
    ck("口径来源字段存在", src in ("lap", "session"))
    # source=session 只在「接收器还没写过 lap_started_at」时出现，
    # 通常意味着 PS5 没在发数据（接收器还没开始过一个场次）。
    if src != "lap":
        print("    ⚠ source=session → 接收器尚未写过 lap_started_at；"
              "PS5 一连上、跑起来后应变成 lap")

    print("\n=== 4. R1.5 本地统计字段已暴露 ===")
    st = get(args.coach + "/api/v1/coach/state")
    for k in ("projected_lap_s", "last_lap", "theory_best_s",
              "potential_gain_s", "fuel_per_lap", "fuel_laps_left"):
        ck(f"state 有 {k}", k in st)
    for k in ("sess_state", "ref_state", "ref_history", "sector_len_m",
              "lap_samples", "fuel_samples"):
        ck(f"stats 有 {k}", k in (st.get("stats") or {}))
    h = st.get("stats", {}).get("ref_history") or {}
    ck("ref_history 结构完整",
       all(k in h for k in ("candidates", "rejected", "adopted")), str(h)[:80])

    print("\n=== 5. 仪表盘页面里有工程师卡片 ===")
    with OPENER.open(args.dash + "/", timeout=30) as r:
        html = r.read().decode("utf-8", "replace")
    for needle in ('id="c-coach"', "coachMute", "coSay", "coHist",
                   "function pollCoach", ".then(renderCoach, renderCoachOff)",
                   "speechSynthesis.cancel()"):
        ck(f"页面含 {needle}", needle in html)

    print("\n=== 6. Coach 自检页含语音抢占 ===")
    with OPENER.open(args.coach + "/", timeout=30) as r:
        demo = r.read().decode("utf-8", "replace")
    ck("自检页 P0 抢占", "speechSynthesis.cancel()" in demo)
    # ⚠️ 别再写 "d.say[0].priority" 这个字面量：R2.2 起自检页先
    #    `var u = (d.say && d.say.length) ? d.say[0] : null;` 再取字段，
    #    字面量早就不存在了 —— 这条断言从那时起一直是假红（实际功能正常）。
    #    改成断言"优先级真的被传下去" + "A 档走立即播"这两件真事。
    ck("自检页传优先级", "say(u.speech || u.text, u.priority)" in demo)
    ck("自检页 A 档立即播", "u.priority < P_NORMAL" in demo)

    print()
    if failures:
        print(f"❌ {len(failures)} 项未通过：{failures}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
