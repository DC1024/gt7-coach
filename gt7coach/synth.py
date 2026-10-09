# -*- coding: utf-8 -*-
"""
合成赛道 —— 可验算的假数据（自检 / 演示 / 单测共用）。
========================================================

真赛道上「参考圈在 812 m 刹车」这个真值没人知道，只能看曲线像不像；
合成圆上每个量都能手算：圈长 = 2πR、横向 G = v²/(R·g)、
刹车区起点就是减速段起点。单测于是能写成**等式**而不是"看起来对"。

`synth_lap_frames()` 给帧序列（走实时管道），
`synth_profile()` 给同一条赛道的剖面 dict（形状与 Dash `/profile` 一致）。
"""

from __future__ import annotations

import math

from .contract import Frame

G = 9.80665


def _v_of_s(s: float, length: float, base_kph: float, dip_kph: float,
            dip_start: float, dip_len: float) -> float:
    """沿弧长的速度（m/s）。减速段用正弦，保证加减速都是连续的。"""
    s = s % length
    v = base_kph
    if dip_start <= s <= dip_start + dip_len:
        k = (s - dip_start) / dip_len
        v = base_kph - (base_kph - dip_kph) * math.sin(math.pi * k)
    return v / 3.6


def _channels(v: float, radius: float, dv_ds: float):
    """由速度与其沿程导数给出 g_force / 踏板。"""
    glat = (v * v / radius) / G
    a_long = dv_ds * v
    glon = a_long / G
    if a_long < -0.3:
        brake, throttle = min(1.0, -glon / 1.2), 0.0
    elif a_long > 0.05:
        brake, throttle = 0.0, min(1.0, glon / 0.6)
    else:
        brake, throttle = 0.0, 0.4
    return glon, glat, brake, throttle


def _params(radius_m: float, base_kph: float, dip_kph: float,
            dip_start_m: float, dip_len_m: float):
    return (2.0 * math.pi * radius_m, base_kph, dip_kph,
            dip_start_m, dip_len_m)


def synth_lap_frames(radius_m: float = 600.0, base_kph: float = 200.0,
                     dip_kph: float = 90.0, dip_start_m: float = 400.0,
                     dip_len_m: float = 160.0, hz: float = 10.0,
                     lap: int = 1, laps: int = 1,
                     with_wheels: bool = True, fuel_start: float = 100.0,
                     fuel_per_lap: float = 8.0,
                     powertrain: str = "fuel",
                     position: int = 0, num_cars: int = 0,
                     laps_in_race: int = 0, car_code: int = 0) -> list[Frame]:
    """一圈（或多圈首尾相接）的合成帧。

    `hz` 缺省 10 —— 故意对齐实时侧 10Hz 轮询的真实采样率，
    这样才能验出「自攒参考圈」在真实采样密度下的精度。

    轮胎角速度按 `ω = v/R_tyre` 造，所以滑移率恒为 0（自由滚动）——
    想要打滑场景自己把某一轮的 ω 乘上去。

    油量按**里程**线性消耗（`fuel_per_lap` 每圈），这样"每圈油耗"正好等于
    `fuel_per_lap`，油耗/续航逻辑就有了可验算的真值。

    比赛信息（`position` / `num_cars` / `laps_in_race` / `car_code`）默认全 0
    = 「未知」，与真机菜单态口径一致；要验名次播报就显式传进去。
    它们在整段里是常量 —— 名次/名次**变化**的测试请自己改 frames 里的字段
    （`dataclasses.replace`），别给这个函数加"第几帧变名次"这类开关。
    """
    length, base_kph, dip_kph, dip_start, dip_len = _params(
        radius_m, base_kph, dip_kph, dip_start_m, dip_len_m)
    dt = 1.0 / hz
    eps = 1e-3
    out: list[Frame] = []
    tyre_r = 0.34
    # 🔴 `t` 必须**跨圈单调**，只有 `lap_time_s` 每圈归零 —— 与记录器一致
    #    （记录器里 t 是全场时钟，t_rel 才是圈内时钟）。
    #    早先这里两圈都把 t 从 0 开始，结果回放时注入的比赛时钟每圈倒回 0，
    #    闸门的「同类冷却」算出来永远是 0 秒，跨圈的提醒全被挡掉，
    #    看起来像"教练只肯说一句话"。夹具错了会让整批行为判断失真。
    t_abs = 0.0
    for lp in range(lap, lap + laps):
        s, t_lap = 0.0, 0.0
        while s < length:
            v = _v_of_s(s, length, base_kph, dip_kph, dip_start, dip_len)
            v1 = _v_of_s(s + eps, length, base_kph, dip_kph, dip_start, dip_len)
            v0 = _v_of_s(s - eps, length, base_kph, dip_kph, dip_start, dip_len)
            glon, glat, brake, throttle = _channels(
                v, radius_m, (v1 - v0) / (2 * eps))
            theta = s / radius_m
            omega = v / tyre_r
            out.append(Frame(
                t=round(t_abs, 4), lap_time_s=round(t_lap, 3),
                speed_kph=round(v * 3.6, 2), rpm=7000.0, max_rpm=8200.0,
                gear=4, throttle=throttle, brake=brake, lap=lp,
                x=round(radius_m * math.cos(theta), 3),
                y=0.0,
                z=round(radius_m * math.sin(theta), 3),
                glat=round(glat, 3), glon=round(glon, 3),
                tyre_temp=(88.0, 89.0, 86.0, 87.0),
                wheel_rads=(omega, omega, omega, omega) if with_wheels else (),
                fuel_pct=round(max(0.0, fuel_start - fuel_per_lap * (s / length)
                                   - fuel_per_lap * (lp - lap)), 3),
                fuel_capacity_l=100.0,
                powertrain=powertrain,
                position=position,
                num_cars=num_cars,
                laps_in_race=laps_in_race,
                car_code=car_code,
                connected=True,
            ))
            s += v * dt
            t_lap += dt
            t_abs += dt
    return out


def synth_profile(radius_m: float = 600.0, base_kph: float = 200.0,
                  dip_kph: float = 90.0, dip_start_m: float = 400.0,
                  dip_len_m: float = 160.0, step_m: float = 5.0,
                  lap: int = 1) -> dict:
    """与 Dash `/profile` 同形状的剖面（供 ReplaySource 走 profile 路径）。"""
    length, base_kph, dip_kph, dip_start, dip_len = _params(
        radius_m, base_kph, dip_kph, dip_start_m, dip_len_m)
    eps = 1e-3
    grid: list[float] = []
    d = 0.0
    while d < length:
        grid.append(round(d, 2))
        d += step_m

    spd, thr, brk, glat, xs, zs = [], [], [], [], [], []
    for q in grid:
        v = _v_of_s(q, length, base_kph, dip_kph, dip_start, dip_len)
        v1 = _v_of_s(q + eps, length, base_kph, dip_kph, dip_start, dip_len)
        v0 = _v_of_s(q - eps, length, base_kph, dip_kph, dip_start, dip_len)
        glon, gl, b, th = _channels(v, radius_m, (v1 - v0) / (2 * eps))
        spd.append(round(v * 3.6, 1))
        thr.append(round(th, 3))
        brk.append(round(b, 3))
        glat.append(round(gl, 3))
        theta = q / radius_m
        xs.append(round(radius_m * math.cos(theta), 2))
        zs.append(round(radius_m * math.sin(theta), 2))

    # 时间轴：按梯形积分 ds/v 累积
    t_rel, acc = [0.0], 0.0
    for i in range(1, len(grid)):
        v_mid = (_v_of_s(grid[i - 1], length, base_kph, dip_kph, dip_start,
                         dip_len)
                 + _v_of_s(grid[i], length, base_kph, dip_kph, dip_start,
                           dip_len)) / 2.0
        acc += (grid[i] - grid[i - 1]) / max(v_mid, 1e-6)
        t_rel.append(round(acc, 3))

    apex_s = dip_start + dip_len / 2.0
    return {
        "lap": lap,
        "length_m": round(length, 1),
        "length_by_speed_m": round(length, 1),
        "length_drift_pct": 0.0,
        "geometry_used": True,
        "lap_time_s": round(acc, 3),
        "step_m": step_m,
        "grid_m": grid,
        "speed_kph": spd, "throttle": thr, "brake": brk,
        "t_rel_s": t_rel, "glat": glat,
        "glon": [0.0] * len(grid),
        "pt": {"x": xs, "z": zs},
        "markers": {
            "brake_in": [{"s_in_m": dip_start, "s_out_m": dip_start + 72.0,
                          "speed_in_kph": base_kph, "peak_brake": 1.0,
                          "duration_s": 2.0, "v_min_kph": dip_kph}],
            "apex": [{"s_m": apex_s, "speed_kph": dip_kph,
                      "glat": round((dip_kph / 3.6) ** 2 / radius_m / G, 3),
                      "radius_m": round(float(radius_m), 1), "turn": "左"}],
            "throttle_on": [{"s_m": dip_start + dip_len, "speed_kph": base_kph,
                             "after_apex_m": dip_len / 2.0}],
            "peak": [], "valley": [],
        },
        "warnings": [],
    }
