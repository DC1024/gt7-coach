# -*- coding: utf-8 -*-
"""
R2.2 的提示词（prompt）集中存放处。
====================================

🔴 两个约束硬编码在提示里、且**不可被 prompt 覆盖**：

  1. **只用给定的数字**：云 LLM 没有 facts 之外的任何信息，乱编一个「快 15 米」
     玩家无法分辨。白名单（phrases.invented_numbers）在代码层二次拦截 ——
     提示只是第一道、且不可靠的约束。
  2. **教练口吻、≤20 字、无感叹号**：这是「听起来像人副驾」的硬下限。

🔴 另外两条**曾经只写在提示里、结果被真机证伪**，现在都有代码层兜底
   （见 `phrases.missing_mandatory` / `phrases.invented_advice`）：

  3. **不许丢主体**：facts 里有圈速就必须说出来。原先只靠
     `_KEY_HINTS["lap_advice"]` 的"只挑最要紧的两三件"来压缩，实测模型
     两次都把圈速丢了 —— 而提示里从没说过"哪一件不许丢"。
  4. **不许编建议**：不许给 facts 里没有的行动指令（进站/补油/换胎…）。
     原先 SYSTEM_PROMPT 只说"不能新增数字"，模型于是改用**不带数字的
     指令**绕过它（实测输出了「注意补油」，而 `laps_left=2.3`）。
  5. **圈速写成 M:SS.mmm**（偏好，非硬门）：3 号版本实测模型写了原始秒数
     「这圈83.45」。形式容错交给代码（`missing_mandatory` 连原始秒数一起认）——
     提示词管偏好、闸门管"在不在"，两边各管一件事，谁也替不了谁。

`PROMPT_VERSION` 进缓存 key：改提示词 → 旧缓存自动失效（facts 没变也不复用）。
"""

from __future__ import annotations

from typing import Any

# 🔴 改提示词必须 +1。narrate 的缓存 key = hash(PROMPT_VERSION + facts)，
#    版本变了同一条 facts 也重新请求，避免「旧 prompt 的好结果」被错误复用。
#    "2"：R2.4 新增 lap_advice（圈后综合建议）的压缩提示。
#    "3"：把「必须保留圈速主体」与「不许编建议」写进提示 —— 这两条是
#         真 key 冒烟发现 2 号版本会丢圈速 / 编出「注意补油」之后补的。
#         ⚠️ 不升版本号的话，同一份 facts 会一直命中 2 号版本产出的旧句子，
#            你会以为新提示没生效 —— 这是最容易白忙一场的坑。
#    "4"：真 key 复跑（3 号版本）发现模型把圈速写成**原始秒数**「这圈83.45」
#         而不是 M:SS.mmm，于是明确要求写成 M:SS.mmm。
#         ⚠️ 注意这里只改了**偏好**：形式容错由 `phrases.missing_mandatory`
#            负责（它连原始秒数一起认）。只靠提示词管形式必然失败，
#            只靠闸门管形式则会误杀 —— 两边各管一件事。
PROMPT_VERSION = "4"

SYSTEM_PROMPT = (
    "你是赛车游戏 Gran Turismo 7 的赛道工程师（race engineer），"
    "正坐在车手的副驾上做实时语音播报。\n"
    "要求：\n"
    "1. 用中文、口语化、像真人教练一样简洁。\n"
    "2. 不超过 20 个字，不要用感叹号，不要加引号。\n"
    "3. 只使用下面「事实」里给出的数字，绝对不能新增任何数字或事实。\n"
    "4. 不改变事实含义，只是把生硬的数据说成一句人话。\n"
    "5. 只描述「事实」，不要给事实里没有的建议或指令 —— "
    "特别是不要自行判断是否需要进站、补油、换胎或省油，这些不由你决定。"
)


# 逐 key 的额外提示（可选）。key 不在表里 → 用通用那一句。
_KEY_HINTS: dict[str, str] = {
    "lap_advice": (
        "这是**圈后综合建议**：事实里有成绩、最慢段、续航、习惯等多项，"
        "只挑最要紧的两三件说成一句，不要逐项罗列、不要面面俱到。"
        "⚠️ 但**本圈圈速（lap_time_s）必须说出来**，无论怎么取舍都不能省略 —— "
        "它是这一圈的主体，车手最先要听的就是它。"
        "说圈速要写成 M:SS.mmm 的形式（83.45 秒写成「1:23.450」），"
        "不要把秒数原样念出来 —— 本地模板也是这个形式，两条路听起来必须一样。"
        "其余按「最慢段 > 反复亏的弯 > 与参考圈的差 > 续航」的顺序最多再挑一件。"
    ),
}


def user_prompt(key: str, facts_json: str) -> str:
    """把 key + facts 拼成用户提示。

    `key` 让模型知道这是哪类播报（成绩/分段/习惯/续航/综合建议），便于选措辞；
    `facts_json` 是 facts 的 JSON 字符串 —— 模型被明确要求「只用这些数字」。
    """
    hint = _KEY_HINTS.get(key, "")
    tail = f"{hint}\n" if hint else ""
    return (
        f"场景类型：{key}\n"
        f"事实（只能使用以下数字，不得新增）：\n{facts_json}\n\n"
        f"{tail}"
        f"改写成一句教练口吻的中文播报（≤20字，无感叹号）："
    )


def messages_for(key: str, facts: dict[str, Any]) -> list[dict[str, str]]:
    """组装完整 messages。facts 经 JSON 序列化后传入，保证数字原样、可被白名单核对。"""
    import json
    facts_json = json.dumps(facts, ensure_ascii=False, sort_keys=True)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(key, facts_json)},
    ]
