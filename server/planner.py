"""
行程编排 + 优化建议。

分工原则:所有事实(景点是否真实存在、站间要多久、几点关门)来自接口计算,
大模型只做推理和表达。这样即使模型发挥不稳定,也不会编造出不存在的地方。
"""
import json
import os
import time
import urllib.request

import amap
import store
from data import CITIES

# ---------- 大模型 ----------
# 只要求接口兼容 OpenAI 的 /chat/completions,换供应商只需改 .env 里的三个变量。
# DeepSeek 的旧变量名继续兼容,写了也算数。
LLM_KEY = (os.getenv("LLM_KEY") or os.getenv("DEEPSEEK_KEY") or "").strip()
MODEL_LIVE = bool(LLM_KEY)
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-plus")
# 名字里带这些的会默认先输出一大段推理再回答,建议生成场景要显式关掉,否则又慢又容易跑偏
THINKING_MODELS = ("qwen3", "qvq", "r1", "reasoning")

PER_DAY = {"轻松": 3, "适中": 4, "紧凑": 5}
# 每天的时间预算(游玩 + 通勤 + 午餐)。到了预算就停止加点,宁可少一个也不排到晚上十点。
BUDGET = {"轻松": 7 * 60, "适中": 9 * 60, "紧凑": 11 * 60}
# 同天两点允许的最大间隔。超过这个值说明不在同一片区域,不该硬排在一起。
NEIGHBOR_KM = 20
# 单段通勤上限。再远就不是"顺路",而是换了个目的地。
MAX_LEG = 90
PREF_TAGS = {
    "自然风光": ["自然"], "历史人文": ["人文"], "美食小吃": ["美食", "小吃"],
    "拍照出片": ["拍照"], "亲子": ["亲子"], "小众": ["小众"],
}
START_MIN = 9 * 60          # 每天 9:00 出发
LUNCH_START = 12 * 60
LUNCH_LEN = 60


# ---------- 编排 ----------

def _median_center(pool: list):
    """用候选池的中位数当城市中心,避免为这件事再调一次接口。"""
    if not pool:
        return None
    lats = sorted(p["lat"] for p in pool)
    lngs = sorted(p["lng"] for p in pool)
    n = len(lats)
    return (lats[n // 2], lngs[n // 2])


def score(pool: list, prefs: list) -> list:
    want = []
    for p in prefs:
        want += PREF_TAGS.get(p, [])
    center = _median_center(pool)
    out = []
    for poi in pool:
        s = poi["rating"] * 10
        for t in poi["tags"]:
            if t in want:
                s += 14
        if center and amap.haversine(poi["lat"], poi["lng"], center[0], center[1]) > 25:
            s -= 6      # 远郊点往后排:刚到一个城市,不该第一天就跑几十公里
        out.append((poi, s))
    out.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in out]


def nearest_order(items: list) -> list:
    """最近邻重排:从第一个点出发,每次走向剩下的最近的点。"""
    if len(items) <= 1:
        return list(items)
    rest = list(items[1:])
    out = [items[0]]
    while rest:
        last = out[-1]
        nxt = min(rest, key=lambda p: amap.haversine(last["lat"], last["lng"], p["lat"], p["lng"]))
        out.append(nxt)
        rest.remove(nxt)
    return out


def _span(items: list) -> int:
    """试算当天总时长。用估算而非真实路径接口,避免试算把调用量放大好几倍。"""
    t = START_MIN
    for i, it in enumerate(items):
        arrive, leave = t, t + it["duration"]
        t = leave
        if i < len(items) - 1:
            if arrive < LUNCH_START <= leave:
                t += LUNCH_LEN
            t += amap.estimate_route(it, items[i + 1])[1]
    return t - START_MIN


def _near(day: list, cand: dict) -> bool:
    """候选点是否和当天已有的某个点在同一片区域。"""
    for p in day:
        if amap.haversine(p["lat"], p["lng"], cand["lat"], cand["lng"]) <= NEIGHBOR_KM:
            return True
    return False


def split_days(ranked: list, days: int, per_day: int, pace: str = "适中") -> list:
    """
    按天分组:取当前最高分作种子,再贪心吸收离它最近的点。
    两道约束 —— 不超过当天时间预算,不跨区域硬凑。
    """
    budget = BUDGET.get(pace, 9 * 60)
    rest = list(ranked)
    groups = []
    for _ in range(days):
        if not rest:
            break
        day = [rest.pop(0)]
        while len(day) < per_day and rest:
            last = day[-1]
            cands = sorted(rest, key=lambda p: amap.haversine(last["lat"], last["lng"], p["lat"], p["lng"]))
            picked = None
            for c in cands:
                if not _near(day, c) or _span(day + [c]) > budget:
                    continue
                if amap.estimate_route(day[-1], c)[1] > MAX_LEG:
                    continue
                picked = c
                break
            # 远郊孤立点当天太空时,至少补一个最近的,免得出现"一天只去一个地方"
            if picked is None and len(day) == 1:
                c = cands[0]
                if _span(day + [c]) <= budget and amap.estimate_route(day[-1], c)[1] <= MAX_LEG:
                    picked = c
            if picked is None:
                break
            day.append(picked)
            rest.remove(picked)
        groups.append(nearest_order(day))
    return groups


def compute_day(items: list, city: str) -> dict:
    """生成当天时间轴。站间耗时来自路径规划接口,绝不交给模型估算。"""
    t = START_MIN
    nodes = []
    for i, it in enumerate(items):
        arrive, leave = t, t + it["duration"]
        nodes.append({"kind": "poi", "poi": it, "arrive": arrive, "leave": leave})
        t = leave
        if i < len(items) - 1:
            if arrive < LUNCH_START <= leave:
                nodes.append({"kind": "meal", "arrive": t, "leave": t + LUNCH_LEN})
                t += LUNCH_LEN
            mode, mins = amap.route_time(it, items[i + 1], city)
            nodes.append({"kind": "move", "mode": mode, "min": mins,
                          "to": items[i + 1]["name"]})
            t += mins
    return {"items": items, "timeline": nodes, "end": t}


def _plan_key(city: str, days: int, prefs: list, pace: str) -> str:
    """同一组输入就是同一份行程 —— 重复生成没必要再烧一次大模型和路径规划。"""
    return f'{city}|{days}|{",".join(sorted(prefs or []))}|{pace}'


def build_plan(city: str, days: int, prefs: list, pace: str, pool: list = None,
               fresh: bool = False) -> dict:
    key = _plan_key(city, days, prefs, pace)
    if not fresh:
        got = store.read("plan", key)
        if isinstance(got, dict) and got.get("days"):
            got["cached"] = True
            return got

    # pool 可由调用方传入(接口层已经取过一次,避免重复请求)
    if pool is None:
        pool = amap.build_pool(city)
    per = PER_DAY.get(pace, 4)
    ranked = score(pool, prefs)
    groups = split_days(ranked, days, per, pace)

    used = {p["id"] for g in groups for p in g}
    plan_days = []
    for i, g in enumerate(groups):
        d = compute_day(g, city)
        d["day"] = i + 1
        plan_days.append(d)

    plan = {
        "city": city,
        "pace": pace,
        "days": plan_days,
        "advice": make_advice(plan_days, city),
        "pool": [p for p in pool if p["id"] not in used],
        "live": amap.LIVE,
        "model": MODEL_LIVE,
        "cached": False,
        "cached_ts": time.time(),
    }
    # 落盘:同样的输入下次直接返回,连大模型都不用叫
    store.write("plan", key, plan)
    return plan


def replan(city: str, groups: list) -> dict:
    """用户改完地点后重排并重算时间轴。groups 是 [[poi, poi, ...], ...]。"""
    plan_days = []
    for i, g in enumerate(groups):
        d = compute_day(nearest_order(g), city)
        d["day"] = i + 1
        plan_days.append(d)
    used = {p["id"] for d in plan_days for p in d["items"]}
    return {
        "city": city, "pace": "", "days": plan_days,
        "advice": make_advice(plan_days, city),
        "pool": [p for p in amap.build_pool(city) if p["id"] not in used],
        "live": amap.LIVE, "model": MODEL_LIVE,
    }


# ---------- 建议 ----------

def hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _digest(plan_days: list) -> str:
    """把已算好的事实压成一段纯文本喂给模型。数字全部来自接口,模型无法篡改。"""
    lines = []
    for d in plan_days:
        lines.append(f"第{d['day']}天(结束 {hhmm(d['end'])}):")
        for n in d["timeline"]:
            if n["kind"] == "poi":
                p = n["poi"]
                lines.append(f"  {hhmm(n['arrive'])}-{hhmm(n['leave'])} {p['name']}"
                             f"(类型:{p['type']}, 评分:{p['rating']}, "
                             f"{'人均' + str(int(p['cost'])) + '元' if p['cost'] else '免费'}, "
                             f"{'室内' if p['indoor'] else '室外'})")
            elif n["kind"] == "move":
                lines.append(f"    → {n['mode']} 约 {n['min']} 分钟")
    return "\n".join(lines)


PROMPT = """你是行程顾问。下面是用户已经选好的行程,里面的时间、评分、通勤时长都是系统算好的真实数据。

{facts}

请指出这份安排里真正值得提醒的问题,挑最重要的 1-3 条。重点关注:
- 相邻两点跨度过大、来回折返
- 某天安排过满或过空
- 室内外搭配、是否需要留机动时间
- 顺序上有没有更省时间的走法

要求:
- 只基于上面给出的数据,不要编造任何景点、时间或价格
- 每条一句话,中文,不超过 40 字,说人话不要客套
- 只输出 JSON 字符串数组,不要其他任何内容,例如 ["第一条","第二条"]"""


def make_advice(plan_days: list, city: str) -> list:
    if MODEL_LIVE:
        try:
            got = _ask_model(_digest(plan_days))
            if got:
                return got
        except Exception:
            pass
    return _rule_advice(plan_days, city)


def _ask_model(facts: str) -> list:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": PROMPT.format(facts=facts)}],
        "temperature": 0.7,
        "max_tokens": 400,
    }
    low = MODEL_NAME.lower()
    if any(k in low for k in THINKING_MODELS):
        payload["enable_thinking"] = False
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        MODEL_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {LLM_KEY}"})
    with urllib.request.urlopen(req, timeout=25) as r:
        d = json.loads(r.read().decode("utf-8"))
    text = d["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):                      # 模型偶尔会包一层代码块
        text = text.strip("`").replace("json", "", 1).strip()
    items = json.loads(text)
    return [str(x) for x in items if isinstance(x, str)][:3]


def _rule_advice(plan_days: list, city: str) -> list:
    """没有模型 Key 时的兜底:规则同样能抓出跨度和过满这两类主要问题。"""
    out = []
    center = CITIES.get(city, {}).get("center")
    for d in plan_days:
        span = d["end"] - START_MIN
        items = d["items"]
        # 每天最多提一条,挑当天最值得说的那件事
        if span > 570:
            out.append(f"第 {d['day']} 天从 9 点排到 {hhmm(d['end'])},约 {span // 60} 小时,"
                       f"偏满。建议删掉一个,或把午餐留长一点。")
            continue
        worst = None
        for a, b in zip(items, items[1:]):
            km = amap.haversine(a["lat"], a["lng"], b["lat"], b["lng"])
            if km > 7 and (worst is None or km > worst[0]):
                worst = (km, a, b)
        if worst:
            _, m = amap.route_time(worst[1], worst[2], city)
            out.append(f"{worst[1]['name']} 到 {worst[2]['name']} 直线 {worst[0]:.1f} 公里,"
                       f"单程约 {m} 分钟,中间会比较赶,可以考虑挪到别的一天。")
            continue
        if len(items) == 1:
            out.append(f"第 {d['day']} 天只排了 {items[0]['name']} 一个点,"
                       f"{items[0]['duration'] // 60} 小时就结束了。可以再加一个顺路的,或者并到别的一天。")
            continue
        if center:
            far = [p for p in items
                   if amap.haversine(p["lat"], p["lng"], center[0], center[1]) > 18]
            if len(far) == len(items):
                out.append(f"第 {d['day']} 天的点都在远郊,路上会花不少时间,建议这天只留行程、不排太紧。")
    if not out:
        out.append("整体节奏合理:相邻两点通勤大多在 25 分钟内,按这个顺序走不会来回折返。")
    return out[:3]
