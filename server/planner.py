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
import hotspots
import llm
import store
from data import CITIES

# ---------- 大模型 ----------
# 配置与 HTTP 细节都在 llm.py 里,这里只做 re-export ——
# app.py 通过 planner.MODEL_LIVE / planner.MODEL_NAME 读这两个量,保持别名不动。
MODEL_LIVE = llm.MODEL_LIVE
MODEL_NAME = llm.MODEL_NAME

# ---------- 当下热点(联网) ----------
# 总开关。联网检索比普通调用慢(实测 5~15 秒),且不是每家供应商都支持联网,
# 不想等、或换供应商后想停掉这个功能时:.env 里写 HOTSPOTS=0
HOTSPOTS_ON = os.getenv("HOTSPOTS", "1").strip().lower() not in ("0", "false", "no", "")

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


def layout_timeline(ordered: list, city: str, on_move=None) -> dict:
    """把"已排好顺序"的景点序列铺成真实时间轴。

    站间耗时来自路径规划接口(绝不交给模型估算);夜景类(best_time=='evening')
    强制排在傍晚(≥17:00)之后;途中跨过 12:00 自动插入午餐。
    和 compute_day 共用,保证 AI 方案与规则方案走同一套"事实落地"逻辑。

    on_move(done, total):每算完一段站间通勤回调一次。这一段要按点对逐个请求
    路径规划接口,天数多点数多时能到十几秒,是加载界面上最需要真实反馈的阶段。
    """
    if not ordered:
        return {"items": [], "timeline": [], "end": START_MIN}
    total_move = max(0, len(ordered) - 1)
    done_move = 0
    # 安全兜底:夜景/傍晚类强制放到当天最后(模型一般也会这么排,这里防止出现
    # "傍晚景点之后又接一个白天景点被推到 18:30" 的违和)。保持组内其余顺序不变。
    ordered = ([p for p in ordered if p.get("best_time") != "evening"]
               + [p for p in ordered if p.get("best_time") == "evening"])
    t = START_MIN
    nodes = []
    for i, it in enumerate(ordered):
        arrive = t
        # 夜景类强制傍晚之后,否则傍晚去就没意义了
        if it.get("best_time") == "evening" and arrive < 17 * 60:
            arrive = 17 * 60
        leave = arrive + it["duration"]
        nodes.append({"kind": "poi", "poi": it, "arrive": arrive, "leave": leave,
                      "best_time_label": it.get("best_time_label", ""),
                      "highlight": it.get("highlight", ""),
                      "why": it.get("why", "")})
        t = leave
        if i < len(ordered) - 1:
            if arrive < LUNCH_START <= leave:
                nodes.append({"kind": "meal", "arrive": t, "leave": t + LUNCH_LEN})
                t += LUNCH_LEN
            mode, mins = amap.route_time(it, ordered[i + 1], city)
            nodes.append({"kind": "move", "mode": mode, "min": mins,
                          "to": ordered[i + 1]["name"]})
            t += mins
            done_move += 1
            if on_move:
                try:
                    on_move(done_move, total_move)
                except Exception:
                    pass
    return {"items": ordered, "timeline": nodes, "end": t}


def compute_day(items: list, city: str) -> dict:
    """生成当天时间轴(规则版:空间最近邻 + 夜景置后)。

    实际落地交给 layout_timeline,这里只决定"顺序"。AI 方案则直接把 AI 给的
    顺序传给 layout_timeline,不重排 —— 这样模型对"先去哪后去哪"的综合考量能保留。"""
    evening = [p for p in items if p.get("best_time") == "evening"]
    normal = [p for p in items if p.get("best_time") != "evening"]
    ordered = []
    if normal:
        ordered += nearest_order(normal)
    if evening:
        ordered += nearest_order(evening)
    return layout_timeline(ordered, city)


def _plan_key(city: str, days: int, prefs: list, pace: str) -> str:
    """同一组输入就是同一份行程 —— 重复生成没必要再烧一次大模型和路径规划。
    v2:AI 综合规划版(选择/分组/排序/经典特色均由模型给出),与旧的规则版缓存区分开。
    v3:再叠一层联网热点。必须换 key —— 否则命中 v2 的旧缓存,热点完全体现不出来。
    """
    return f'v3|{city}|{days}|{",".join(sorted(prefs or []))}|{pace}'


# ---------- AI 综合规划 ----------
# 思路:模型只从"真实候选池"里挑、排、组合,并给每个景点写一句真正的经典特色;
# 站间真实耗时、时段硬性约束仍由接口/规则落地,模型无法编造。这样既有"综合考量",
# 又不会跑出不存在的景点或编出假的交通时间。

PLAN_PROMPT = """你是一位资深旅行行程规划师。下面是一座城市的真实候选景点(数据来自地图接口,均为真实存在的地方)。请综合"每个景点的经典特色、最佳游玩时段、地理位置(用给出的坐标判断远近)、用户偏好、整体节奏"来制定一份多日行程。

城市: {city}
天数: {days} 天
节奏: {pace}(轻松≈每天3个,适中≈4个,紧凑≈5个)
用户偏好: {prefs}

候选景点(每行一个,name | 类型 | 评分 | 建议游玩分钟 | 最佳时段 | 人均 | 室内/室外 | 坐标 | 临近景点):
{pois}
{hot}
要求:
1. 只从上述候选里挑选,不要编造任何景点。每天点数大致符合节奏。
2. 同一天的景点要在地理上尽量紧凑(用坐标与"临近"信息判断,避免跨城折返);夜景/傍晚类(最佳时段含"傍晚/夜间")放当天最后;户外/自然类尽量排上午或下午、避开正午暴晒;上午类排前面。
3. 兼顾用户偏好与类型多样(历史、自然、美食、拍照不要全挤在一天)。
4. 为每个被你选中的景点写一句"经典特色"(它最值得看的是什么,例如某宫殿的世界之最、某古街的历史地位、某博物馆的镇馆之宝),以及一句"为什么排这里"(结合时段/位置/与其他景点的搭配)。
5. 每天写一句"本日思路",说明你当天安排的逻辑(为何这样串联、如何照顾特色与体力)。
6. 若有"当下热点"段落:这批是联网核实过、确在地图上的当下热点,请尽量安排进合适的一天
   —— 限时展演务必留意它的有效期,季节景观要对得上季节。写"经典特色"时要点出它此刻特别
   值得去的理由。但它们是加分项:不要为了让位给热点而挤掉这座城市真正的地标与经典景点。

只输出如下结构的 JSON,不要任何解释文字、不要代码块标记:
{{
  "summary": "整体思路一句话",
  "classic": {{ "<景点名>": "<一句经典特色>", ... }},
  "days": [
    {{ "note": "本日思路...", "spots": [ {{"name":"<景点名>", "why":"<为什么排这里>"}}, ... ] }},
    ...
  ]
}}"""


def _norm_name(s: str) -> str:
    return (s or "").replace(" ", "").replace("·", "").replace("•", "").lower()


def _build_plan_prompt(city: str, pool: list, days: int, prefs: list, pace: str,
                       hot: dict = None) -> str:
    """挑出要喂给模型的候选(按评分 + 偏好加权截断到可控规模),并附上临近景点提示,
    让模型有地理聚类所需的上下文。

    hot: hotspots.discover() 的结果。热点额外加权,否则新开的展/馆评分不高,
    会被截断在这批候选之外 —— 模型根本看不到它们。
    """
    want = []
    for p in prefs or []:
        want += PREF_TAGS.get(p, [])
    scored = []
    for p in pool:
        s = p["rating"]
        if any(t in want for t in p.get("tags", [])):
            s += 0.6
        if p.get("hot"):
            s += 2.0          # 当下热点:评分再低也要进模型视野
        scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for _, p in scored[:30]]
    extra = [p for _, p in scored[30:] if any(t in want for t in p.get("tags", []))]
    sel = (top + extra)[:45]

    # 预计算临近景点,给模型地理聚类依据
    nb = {}
    for p in sel:
        others = sorted((q for q in sel if q is not p),
                        key=lambda q: amap.haversine(p["lat"], p["lng"], q["lat"], q["lng"]))
        nb[p["id"]] = [(q["name"], round(amap.haversine(p["lat"], p["lng"], q["lat"], q["lng"]), 1))
                       for q in others[:3]]
    lines = []
    for p in sel:
        near = ", ".join(f"{n}({d}km)" for n, d in nb[p["id"]])
        flag = "[当下热点] " if p.get("hot") else ""
        lines.append(
            f"- {flag}{p['name']} | 类型:{p['type']} | 评分:{p['rating']} | "
            f"建议游玩:{p['duration']}分钟 | 最佳时段:{p.get('best_time_label','')} | "
            f"人均{int(p['cost']) if p['cost'] else 0}元 | {'室内' if p['indoor'] else '室外'} | "
            f"坐标:{p['lat']:.4f},{p['lng']:.4f} | 临近:{near}")

    return PLAN_PROMPT.format(city=city, days=days, pace=pace,
                              prefs=", ".join(prefs) if prefs else "（无特别偏好）",
                              pois="\n".join(lines),
                              hot=_hot_section(hot, sel))


def _hot_section(hot: dict, sel: list) -> str:
    """给模型看的热点段落。只列已经被核实在目标候选里出现的热点 ——
    进不了视野的名字列出来也没用,反而勾着模型去写它。"""
    if not hot or not hot.get("items"):
        return ""
    shown = {p["name"] for p in sel}
    rows = [p for p in hot["items"] if p["name"] in shown]
    if not rows:
        return ""
    body = "\n".join(
        f"- {p['name']} | {p.get('hot_kind') or '当下热点'} | {p.get('hot_why') or ''}"
        for p in rows)
    return ("\n当下热点(联网核实过、地图上确有其地,请留意限时信息的有效期):\n"
            + body + "\n")


def ai_plan(city: str, pool: list, days: int, prefs: list, pace: str, hot: dict = None):
    """让大模型综合候选池给出:分组(哪天去哪几个)、顺序、每点经典特色与排布理由。

    返回 dict 或 None(网络/解析失败时返回 None,由调用方回退到规则版)。
    注意:模型只能从 pool 里挑,返回的景点名必须能在 pool 中匹配到,匹配不上的直接丢弃,
    因此绝不会把不存在的景点排进行程。"""
    try:
        raw = llm.ask_json(_build_plan_prompt(city, pool, days, prefs, pace, hot),
                          max_tokens=1800, timeout=40)
    except Exception as e:
        import logging
        logging.getLogger("planner").warning("ai_plan 模型调用失败: %s", e)
        return None
    if not isinstance(raw, dict) or not raw.get("days"):
        return None

    by_name = {_norm_name(p["name"]): p for p in pool}
    classic = raw.get("classic") or {}
    groups, day_notes, used = [], [], set()
    for d in raw["days"]:
        grp = []
        for item in d.get("spots", []):
            nm = item.get("name", "")
            poi = by_name.get(_norm_name(nm))
            if not poi or poi["id"] in used:
                continue
            # 把模型给的"经典特色"和"为什么排这里"挂回 POI,供时间轴/地图/建议复用
            if nm in classic:
                poi["highlight"] = classic[nm]
            if item.get("why"):
                poi["why"] = item["why"]
            used.add(poi["id"])
            grp.append(poi)
        if grp:
            groups.append(grp)
            day_notes.append(d.get("note", ""))
        if len(groups) >= days:
            break

    if not groups:
        return None
    return {"groups": groups, "day_notes": day_notes,
            "summary": raw.get("summary", ""), "classic": classic}


def build_plan(city: str, days: int, prefs: list, pace: str, pool: list = None,
               fresh: bool = False, emit=None) -> dict:
    """生成完整行程。emit(dict) 用于向前端推送真实进度事件,没传就完全静默。

    各阶段的 pct 区间是估的:

        pool  0.00 - 0.38   多个关键词串行搜索(可能触发限流退避)
        hot   0.38 - 0.50   联网检索当下热点
        ai    0.50 - 0.72   大模型综合编排(含一次可能的重试)
        route 0.72 - 0.95   逐段请求路径规划,算真实通勤时间
        save  0.95 - 1.00   整理输出

    emit 抛异常一律吞掉 —— 进度是给人看的,不能反过来拖垮生成。
    """
    def _emit(stage, detail, pct):
        if emit:
            try:
                emit({"stage": stage, "detail": detail,
                      "pct": round(max(0.0, min(1.0, pct)), 3)})
            except Exception:
                pass

    key = _plan_key(city, days, prefs, pace)
    if not fresh:
        got = store.read("plan", key)
        if isinstance(got, dict) and got.get("days"):
            got["cached"] = True
            _emit("done", "命中本地缓存", 1.0)
            return got

    # pool 可由调用方传入(接口层已经取过一次,避免重复请求)。
    # 传进来的话,pool 阶段的进度一律由调用方负责上报 —— 否则这里会先报一个 0.02,
    # 排在人家已经报过的 0.38 后面,前端进度条会肉眼可见地往回缩。
    if pool is None:
        _emit("pool", "读取城市景点数据", 0.02)
        pool = amap.build_pool(
            city,
            on_kw=lambda d, t, f: _emit(
                "pool",
                f"检索景点 {d}/{t}" + (f" · 已收录 {f} 个" if f else ""),
                0.02 + 0.36 * (d / max(1, t))))
        _emit("pool", f"候选池就绪,共 {len(pool)} 个可排景点", 0.38)
    per = PER_DAY.get(pace, 4)

    # 当下热点:联网发现 + 地图核实。失败内部静默,最多慢几秒。
    # 好:hotspots 里命中已有 POI 时是原地打标,池子自动带上 hot 标记,
    #    只有"地图上能搜到但候选池原本没有"的那些才需要并进来。
    _emit("hot", "联网检索当下热点", 0.40)
    hot = hotspots.discover(city, pool) if HOTSPOTS_ON else None
    hot_notes = []
    if hot and hot.get("items"):
        have = {p["name"] for p in pool}
        for p in hot["items"]:
            if p["name"] not in have:
                pool.append(p)
                have.add(p["name"])
    if hot and hot.get("notes"):
        hot_notes = [{"name": n, "why": w} for n, w in hot["notes"]]
    got_n = len((hot or {}).get("items") or [])
    _emit("hot", f"补充 {got_n} 个当下热点" if got_n else "未发现新增热点", 0.50)

    # 优先走 AI 综合规划;只有没配模型 Key、或模型调用/解析失败时才回退到规则版。
    groups, day_notes, summary = None, [], ""
    if MODEL_LIVE:
        _emit("ai", "AI 正在综合特色、时段、体力编排行程", 0.52)
        res = ai_plan(city, pool, days, prefs, pace, hot)
        if res:
            groups, day_notes, summary = res["groups"], res["day_notes"], res["summary"]
            # 把模型给的经典特色落盘,下次 /api/pois、replan 也能复用
            if res.get("classic"):
                store.write("feature", city, res["classic"])
            n = sum(len(g) for g in groups)
            _emit("ai", f"编排完成,共 {len(groups)} 天 {n} 个景点", 0.72)
        else:
            _emit("ai", "AI 未返回结果,改用规则版编排", 0.72)

    if not groups:
        # 规则版兜底:评分排序 → 最近邻贪心分组 → 时间轴
        _emit("ai", "按评分与偏好分组", 0.60)
        ranked = score(pool, prefs)
        groups = split_days(ranked, days, per, pace)
        day_notes = [""] * len(groups)
        _emit("ai", "分组完成", 0.72)

    # 轻量纠偏:防止模型把某天排得过满/过空。每天数控制在 per..per+1;
    # 总点数上限 days*per+2,超出的从后几天末尾摘除。
    capped = []
    total = 0
    for g in groups:
        cap = per + 1
        if len(g) > cap:
            g = g[:cap]
        if total >= days * per + 2 and g:
            g = g[:1] if len(g) > 1 else g
        capped.append(g)
        total += len(g)
    groups = capped

    used = {p["id"] for g in groups for p in g}
    plan_days = []
    total_move = sum(max(0, len(g) - 1) for g in groups)
    done_move = 0
    _emit("route", "计算站间真实通勤时间", 0.73)

    def _make_move_cb(base):
        """layout_timeline 的计数是每天内部从 0 开始,这里折算成全局进度。"""
        def cb(done_in_day, _total_in_day):
            nonlocal done_move
            done_move = base + done_in_day
            if total_move:
                _emit("route", f"通勤计算 {done_move}/{total_move} 段",
                      0.73 + 0.22 * done_move / total_move)
        return cb

    for i, g in enumerate(groups):
        d = layout_timeline(g, city, on_move=_make_move_cb(done_move))
        d["day"] = i + 1
        d["note"] = day_notes[i] if i < len(day_notes) else ""
        plan_days.append(d)
    _emit("route", f"通勤时间已落地,共 {total_move} 段", 0.95)

    _emit("save", "整理每日安排与出行建议", 0.97)

    plan = {
        "city": city,
        "pace": pace,
        "days": plan_days,
        "summary": summary,
        "advice": make_advice(plan_days, city),
        "hot_notes": hot_notes,
        "pool": [p for p in pool if p["id"] not in used],
        "live": amap.LIVE,
        "model": MODEL_LIVE,
        "cached": False,
        "cached_ts": time.time(),
    }
    # 落盘:同样的输入下次直接返回,连大模型都不用叫
    store.write("plan", key, plan)
    _emit("done", "行程已就绪", 1.0)
    return plan


def replan(city: str, groups: list) -> dict:
    """用户改完地点后重排并重算时间轴。groups 是 [[poi, poi, ...], ...]。

    尊重用户手动排好的顺序(不再用最近邻重排),只落地"事实":夜景置后、真实通勤时间。"""
    plan_days = []
    for i, g in enumerate(groups):
        d = layout_timeline(g, city)
        d["day"] = i + 1
        plan_days.append(d)
    used = {p["id"] for d in plan_days for p in d["items"]}
    return {
        "city": city, "pace": "", "days": plan_days,
        "advice": make_advice(plan_days, city),
        "pool": [p for p in amap.build_pool(city) if p["id"] not in used],
        "live": amap.LIVE, "model": MODEL_LIVE,
    }


# ---------- 对话式调整计划 ----------
# 用户用自然语言提意见(太赶/想多看人文/第二天不想爬山...),由大模型在
# "真实候选池"内重新编排,产出一份新行程 + 一段回复。
# 硬约束和 ai_plan 一样:模型只能从候选池里挑,名字对不上的直接丢弃,
# 因此绝不会凭空多出景点;时间轴仍由 layout_timeline 用真实接口数据落地。

CHAT_PROMPT = """你是一位资深旅行规划师，正在和一位已拿到行程的游客讨论并修改他的行程。
请理解他的意见，并给出修改后的行程。

城市: {city}
总天数: {days} 天
节奏: {pace}

【当前行程】
{current}

【可用的真实景点】(只能从这里挑选，绝不能编造新的地方)
{pois}

【游客的意见】
{msg}

要求：
1. 先理解意见的意图：如果他嫌太赶，就减少点数或把远的挪到一起；如果他想去某类主题，就用该主题的景点替换；如果他不想去某个地方，就把它删掉换成别的。
2. 只从"可用的真实景点"中挑选。总数仍要符合节奏与总天数。
3. 每天内部保持地理紧凑，夜景/傍晚类放当天最后。
4. 为每个景点写一句"为什么这样排"，每天写一句"本日思路"。
5. 如果你认为他的意见不合理（比如会绕远路），照做但在 reply 里简短说明理由。

只输出如下 JSON，不要解释文字、不要代码块标记：
{{
  "reply": "两三句话回应他的意见：说明你怎么改的、为什么这么改",
  "summary": "修改后的整体思路一句话",
  "days": [
    {{ "note": "本日思路...", "spots": [ {{"name":"<景点名>", "why":"<为什么排这里>"}}, ... ] }},
    ...
  ]
}}"""


def _chat_current(plan_days: list) -> str:
    """把当前行程压成纯文本(名字+时段+特色),作为对话上下文。"""
    lines = []
    for d in plan_days:
        lines.append(f"第{d['day']}天(结束 {hhmm(d.get('end', 0))}):")
        for n in d.get("timeline", []):
            if n.get("kind") != "poi":
                continue
            p = n["poi"]
            extra = f"，{p['highlight']}" if p.get("highlight") else ""
            lines.append(f"  {hhmm(n['arrive'])}-{hhmm(n['leave'])} {p['name']}({p['type']},"
                         f"{p.get('best_time_label','')}{extra})")
        if d.get("note"):
            lines.append(f"  思路: {d['note']}")
    return "\n".join(lines) or "(暂无)"


def _chat_pois(plan_days: list, pool: list, limit: int = 45) -> str:
    """候选池 = 当前行程里已在的点 + 未用的候选点,去重后按评分截断。"""
    seen, out = set(), []
    for d in plan_days:
        for n in d.get("timeline", []):
            if n.get("kind") == "poi":
                p = n["poi"]
                if p["id"] not in seen:
                    seen.add(p["id"])
                    out.append(p)
    for p in sorted(pool, key=lambda x: x.get("rating", 0), reverse=True):
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        out.append(p)
    lines = []
    for p in out[:limit]:
        lines.append(
            f"- {p['name']} | 类型:{p['type']} | 评分:{p['rating']} | "
            f"建议游玩:{p['duration']}分钟 | 最佳时段:{p.get('best_time_label','')} | "
            f"人均{int(p['cost']) if p['cost'] else 0}元 | {'室内' if p['indoor'] else '室外'}")
    return "\n".join(lines)


def chat_plan(city: str, plan_days: list, message: str, pool: list = None,
              pace: str = "适中", prefs: list = None):
    """按用户意见重排行程。返回 {reply, groups, day_notes, summary, classic} 或 None。

    None 表示没配模型或调用/解析失败,调用方应回退到规则版(只重排顺序)。
    """
    if not MODEL_LIVE or not plan_days:
        return None
    if pool is None:
        pool = amap.build_pool(city)
    prompt = CHAT_PROMPT.format(
        city=city, days=len(plan_days), pace=pace or "适中",
        current=_chat_current(plan_days),
        pois=_chat_pois(plan_days, pool),
        msg=(message or "").strip() or "请在保持整体均衡的前提下,优化这份行程。")
    try:
        raw = llm.ask_json(prompt, max_tokens=2200)
    except Exception as e:
        import logging
        logging.getLogger("planner").warning("chat_plan 模型调用失败: %s", e)
        return None
    if not isinstance(raw, dict) or not raw.get("days"):
        return None

    # 可选集合 = 当前行程里的点 + 候选池,模型选的名字必须命中其一
    by_name = {}
    for d in plan_days:
        for n in d.get("timeline", []):
            if n.get("kind") == "poi":
                by_name[_norm_name(n["poi"]["name"])] = n["poi"]
    for p in pool:
        by_name.setdefault(_norm_name(p["name"]), p)

    groups, day_notes, used = [], [], set()
    for d in raw["days"]:
        grp = []
        for item in d.get("spots", []):
            nm = item.get("name", "")
            poi = by_name.get(_norm_name(nm))
            if not poi or poi["id"] in used:
                continue
            if item.get("why"):
                poi["why"] = item["why"]
            used.add(poi["id"])
            grp.append(poi)
        if grp:
            groups.append(grp)
            day_notes.append(d.get("note", ""))
        if len(groups) >= len(plan_days):
            break

    if not groups:
        return None
    return {
        "reply": (raw.get("reply") or "").strip(),
        "groups": groups, "day_notes": day_notes,
        "summary": (raw.get("summary") or "").strip(),
        "classic": raw.get("classic") or {},
    }


def chat_fallback(city: str, plan_days: list, message: str) -> dict:
    """没接大模型时的兜底:按关键词做最朴素的调整(减点/换顺序),并如实说明。"""
    items = [n["poi"] for d in plan_days for n in d.get("timeline", []) if n.get("kind") == "poi"]
    msg = (message or "")
    groups = [d["items"] for d in plan_days if d.get("items")]
    if any(k in msg for k in ("少一点", "太赶", "轻松", "别太累", "减一个")):
        # 从点数最多的那天减一个(list.index 对重复元素不可靠,按序号遍历)
        for i in range(len(groups)):
            if len(groups[i]) > 2:
                groups[i] = groups[i][:-1]
                break
        reply = "已按你说的减了一个点,行程会松一些。不过更细的调整(换景点、控节奏)需要接上大模型。"
    else:
        for i, g in enumerate(groups):
            groups[i] = compute_day(g, city)["items"]
        reply = "已按地理位置重排了每天的顺序(就近串着走)。想换景点或改主题,需要接上大模型。"
    plan_days = []
    for i, g in enumerate(groups):
        d = layout_timeline(g, city)
        d["day"] = i + 1
        plan_days.append(d)
    used = {p["id"] for d in plan_days for p in d["items"]}
    return {
        "reply": reply, "summary": "",
        "plan": {
            "city": city, "pace": "", "days": plan_days,
            "advice": make_advice(plan_days, city),
            "pool": [p for p in amap.build_pool(city) if p["id"] not in used],
            "live": amap.LIVE, "model": False,
        },
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
                             f"建议游玩:{p['duration']}分钟, 最佳时段:{p.get('best_time_label','')}, "
                             f"特色:{p.get('highlight','')}, "
                             f"{'人均' + str(int(p['cost'])) + '元' if p['cost'] else '免费'}, "
                             f"{'室内' if p['indoor'] else '室外'})")
            elif n["kind"] == "move":
                lines.append(f"    → {n['mode']} 约 {n['min']} 分钟")
    return "\n".join(lines)


PROMPT = """你是行程顾问。下面是用户已经选好的行程,里面的时间、评分、通勤时长、建议游玩时长、最佳时段、特色都是系统算好的真实数据。

{facts}

请指出这份安排里真正值得提醒的问题,挑最重要的 1-3 条。重点关注:
- 相邻两点跨度过大、来回折返
- 某天安排过满或过空
- 室内外搭配、是否需要留机动时间
- 顺序上有没有更省时间的走法
- 是否有景点的最佳时段被排错(例如夜景类没安排在傍晚、户外景点排到正午暴晒)
- 是否真正体现了每个景点的经典特色(可提示"XX 以 XX 为经典特色,建议多留时间/请讲解")

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
    return llm.ask_list(PROMPT.format(facts=facts), max_tokens=400, limit=3)


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
