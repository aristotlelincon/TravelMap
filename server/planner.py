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

# 用户自己收藏过的景点在打分里的加权。
# 要比"命中偏好"(+14)高得多 —— 那是勾选来的,这是专门攒的,意愿更强。
# 但它只是一道加分,不是钉死:后面 split_days 的预算/跨区/单段上限照样能把它挡下来,
# 这正是"优先排入,但不硬塞"想要的 —— 收藏不会因为绕路把行程毁掉。
WISH_BOOST = 40
# 喂给模型时也要加权,否则收藏的点评分不高会被截断在候选之外,模型根本看不见它。
WISH_PROMPT_BOOST = 3.0


# ---------- 编排 ----------

def _median_center(pool: list):
    """用候选池的中位数当城市中心,避免为这件事再调一次接口。"""
    if not pool:
        return None
    lats = sorted(p["lat"] for p in pool)
    lngs = sorted(p["lng"] for p in pool)
    n = len(lats)
    return (lats[n // 2], lngs[n // 2])


def score(pool: list, prefs: list, wish: list = None) -> list:
    """候选池打分排序。高分的先被 split_days 取作每天的种子,也就更可能排进行程。

    wish 是用户收藏的景点名:命中就加权,让收藏优先成为种子。它只影响顺序,
    不绕过时间预算与跨区约束 —— 收藏不该把行程排成绕路。
    """
    want = []
    for p in prefs:
        want += PREF_TAGS.get(p, [])
    wset = {_norm_name(x) for x in (wish or [])}
    center = _median_center(pool)
    out = []
    for poi in pool:
        s = poi["rating"] * 10
        for t in poi["tags"]:
            if t in want:
                s += 14
        if _norm_name(poi.get("name")) in wset:
            s += WISH_BOOST
        if center and amap.haversine(poi["lat"], poi["lng"], center[0], center[1]) > 25:
            s -= 6      # 远郊点往后排:刚到一个城市,不该第一天就跑几十公里
        out.append((poi, s))
    out.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in out]


def nearest_order(items: list, start: dict = None) -> list:
    """最近邻重排:从第一个点出发,每次走向剩下的最近的点。

    start 传落脚点时,头一个点改成"离落脚点最近的那个" —— 早上不用先横穿半座城
    再折回来。不给 start 就退回原来的行为(保住调用方已有的顺序)。"""
    if len(items) <= 1:
        return list(items)
    rest = list(items)
    if start and start.get("lat") and start.get("lng"):
        first = min(rest, key=lambda p: amap.haversine(
            start["lat"], start["lng"], p["lat"], p["lng"]))
    else:
        first = rest[0]
    rest.remove(first)
    out = [first]
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


def _move(a: dict, b: dict, city: str, home: bool = False) -> dict:
    """一段通勤。去程 / 回程(home=True)都从这里出,省得两处写得不一致。"""
    mode, mins = amap.route_time(a, b, city)
    node = {"kind": "move", "mode": mode, "min": mins,
            "to": b.get("name") or "落脚点", "from": a.get("name") or "落脚点"}
    if home:
        node["home"] = True
    return node


def layout_timeline(ordered: list, city: str, base: dict = None,
                    on_move=None) -> dict:
    """把"已排好顺序"的景点序列铺成真实时间轴。

    站间耗时来自路径规划接口(绝不交给模型估算);夜景类(best_time=='evening')
    强制排在傍晚(≥17:00)之后;途中跨过 12:00 自动插入午餐。
    和 compute_day 共用,保证 AI 方案与规则方案走同一套"事实落地"逻辑。

    base 是落脚点(酒店/住处)。给了的话,每天 9:00 从落脚点出发、最后回到落脚点,
    这两段通勤同样走真实接口 —— 之前的行程是"每天凭空出现在第一个景点门口",
    换个落脚点通勤差别很大,按直线估算出来的到店时间不可信。

    on_move(done, total):每算完一段站间通勤回调一次。这一段要按点对逐个请求
    路径规划接口,天数多点数多时能到十几秒,是加载界面上最需要真实反馈的阶段。
    """
    if not ordered:
        return {"items": [], "timeline": [], "end": START_MIN}
    home = bool(base and base.get("lat") and base.get("lng"))
    # 站间段数:点之间的,加上去程 / 回程两段
    total_move = max(0, len(ordered) - 1) + (2 if home else 0)
    done_move = 0
    # 安全兜底:夜景/傍晚类强制放到当天最后(模型一般也会这么排,这里防止出现
    # "傍晚景点之后又接一个白天景点被推到 18:30" 的违和)。保持组内其余顺序不变。
    ordered = ([p for p in ordered if p.get("best_time") != "evening"]
               + [p for p in ordered if p.get("best_time") == "evening"])
    t = START_MIN
    nodes = []

    def _tick():
        nonlocal done_move
        done_move += 1
        if on_move:
            try:
                on_move(done_move, total_move)
            except Exception:
                pass

    if home:                                   # 9:00 从落脚点动身
        mv = _move(base, ordered[0], city)
        nodes.append(mv)
        t += mv["min"]
        _tick()

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
                          "to": ordered[i + 1]["name"], "from": it["name"]})
            t += mins
            _tick()

    if home:                                   # 玩完回到落脚点,这天到这里才算结束
        mv = _move(ordered[-1], base, city, home=True)
        nodes.append(mv)
        t += mv["min"]
        _tick()
    return {"items": ordered, "timeline": nodes, "end": t}


def _leg_min(a: dict, b: dict, city: str) -> int:
    """一段通勤的分钟数,拿不到就当 0(不能因为拿不到就整段崩掉)。"""
    try:
        return int(amap.route_time(a, b, city)[1])
    except Exception:
        return 0


def _worth(p: dict) -> float:
    """这个点值不值得留。删点时按它排序 —— 同样省时间,先砍不值钱的。"""
    s = float(p.get("rating") or 0) * 10
    if p.get("hot"):
        s += 20        # 当期限定展演,过滤掉就没了,优先保
    return s


def _drop_gain(ordered: list, i: int, city: str, base: dict = None):
    """移除第 i 个点能省下多少分钟 = 绕开它的通勤差 + 它本身的游玩时长。

    只多请求一段"跳接段" prev→next,其余段 layout_timeline 已经算过并进了缓存,
    所以评估一整轮候选的额外接口开销约等于点数,几个而已。
    """
    n = len(ordered)
    if n <= 1:
        return None
    home = bool(base and base.get("lat") and base.get("lng"))
    p = ordered[i]
    prev = base if (i == 0 and home) else (ordered[i - 1] if i > 0 else None)
    nxt = base if (i == n - 1 and home) else (ordered[i + 1] if i < n - 1 else None)
    saved = int(p.get("duration") or 0)
    if prev is not None and nxt is not None:
        saved += (_leg_min(prev, p, city) + _leg_min(p, nxt, city)
                  - _leg_min(prev, nxt, city))
    elif prev is not None:
        saved += _leg_min(prev, p, city)
    elif nxt is not None:
        saved += _leg_min(p, nxt, city)
    return max(0, saved)


def fit_budget(ordered: list, city: str, budget: int = None, base: dict = None,
               pace: str = "适中", max_drop: int = 2, min_keep: int = 2,
               on_move=None) -> dict:
    """用真实通勤校验当天预算,超了就从当天摘掉性价比最低的点,最多摘 max_drop 个。

    为什么需要这一步:分组时用 estimate_route 估算,真正铺时间轴时才请求真实
    路径接口 —— 真实通勤几乎总比估算长,于是"预算 9 小时"的一天落地后可能
    拖到 20:00 才结束。以前没人管这个差,现在用真实结果兜底。

    摘谁不是简单砍最后一个:对每个点算"去掉它能省几分钟",在所有能压回预算的
    候选里挑价值(_worth)最低的;一个都压不回来就挑省时间最多的那个,再迭代一轮。
    """
    if budget is None:
        budget = BUDGET.get(pace, 9 * 60)
    cur = list(ordered)
    res = layout_timeline(cur, city, base=base, on_move=on_move)
    dropped = []          # 攒在循环外:每次重算都换一个新 dict,挂在 res 上会丢
    for _ in range(max_drop):
        if len(cur) <= min_keep:
            break
        span = res["end"] - START_MIN
        if span <= budget:
            break
        cands = []
        for i in range(len(cur)):
            g = _drop_gain(cur, i, city, base)
            if g is None:
                continue
            cands.append((i, g, span - g))       # (下标, 能省, 删后时长)
        if not cands:
            break
        ok = [c for c in cands if c[2] <= budget]
        if ok:                                    # 能压回预算的里,砍最不值的
            pick = min(ok, key=lambda c: _worth(cur[c[0]]))
        else:                                     # 都压不回,砍省时间最多的
            pick = max(cands, key=lambda c: (c[1], -_worth(cur[c[0]])))
        gone = cur[pick[0]]
        cur = cur[:pick[0]] + cur[pick[0] + 1:]
        # 重算只走缓存,不再上报进度,免得进度条来回滚
        res = layout_timeline(cur, city, base=base)
        dropped.append({"name": gone.get("name"), "id": gone.get("id"),
                        "min": pick[1]})
    if dropped:
        res["dropped"] = dropped
    return res


def compute_day(items: list, city: str, base: dict = None) -> dict:
    """生成当天时间轴(规则版:空间最近邻 + 夜景置后)。

    实际落地交给 layout_timeline,这里只决定"顺序"。AI 方案则直接把 AI 给的
    顺序传给 layout_timeline,不重排 —— 这样模型对"先去哪后去哪"的综合考量能保留。"""
    evening = [p for p in items if p.get("best_time") == "evening"]
    normal = [p for p in items if p.get("best_time") != "evening"]
    ordered = []
    if normal:
        ordered += nearest_order(normal, base)
    if evening:
        ordered += nearest_order(evening)
    return layout_timeline(ordered, city, base)


def _plan_key(city: str, days: int, prefs: list, pace: str, base: dict = None,
              wish: list = None) -> str:
    """同一组输入就是同一份行程 —— 重复生成没必要再烧一次大模型和路径规划。
    v2:AI 综合规划版(选择/分组/排序/经典特色均由模型给出),与旧的规则版缓存区分开。
    v3:再叠一层联网热点。必须换 key —— 否则命中 v2 的旧缓存,热点完全体现不出来。
    v4:再加落脚点。落脚点不同则每天的出发/回程通勤都不一样,必须各自缓存 ——
       这也是为什么 key 里只放地点名:坐标小数位漂移不该导致缓存永远不命中。
    v5:再加心愿清单。收藏的内容不一样,排出来的行程就不一样,不能共用同一份缓存 ——
       否则用户新收一个地方再生成,命中的还是那份没带收藏的旧行程,他会以为功能没生效。
    """
    b = (base or {}).get("name") or "-"
    w = ",".join(sorted({_norm_name(x) for x in (wish or [])}))
    return f'v5|{city}|{days}|{",".join(sorted(prefs or []))}|{pace}|{b}|{w}'


# ---------- AI 综合规划 ----------
# 思路:模型只从"真实候选池"里挑、排、组合,并给每个景点写一句真正的经典特色;
# 站间真实耗时、时段硬性约束仍由接口/规则落地,模型无法编造。这样既有"综合考量",
# 又不会跑出不存在的景点或编出假的交通时间。

PLAN_PROMPT = """你是一位资深旅行行程规划师。下面是一座城市的真实候选景点(数据来自地图接口,均为真实存在的地方)。请综合"每个景点的经典特色、最佳游玩时段、地理位置(用给出的坐标判断远近)、用户偏好、整体节奏"来制定一份多日行程。

城市: {city}
天数: {days} 天
节奏: {pace}(轻松≈每天3个,适中≈4个,紧凑≈5个)
用户偏好: {prefs}
落脚点(住宿/出发点): {base}

候选景点(每行一个,name | 类型 | 评分 | 建议游玩分钟 | 最佳时段 | 人均 | 室内/室外 | 距落脚点 | 坐标 | 临近景点):
{pois}
{hot}
{wish}
要求:
1. 只从上述候选里挑选,不要编造任何景点。每天点数大致符合节奏。
2. 同一天的景点要在地理上尽量紧凑(用坐标与"临近"信息判断,避免跨城折返);夜景/傍晚类(最佳时段含"傍晚/夜间")放当天最后;户外/自然类尽量排上午或下午、避开正午暴晒;上午类排前面。
3. 兼顾用户偏好与类型多样(历史、自然、美食、拍照不要全挤在一天)。{base_req}
4. 为每个被你选中的景点写一句"经典特色"(它最值得看的是什么,例如某宫殿的世界之最、某古街的历史地位、某博物馆的镇馆之宝),以及一句"为什么排这里"(结合时段/位置/与其他景点的搭配)。
5. 每天写一句"本日思路",说明你当天安排的逻辑(为何这样串联、如何照顾特色与体力)。
6. 若有"当下热点"段落:这批是联网核实过、确在地图上的当下热点,请尽量安排进合适的一天
   —— 限时展演务必留意它的有效期,季节景观要对得上季节。写"经典特色"时要点出它此刻特别
   值得去的理由。但它们是加分项:不要为了让位给热点而挤掉这座城市真正的地标与经典景点。
7. 若有"用户收藏"段落:那是用户自己攒的、一直想去的地方,候选里标了 [你收藏过]。
   请优先把它们排进行程 —— 这是他明确表达过的意愿,权重高于普通高分景点。
   但第 2 条的地理紧凑仍然优先:要是某个收藏离当天其他点太远、排进去会明显绕路,
   就别硬塞,让它留给别的机会,行程整体的顺畅比单点意愿更重要。

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
                       hot: dict = None, base: dict = None, wish: list = None) -> str:
    """挑出要喂给模型的候选(按评分 + 偏好加权截断到可控规模),并附上临近景点提示,
    让模型有地理聚类所需的上下文。

    hot: hotspots.discover() 的结果。热点额外加权,否则新开的展/馆评分不高,
    会被截断在这批候选之外 —— 模型根本看不到它们。
    wish: 用户收藏的景点名。同理要加权,而且必须顶到候选最前面 —— 收藏的点未必高分,
    按评分排序再截断的话它第一个被切掉,模型连看都看不见它。
    """
    want = []
    for p in prefs or []:
        want += PREF_TAGS.get(p, [])
    wset = {_norm_name(x) for x in (wish or [])}
    scored = []
    for p in pool:
        s = p["rating"]
        if any(t in want for t in p.get("tags", [])):
            s += 0.6
        if p.get("hot"):
            s += 2.0          # 当下热点:评分再低也要进模型视野
        if _norm_name(p.get("name")) in wset:
            s += WISH_PROMPT_BOOST
        scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for _, p in scored[:30]]
    extra = [p for _, p in scored[30:] if any(t in want for t in p.get("tags", []))]
    # 收藏顶到最前:45 条是从前往后切的,排在后面等于白收藏
    ordered = top + extra
    sel = ([p for p in ordered if _norm_name(p["name"]) in wset]
           + [p for p in ordered if _norm_name(p["name"]) not in wset])[:45]

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
        if _norm_name(p["name"]) in wset:
            flag += "[你收藏过] "
        # 到落脚点的直线距离。直线而非路程是因为这条路模型没法算准确,而"远不远"
        # 只需要个量级;真正的通勤时间仍由路径规划接口落地,不交给模型。
        far = ""
        if _valid_base(base):
            far = (f"距落脚点:{amap.haversine(base['lat'], base['lng'], p['lat'], p['lng']):.1f}km | ")
        lines.append(
            f"- {flag}{p['name']} | 类型:{p['type']} | 评分:{p['rating']} | "
            f"建议游玩:{p['duration']}分钟 | 最佳时段:{p.get('best_time_label','')} | "
            f"人均{int(p['cost']) if p['cost'] else 0}元 | {'室内' if p['indoor'] else '室外'} | "
            f"{far}坐标:{p['lat']:.4f},{p['lng']:.4f} | 临近:{near}")

    return PLAN_PROMPT.format(city=city, days=days, pace=pace,
                              prefs=", ".join(prefs) if prefs else "（无特别偏好）",
                              pois="\n".join(lines),
                              hot=_hot_section(hot, sel),
                              wish=_wish_section(wish, sel),
                              base=_base_text(base),
                              base_req=_base_req(base))


def _valid_base(base) -> bool:
    """落脚点要么不设,要么就得能算距离 —— 只有名字没有坐标的话宁可按没设处理。"""
    return bool(base and base.get("lat") is not None and base.get("lng") is not None)


def _base_text(base) -> str:
    if not _valid_base(base):
        return "（未设定,默认每天从市中心机动出发）"
    nm = base.get("name") or "落脚点"
    return f"{nm}（坐标 {base['lat']:.4f},{base['lng']:.4f}）"


def _base_req(base) -> str:
    if not _valid_base(base):
        return ""
    return ("\n   - 每天 09:00 从落脚点出发、玩完回到落脚点，这两段通勤时间是实际存在的,\n"
            "     已计入当天总时长(预算约9小时),因此请据此控制当天的点数;\n"
            "     每天的头一个景点尽量离落脚点近一些或顺路,避免一早先横穿半座城;\n"
            "     最后一天要为返程留出余地(行李、赶车),不要排得太满太远;")


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


def _wish_section(wish: list, sel: list) -> str:
    """给模型看的收藏段落。只列已经在候选里出现的 —— 候选里没有的名字列出来
    只会勾着模型去硬编一个不存在的地方,那正好是这个方案一直在防的事。"""
    if not wish:
        return ""
    wset = {_norm_name(x) for x in wish}
    rows = [p["name"] for p in sel if _norm_name(p["name"]) in wset]
    if not rows:
        return ""
    return ("\n用户收藏(他自己攒的、一直想去的地方,请优先安排):\n"
            + "\n".join(f"- {n}" for n in rows) + "\n")


def ai_plan(city: str, pool: list, days: int, prefs: list, pace: str, hot: dict = None,
            base: dict = None, wish: list = None):
    """让大模型综合候选池给出:分组(哪天去哪几个)、顺序、每点经典特色与排布理由。

    返回 dict 或 None(网络/解析失败时返回 None,由调用方回退到规则版)。
    注意:模型只能从 pool 里挑,返回的景点名必须能在 pool 中匹配到,匹配不上的直接丢弃,
    因此绝不会把不存在的景点排进行程。"""
    try:
        raw = llm.ask_json(_build_plan_prompt(city, pool, days, prefs, pace, hot, base, wish),
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
               fresh: bool = False, emit=None, base: dict = None,
               wish: list = None) -> dict:
    """生成完整行程。emit(dict) 用于向前端推送真实进度事件,没传就完全静默。

    base 是落脚点 {name, lat, lng}:给了就每天从这里出发、回到这里,并且 AI 编排
    时会参考它到各点的距离。没有落脚点的行程默认从第一站开始,这在换了住处、
    通勤动辄四五十分钟的城市里,时间点是不准的。

    wish 是用户收藏的景点名列表(只对当前城市生效)。命中就加权、优先排入,
    但时间预算 / 跨区 / 单段上限这些约束照旧生效 —— 排不下的会连同原因一并返回
    (wish_miss),前端如实告诉用户,绝不假装排进去了。

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

    key = _plan_key(city, days, prefs, pace, base, wish)
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
        res = ai_plan(city, pool, days, prefs, pace, hot, base, wish)
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
        ranked = score(pool, prefs, wish)
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
    # 落脚点会让每天多出去程+回程两段,段数要算进去,否则进度会冲过 100%
    total_move = sum(max(0, len(g) - 1) + (2 if g and _valid_base(base) else 0)
                     for g in groups)
    done_move = 0
    _emit("route", "计算站间真实通勤时间", 0.73)

    def _make_move_cb(offset):
        """layout_timeline 的计数是每天内部从 0 开始,这里折算成全局进度。

        取 max 是防回退:fit_budget 可能删点重算,回算的那次段数更少,
        进度条如果跟着往回缩,用户看着会以为卡了。
        """
        def cb(done_in_day, _total_in_day):
            nonlocal done_move
            done_move = max(done_move, offset + done_in_day)
            if total_move:
                _emit("route", f"通勤计算 {done_move}/{total_move} 段",
                      0.73 + 0.22 * min(1.0, done_move / total_move))
        return cb

    dropped_ids = set()
    for i, g in enumerate(groups):
        # 用真实通勤校验当天预算,超时就摘点(见 fit_budget)
        d = fit_budget(g, city, base=base, pace=pace,
                       on_move=_make_move_cb(done_move))
        d["day"] = i + 1
        d["note"] = day_notes[i] if i < len(day_notes) else ""
        plan_days.append(d)
        for x in (d.get("dropped") or []):
            if x.get("id"):
                dropped_ids.add(x["id"])
    _emit("route", f"通勤时间已落地,共 {total_move} 段", 0.95)

    _emit("save", "整理每日安排与出行建议", 0.97)

    # 收藏兑现情况:排进去了要让前端标出来(那是用户自己攒的),排不进去必须给个说法。
    # 默默吞掉是最糟的情况 —— 用户是带着期待点的收藏,他会以为这个功能根本没生效。
    used_names = {_norm_name(p["name"]) for d in plan_days for p in (d.get("items") or [])}
    in_pool = {}
    for p in pool:
        in_pool.setdefault(_norm_name(p["name"]), p)
    wish_hit, wish_miss = [], []
    for w in (wish or []):
        n = _norm_name(w)
        if n in used_names:
            wish_hit.append(w)
            continue
        p = in_pool.get(n)
        if not p:
            wish_miss.append({"name": w, "reason": "这座城的候选里还没有它"})
            continue
        # 在候选池里、却没排进行程:要说清楚卡在哪一步 ——
        # 是离已排的动线太远,还是当天的时间预算被别的点占满了。
        near_any = any(
            amap.haversine(p["lat"], p["lng"], q["lat"], q["lng"]) <= NEIGHBOR_KM
            for d in plan_days for q in (d.get("items") or []))
        wish_miss.append({"name": w, "reason": (
            "离已排的动线太远,排进去要多绕路" if not near_any
            else "当天时间预算排满了,这次没排下")})

    plan = {
        "city": city,
        "pace": pace,
        "base": base if _valid_base(base) else None,   # 回写给前端,刷新页面后还在
        "days": plan_days,
        "summary": summary,
        "advice": make_advice(plan_days, city),
        "hot_notes": hot_notes,
        "wish_hit": wish_hit,
        "wish_miss": wish_miss,
        # 被 fit_budget 摘掉的点不算"已用",放回备选池,用户可以手动加回来
        "pool": [p for p in pool if p["id"] not in used or p["id"] in dropped_ids],
        "live": amap.LIVE,
        "model": MODEL_LIVE,
        "cached": False,
        "cached_ts": time.time(),
    }
    # 落盘:同样的输入下次直接返回,连大模型都不用叫
    store.write("plan", key, plan)
    _emit("done", "行程已就绪", 1.0)
    return plan


def replan(city: str, groups: list, base: dict = None) -> dict:
    """用户改完地点后重排并重算时间轴。groups 是 [[poi, poi, ...], ...]。

    尊重用户手动排好的顺序(不再用最近邻重排),只落地"事实":夜景置后、真实通勤时间。"""
    plan_days = []
    for i, g in enumerate(groups):
        d = layout_timeline(g, city, base=base)
        d["day"] = i + 1
        plan_days.append(d)
    used = {p["id"] for d in plan_days for p in d["items"]}
    return {
        "city": city, "pace": "", "days": plan_days,
        "base": base if _valid_base(base) else None,
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
落脚点(每天从这里出发、最后回到这里): {base}

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
6. 已设落脚点时：每天的头一个景点不要离落脚点太远（避免一早横穿半座城），
   且要考虑从落脚点出发与回到落脚点的通勤时间，别把当天排到太晚。

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
              pace: str = "适中", prefs: list = None, base: dict = None):
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
        msg=(message or "").strip() or "请在保持整体均衡的前提下,优化这份行程。",
        base=_base_text(base))
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


def chat_fallback(city: str, plan_days: list, message: str, base: dict = None) -> dict:
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
            groups[i] = compute_day(g, city, base)["items"]
        reply = "已按地理位置重排了每天的顺序(就近串着走)。想换景点或改主题,需要接上大模型。"
    plan_days = []
    for i, g in enumerate(groups):
        d = layout_timeline(g, city, base=base)
        d["day"] = i + 1
        plan_days.append(d)
    used = {p["id"] for d in plan_days for p in d["items"]}
    return {
        "reply": reply, "summary": "",
        "plan": {
            "city": city, "pace": "", "days": plan_days,
            "base": base if _valid_base(base) else None,
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
