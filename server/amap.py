"""
高德 API 封装。

有 AMAP_KEY 时走真实接口(地理编码 / POI 搜索 / 路径规划);
没有 Key 时自动降级为内置示例数据 + 直线距离估算,保证项目零配置也能跑通。
"""
import json
import logging
import math
import os
import time
import urllib.parse
import urllib.request

from data import (CAMPUS_SIGHTS, CITIES, DEFAULT_PROFILE, SEARCH_KEYWORDS,
                  TYPE_PROFILE, is_campus_sight, is_sight, is_sight_name)

KEY = os.getenv("AMAP_KEY", "").strip()
LIVE = bool(KEY)
BASE = "https://restapi.amap.com/v3"
TIMEOUT = 8

import store

# 进程内内存缓存。磁盘缓存挡的是"重启之后",内存缓存挡的是"同一次会话里"。
_POOL_CACHE = {}
_GEO_CACHE = {}
_ROUTE_CACHE = {}
_BOUNDARY_CACHE = {}
POOL_TTL = 600


# ---------- 基础 ----------

def _get(path: str, **params) -> dict:
    params["key"] = KEY
    params["output"] = "JSON"
    url = f"{BASE}{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "travel-planner/0.1"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def haversine(a_lat, a_lng, b_lat, b_lng) -> float:
    """两点直线距离,单位公里。"""
    r = 6371.0
    p = math.pi / 180
    dlat = (b_lat - a_lat) * p
    dlng = (b_lng - a_lng) * p
    h = (math.sin(dlat / 2) ** 2
         + math.cos(a_lat * p) * math.cos(b_lat * p) * math.sin(dlng / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


def estimate_route(a: dict, b: dict) -> tuple:
    """兜底估算:近的走路,远的按城市均速算。返回 (方式, 分钟)。"""
    d = haversine(a["lat"], a["lng"], b["lat"], b["lng"])
    if d < 1.2:
        return "步行", max(6, round(d / 4.6 * 60))
    return ("驾车" if d > 15 else "地铁"), round(10 + d / 21 * 60)


# ---------- 行程规划需要的"软"字段:最佳时段 + 特色 ----------
# 高德不返回这些,但规划时必须考虑(什么时候去、有什么特色),否则排出来的行程
# 和景点真实情况对不上。用类型 + 标签规则推断,稳定可解释,不依赖模型。

_BEST_TIME_LABEL = {
    "morning": "上午(9-11点)",
    "afternoon": "下午(14-17点)",
    "evening": "傍晚/夜间",
    "any": "全天皆宜",
}

# 类型 → 默认最佳时段
_TYPE_BEST = {
    "动物园": "morning", "植物园": "morning", "自然风光": "morning",
    "公园": "morning", "城市公园": "morning", "乡村休闲": "morning",
    "海滨浴场": "morning", "风景名胜区": "morning",
    "历史古迹": "morning", "文化遗址": "morning", "文物古迹": "morning",
    "风景名胜": "morning", "寺庙": "morning", "寺庙道观": "morning",
    "宗教建筑": "morning", "国家级景点": "morning", "旅游景点": "morning",
    "博物馆": "afternoon", "美术馆": "afternoon", "纪念馆": "afternoon",
    "艺术馆": "afternoon", "科技馆": "afternoon", "图书馆": "afternoon",
    "水族馆": "afternoon", "教堂": "afternoon",
    "主题乐园": "afternoon", "游乐园": "afternoon", "剧院": "afternoon",
    "度假村": "afternoon",
}

# 类型 → 特色一句话(模板)
_TYPE_HL = {
    "博物馆": "馆藏丰富,建议慢慢看并听讲解",
    "美术馆": "展品更新快,适合安静欣赏",
    "纪念馆": "历史厚重,值得细读",
    "艺术馆": "艺术氛围浓,出片好看",
    "科技馆": "互动展项多,亲子首选",
    "图书馆": "安静阅读,适合歇脚",
    "水族馆": "海底世界,室内不晒",
    "动物园": "看动物,上午最活跃",
    "植物园": "四季花木,散步放松",
    "自然风光": "自然景色好,适合拍照散步",
    "公园": "本地人休闲地,生活气息浓",
    "城市公园": "城市绿肺,适合遛弯",
    "乡村休闲": "近郊田园,节奏慢",
    "海滨浴场": "看海玩水,注意防晒",
    "历史古迹": "历史厚重,建议请讲解",
    "文化遗址": "遗址现场感强,配讲解最佳",
    "文物古迹": "古迹斑驳,适合慢慢看",
    "风景名胜": "经典打卡,人多建议早去",
    "宗教建筑": "清静庄严,注意着装",
    "寺庙": "香火旺,上午去更清净",
    "寺庙道观": "晨钟暮鼓,上午最静",
    "教堂": "建筑精美,室内安静",
    "历史文化街区": "老街巷弄,吃逛一体",
    "美食街区": "本地小吃集中,饿了就去",
    "特色商业街": "吃喝玩乐一站式",
    "商业街": "热闹商圈,适合夜游",
    "步行街": "逛街首选,灯光好看",
    "商圈": "购物餐饮集中,留足时间",
    "文创园区": "文艺小店多,适合拍照",
    "主题乐园": "项目多,建议一整天",
    "游乐园": "亲子刺激,体力活",
    "度假村": "休闲度假,节奏慢",
    "剧院": "演出为主,注意开场时间",
    "国家级景点": "必去地标,早去避人流",
    "旅游景点": "常规景点,按需安排",
}


def _enrich(poi: dict) -> dict:
    """补全规划需要的软字段:最佳时段 + 特色。规则推断,稳定可解释。"""
    typ = poi.get("type", "")
    tags = poi.get("tags", []) or []
    indoor = poi.get("indoor", 0)
    name = poi.get("name", "")

    # 最佳时段
    if any("夜景" in t for t in tags):
        bt = "evening"
    elif typ in _TYPE_BEST:
        bt = _TYPE_BEST[typ]
    elif indoor:
        bt = "afternoon"        # 室内景点下午去,避开正午暴晒
    else:
        bt = "any"
    poi["best_time"] = bt
    poi["best_time_label"] = _BEST_TIME_LABEL.get(bt, "全天皆宜")

    # 特色一句话:标签优先,其次类型模板
    hl = None
    if "夜景" in tags:
        hl = "夜景出片,建议傍晚去"
    elif "亲子" in tags:
        hl = "适合带娃,互动项目多"
    elif "拍照" in tags:
        hl = "出片率高,带好相机"
    elif "美食" in tags or "小吃" in tags:
        hl = "本地味道集中,饿了就去"
    elif "人文" in tags:
        hl = "人文气息浓,建议听讲解"
    elif "自然" in tags:
        hl = "自然风景好,适合散步"
    if not hl:
        hl = _TYPE_HL.get(typ, "值得一去")
    poi["highlight"] = hl
    return poi


# ---------- 地理编码 ----------

_PROBE_CACHE = {"t": 0.0, "ok": None}


def probe() -> bool:
    """
    探一次真实接口,确认 Key 真的可用。
    LIVE 只表示"填了 Key",填错或额度用尽时它依然是 True —— 徽标会骗人。
    结果缓存 60 秒,避免每次健康检查都发请求。
    """
    if not LIVE:
        return False
    import time as _t
    if _PROBE_CACHE["ok"] is not None and _t.time() - _PROBE_CACHE["t"] < 60:
        return _PROBE_CACHE["ok"]
    ok = geocode("成都") is not None
    _PROBE_CACHE.update(t=_t.time(), ok=ok)
    return ok


def geocode(city: str):
    """城市名 → (纬度, 经度)。失败返回 None。"""
    if not LIVE:
        c = CITIES.get(city)
        return c["center"] if c else None
    key = (city or "").strip()
    hit = _GEO_CACHE.get(key)
    if hit:
        return tuple(hit)
    got = store.read("geo", key)
    if got:
        _GEO_CACHE[key] = got
        return tuple(got)
    try:
        d = _get("/geocode/geo", address=city)
        loc = d["geocodes"][0]["location"]      # "lng,lat"
        lng, lat = (float(x) for x in loc.split(","))
    except Exception:
        return None
    val = [lat, lng]
    _GEO_CACHE[key] = val
    store.write("geo", key, val)
    return lat, lng


def search_place(city: str, kw: str, limit: int = 8) -> list:
    """在本市按关键词找地点 —— 落脚点专用。

    落脚点不是景点:酒店、民宿、小区、地铁站、机场都可能是,所以这里**不过 is_sight**
    (那条规则本来就是要滤掉住宿服务大类的)。返回值只带编排要用到的最小字段。
    """
    kw = (kw or "").strip()
    if not LIVE or not kw:
        return []
    try:
        d = _get("/place/text", keywords=kw, city=city, citylimit="true",
                 offset=min(max(limit, 1), 10), page=1, extensions="base")
    except Exception:
        return []
    if str(d.get("status")) != "1":
        return []
    out = []
    for raw in d.get("pois") or []:
        loc = raw.get("location") or ""
        if "," not in loc:
            continue
        try:
            lng, lat = (float(x) for x in loc.split(",")[:2])
        except ValueError:
            continue
        name = (raw.get("name") or "").strip()
        if not name:
            continue
        seg = [s for s in (raw.get("type") or "").replace("|", ";").split(";") if s]
        out.append({"name": name, "lat": lat, "lng": lng,
                    "addr": (raw.get("address") or "").strip(),
                    "type": seg[-1] if seg else ""})
        if len(out) >= limit:
            break
    return out


def rgeo(lng: float, lat: float) -> dict:
    """逆地理编码:在地图上点一个位置 → 变成能落脚的地址。

    名称的优先级按"人会不会这么称呼"排:最近的 POI(如某酒店) > 小区/楼宇 > 乡镇街道。
    """
    try:
        d = _get("/geocode/regeo", location=f"{lng},{lat}", extensions="base")
    except Exception:
        return None
    if str(d.get("status")) != "1":
        return None
    rg = d.get("regeocode") or {}
    comp = rg.get("addressComponent") or {}
    name, addr = "", (rg.get("formatted_address") or "").strip()
    pois = rg.get("pois") or []
    if isinstance(pois, list) and pois:
        name = (pois[0].get("name") or "").strip()
    if not name:
        for k in ("neighborhood", "building"):
            v = comp.get(k) or {}
            if isinstance(v, dict) and (v.get("name") or "").strip():
                name = v["name"].strip()
                break
    if not name:
        name = (comp.get("township") or comp.get("street") or "").strip()
    if not name:
        name = addr[:20] if addr else f"地图选点 {lat:.4f},{lng:.4f}"
    return {"name": name, "lat": lat, "lng": lng, "addr": addr, "type": "落脚点"}


# ---------- 候选池 ----------

def _read_cache(city: str):
    data = store.read("pois", city)
    return data if isinstance(data, list) and data else None


def _write_cache(city: str, pool: list):
    store.write("pois", city, pool)

def _mock_pool(city: str) -> list:
    c = CITIES.get(city)
    if not c:
        return []
    return [{
        "id": f"{city}_{i}",
        "name": p[0], "type": p[1], "rating": p[2], "cost": float(p[3]),
        "duration": p[4], "indoor": p[5],
        "lat": p[6], "lng": p[7], "tags": p[8].split(","),
    } for i, p in enumerate(c["pois"])]


def _normalize(raw: dict, idx: int) -> dict:
    """把高德返回的 POI 规整成统一结构,并补全高德不提供的字段。"""
    loc = raw.get("location", "")
    parts = loc.split(",")
    if len(parts) != 2:
        return None
    try:
        lng, lat = float(parts[0]), float(parts[1])
    except ValueError:
        return None

    biz = raw.get("biz_ext") or {}
    rating = 0.0
    for src in (raw.get("rating"), biz.get("rating")):
        try:
            rating = float(src)
            if rating > 0:
                break
        except (TypeError, ValueError):
            continue
    if rating <= 0:
        rating = 4.0          # 高德没评分时给个中性值,避免排序失真

    cost = 0.0
    try:
        cost = float(biz.get("cost") or 0)
    except (TypeError, ValueError):
        cost = 0.0

    # 高德的 type 是「大类;中类;小类」(例:"科教文化服务;博物馆;博物馆")。
    # 取最后一段才是细分类型 —— 取第一段会得到"风景名胜"这种只分了大类的值,
    # 于是 TYPE_PROFILE 全部落空、所有景点都是同一个兜底时长,类型筛选也没意义。
    segs = [s for s in (raw.get("type") or "").replace("|", ";").split(";") if s]
    typ = segs[-1] if segs else "景点"
    duration, indoor = TYPE_PROFILE.get(typ, DEFAULT_PROFILE)

    return {
        "id": raw.get("id") or f"amap_{idx}",
        "name": raw.get("name", "").strip(),
        "type": typ,
        "rating": rating,
        "cost": cost,
        "duration": duration,
        "indoor": indoor,
        "lat": lat,
        "lng": lng,
        "tags": [t for t in raw.get("type", "").replace("|", ";").split(";") if t][:2],
        # 所属城市。高德在 city 参数无法识别时不会报错,而是悄悄返回另一座城市的 POI,
        # 靠这个字段才能把"输入了不存在的城市"和"正常结果"区分开。
        "city": raw.get("cityname") or raw.get("pname") or "",
        # 区县级名。县级市必须看这个:搜「敦煌」时 cityname 返回的是上级「酒泉市」,
        # adname 才是「敦煌市」。只比对 cityname 会把敦煌/香格里拉/延吉整城误杀成 0 条。
        "district": raw.get("adname") or "",
    }


def _fetch_campus(city: str, seen: set, limit: int) -> list:
    """
    观光型校园定向补搜。

    为什么必须定向:通用关键词一条大学都搜不到(实测搜「景点」在厦门/武汉/北京/郑州
    学校类命中均为 0),而加「大学」「名校」「校园」关键词又是灾难 ——
    分别返回老年大学/院系/分校、新东方留学培训、幼儿园与校园卡服务部。
    所以只对 CAMPUS_SIGHTS 名单里的校园按名字精确搜一次。
    在打这个补丁之前,厦门大学、武汉大学这类真景点其实是**缺失**的,不只是没挡住噪音。
    """
    out = []
    for nm in CAMPUS_SIGHTS.get(city, ()):
        if len(out) >= limit:
            break
        time.sleep(0.35)
        try:
            d = _get("/place/text", keywords=nm, city=city, citylimit="true",
                     offset=5, page=1, extensions="all")
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("status") != "1":
            continue
        for raw in d.get("pois", []):
            item = _normalize(raw, len(out))
            if not item or not item["name"] or item["name"] in seen:
                continue
            if item["rating"] < 3.5:
                continue
            # 只认校园本体:「厦门大学(思明校区)」要,
            # 「厦门大学思明校区经济学院」「厦门大学附属中山医院」不要。
            if not is_campus_sight(item["name"], city):
                continue
            if city:
                blob = (item.get("city") or "") + (item.get("district") or "")
                if blob and city not in blob:
                    continue
            seen.add(item["name"])
            out.append(item)
            break          # 一个校园只取第一条匹配的,避免各校区刷屏
    return out


def build_pool(city: str, limit: int = 100, on_kw=None) -> list:
    """
    构建候选池:真实模式下按多个关键词分别搜索再合并去重。
    这一步是防幻觉的前提 —— 模型只能从这个池子里挑,不能凭空造景点。

    on_kw(done, total, found):每个关键词搜完回调一次,给前端喂真实进度。
    这一步有多个串行请求 + 限流退避,是整个规划里最不可预估耗时的一段,
    所以在UI上必须是有真实反馈的,不能让转圈假装在工作。
    """
    def _beat(done, total, found):
        if on_kw:
            try:
                on_kw(done, total, found)
            except Exception:      # 进度回调用来显示的,绝不能拖垮主流程
                pass

    if not LIVE:
        return [_enrich(p) for p in _mock_pool(city)]

    now = time.time()
    hit = _POOL_CACHE.get(city)
    # 第三项是这条缓存自己的 TTL:完整结果 10 分钟,残缺结果只有 1 分钟
    if hit and now - hit[0] < (hit[2] if len(hit) > 2 else POOL_TTL):
        _beat(len(SEARCH_KEYWORDS), len(SEARCH_KEYWORDS), len(hit[1]))
        return hit[1]

    disk = _read_cache(city)          # 重启之后仍然命中,省掉一整轮搜索
    if disk:
        # 过滤规则会随审计迭代(比如新增了 tags 大类判定),旧缓存里可能还留着
        # 手机店/KTV/民宿。读取时再过一遍,顺手把清理后的结果写回,缓存自我修复。
        cleaned = [p for p in disk
                   if is_sight(p.get("type", ""), p.get("tags"),
                               p.get("name", ""), city)
                   and is_sight_name(p.get("name", ""), p.get("type", ""))]
        if len(cleaned) != len(disk):
            logging.getLogger("amap").info(
                "旧缓存清洗: %s %d -> %d 条", city, len(disk), len(cleaned))
            _write_cache(city, cleaned)
        if cleaned:
            # 观光校园是后加的规则,旧缓存里没有。补一次再写回,不必整城重拉。
            extra = _fetch_campus(city, {p.get("name", "") for p in cleaned}, limit)
            if extra:
                cleaned += [_enrich(p) for p in extra]
                logging.getLogger("amap").info(
                    "观光校园补入: %s %d 条 —— %s", city, len(extra),
                    "、".join(p["name"] for p in extra))
                _write_cache(city, cleaned)
            enriched = [(_enrich(p) if "best_time" not in p else p) for p in cleaned]
            _POOL_CACHE[city] = (now, enriched, POOL_TTL)
            _beat(len(SEARCH_KEYWORDS), len(SEARCH_KEYWORDS), len(enriched))
            return enriched
        # 全部被新规则判为非景点 —— 缓存已废,往下走重新搜索

    seen, pool = set(), []
    ok_kw = 0                      # 成功拿到结果的关键词数
    for i, kw in enumerate(SEARCH_KEYWORDS):
        _beat(i, len(SEARCH_KEYWORDS), len(pool))
        if i:                      # 关键词之间拉开间隔,避免触发高德 QPS 限流
            time.sleep(0.35)
        d = None
        for attempt in range(3):   # CUQPS_HAS_EXCEEDED_THE_LIMIT 是限流,退避重试
            try:
                d = _get("/place/text", keywords=kw, city=city,
                        citylimit="true", offset=20, page=1, extensions="all")
            except Exception:
                d = None
            if isinstance(d, dict) and d.get("status") == "1":
                break
            info = (d or {}).get("info", "")
            if info == "CUQPS_HAS_EXCEEDED_THE_LIMIT":
                time.sleep(0.9 * (attempt + 1))   # 1 次不够就退久一点
                continue
            break
        if not isinstance(d, dict) or d.get("status") != "1":
            continue               # 这个关键词彻底失败,静默跳过(但不能假装它成功过)
        ok_kw += 1
        for raw in d.get("pois", []):
            item = _normalize(raw, len(pool))
            if not item or not item["name"] or item["name"] in seen:
                continue
            if item["rating"] < 3.5:
                continue
            # 写字楼、道路名、餐厅、商场、派出所这类不是景点,排进行程会闹笑话,直接剔掉。
            # 用 is_sight() 做关键词包含判断 —— 精确匹配挡不住「中餐厅」「火锅店」这些细分名。
            # 传 tags(高德类型串的前两段 = 大类/中类)进去:顶层大类比细分名可靠,
            # 「山西会馆」type 是博物馆、tags 却是餐饮服务,靠这个才认得出是饭店。
            if not is_sight(item["type"], item.get("tags"), item["name"], city):
                continue
            # 名字里带楼栋号的(「九街文创园2栋」「XX中心1期T3栋」)是写字楼,
            # 它们的 type 一律叫「产业园区」,只能靠名字认。
            if not is_sight_name(item["name"], item["type"]):
                continue
            # 城市名对不上就丢掉。citylimit=true 在 city 无法识别时是失效的 ——
            # 输入"不存在的城市",高德会静默返回北京的结果,行程就整个串了城。
            # 注意要同时比对 district(区县级名):敦煌/香格里拉/延吉都是县级市,
            # 高德的 cityname 返回的是上级地级名(酒泉市/迪庆州/延边州),只比对
            # cityname 会把这三座城的景点整城过滤成 0 条。
            if city:
                blob = (item.get("city") or "") + (item.get("district") or "")
                if blob and city not in blob:
                    continue
            seen.add(item["name"])
            pool.append(item)
        if len(pool) >= limit:
            break

    pool += _fetch_campus(city, seen, limit)

    # 残缺结果绝不能落盘。高德 QPS 限流时,7 个关键词常常只成功 1 个 ——
    # 成都就曾经只剩「文创园」这一个词的结果,14 条全是产业园区写字楼,
    # 一旦写进 7 天缓存,这座城市就被锁死成"只有写字楼"了。
    complete = ok_kw >= max(2, len(SEARCH_KEYWORDS) // 2)

    if pool:
        enriched = [ _enrich(p) for p in pool[:limit] ]
        # 若大模型之前给过这座城市的"经典特色",用它覆盖模板默认文案,
        # 这样地图标点、景点列表、手动改行程都能显示真正的特色而非通用话术。
        feat = store.read("feature", city)
        if isinstance(feat, dict):
            by_name = {p["name"]: p for p in enriched}
            for nm, hl in feat.items():
                if nm in by_name and hl:
                    by_name[nm]["highlight"] = hl
        if complete:
            _POOL_CACHE[city] = (now, enriched, POOL_TTL)
            _write_cache(city, enriched)
        else:
            # 残缺:只在内存里放 1 分钟,下次访问很快就重试;绝不写磁盘。
            logging.getLogger("amap").warning(
                "候选池不完整: %s 仅 %d/%d 个关键词成功(%d 条),不写缓存",
                city, ok_kw, len(SEARCH_KEYWORDS), len(enriched))
            _POOL_CACHE[city] = (now, enriched, 60)
        _beat(len(SEARCH_KEYWORDS), len(SEARCH_KEYWORDS), len(enriched))
        return enriched
    _beat(len(SEARCH_KEYWORDS), len(SEARCH_KEYWORDS), 0)
    return []


# ---------- 站间耗时 ----------

def _route_key(a, b, city: str) -> str:
    # 5 位小数约等于 1 米,足够区分两个景点,又能让"同一对点"稳定命中。
    # 结果里含"方式",所以算出方式判断改了必须换 key,否则会一直读到老数值。
    return f'v2|{city}|{a["lng"]:.5f},{a["lat"]:.5f}|{b["lng"]:.5f},{b["lat"]:.5f}'


def _drive_min(a: dict, b: dict) -> int:
    """打车过去要多久。景点之间直线不远但没有地铁方案时(或要绕山、跨江),
    这才是游客真实的走法 —— 也用来替掉"步行 38 分钟""地铁 77 分钟"这种虚高方案。"""
    d = _get("/direction/driving", origin=f'{a["lng"]},{a["lat"]}',
             destination=f'{b["lng"]},{b["lat"]}')
    return max(5, int(float(d["route"]["paths"][0]["duration"])) // 60)


# 宁可打车也不接受的步行距离/时长。超过这两个值说明"走过去"或"坐着公交绕"不现实。
WALK_MAX_MIN = 30
TRANSIT_MAX_WALK_M = 2000
TRANSIT_OK_RATIO = 1.6     # 公交超过打车时长的这么多倍,就判定为"绕路",改打车


def _parse_polyline(polyline: str):
    """
    把高德行政区划 polyline 解析成 GeoJSON Polygon 列表。
    高德格式:lng,lat;lng,lat|lng,lat;...   '|' 分隔多个环( Polygon ),同一个环里 '|' 也可能分隔子环。
    这里按最常用方式处理:用 '|' 先拆成 polygon,每个 polygon 再用 ';' 拆点。
    """
    if not polyline:
        return []
    polygons = []
    for part in polyline.split("|"):
        ring = []
        for pt in part.split(";"):
            pt = pt.strip()
            if not pt:
                continue
            try:
                lng, lat = (float(x) for x in pt.split(","))
                ring.append([lng, lat])
            except Exception:
                continue
        if len(ring) >= 3:
            ring.append(ring[0])  # 闭合
            polygons.append([ring])
    return polygons


def boundary(city: str):
    """
    获取城市行政边界 GeoJSON。失败返回 None。
    优先读磁盘/内存缓存,避免每次切城市都请求。
    """
    if not LIVE:
        return None
    key = (city or "").strip()
    if not key:
        return None

    hit = _BOUNDARY_CACHE.get(key)
    if hit:
        return hit

    got = store.read("boundary", key)
    if got:
        _BOUNDARY_CACHE[key] = got
        return got

    # 关键词歧义处理:高德 district 接口按 keywords 模糊匹配,
    # 比如 "西安" 会同时返回吉林的「西安区」和陕西的「西安市」,且区可能排在市前面。
    # 策略:先试原始名(已去「市」),再试带「市」;每轮遍历结果,优先取 市级(level=='city')。
    polyline = None
    chosen_name = None
    chosen_adcode = None
    for kw in (key, key + "市"):
        if polyline:
            break
        try:
            d = _get("/config/district", keywords=kw, subdistrict=0, extensions="all")
        except Exception:
            continue
        districts = d.get("districts") or []
        chosen = None
        for dist in districts:                       # 第一优先:市级行政区
            if dist.get("level") == "city":
                chosen = dist
                break
        if not chosen:                              # 退而求其次:名字以城市名开头
            for dist in districts:
                if (dist.get("name") or "").startswith(key):
                    chosen = dist
                    break
        if not chosen and districts:                # 最后兜底:取第一个
            chosen = districts[0]
        if chosen and chosen.get("polyline"):
            polyline = chosen["polyline"]
            chosen_name = chosen.get("name", city)
            chosen_adcode = chosen.get("adcode", "")
    if not polyline:
        return None
    polygons = _parse_polyline(polyline)
    if not polygons:
        return None
    geojson = {
        "type": "Feature",
        "properties": {"name": chosen_name, "adcode": chosen_adcode},
        "geometry": {"type": "MultiPolygon", "coordinates": polygons},
    }

    _BOUNDARY_CACHE[key] = geojson
    store.write("boundary", key, geojson)
    return geojson


def route_time(a: dict, b: dict, city: str = "") -> tuple:
    """
    站间真实耗时。这是绝不能让模型估算的东西 —— 它一定会编。

    真实接口失败时降级为直线估算,并在方式后不加标记(调用方无需感知)。
    结果按点对缓存:用户每改一次地点都要重算十几个点对,不缓存的话调用量涨得很快。
    """
    if not LIVE:
        return estimate_route(a, b)

    key = _route_key(a, b, city)
    hit = _ROUTE_CACHE.get(key)
    if hit:
        return tuple(hit)
    got = store.read("route", key)
    if got and isinstance(got, list) and len(got) == 2:
        _ROUTE_CACHE[key] = got
        return tuple(got)

    origin = f'{a["lng"]},{a["lat"]}'
    dest = f'{b["lng"]},{b["lat"]}'
    straight = haversine(a["lat"], a["lng"], b["lat"], b["lng"])

    try:
        if straight < 1.5:
            d = _get("/direction/walking", origin=origin, destination=dest)
            mins = int(float(d["route"]["paths"][0]["duration"])) // 60
            # 走路超过半小时基本等于"没打算走过去"(可能要绕山/跨江),改打车
            res = ["打车", _drive_min(a, b)] if mins > WALK_MAX_MIN else ["步行", max(5, mins)]
        else:
            drive = transit = None
            try:
                drive = _drive_min(a, b)
            except Exception:
                pass
            try:
                d = _get("/direction/transit/integrated", origin=origin, destination=dest,
                         city=city, cityd=city, strategy=0, nightflag=0)
                tr = d["route"]["transits"][0]
                mins = int(float(tr.get("duration") or 0)) // 60
                if mins > 0 and float(tr.get("walking_distance") or 0) <= TRANSIT_MAX_WALK_M:
                    transit = max(8, mins)
            except Exception:
                pass
            # 公交比打车慢一大截时,游客不会坐;差得不多时地铁更划算。
            # 只按这一条裁定,别再叠加别的规则 —— 通勤时间虚高会让每天白少排一个点。
            if transit and (drive is None or transit <= drive * TRANSIT_OK_RATIO):
                res = ["地铁", transit]
            elif drive is not None:
                res = ["打车", drive]
            else:
                raise ValueError("transit/driving 都没拿到")
    except Exception:
        return estimate_route(a, b)     # 失败不缓存,下一次还有机会拿到真值

    _ROUTE_CACHE[key] = res
    store.write("route", key, res)
    return tuple(res)
