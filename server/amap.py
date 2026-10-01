"""
高德 API 封装。

有 AMAP_KEY 时走真实接口(地理编码 / POI 搜索 / 路径规划);
没有 Key 时自动降级为内置示例数据 + 直线距离估算,保证项目零配置也能跑通。
"""
import json
import math
import os
import time
import urllib.parse
import urllib.request

from data import (CITIES, DEFAULT_PROFILE, NON_SIGHT, SEARCH_KEYWORDS,
                  TYPE_PROFILE)

KEY = os.getenv("AMAP_KEY", "").strip()
LIVE = bool(KEY)
BASE = "https://restapi.amap.com/v3"
TIMEOUT = 8

import store

# 进程内内存缓存。磁盘缓存挡的是"重启之后",内存缓存挡的是"同一次会话里"。
_POOL_CACHE = {}
_GEO_CACHE = {}
_ROUTE_CACHE = {}
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
    }


def build_pool(city: str, limit: int = 100) -> list:
    """
    构建候选池:真实模式下按多个关键词分别搜索再合并去重。
    这一步是防幻觉的前提 —— 模型只能从这个池子里挑,不能凭空造景点。
    """
    if not LIVE:
        return _mock_pool(city)

    now = time.time()
    hit = _POOL_CACHE.get(city)
    if hit and now - hit[0] < POOL_TTL:
        return hit[1]

    disk = _read_cache(city)          # 重启之后仍然命中,省掉一整轮搜索
    if disk:
        _POOL_CACHE[city] = (now, disk)
        return disk

    seen, pool = set(), []
    for kw in SEARCH_KEYWORDS:
        try:
            d = _get("/place/text", keywords=kw, city=city,
                    citylimit="true", offset=20, page=1, extensions="all")
        except Exception:
            continue
        for raw in d.get("pois", []):
            item = _normalize(raw, len(pool))
            if not item or not item["name"] or item["name"] in seen:
                continue
            if item["rating"] < 3.5:
                continue
            # 写字楼、道路名这类不是景点,排进行程会闹笑话,直接剔掉
            if item["type"] in NON_SIGHT:
                continue
            # 城市名对不上就丢掉。citylimit=true 在 city 无法识别时是失效的 ——
            # 输入"不存在的城市",高德会静默返回北京的结果,行程就整个串了城。
            if city and item["city"] and city not in item["city"]:
                continue
            seen.add(item["name"])
            pool.append(item)
        if len(pool) >= limit:
            break
    # 空结果不缓存:否则一次网络抖动会把这个城市锁死
    if pool:
        _POOL_CACHE[city] = (now, pool[:limit])
        _write_cache(city, pool[:limit])
    return pool[:limit]


# ---------- 站间耗时 ----------

def _route_key(a, b, city: str) -> str:
    # 5 位小数约等于 1 米,足够区分两个景点,又能让"同一对点"稳定命中
    return f'{city}|{a["lng"]:.5f},{a["lat"]:.5f}|{b["lng"]:.5f},{b["lat"]:.5f}'


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
            dur = d["route"]["paths"][0]["duration"]
            res = ["步行", max(5, int(dur) // 60)]
        else:
            d = _get("/direction/transit/integrated", origin=origin, destination=dest,
                     city=city, cityd=city, strategy=0, nightflag=0)
            dur = d["route"]["transits"][0]["duration"]
            res = ["地铁", max(8, int(float(dur)) // 60)]
    except Exception:
        return estimate_route(a, b)     # 失败不缓存,下一次还有机会拿到真值

    _ROUTE_CACHE[key] = res
    store.write("route", key, res)
    return tuple(res)
