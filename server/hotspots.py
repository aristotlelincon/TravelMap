"""
当下热点发现:联网检索这座城市"这阵子什么在火",再拿地图数据核实。

为什么需要这一步
----------------
候选池来自高德关键词搜索,它回答的是"这座城市有哪些景点";
但"最近在火什么"是另一回事 —— 限时展、新开的打卡地、因为某部剧重新
走红的老景点,这些既搜不到,也不在大模型的训练数据里。
实测:不开联网模型会坚定地认为"现在是 2024 年",对实时问题直接拒答。

原则:模型只能提名,入池由高德说了算
----------------------------------
联网拿到的名字一律回查真实数据:
  1. 先在本城候选池里按名字找 —— 找得到就原地打标
  2. 找不到再拿这个名字去高德精确搜 —— 搜到且是真景点就补进池子
  3. 两处都查无此地 —— 丢弃,只留一句文字提示,绝不排进行程
这和全项目"所有事实来自接口计算,模型只做推理表达"的分工一致,
联网环节再多幻觉也不会污染行程。

失败静默:联网慢(实测 5~15 秒)、可能超时、供应商可能不支持。
任何一步出错都返回空结果,主流程照旧跑,不让热点拖垮规划。
"""
import datetime
import logging
import time

import amap
import llm
import store
from data import is_sight, is_sight_name

LOG = logging.getLogger("hotspots")

HOT_TTL = 6 * 3600          # 限时信息时效性强,6 小时后重新联网
MAX_VERIFY = 8              # 最多核实 8 条,控制高德调用量与整体耗时
MAX_RETURN = 6              # 最终最多返回 6 个热点,避免挤占行程

HOT_PROMPT = """你要为一个旅行规划系统提供「{city}」的当下热点情报。

今天是 {date}。请**联网检索**近期真实在发生的信息 —— 社媒讨论(小红书、抖音等)、本地媒体报道、展演资讯、新开业的商业空间都可以作为来源。

我要的是这三类:
1. 限时展演/活动:这段时间正在举办的艺术展、快闪、灯光节、市集、演出,有明确起止日期的
2. 新晋打卡地:近一两年才开业、或才火起来的具体场所
3. 重新走红的老景点:因为影视剧取景、季节限定景观等原因又被热议的经典景点

硬性要求:
- 必须给出**能在地图上搜到的具体场所名称**,用它在地图/点评上的官方名。
  不要写"某某街区漫步""citywalk路线"这类没有确切坐标的活动形式。
- 不要收录纯粹的餐厅、酒店、酒吧、商场本身(商场里正在办的展览可以收)。
- 每条一句话说清"为什么现在值得去",必须带时间限定(如"展期至10月20日""每年10月下旬银杏最佳")。
- 最多 {n} 条。宁缺毋滥:给 3 条确凿的,也好过凑 10 条含糊的。
- 不确定、查不到的就不要写。

只输出 JSON,不要代码块标记、不要解释文字:
{{
  "hotspots": [
    {{"name": "场所官方名", "why": "为什么现在值得去,带时间限定", "kind": "限时展演|新晋打卡|季节限定|重新走红"}}
  ]
}}
"""


def _today() -> str:
    """北京时间今天。-agent 本机时区可能不是东八区,写死偏移才可靠。"""
    try:
        tz = datetime.timezone(datetime.timedelta(hours=8))
        return datetime.datetime.now(tz).strftime("%Y年%m月%d日")
    except Exception:
        return time.strftime("%Y年%m月%d日")


def _norm_name(s: str) -> str:
    return (s or "").replace(" ", "").replace("·", "").replace("•", "").lower()


def _search_map(query: str, city: str, idx: int):
    """拿一个名字去高德精确核实。返回规整后的 POI,查不到或非景点返回 None。"""
    # offset 给 5:相关度排第一的未必是真景点(搜「沙坡尾艺术西区」先返回一个停车场),
    # 多看两条能让真正的景点浮上来。反正每条都要过 is_sight,放宽不会放进噪音。
    try:
        d = amap._get("/place/text", keywords=query, city=city,
                      citylimit="true", offset=5, page=1, extensions="all")
    except Exception as e:
        LOG.debug("热点核实网络失败 %s: %s", query, e)
        return None
    if not isinstance(d, dict) or d.get("status") != "1":
        return None
    for raw in (d.get("pois") or []):
        item = amap._normalize(raw, idx)
        if not item or not item["name"]:
            continue
        # 核验是否是真景点(与候选池同一套标准)
        if not is_sight(item["type"], item.get("tags"), item["name"], city):
            continue
        if not is_sight_name(item["name"], item["type"]):
            continue
        # 城市错配一律不要:高德在搜不到时会悄悄返回别的城市的结果
        blob = (item.get("city") or "") + (item.get("district") or "")
        if blob and city not in blob:
            continue
        return item
    return None


def discover(city: str, pool: list, max_n: int = MAX_VERIFY) -> dict:
    """联网发现本城当下热点,并核实成真实 POI。

    返回 {"items": [已核实并打好标记的 POI], "notes": [(name, why)], "asof": ts}。
    items 里的 POI 已补字典段,可直接并入候选池;notes 是查无此地的热点,
    只作为"本地可能还有这些"的文字情报。失败返回空结果,不抛异常。
    """
    empty = {"items": [], "notes": [], "asof": time.time()}
    if not llm.MODEL_LIVE:
        return empty

    cached = store.read("hot", city)
    # 注意:落盘的是 {"spots": [...]} 纯情报。判空要查 spots —— 查 items 永远读不到,
    # 结果就是每次生成行程都重新联网一次,白白多花 6 秒。
    if isinstance(cached, dict) and cached.get("spots") is not None:
        # 核实用当下的池子重跑一次,很便宜(命中的话连高德都不用调)
        return _hydrate(cached, city, pool)

    try:
        raw = llm.ask_json(
            HOT_PROMPT.format(city=city, date=_today(), n=max_n),
            max_tokens=900, temperature=0.3, search=True, timeout=60)
    except Exception as e:
        LOG.warning("热点联网检索失败(%s),跳过热点: %s", city, e)
        return empty

    spots = (raw or {}).get("hotspots") if isinstance(raw, dict) else None
    if spots is None:
        LOG.warning("热点返回缺 hotspots 字段(%s): %s", city, str(raw)[:200])
        return empty
    if not isinstance(spots, list) or not spots:
        LOG.info("热点: %s 本次未检索到可用条目", city)
        return empty

    cleaned = []
    for s in spots[:max_n]:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or "").strip()
        why = str(s.get("why") or "").strip()
        kind = str(s.get("kind") or "").strip()
        if len(name) < 2 or len(name) > 40:
            continue
        cleaned.append({"name": name, "why": why, "kind": kind})
    if not cleaned:
        return empty

    # 情报本身先落盘 —— 不含坐标,下次换个池子也能复用
    store.write("hot", city, {"asof": time.time(), "spots": cleaned})
    return _hydrate({"spots": cleaned, "asof": time.time()}, city, pool)


def _hydrate(data: dict, city: str, pool: list) -> dict:
    """把纯情报(name/why/kind)核实成真实 POI。

    分两步查:先在已有候选池里找(零成本),找不到才调高德(要额度、
    也可能触发限流)。联网一次能拿到好几条,这里值得花钱核实。
    """
    spots = data.get("spots") or []
    if not spots:
        return {"items": [], "notes": [], "asof": data.get("asof", time.time())}

    index = {_norm_name(p["name"]): p for p in pool}
    items, notes, seen = [], [], set()
    idx = 0
    for s in spots:
        name, why, kind = s.get("name"), s.get("why"), s.get("kind")
        key = _norm_name(name)
        if not key or key in seen:
            continue
        poi = index.get(key)

        if poi is None:
            time.sleep(0.35)            # 串行 + 间隔,防高德限流
            poi = _search_map(name, city, idx)
            idx += 1
            if poi is not None:
                poi = amap._enrich(poi)

        if poi is None:
            notes.append((name, why))   # 地图上查无此地:只当情报,不进池
            continue

        seen.add(key)
        poi["hot"] = True
        poi["hot_why"] = why
        poi["hot_kind"] = kind
        items.append(poi)
        if len(items) >= MAX_RETURN:
            break

    return {"items": items, "notes": notes, "asof": data.get("asof", time.time())}
