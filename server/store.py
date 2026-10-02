"""
本地磁盘缓存。

目的只有一个:少调接口。所有"查过一次就不会变"的东西都落盘,
重启服务、刷新页面都还能用,不用重新烧一遍额度。

缓存了五类:
    pois_<城市>     候选池           7 天   一次 7 次高德搜索
    geo_<城市>      城市中心点       30 天  几乎不会变
    route_<点对>    站间耗时         7 天   重排一次就要重算十几个点对
    plan_<签名>     完整行程         12 小时 含大模型建议,是最贵的一次调用
    boundary_<城市> 行政边界         30 天  高德 district 接口,几乎不会变
    hot_<城市>      当下热点情报     6 小时 限时展/新晋打卡地时效性强,过期就得重新联网

任何一步写盘失败都静默跳过 —— 缓存只是省钱,不是功能的一部分。
"""

import json
import os
import time
import urllib.parse

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

DAY = 24 * 3600
TTL = {
    "pois": 7 * DAY,
    "geo": 30 * DAY,
    "route": 7 * DAY,
    "plan": 12 * 3600,
    "boundary": 30 * DAY,
    "hot": 6 * 3600,
    # feature(经典特色)没单列:跟着城市走但会随模型更新,用默认的 1 天正合适
}


def _file(kind: str, key: str) -> str:
    # 城市名直接当文件名不保险(空格、斜杠、奇怪字符),统一转义
    safe = urllib.parse.quote(str(key), safe="")
    if len(safe) > 120:                 # 路由缓存的 key 很长,截断后补个短哈希
        import hashlib
        safe = safe[:80] + hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{kind}_{safe}.json")


def read(kind: str, key: str):
    """读到有效缓存返回内容,过期/损坏/不存在都返回 None。"""
    path = _file(kind, key)
    try:
        if time.time() - os.stat(path).st_mtime > TTL.get(kind, DAY):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write(kind: str, key: str, data) -> bool:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = _file(kind, key)
        tmp = path + ".tmp"             # 先写临时文件再改名,避免写一半被读走
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except (OSError, TypeError, ValueError):
        return False


def clear(kind: str = None) -> int:
    """清缓存。kind 为空则全清。返回删掉的文件数。"""
    if not os.path.isdir(CACHE_DIR):
        return 0
    n = 0
    for name in os.listdir(CACHE_DIR):
        if not name.endswith(".json"):
            continue
        if kind and not name.startswith(kind + "_"):
            continue
        try:
            os.remove(os.path.join(CACHE_DIR, name))
            n += 1
        except OSError:
            pass
    return n
