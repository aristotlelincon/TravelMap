"""
想去的地方(心愿清单)。

为什么不用 store.py
------------------
store 是**缓存**:有 TTL,过期即废,`store.clear()` 还会整个清掉。
心愿清单是用户自己攒的数据 —— 不能过期,更不能因为哪天清了一次路线缓存
就连同"我攒了半年的想去清单"一起没了。所以单独存一份,放 .data/ 而不是
.cache/,也不进 TTL 表。

条目有两种:
    city  想去的城市(还没决定怎么玩,先记下来)
    poi   某个具体景点(已经有了行程、或在全城列表里看到的)

同一个 (kind, city, name) 只留一条 —— 重复点是收藏功能最常见的误操作,
与其让用户事后清理,不如加的时候就合并(备注取新的,时间取最早的)。
"""

import json
import os
import threading
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
FILE = os.path.join(DATA_DIR, "wish.json")

_lock = threading.Lock()          # 前端可能连着点几下,写文件不能交错


def _read() -> list:
    try:
        with open(FILE, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("items"), list):
            return d["items"]
    except (OSError, ValueError):
        pass
    return []


def _write(items: list) -> bool:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = FILE + ".tmp"        # 先写临时文件再改名,避免写一半断电留下坏文件
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"items": items, "updated": time.time()}, f,
                      ensure_ascii=False)
        os.replace(tmp, FILE)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _mk_id(kind: str, name: str, city: str) -> str:
    return "%s|%s|%s" % (kind, (city or "").strip(), (name or "").strip())


def _clean(item: dict) -> dict:
    """只留认识的字段,顺手规整类型。脏数据进来会污染整个清单。"""
    kind = (item.get("kind") or "city").strip()
    if kind not in ("city", "poi"):
        kind = "city"
    name = str(item.get("name") or "").strip()
    city = str(item.get("city") or "").strip() or name
    try:
        lat = float(item.get("lat") or 0)
        lng = float(item.get("lng") or 0)
    except (TypeError, ValueError):
        lat = lng = 0.0
    return {
        "id": _mk_id(kind, name, city),
        "kind": kind,
        "name": name,
        "city": city,
        "lat": lat,
        "lng": lng,
        "note": str(item.get("note") or "").strip()[:200],
        # 详细地址。从高德自动补全选中的地点会带上真实门牌地址 —— 名字常常重复
        # ("万达广场"全国几十个),光看名字分不清是哪个,存下地址才认得出来。
        "addr": str(item.get("addr") or "").strip()[:200],
        "done": bool(item.get("done")),
        "added": int(item.get("added") or time.time()),
        # 标记去过的时刻。首页「最近去过」按它排序,只靠 added 会排成收藏顺序。
        "done_at": int(item.get("done_at") or 0),
    }


def list_all() -> list:
    """未去的排前面、新的排前面;已去过的沉到底部但不删除。"""
    items = [x for x in _read() if isinstance(x, dict) and x.get("name")]
    items.sort(key=lambda x: (bool(x.get("done")), -(x.get("added") or 0)))
    return items


def add(item: dict) -> list:
    """新增或合并。返回合并后的完整清单。"""
    it = _clean(item)
    if not it["name"]:
        return list_all()
    with _lock:
        items = _read()
        for i, old in enumerate(items):
            if old.get("id") == it["id"]:
                # 已存在:保留最早的收藏时间,其余用新的(备注可能被改过)
                it["added"] = int(old.get("added") or it["added"])
                items[i] = it
                break
        else:
            items.append(it)
        _write(items)
    return list_all()


def remove(wid: str) -> list:
    with _lock:
        items = [x for x in _read() if x.get("id") != wid]
        _write(items)
    return list_all()


def mark(wid: str, done: bool) -> list:
    """标记去过 / 撤回去过。找不到就算了,不报错。"""
    with _lock:
        items = _read()
        for x in items:
            if x.get("id") == wid:
                x["done"] = bool(done)
                x["done_at"] = int(time.time()) if done else 0
                break
        _write(items)
    return list_all()


def has(kind: str, name: str, city: str) -> bool:
    wid = _mk_id(kind, name, city)
    return any(x.get("id") == wid for x in _read())
