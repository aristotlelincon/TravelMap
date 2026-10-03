"""行程规划后端。启动: uvicorn app:app --reload  →  http://localhost:8000"""
import json
import os
import queue
import threading

# 必须先于 amap / planner 导入,那两个模块在导入时就读 Key
import env                                             # noqa: E402

env.load_env()

from typing import List, Optional                      # noqa: E402

from fastapi import FastAPI, HTTPException             # noqa: E402
from fastapi.middleware.cors import CORSMiddleware     # noqa: E402
from fastapi.staticfiles import StaticFiles            # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel                         # noqa: E402

import amap                                            # noqa: E402
import planner                                         # noqa: E402
import store                                           # noqa: E402
from data import CITY_NAMES                            # noqa: E402

app = FastAPI(title="行程规划后端")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
VENDOR_DIR = os.path.join(WEB_DIR, "vendor")           # 本地托管的第三方前端库(MapLibre 等)


class PlanReq(BaseModel):
    city: str
    days: int = 3
    prefs: List[str] = []
    pace: str = "适中"
    fresh: bool = False        # 「重新生成」时传 true,跳过缓存重新叫一次大模型
    base: Optional[dict] = None  # 落脚点 {name, lat, lng},没有就是每天凭空开始


class ReplanReq(BaseModel):
    city: str
    days: List[List[dict]]      # [[poi, poi, ...], ...] 每天一组
    base: Optional[dict] = None


class ChatReq(BaseModel):
    city: str
    days: List[dict]            # 当前行程(前端传的 plan.days,含 items/timeline)
    message: str = ""           # 用户提的意见
    pace: str = "适中"
    prefs: List[str] = []
    base: Optional[dict] = None


@app.get("/api/health")
def health():
    return {
        "ok": True,
        # probe() 而不是 LIVE:LIVE 只说明填了 Key,填错时它也是 True,徽标会骗人
        "live": amap.probe(),
        "model": planner.MODEL_LIVE,  # 是否接了大模型
        "model_name": planner.MODEL_NAME if planner.MODEL_LIVE else "",
    }


@app.get("/api/config")
def config():
    """
    给前端的非敏感配置。
      tmap_key  : 自己 uvicorn 起服务时,腾讯地图要求显式带 key 才用得到(预览走密钥代理不用)。
      tianditu_tk: 天地图卫星/地形底图 Key(合规白名单内),免费申请。留空则用 Esri 卫星兜底。
    """
    return {
        "tmap_key": os.getenv("TMAP_KEY", "").strip(),
        "tianditu_tk": os.getenv("TIANDITU_TK", "").strip(),
    }


@app.get("/api/cities")
def cities():
    return {"cities": CITY_NAMES, "live": amap.LIVE}


@app.get("/api/pois")
def pois(city: str):
    """候选池。真实模式下来自高德 POI 搜索,这也是模型挑选的唯一范围。"""
    return {"city": city, "pois": amap.build_pool(city), "live": amap.LIVE}


@app.get("/api/places")
def places(city: str, kw: str, limit: int = 8):
    """落脚点候选搜索。落脚点不是景点,所以走的是另一条查询(不过 is_sight)。

    搜不到返回空列表而不是报错:城市名没错只是这里没结果,让用户再换个词搜就行。
    """
    if not amap.LIVE:
        return {"places": [], "live": False}
    kw = (kw or "").strip()
    if len(kw) < 1:
        return {"places": [], "live": True}
    return {"places": amap.search_place(city, kw, max(1, min(limit, 10))), "live": True}


@app.get("/api/rgeo")
def rgeo(loc: str):
    """逆地理编码:地图点选落脚点后把坐标变成地址名。loc 传 "lng,lat"。"""
    try:
        lng, lat = (float(x) for x in (loc or "").split(",")[:2])
    except ValueError:
        raise HTTPException(400, "loc 需要形如 116.39,39.90")
    got = amap.rgeo(lng, lat)
    if not got:
        raise HTTPException(404, "这个位置没能解析出地址")
    return got


@app.post("/api/plan")
def plan(req: PlanReq):
    days = max(1, min(req.days, 5))
    pace = req.pace if req.pace in planner.PER_DAY else "适中"
    pool = amap.build_pool(req.city)
    # 真实模式下拿不到候选池就是拿不到,不能返回空行程 —— 宁可报错让前端降级到示例数据
    if amap.probe() and not pool:
        raise HTTPException(503, f"高德未返回「{req.city}」的候选点,请检查城市名或 Key 额度")
    return planner.build_plan(req.city, days, req.prefs, pace, pool, fresh=req.fresh,
                              base=req.base)


@app.post("/api/plan_stream")
def plan_stream(req: PlanReq):
    """流式生成行程:每推进一个阶段就吐一行 JSON(NDJSON)。

    为什么是 NDJSON + fetch 而不是 SSE:EventSource 只支持 GET,而这个请求要带
    days/prefs/pace/fresh,写在 query 里既丑又容易被代理缓存。

    为什么非流式不可:一次真实生成要经过 候选池多关键词串行搜索(还可能触发高德
    限流退避)→ 联网检索热点 → 大模型编排(可能重试)→ 逐段路径规划,实测 20~40 秒。
    在这期间如果前端只是干转圈,用户无法判断是卡了还是慢 —— 所以进度必须是真的。

    worker 放在子线程:build_plan 全程是同步阻塞调用,直接在生成器里跑会把这两件事
    串成"全部做完才吐第一行"。Queue 是这里唯一需要的线程同步。
    """
    days = max(1, min(req.days, 5))
    pace = req.pace if req.pace in planner.PER_DAY else "适中"
    q: queue.Queue = queue.Queue()

    def emit(ev):
        ev.setdefault("type", "stage")
        q.put(ev)

    def work():
        try:
            emit({"stage": "pool", "detail": "读取城市景点数据", "pct": 0.02})
            pool = amap.build_pool(
                req.city,
                on_kw=lambda done, total, found: emit({
                    "stage": "pool",
                    "detail": (f"检索景点 {done}/{total}"
                               + (f" · 已收录 {found} 个" if found else "")),
                    "pct": 0.02 + 0.36 * (done / max(1, total)),
                }))
            emit({"stage": "pool",
                  "detail": f"候选池就绪,共 {len(pool)} 个可排景点", "pct": 0.38})
            if amap.probe() and not pool:
                raise HTTPException(
                    503, f"高德未返回「{req.city}」的候选点,请检查城市名或 Key 额度")
            plan = planner.build_plan(req.city, days, req.prefs, pace, pool,
                                      fresh=req.fresh, emit=emit, base=req.base)
            emit({"type": "done", "plan": plan, "pct": 1.0})
        except Exception as e:                          # 进度流里报错也要告诉前端
            emit({"type": "error", "msg": str(e) or type(e).__name__})
        finally:
            q.put(None)                                 # 哨兵:生成器据此结束

    threading.Thread(target=work, daemon=True).start()

    def gen():
        while True:
            ev = q.get()
            if ev is None:
                break
            try:
                yield json.dumps(ev, ensure_ascii=False, default=str) + "\n"
            except (TypeError, ValueError):
                continue

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/replan")
def replan(req: ReplanReq):
    """用户换/删/加之后重新排序并重算时间轴。"""
    return planner.replan(req.city, [g for g in req.days if g], base=req.base)


@app.post("/api/chat")
def chat(req: ChatReq):
    """对话式调整行程:用户提意见,大模型在真实候选池内重排,回一段说明 + 新行程。

    没配模型 Key(或调用失败)时走 chat_fallback 的规则兜底,不会让用户看到一个死掉的输入框。
    """
    days = [d for d in req.days if d.get("items")]
    if not days:
        raise HTTPException(400, "还没有行程可以调整")

    pool = amap.build_pool(req.city)
    res = planner.chat_plan(req.city, days, req.message, pool,
                            pace=req.pace, prefs=req.prefs, base=req.base)
    if res:
        plan_days = []
        for i, g in enumerate(res["groups"]):
            # 落脚点要一直带着:改行程不该把"住哪儿"改没了
            d = planner.layout_timeline(g, req.city, base=req.base)
            d["day"] = i + 1
            d["note"] = res["day_notes"][i] if i < len(res["day_notes"]) else ""
            plan_days.append(d)
        if res.get("classic"):
            store.write("feature", req.city, res["classic"])
        used = {p["id"] for d in plan_days for p in d["items"]}
        return {
            "reply": res["reply"],
            "summary": res["summary"],
            "by_model": True,
            "plan": {
                "city": req.city, "pace": req.pace, "days": plan_days,
                "base": req.base,
                "advice": planner.make_advice(plan_days, req.city),
                "pool": [p for p in pool if p["id"] not in used],
                "live": amap.LIVE, "model": True,
            },
        }

    fb = planner.chat_fallback(req.city, days, req.message, req.base)
    return {"reply": fb["reply"], "summary": "", "by_model": False, "plan": fb["plan"]}


@app.get("/api/boundary")
def city_boundary(city: str):
    """城市行政边界 GeoJSON,用于前端把地图裁剪成城市形状。"""
    geo = amap.boundary(city)
    if not geo:
        raise HTTPException(503, f"无法获取「{city}」的行政边界,请检查城市名或 Key 额度")
    return {"city": city, "boundary": geo}


# 本地第三方前端库(如 MapLibre GL JS,用于真三维地形)。先于此挂载的静态根,优先匹配
@app.get("/vendor/{file:path}")
def vendor_file(file: str):
    import mimetypes
    p = os.path.join(VENDOR_DIR, file)
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(p, media_type=mimetypes.guess_type(p)[0] or "application/octet-stream")


# 前端静态托管:启动后直接访问 http://localhost:8000 即可
if os.path.isdir(WEB_DIR):
    # 首页禁用缓存:改前端后浏览器总是拿到新版本。
    # 否则改了 index.html 页面仍是旧版(出现过"截图里还是有半透明色块,
    # 但代码里早就换成细线了"的情况),非常难排查。
    @app.get("/")
    def index():
        return FileResponse(
            os.path.join(WEB_DIR, "index.html"),
            media_type="text/html",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                     "Pragma": "no-cache"},
        )

    @app.get("/index.html")
    def index_html():
        return index()

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
