"""行程规划后端。启动: uvicorn app:app --reload  →  http://localhost:8000"""
import os

# 必须先于 amap / planner 导入,那两个模块在导入时就读 Key
import env                                             # noqa: E402

env.load_env()

from typing import List, Optional                      # noqa: E402

from fastapi import FastAPI, HTTPException             # noqa: E402
from fastapi.middleware.cors import CORSMiddleware     # noqa: E402
from fastapi.staticfiles import StaticFiles            # noqa: E402
from fastapi.responses import FileResponse             # noqa: E402
from pydantic import BaseModel                         # noqa: E402

import amap                                            # noqa: E402
import planner                                         # noqa: E402
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


class ReplanReq(BaseModel):
    city: str
    days: List[List[dict]]      # [[poi, poi, ...], ...] 每天一组


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


@app.post("/api/plan")
def plan(req: PlanReq):
    days = max(1, min(req.days, 5))
    pace = req.pace if req.pace in planner.PER_DAY else "适中"
    pool = amap.build_pool(req.city)
    # 真实模式下拿不到候选池就是拿不到,不能返回空行程 —— 宁可报错让前端降级到示例数据
    if amap.probe() and not pool:
        raise HTTPException(503, f"高德未返回「{req.city}」的候选点,请检查城市名或 Key 额度")
    return planner.build_plan(req.city, days, req.prefs, pace, pool, fresh=req.fresh)


@app.post("/api/replan")
def replan(req: ReplanReq):
    """用户换/删/加之后重新排序并重算时间轴。"""
    return planner.replan(req.city, [g for g in req.days if g])


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
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
