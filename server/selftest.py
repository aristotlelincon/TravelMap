# 本地自检:不依赖 fastapi,只验证编排逻辑。可删。
import env

env.load_env()   # 不加载就永远只跑示例数据,填了 Key 也测不到真实链路

import json
import planner
import amap

print("LIVE:", amap.LIVE, "| MODEL:", planner.MODEL_LIVE)

for city, prefs, pace, days in [
    ("成都", ["历史人文", "拍照出片"], "适中", 3),
    ("杭州", ["自然风光"], "紧凑", 2),
    ("西安", [], "轻松", 3),
]:
    p = planner.build_plan(city, days, prefs, pace)
    print("\n=== %s %d天 %s ===" % (p["city"], len(p["days"]), p["pace"]))
    print("候选池剩余:", len(p["pool"]))
    for d in p["days"]:
        print("-- day %d, 结束 %s" % (d["day"], planner.hhmm(d["end"])))
        for n in d["timeline"]:
            if n["kind"] == "poi":
                print("   %s-%s %s (%s, %.1f)" % (
                    planner.hhmm(n["arrive"]), planner.hhmm(n["leave"]),
                    n["poi"]["name"], n["poi"]["type"], n["poi"]["rating"]))
            elif n["kind"] == "move":
                print("      -> %s %d min" % (n["mode"], n["min"]))
            else:
                print("      午餐 %s" % planner.hhmm(n["arrive"]))
    print("建议:")
    for a in p["advice"]:
        print("  *", a)

# 换一个之后重排
p = planner.build_plan("成都", 3, [], "适中")
g0 = p["days"][0]["items"]
g0[0] = p["pool"][0]
r = planner.replan("成都", [d["items"] for d in p["days"]])
print("\n=== 替换后重排 ===")
print("day1:", [x["name"] for x in r["days"][0]["items"]])
print("end:", planner.hhmm(r["days"][0]["end"]))
print("建议:", r["advice"])

# 确认 JSON 可序列化(接口返回用)
json.dumps(planner.build_plan("杭州", 2, ["亲子"], "适中"), ensure_ascii=False)
print("\nJSON_SERIALIZE_OK")
