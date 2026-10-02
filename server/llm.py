"""
大模型调用层。

单独抽出来是为了让 planner(行程编排)和 hotspots(联网热点发现)共用同一套配置,
同时保持依赖单向 —— 两者都依赖本模块,彼此互不依赖,不会形成 import 环。

只要求接口兼容 OpenAI 的 /chat/completions,换供应商改 .env 三个变量即可。

关于联网搜索 enable_search(踩过的坑,别改回 extra_body):
  实测阿里云百炼只认**顶层** enable_search;
  塞进 extra_body 会被静默忽略 —— 不报错,但回答退回训练数据,
  表现为模型坚信"现在是 2024 年"并拒绝回答实时问题,极难察觉。
  别家供应商若不认这个顶层参数会报 400,届时自动降级为不联网重试一次。
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

# DeepSeek 的旧变量名继续兼容,写了也算数。
LLM_KEY = (os.getenv("LLM_KEY") or os.getenv("DEEPSEEK_KEY") or "").strip()
MODEL_LIVE = bool(LLM_KEY)
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-plus")
# 名字里带这些的会默认先输出一大段推理再回答,生成场景要显式关掉,否则又慢又容易跑偏
THINKING_MODELS = ("qwen3", "qvq", "r1", "reasoning")


def _payload(prompt: str, max_tokens: int, temperature: float,
             search: bool = False) -> dict:
    p = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    low = MODEL_NAME.lower()
    if any(k in low for k in THINKING_MODELS):
        p["enable_thinking"] = False
    if search:
        p["enable_search"] = True
    return p


def _post(payload: dict, timeout: int, retries: int = 1) -> dict:
    """网络层。偶发的 RemoteDisconnected / URLError 重试一次就好 ——
    短时间内连续调用百炼会时不时直接断连,不重试就等于白白丢一次生成。
    HTTP 错误不在这里重试:那是请求本身的问题,得交给上层按状态码判断。"""
    last = None
    for i in range(retries + 1):
        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                MODEL_URL, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {LLM_KEY}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError:
            raise
        except Exception as e:
            last = e
            if i < retries:
                time.sleep(1.2)
    raise last


def chat(prompt: str, max_tokens: int = 1800, temperature: float = 0.5,
         search: bool = False, timeout: int = 60) -> str:
    """拿模型的一段纯文本回答。search=True 时开联网搜索。"""
    if not LLM_KEY:
        raise RuntimeError("未配置 LLM_KEY")
    payload = _payload(prompt, max_tokens, temperature, search)
    try:
        d = _post(payload, timeout)
    except urllib.error.HTTPError as e:
        # 供应商不认 enable_search:降级重试一次。拿不到实时信息也比整个失败好。
        if not (search and e.code in (400, 422)):
            raise
        payload.pop("enable_search", None)
        d = _post(payload, timeout)
    return (d["choices"][0]["message"]["content"] or "").strip()


def _strip(text: str) -> str:
    """模型偶尔会给 JSON 包一层 ```json 代码块。"""
    if text.startswith("```"):
        text = text.strip("`").replace("json", "", 1).strip()
    return text


# 模型时不时会用单引号当 JSON 的字符串分隔符:
#   "陕西历史博物馆": '被誉为"古都明珠"的博物馆',    ← 这不是合法 JSON
# 实测约四分之一的概率。直接解析失败会白白丢掉这次生成,先修一轮再放弃。
# 只替换处在 JSON 结构位置(: { [ , 之后,到 , } ] 之前)的单引号串,
# 句子内部的引号(如引用某个人的话)不动,避免把正文改坏。
_QUOTED = re.compile(r"(?<=[:{\[,])\s*[‘'’]\s*([^'’]*?)\s*[‘'’]\s*(?=[,}\]])")


def _loads(text: str) -> dict:
    try:
        return json.loads(text)
    except ValueError:
        fixed = _QUOTED.sub(r'"\1"', text)
        if fixed != text:
            try:
                return json.loads(fixed)
            except ValueError:
                pass
        raise


def ask_json(prompt: str, max_tokens: int = 1800, temperature: float = 0.5,
             search: bool = False, timeout: int = 60) -> dict:
    """要求模型返回 JSON。

    三级容错:直接解析 → 修掉误用的单引号再解析 → 带着报错原因重问一次。
    全都解析不出来才抛异常(上层会退回规则版规划)。
    """
    err = None
    try:
        return _loads(_strip(chat(prompt, max_tokens, temperature, search, timeout)))
    except ValueError as ex:
        # Python 3 出了 except 块就会把这个名字删掉,必须先存下去
        err = ex

    retry = (prompt + "\n\n注意:你上一次的输出无法解析成 JSON(原因:%s)。\n"
                      "请重新输出,键名和字符串值一律用英文双引号包裹,不要用单引号,\n"
                      "值内部如果出现引号请转义,末尾不要有多余逗号。" % err)
    return _loads(_strip(chat(retry, max_tokens, max(temperature - 0.2, 0.1),
                              search, timeout)))


def ask_list(prompt: str, max_tokens: int = 400, limit: int = 3) -> list:
    """要求模型返回一个字符串数组。"""
    items = json.loads(_strip(chat(prompt, max_tokens, 0.7, False, 25)))
    return [str(x) for x in items if isinstance(x, str)][:limit]
