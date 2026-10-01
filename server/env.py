"""
最小 .env 加载器。无第三方依赖,必须在 amap / planner 之前导入 —— 那两个模块在导入时就读 Key。

只做三件事:忽略空行和注释、按第一个等号切分、剥掉值两侧的引号。
第三点很关键:带引号的 Key 发出去会直接返回 INVALID_USER_KEY,而且不会报错,
只会安静地退化成空候选池。
"""


def load_env(path: str = None, override: bool = False) -> None:
    import os

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if override or k not in os.environ:
                os.environ[k] = v
