import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# torchvisusion 默认把权重下载到 ~/.cache/torch，某些环境不可写。
# 统一改到项目内的 .cache，保证 embedding 用真模型而不是退化成 pHash。
_CACHE = os.path.join(_PROJECT, ".cache", "torch")
try:
    os.makedirs(_CACHE, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", _CACHE)
except Exception:  # 只读环境就算了，后续会自动降级
    pass

DEFAULT_CONFIG = os.path.join(_PROJECT, "config.json")


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=dict)
    path: str = ""

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        p = path or DEFAULT_CONFIG
        with open(p, "r", encoding="utf-8") as f:
            return cls(raw=json.load(f), path=p)

    def get(self, dotted: str, default=None):
        cur: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def abspath(self, dotted: str, default: str = "") -> str:
        v = self.get(dotted, default)
        return v.replace("/", os.sep) if v else v
