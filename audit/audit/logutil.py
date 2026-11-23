"""统一日志：链路各环节实时输出进度，不再等全部跑完才看到内容。

用法约定（改这块代码前先看）：

1. 库代码（audit/*.py）只用 `get_logger(__name__)` 打日志，**不配置 handler**；
   handler 只由 scripts/ 下的入口调一次 `setup_logging()`。
   这样 path_check.py 这类不 setup 的脚本自动安静（走 logging.lastResort，只留 WARNING 以上）。
2. 输出走 **stdout 而不是 stderr**：PowerShell 会把 stderr 渲染成红色错误文本，
   正常进度看起来像报错，反而更难排查。
3. 长耗时环节（YOLO、VLM）必须在**开始之前**就打一条，否则单图 20~80 秒期间终端一片空白。
4. 级别口径：
   - INFO  = 阶段边界 + 该阶段决定了什么（命中/推导/封顶）
   - DEBUG = 原始数值与中间量（HTTP 元数据、框坐标统计、模型原始返回）
   - WARNING = 降级、兜底、可疑（degraded 项、pHash 退化、类别过滤后为 0）
   - ERROR = 该图必定转人工的失败（VLM 空响应、解析失败、HTTP 非 200）
"""
from __future__ import annotations

import logging
import sys
import time
from typing import Optional

ROOT_NAME = "audit"
_T0 = [time.time()]

_FMT = "[+%(dt)s] %(levelname)-7s %(src)-9s %(message)s"
_DATEFMT = "%H:%M:%S"


class _Formatter(logging.Formatter):
    """带"进程启动以来秒数"的格式，便于把日志和耗时对上。"""

    def format(self, record: logging.LogRecord) -> str:
        record.dt = f"{record.created - _T0[0]:6.1f}s"
        record.src = record.name.replace(f"{ROOT_NAME}.", "").replace(ROOT_NAME, "audit")
        return super().format(record)


def get_logger(name: str = "") -> logging.Logger:
    """get_logger(__name__) 或 get_logger("vlm")；一律挂在 audit 命名空间下。"""
    if not name:
        return logging.getLogger(ROOT_NAME)
    short = name.split(".")[-1]
    return logging.getLogger(f"{ROOT_NAME}.{short}")


def setup_logging(level: str = "info", stream: Optional[object] = None) -> logging.Logger:
    """入口脚本调用。重复调用是幂等的（不会叠加多个 handler）。"""
    lg = logging.getLogger(ROOT_NAME)
    lg.handlers.clear()

    if str(level).lower() in ("quiet", "none", "off"):
        lg.setLevel(logging.CRITICAL + 1)
        lg.addHandler(logging.NullHandler())
        lg.propagate = False
        return lg

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(_Formatter(_FMT, datefmt=_DATEFMT))
    lg.addHandler(handler)
    lg.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    lg.propagate = False
    _T0[0] = time.time()
    return lg
