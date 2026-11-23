"""临时诊断：qwen3-vl 空响应根因。A=think:False（现状） B=允许思考+2500 tokens。"""
import base64
import json
import os
import sys
import time
from io import BytesIO

import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from audit.vlm import PROMPT, _b64  # noqa: E402

IMG = r"d:\Users\Desktop\巡店AI\data\images\20251004\southwest\00947_cur.jpg"

PROMPT_V2 = """你是零售巡店的图像审核员。先观察后结论，看不到的填 unknown，禁止猜测。
注意：摆拍陈列往往恰恰表现为“单排、等距、正对镜头”，整齐不是合格依据。
元气森林产品：气泡水、外星人电解质水、维C水、维B水、冰茶、好望水。

只输出 JSON，字段按序：
1. around：一句话，陈列面周围是什么非元气商品（零食/文具/洗涤/烟酒/母婴/餐具/无…）
2. behind：一句话，前排元气瓶正后方是什么（仍是元气产品/其他品牌产品/空层板或纸箱/被挡看不清）
3. empty_obs：一句话，陈列面有无无液体的透明空瓶、约几支
4. scene：枚举 drink_shelf(饮料货架内)/drink_freezer(冰柜柜体内部,含他司;冰柜顶/盖上摆瓶不算)/case_display(割箱堆箱)/endcap/snack_shelf/other_shelf(文具/洗涤/烟酒/母婴等非饮料货架)/counter/table(餐区或冰柜顶桌面)/outdoor/storage/screen/other/unknown
5. is_screen：拍屏幕/翻拍=true（UI、状态栏、机身边框、摩尔纹、拼接缝）
6. empty_bottle_count：整数，依据 empty_obs；没有=0，看不清=-1
7. depth：依据 behind：后方有元气=yes；后方空或仅他牌=no；看不清=unknown
8. single_row：元气只摆一排等距正对镜头=true
9. confidence：0到1
10. reason：一句中文总结"""


def call(mode: str, img: str = IMG) -> None:
    im = Image.open(img).convert("RGB")
    prompt = PROMPT
    if mode in ("C", "D"):
        prompt = "/no_think\n" + PROMPT
    if mode == "E":
        prompt = PROMPT + "\n/no_think"
    if mode == "H":
        prompt = PROMPT_V2
    if mode == "F":
        # 纯文本对照：验证 think 参数在本 build 是否生效
        payload = {"model": "qwen3-vl:2b",
                   "messages": [{"role": "user", "content": "1+1等于几？只回答数字。"}],
                   "stream": False, "think": False,
                   "options": {"num_predict": 100}}
        t = time.time()
        d = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=120).json()
        msg = d.get("message") or {}
        print(f"[F] {time.time() - t:.0f}s content={msg.get('content', '')!r} "
              f"thinking_len={len(msg.get('thinking', '') or '')}")
        return
    if mode == "D":
        prompt = ("/no_think\n你是巡店照片审核员。只输出 JSON，不要解释。"
                  "字段：scene(选一个:drink_shelf/drink_freezer/case_display/endcap/snack_shelf/"
                  "other_shelf/counter/table/outdoor/storage/screen/other/unknown)；"
                  "is_screen(拍屏幕/翻拍=true)；empty_bottle_count(无液体透明空瓶支数,没有0,看不清-1)；"
                  "depth(前排元气瓶后方有元气产品=yes,后方空或仅他牌=no,被挡=unknown)；"
                  "single_row(单排等距正对镜头=true)；confidence(0-1)；reason(一句中文)。"
                  "元气森林产品:气泡水/外星人电解质水/维C水/维B水/冰茶/好望水。")
    payload = {
        "model": "qwen3-vl:2b-instruct" if mode in ("G", "H") else "qwen3-vl:2b",
        "messages": [{"role": "user", "content": prompt, "images": [_b64(im, 1024 if mode == "H" else 768)]}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": 1500 if mode in ("A", "C", "G", "H") else 2500},
    }
    if mode in ("A", "C", "D", "E"):
        payload["think"] = False
    t = time.time()
    r = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=600)
    d = r.json()
    msg = d.get("message") or {}
    content = msg.get("content", "") or ""
    thinking = msg.get("thinking", "") or ""
    print(f"[{mode}] {time.time() - t:.0f}s keys={list(msg.keys())} "
          f"content_len={len(content)} thinking_len={len(thinking)} "
          f"eval_count={d.get('eval_count')} done_reason={d.get('done_reason')}")
    print(f"[{mode}] content head: {content[:200]!r}")
    print(f"[{mode}] thinking tail: {thinking[-200:]!r}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = sys.argv[1:]
    modes = [a for a in args if a in "ABCDEFGH" and len(a) == 1]
    img = next((a for a in args if a.endswith(".jpg")), IMG)
    for m in modes or ["A", "B"]:
        call(m, img)