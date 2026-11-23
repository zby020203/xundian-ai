"""临时校准脚本：在真实样本上看 moire / composite_corr 分布，定阈值用。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from audit import imageops  # noqa: E402

IMGS = {
    "正常-割箱": "images/20251001/north/00002_cur.jpg",
    "正常-冰柜": "images/20251001/northeast/00021_cur.jpg",
    "空瓶": "images/20251004/southwest/00947_cur.jpg",
    "纵深-洗涤架": "images/20251004/central/00987_cur.jpg",
    "场景-餐区": "images/20251002/north/00582_cur.jpg",
    "货架摆拍": "images/20251001/central/00006_cur.jpg",
    "拍屏幕1": "images/20251016/southwest/15164_cur.jpg",
    "拍屏幕2": "images/20251016/southwest/15163_cur.jpg",
    "拼接屏": "images/20251016/southwest/14816_cur.jpg",
    "翻拍货架": "images/20251028/northwest/30530_cur.jpg",
}

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    data_root = r"d:\Users\Desktop\巡店AI\data"
    for name, rel in IMGS.items():
        im = imageops.load_image(os.path.join(data_root, rel))
        s = imageops.summarize(im, {})
        print(f"{name:10s} moire={s['moire']:8.3f} comp_corr={s['composite_corr']:6.3f} "
              f"blur={s['blur']:9.1f} {s['width']}x{s['height']}")