# -*- coding: utf-8 -*-
r"""
巡店AI 数据集自检脚本
==========================
用法：
  1. 把本文件放到 D:\Users\Desktop\巡店AI\data 下（和 images 同级）
  2. 双击运行：check.py
     或命令行：python check.py

它会检查：
  - images 文件夹位置对不对
  - 每个日期/战区文件夹的张数对不对
  - CSV 里每一条图片路径 是否真实存在
  - 有没有多余文件
"""
import os
import sys
import csv
import collections

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, 'images')

OK = '[OK]'
NG = '[!!]'
WARN = '[??]'

print('=' * 60)
print('巡店AI 数据集自检')
print('=' * 60)
print('当前目录:', ROOT)
print()

# ---------- 1. 检查 images 位置 ----------
print('【1】检查 images 文件夹位置')
if not os.path.isdir(IMG):
    print(NG, '找不到 images 文件夹！')
    print('    应该在:', IMG)
    print('    请确认 images 文件夹直接放在本脚本旁边（同一层），')
    print('    不要多套一层 巡店AI/data/images 或 images/images')
    input('\n按回车退出...')
    sys.exit(1)
print(OK, '找到 images 文件夹')
print()

# ---------- 2. 实际扫描 ----------
print('【2】扫描实际文件...')
actual = collections.Counter()   # (date, region) -> count
all_files = set()
bad_ext = []
for dp, dns, fns in os.walk(IMG):
    for fn in fns:
        full = os.path.join(dp, fn)
        rel = os.path.relpath(full, ROOT).replace('\\', '/')
        all_files.add(rel)
        if not fn.lower().endswith('.jpg'):
            bad_ext.append(rel)
        parts = rel.split('/')
        if len(parts) == 4:
            actual[(parts[1], parts[2])] += 1

total_actual = sum(actual.values())
print(OK, '扫描到 %d 张图片' % total_actual)
if bad_ext:
    print(WARN, '有 %d 个非 .jpg 文件，例如: %s' % (len(bad_ext), bad_ext[:3]))
print()

# ---------- 3. 张数校验 ----------
print('【3】比对应有张数（共应有 71,320 张）')
plan_file = os.path.join(ROOT, '文件夹张数清单.csv')
if os.path.isfile(plan_file):
    plan = {}
    with open(plan_file, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            plan[(row['日期目录'], row['战区目录'])] = int(row['应有张数'])
    miss = []
    extra = []
    for k, v in plan.items():
        a = actual.get(k, 0)
        if a < v:
            miss.append((k, v, a))
        elif a > v:
            extra.append((k, v, a))
    for k in actual:
        if k not in plan:
            extra.append((k, 0, actual[k]))

    if not miss and not extra:
        print(OK, '237 个日期/战区文件夹 张数全部正确')
    else:
        if miss:
            print(NG, '缺少图片的文件夹 %d 个:' % len(miss))
            for k, v, a in sorted(miss)[:15]:
                print('      %s/%s  应有%d 实际%d 缺%d' % (k[0], k[1], v, a, v - a))
            if len(miss) > 15:
                print('      ...还有 %d 个' % (len(miss) - 15))
        if extra:
            print(WARN, '多余图片的文件夹 %d 个:' % len(extra))
            for k, v, a in sorted(extra)[:10]:
                print('      %s/%s  应有%d 实际%d 多%d' % (k[0], k[1], v, a, a - v))
else:
    print(WARN, '没找到 文件夹张数清单.csv，跳过张数比对')
print()

# ---------- 4. CSV 路径校验 ----------
print('【4】校验 CSV 里的图片路径是否真实存在')
csvs = [f for f in os.listdir(ROOT) if f.lower().endswith('.csv')
        and f != '文件夹张数清单.csv']
if not csvs:
    print(WARN, '当前目录没有 CSV 文件')
else:
    for fn in sorted(csvs):
        path = os.path.join(ROOT, fn)
        try:
            rows = list(csv.reader(open(path, encoding='utf-8-sig')))
        except Exception as e:
            print(NG, fn, '读取失败:', e)
            continue
        if not rows:
            continue
        hdr, body = rows[0], rows[1:]
        # 找所有值像图片路径的列
        idxs = []
        for i in range(len(hdr)):
            for r in body[:50]:
                if i < len(r) and r[i].startswith('images/'):
                    idxs.append(i)
                    break
        if not idxs:
            print('   %-16s %5d行  无图片列' % (fn, len(body)))
            continue
        ps = []
        for r in body:
            for i in idxs:
                if i < len(r) and r[i]:
                    ps.extend(r[i].split('|'))
        absn = sum(1 for p in ps if os.path.isabs(p) or ':' in p)
        hit = sum(1 for p in ps if os.path.isfile(os.path.join(ROOT, p)))
        status = OK if hit == len(ps) and absn == 0 else NG
        print('   %s %-16s %5d行  路径%6d  存在%6d  缺失%4d  绝对路径%d'
              % (status, fn, len(body), len(ps), hit, len(ps) - hit, absn))
print()

# ---------- 5. 多余文件 ----------
print('【5】检查多余文件')
if plan_file:
    plan_paths = set()
    print('   （本项仅提示，不影响使用）')
    extra_n = len(all_files) - 71320
    if extra_n > 0:
        print(WARN, 'images 内比预期多 %d 个文件' % extra_n)
    elif extra_n < 0:
        print(NG, 'images 内比预期少 %d 个文件' % (-extra_n))
    else:
        print(OK, '文件数量正好 71,320')
print()
print('=' * 60)
print('自检结束')
print('=' * 60)
input('\n按回车退出...')
