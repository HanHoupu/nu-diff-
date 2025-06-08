# parser_lgq.py
# ——————————————————————————————————————
# 读 ENSDF  → 产出 QRecord / LRecord / GRecord
# 把 records.py 放在同目录即可 import

# ── 1. 依赖与记录类型 ─────────────────────
import re
from pathlib import Path
from typing import List, Dict, Iterator
from records import QRecord, LRecord, GRecord           # ← 你的 dataclass 文件

# ── 2. 通用小工具 ─────────────────────────
FLOAT = lambda s: float(s.replace('E', 'e')) if s.strip() else None
KV_RE = re.compile(r'([A-Za-z%]+)\s*=?\s*([-\w.+]+)')   # 抠 "KEY=VAL"

# ── 3. 读 Q 文件 ──────────────────────────
def iter_q_records(fp: Path) -> Iterator[QRecord]:
    with fp.open('r', encoding='utf-8', errors='ignore') as fin:
        for ln in fin:
            if ln[6:8].strip() != 'Q':                  # 只有 Q 行才读
                continue
            z   = int(ln[0:3])
            a   = int(ln[3:6])
            ele = ln[6:9].strip()
            qv  = FLOAT(ln[9:19])
            me  = FLOAT(ln[19:29])
            src = ln[55:].strip() or None
            yield QRecord(z, a, ele, qv, me, src)

# ── 4. 读 L 文件（自动编号）─────────────────
def iter_l_records(fp: Path) -> Iterator[LRecord]:
    buf, idx = [], 0

    def flush():
        nonlocal buf, idx
        if not buf:
            return
        base = buf[0]                                   # L 主行
        idx += 1
        en   = FLOAT(base[9:19])
        jp   = base[21:39].strip()
        hl   = base[39:55].strip()
        ex   = {k.upper(): v for l in buf
                 for k, v in KV_RE.findall(l[55:].replace('$', ' '))}
        buf.clear()
        return LRecord(idx, en, jp, hl, extras=ex)

    for ln in fp.open('r', encoding='utf-8', errors='ignore'):
        flag = ln[6:8].strip()                          # "L", "L2"…
        if flag == 'L':
            rec = flush()
            if rec:  yield rec
        if flag.startswith('L'):
            buf.append(ln)
    rec = flush()
    if rec:  yield rec

# ── 5. 读 G 文件并配对 from_id / to_id ──────
def iter_g_records(fp: Path,
                   level_idx: Dict[float, int],
                   tol: float = 1.0) -> Iterator[GRecord]:

    def match(e):                                       # 能量 → level_id
        return level_idx.get(round(e / tol) * tol)

    buf: List[str] = []

    def flush():
        if not buf:
            return
        main = buf[0]
        eg   = FLOAT(main[9:19])
        inten= main[21:29].strip() or None
        mul  = main[29:39].strip() or None
        g    = GRecord(eg, inten, mul)

        for l in buf:                                   # 扫续行
            tail = l[55:].replace('$', ' ')
            if m := re.search(r'WIDTHG\s*=?\s*([-\d.E+]+)\s*EV\s*([\d.]*)', tail, re.I):
                g.width_ev = float(m.group(1))
                g.width_unc_ev = FLOAT(m.group(2))
            for key in ('BM1W', 'BE2W'):
                if m := re.search(fr'{key}\s*=?\s*([-\d.E+]+)', tail, re.I):
                    setattr(g, key.lower(), float(m.group(1)))
            for k, v in KV_RE.findall(tail):
                if k.upper() not in ('WIDTHG', 'BM1W', 'BE2W'):
                    g.extras[k.upper()] = v
            if m := re.search(r'FL\s*=\s*([-\d.E+]+)', tail, re.I):
                g.to_id = match(float(m.group(1)))

        if g.to_id and eg:                              # 能找终点就推起点
            to_e = next(k for k, v in level_idx.items() if v == g.to_id)
            g.from_id = match(to_e + eg)
        if g.from_id is None and eg:                    # 兜底：就近能级
            g.from_id = match(eg)

        buf.clear()
        return g

    for ln in fp.open('r', encoding='utf-8', errors='ignore'):
        flag = ln[6:8].strip()                          # "G", "G2"…
        if flag == 'G':
            rec = flush()
            if rec: yield rec
        if flag.startswith('G'):
            buf.append(ln)
    rec = flush()
    if rec: yield rec
