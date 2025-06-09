import re
from pathlib import Path
from typing import List, Dict, Iterator
from records import QRecord, LRecord, GRecord

FLOAT = lambda s: float(s.replace('E', 'e')) if s.strip() else None
KV_RE = re.compile(r'([A-Za-z%]+)\s*=?\s*([-\w.+]+)')
YEAR_RE = re.compile(r'(\d{4})')            # 抓文件名里的 4 位数

# ---------- Q ----------
def iter_q_records(fp: Path) -> Iterator[QRecord]:
    year = _year_from_path(fp)
    with fp.open('r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            if ln[6:8].strip() != 'Q':
                continue
            z, a = int(ln[:3]), int(ln[3:6])
            yield QRecord(
                z, a, ln[6:9].strip(),
                FLOAT(ln[9:19]), FLOAT(ln[19:29]),
                ln[55:].strip() or None, year
            )

# ---------- L ----------
def iter_l_records(fp: Path) -> Iterator[LRecord]:
    year = _year_from_path(fp)
    buf, idx = [], 0
    def flush():
        nonlocal idx, buf
        if not buf: return
        idx += 1
        base = buf[0]
        rec = LRecord(
            idx, FLOAT(base[9:19]), base[21:39].strip(), base[39:55].strip(),
            {k.upper(): v for l in buf for k, v in KV_RE.findall(l[55:].replace('$',' '))},
            dataset_year=year
        )
        buf.clear(); return rec
    for ln in fp.open('r', encoding='utf-8', errors='ignore'):
        flag = ln[6:8].strip()
        if flag == 'L':
            r = flush(); 0 if not r else (yield r)
        if flag.startswith('L'): buf.append(ln)
    r = flush(); 0 if not r else (yield r)

# ---------- G ----------
def iter_g_records(fp: Path,
                   level_idx: Dict[float,int],
                   tol: float = 1.0) -> Iterator[GRecord]:
    year = _year_from_path(fp)
    buf: List[str] = []
    def match(e): return level_idx.get(round(e/tol)*tol)
    def flush():
        if not buf: return
        main = buf[0]
        g = GRecord(FLOAT(main[9:19]), main[21:29].strip() or None,
                    main[29:39].strip() or None, dataset_year=year)
        for l in buf:
            tail = l[55:].replace('$',' ')
            if m:=re.search(r'WIDTHG\s*=?\s*([-\d.E+]+)\s*EV\s*([\d.]*)', tail, re.I):
                g.width_ev, g.width_unc_ev = float(m[1]), FLOAT(m[2])
            for key in ('BM1W','BE2W'):
                if m:=re.search(fr'{key}\s*=?\s*([-\d.E+]+)', tail, re.I):
                    setattr(g, key.lower(), float(m[1]))
            for k,v in KV_RE.findall(tail):
                if k.upper() not in ('WIDTHG','BM1W','BE2W'):
                    g.extras[k.upper()]=v
            if m:=re.search(r'FL\s*=\s*([-\d.E+]+)', tail, re.I):
                g.to_id = match(float(m[1]))
        if g.to_id and g.e_gamma_keV:
            to_e = next(k for k,v in level_idx.items() if v==g.to_id)
            g.from_id = match(to_e + g.e_gamma_keV)
        if g.from_id is None and g.e_gamma_keV:
            g.from_id = match(g.e_gamma_keV)
        buf.clear(); return g
    for ln in fp.open('r', encoding='utf-8', errors='ignore'):
        flag = ln[6:8].strip()
        if flag == 'G':
            r = flush(); 0 if not r else (yield r)
        if flag.startswith('G'): buf.append(ln)
    r = flush(); 0 if not r else (yield r)

# ---------- 辅助 ----------
def _year_from_path(fp: Path) -> int | None:
    m = YEAR_RE.search(fp.stem)
    return int(m.group(1)) if m else None
