# parser.py  --  split + regex parse ENSDF (Q / L / G)
import re
from pathlib import Path
from typing import List, Dict, Iterator, Optional
from records import QRecord, LRecord, GRecord

# ---------------- common tools -----------------
NUM_RE = re.compile(r"[+\-]?\d+(?:\.\d+)?(?:[Ee][+\-]?\d+)?")  # match float/scientific
KV_RE = re.compile(r"([A-Za-z%]+)\s*=?\s*([\-\w.+]+)")  # match KEY=VAL
YEAR_RE = re.compile(r"(\d{4})")


def _first_float(text: str) -> Optional[float]:
    m = NUM_RE.search(text)
    return float(m.group()) if m else None


def _year(fp: Path) -> Optional[int]:
    m = YEAR_RE.search(fp.stem)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- Q ----------
def iter_q_records(fp: Path) -> Iterator[QRecord]:
    year = _year(fp)
    with fp.open("r", encoding="utf-8", errors="ignore") as fin:
        for ln in fin:
            cols = ln.strip().split(maxsplit=4)
            if len(cols) < 2 or cols[1] != "Q":
                continue

            # -------- Z, A, Element --------
            ZA = cols[0]
            z_match = re.match(r"\d+", ZA)
            z = int(z_match.group()) if z_match else 0
            rest = ZA[z_match.end() :] if z_match else ZA
            a_match = re.match(r"\d+", rest)
            if a_match:
                a = int(a_match.group())
                element = rest[a_match.end() :]
            else:
                a = 0
                element = rest

            q_val = _first_float(cols[2]) if len(cols) > 2 else None
            m_exc = _first_float(cols[3]) if len(cols) > 3 else None
            src = cols[4] if len(cols) > 4 else None

            yield QRecord(z, a, element, q_val, m_exc, src, year)


# ---------------------------------------------------------------- L ----------
def iter_l_records(fp: Path) -> Iterator[LRecord]:
    year = _year(fp)
    buf: List[str] = []
    level_id = 0

    def flush():
        nonlocal level_id, buf
        if not buf:
            return
        main = buf[0].strip().split(maxsplit=4)
        energy = _first_float(main[2]) if len(main) > 2 else None
        jp = main[3] if len(main) > 3 else ""
        hl = main[4] if len(main) > 4 else ""
        extras = {}
        for l in buf:
            extras.update({k.upper(): v for k, v in KV_RE.findall(l)})
        level_id += 1
        rec = LRecord(level_id, energy, jp, hl, extras, dataset_year=year)
        buf.clear()
        return rec

    for ln in fp.open("r", encoding="utf-8", errors="ignore"):
        parts = ln.strip().split(maxsplit=2)
        flag = parts[1] if len(parts) > 1 else ""
        if flag == "L":
            r = flush()
            if r:
                yield r
        if flag.startswith("L"):
            buf.append(ln)
    r = flush()
    if r:
        yield r


# ---------------------------------------------------------------- G ----------
def iter_g_records(
    fp: Path, level_idx: Dict[float, int], tol: float = 1.0
) -> Iterator[GRecord]:

    def match(e: float) -> Optional[int]:
        return level_idx.get(round(e / tol) * tol)

    year = _year(fp)
    buf: List[str] = []

    def flush():
        if not buf:
            return

        main = buf[0].strip().split(maxsplit=6)
        e_g = _first_float(main[2]) if len(main) > 2 else None
        inten = main[3] if len(main) > 3 else None
        mul = main[5] if len(main) > 5 else None

        g = GRecord(e_g, inten, mul, dataset_year=year)

        for l in buf:
            tail = l.strip()

            # WIDTHG
            if m := re.search(r"WIDTHG\s*=?\s*([-\d.E+]+)\s*EV\s*([\d.]*)", tail, re.I):
                g.width_ev = _first_float(m.group(1))
                g.width_unc_ev = _first_float(m.group(2))

            # BM1W / BE2W
            for key in ("BM1W", "BE2W"):
                if m := re.search(rf"{key}\s*=?\s*([-\d.E+]+)", tail, re.I):
                    setattr(g, key.lower(), _first_float(m.group(1)))

            # 其他键值
            for k, v in KV_RE.findall(tail):
                if k.upper() not in ("WIDTHG", "BM1W", "BE2W"):
                    g.extras[k.upper()] = v

            # 终止能级 FL=
            if m := re.search(r"FL\s*=\s*([-\d.E+]+)", tail, re.I):
                g.to_id = match(_first_float(m.group(1)))

        # 计算 from_id
        if g.to_id is not None and e_g is not None:
            to_e = next(k for k, v in level_idx.items() if v == g.to_id)
            g.from_id = match(to_e + e_g)
        if g.from_id is None and e_g is not None:
            g.from_id = match(e_g)

        buf.clear()
        return g

    # -------------- 逐行读取并收集块 ----------------
    for ln in fp.open("r", encoding="utf-8", errors="ignore"):
        cols = ln.strip().split(maxsplit=2)
        flag = cols[1] if len(cols) > 1 else ""
        if flag == "G":
            r = flush()
            if r:
                yield r
        if flag.startswith("G"):
            buf.append(ln)
    r = flush()
    if r:
        yield r
