#!/usr/bin/env python3
# 只保留每个记录类型为 L / G / Q 的行，分别写入三个文件
# 用法: python divide.py raw.ens

import sys, pathlib

KEEP = ["L", "G", "Q"]


def filter_file(src: pathlib.Path):

    out_files = {
        k: open(src.with_suffix(f".{k}.ens"), "w", encoding="utf-8") for k in KEEP
    }

    with open(src, "r", encoding="utf-8", errors="ignore") as fin:
        for ln in fin:
            if not ln.strip():
                continue
            parts = ln.expandtabs().strip().split(maxsplit=2)
            if len(parts) > 1 and parts[1] in KEEP:
                out_files[parts[1]].write(ln)

    for f in out_files.values():
        f.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("how to use: python divide.py <raw.ens>")
        sys.exit(1)

    src = pathlib.Path(sys.argv[1])
    filter_file(src)
    print("√ L/G/Q ")
