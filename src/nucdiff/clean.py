#!/usr/bin/env python3
# 只保留第二个词组是 L / G / Q 的行
# 用法: python keep_LGQ.py raw.ens cleaned.ens

import sys, pathlib

KEEP = {"L", "G", "Q"}

def filter_file(src: pathlib.Path, dst: pathlib.Path):
    with open(src, "r", encoding="utf-8", errors="ignore") as fin, \
         open(dst, "w", encoding="utf-8") as fout:
        for ln in fin:
            if not ln.strip():                # 跳过空行
                continue
            parts = ln.expandtabs().strip().split(maxsplit=2)
            if len(parts) > 1 and parts[1] in KEEP:
                fout.write(ln)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("how to use: python clean.py <raw.ens> <cleaned.ens>")
        sys.exit(1)

    src, dst = map(pathlib.Path, sys.argv[1:])
    filter_file(src, dst)
    print("✔ Only Keep L/G/Q →", dst)
