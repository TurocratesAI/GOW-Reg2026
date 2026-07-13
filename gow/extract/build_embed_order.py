#!/usr/bin/env python3
"""Build the stratified embedding order: rare histologic types + cervix/lung FIRST, then common.
So a throughput-limited run spends its budget on the labels that move the score.
  python gow/extract/build_embed_order.py <train_CoT.json> data/embed_order.txt [s3://bucket/train]
"""
import json, sys
from collections import Counter

cot, out = sys.argv[1], sys.argv[2]
bucket = sys.argv[3] if len(sys.argv) > 3 else "s3://reg2026-challenge-do-not-touch/train"
data = json.load(open(cot))


def htype(c):
    for s in c["chain-of-thought"]:
        if s.get("question", "").strip().lower().startswith("what is the histologic type of neoplasm"):
            return s.get("answer", "").strip()
    return None


tcount = Counter(t for c in data if (t := htype(c)))


def priority(c):
    t = htype(c)
    r = 1.0 / tcount[t] if t else 0.5                 # rarer type -> higher; benign/no-type -> mid
    b = 0.5 if c.get("organ") in ("cervix", "lung") else 0.0
    return r + b


order = sorted(data, key=priority, reverse=True)
with open(out, "w") as f:
    for c in order:                                   # c['id'] already ends in .tiff
        f.write(f"{bucket}/{c['id']}\n")
print(f"{len(order)} slides -> {out}  (rarest type first, e.g. {htype(order[0])!r} / {order[0]['organ']})")
