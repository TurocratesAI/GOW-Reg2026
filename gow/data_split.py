#!/usr/bin/env python3
"""
One shared, reproducible, PATIENT-GROUPED, STRATIFIED train/val/test split for the GOW heads.

Why this exists (from the training-pipeline review):
  * IDs are PIT_<source>_<case>_<section>.tiff. Sections of one block share a patient and near
    -identical labels -> a per-SLIDE split leaks patients across train/val and inflates val acc.
    => split unit = PATIENT STEM  PIT_<source>_<case>  (all _SS sections stay together).
  * The embed order is rare-histologic-type-first, so any order-based split is organ/type-biased.
    => stratify jointly by (organ, #1-diagnosis) [present in 100% of cases, 118-way] so every
       diagnosis that appears in val/test also appears in train (rare-safe rounding).
  * train_heads.py and eval_model.py must read the SAME persisted manifest (no os.path.getmtime,
    no independent random shuffle) so numbers are reproducible and train/test are disjoint by
    construction while extraction is still writing new bags.

Manifest = gow/artifacts/split.json:
  {"meta": {...}, "stem_split": {stem: "train"|"val"|"test"}, "counts": {...}}

Source is recoverable from the id (PIT_<source>_...) for per-source / Leave-One-Source-Out
reporting; LOSO itself is a training-time flag (exclude a source from train, test on it), not a
field here.
"""
import os, re, json, random
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPLIT = os.path.join(HERE, "artifacts", "split.json")

STEM_RE = re.compile(r"^(PIT_\d+_\d+)_\d+$")
_DIAG_Q = "what is the #1 diagnosis"


def _canon(s):
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return re.sub(r"[.,;:!?]+$", "", s).strip()


def stem(slide_id):
    """PIT_01_00019_01[.tiff] -> PIT_01_00019 ; ids without a section suffix pass through."""
    sid = os.path.splitext(os.path.basename(slide_id))[0]
    m = STEM_RE.match(sid)
    return m.group(1) if m else sid


def source(slide_id):
    """PIT_01_00019_01 -> PIT_01 (scanner/site) ; for per-source + LOSO reporting."""
    sid = os.path.splitext(os.path.basename(slide_id))[0]
    m = re.match(r"^(PIT_\d+)_", sid)
    return m.group(1) if m else sid.split("_")[0]


def strat_label(case):
    """(organ, canon #1-diagnosis) -> the joint stratification key (rare-diagnosis-safe)."""
    organ = case.get("organ", "?")
    dx = ""
    for s in case.get("chain-of-thought", []):
        if _canon(s.get("question", "")) == _DIAG_Q:
            dx = _canon(s.get("answer", ""))
            break
    return (organ, dx)


def build(cot_path, out_path=DEFAULT_SPLIT, seed=0, ratios=(0.8, 0.1, 0.1)):
    """Group cases by patient stem, stratify by (organ,#1-dx), write the split manifest."""
    cot = json.load(open(cot_path))
    # one representative label per stem (siblings share organ + near-identical dx)
    stem_label, stem_organ, stem_source = {}, {}, {}
    for c in cot:
        sid = c.get("id")
        if not sid:
            continue
        st = stem(sid)
        if st not in stem_label:                       # first section wins (deterministic by file order)
            stem_label[st] = strat_label(c)
            stem_organ[st] = c.get("organ", "?")
            stem_source[st] = source(sid)

    groups = defaultdict(list)
    for st, key in stem_label.items():
        groups[key].append(st)

    rng = random.Random(seed)
    assign = {}
    for key in sorted(groups):                         # deterministic group order
        stems = sorted(groups[key])                    # deterministic within group
        rng.shuffle(stems)
        n = len(stems)
        # rare-safe: train gets >=1 always; val/test only receive types already in train
        n_tr = max(1, int(round(ratios[0] * n)))
        n_tr = min(n_tr, n)                             # never exceed
        rest = n - n_tr
        n_va = int(round(ratios[1] / (ratios[1] + ratios[2]) * rest)) if rest else 0
        for i, st in enumerate(stems):
            assign[st] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    counts = Counter(assign.values())
    # organ x split sanity table
    org_split = defaultdict(Counter)
    for st, sp in assign.items():
        org_split[stem_organ[st]][sp] += 1
    src_split = defaultdict(Counter)
    for st, sp in assign.items():
        src_split[stem_source[st]][sp] += 1

    manifest = {
        "meta": {"seed": seed, "ratios": list(ratios), "unit": "patient_stem",
                 "stratify": "organ+#1diagnosis", "cot": os.path.basename(cot_path),
                 "n_stems": len(assign), "n_slides_in_cot": sum(1 for c in cot if c.get("id")),
                 "n_strat_groups": len(groups)},
        "counts": dict(counts),
        "organ_split": {o: dict(c) for o, c in sorted(org_split.items())},
        "source_split": {s: dict(c) for s, c in sorted(src_split.items())},
        "stem_split": assign,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return manifest


def load(path=DEFAULT_SPLIT):
    """-> dict stem -> 'train'|'val'|'test'."""
    return json.load(open(path))["stem_split"]


def split_of(slide_id, stem_split):
    """slide id/path -> 'train'|'val'|'test'|None (None = not in the manifest / excluded)."""
    return stem_split.get(stem(slide_id))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", default="data/train_CoT_v01.json")
    ap.add_argument("--out", default=DEFAULT_SPLIT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = build(args.cot, args.out, args.seed)
    print(f"[split] {m['meta']['n_stems']} stems -> {m['counts']}  "
          f"({m['meta']['n_strat_groups']} strat groups)  -> {args.out}\n")
    print(f"{'organ':10} {'train':>6} {'val':>5} {'test':>5}")
    for o, c in m["organ_split"].items():
        print(f"{o:10} {c.get('train',0):6d} {c.get('val',0):5d} {c.get('test',0):5d}")
    print(f"\n{'source':8} {'train':>6} {'val':>5} {'test':>5}")
    for s, c in m["source_split"].items():
        print(f"{s:8} {c.get('train',0):6d} {c.get('val',0):5d} {c.get('test',0):5d}")
