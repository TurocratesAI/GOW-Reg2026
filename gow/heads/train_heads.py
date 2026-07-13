#!/usr/bin/env python3
"""
Train the GOW heads (organ + question-conditioned answer/grade) on frozen Virchow2 bags.

For each embedded slide (npz) matched to its CoT case we build supervision:
  - organ label (node 1)
  - per answered node: (question CONCH-emb, candidate-answer CONCH-embs for that (organ,question),
    GT answer index)   -> CLIP-style CE.  Grades are just nodes -> trains on the official CoT labels.

Text embeddings come from CONCH (precompute_conch_text.py); a deterministic-random fallback lets the
loop run/smoke before CONCH embeddings exist.

  smoke:  python gow/heads/train_heads.py --smoke --device cuda:1
  train:  python gow/heads/train_heads.py --features-dir data/feats --cot <cot> --text-emb gow/artifacts/text_emb.npz
"""
import argparse, json, os, glob, sys, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                  # gow_model
sys.path.insert(0, os.path.join(HERE, "..", "walker"))   # gow_walker (canon, artifacts)
sys.path.insert(0, os.path.join(HERE, ".."))             # data_split (shared split manifest)
import gow_model
import gow_walker as W
import data_split as DS

ORGANS = ["prostate", "breast", "colon", "stomach", "bladder", "lung", "cervix"]
O2I = {o: i for i, o in enumerate(ORGANS)}
MAX_CAND = 48                                             # cap candidate answers per node


class TextEmb:
    """string -> CONCH text vector; deterministic-random fallback until CONCH embeddings exist."""
    def __init__(self, path=None, dim=512):
        self.dim = dim
        self.cache = {}
        if path and os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            self.cache = {k: v for k, v in zip(d["keys"], d["emb"])}
            self.dim = d["emb"].shape[1]

    def __call__(self, s):
        if s in self.cache:
            return self.cache[s].astype(np.float32)
        rng = np.random.default_rng(abs(hash(s)) % (2 ** 32))
        return rng.normal(0, 1, self.dim).astype(np.float32)


def build_examples(case, text, AV):
    """-> (organ_idx, [ (q_emb[512], cand_emb[C,512], gt_idx) ... ]) using the CoT answers."""
    organ = case["organ"]
    if organ not in O2I:
        return None
    ans, nodes = {}, []
    for s in case["chain-of-thought"]:
        cq = W.canon(s.get("question", ""))
        if cq and cq not in ans:
            ans[cq] = s.get("answer", "")
    for cq, gt in ans.items():
        if cq == W.REPORT_Q or organ not in AV or cq not in AV[organ]:
            continue
        # answer_vocab is dumped with sort_keys=True (alphabetical), NOT by count -> sort by count
        # here so the MAX_CAND=48 cap keeps the COMMON answers, not an arbitrary alphabetical slice.
        cands = [a for a, _ in sorted(AV[organ][cq].items(), key=lambda kv: -kv[1])][:MAX_CAND]
        if gt not in cands:
            cands = [gt] + cands[:MAX_CAND - 1]                 # always keep GT reachable
        q_emb = text(cq)
        cand_emb = np.stack([text(c) for c in cands])
        nodes.append((q_emb, cand_emb, cands.index(gt)))
    return O2I[organ], nodes


ORGAN_W = 8.0            # up-weight organ CE (it gates 93% of edges) vs the many per-node answer CEs
FOCAL_GAMMA = 1.5        # focus answer loss on HARD nodes (the trivial binary-presence nodes dominate)
ACCUM = 8               # grad accumulation over K bags -> de-noise the effective batch-1 updates


def run(tr, va, load_bag, args):
    import torch, torch.nn.functional as F, math
    dev = args.device
    model = gow_model.build().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    # warmup (5% of steps) then cosine decay
    total_steps = max(1, args.epochs * math.ceil(len(tr) / ACCUM))
    warm = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    print(f"train slides {len(tr)}  val slides {len(va)}  (organ_w {ORGAN_W}, focal {FOCAL_GAMMA}, accum {ACCUM})")

    def focal_ce(logits, target):
        ce = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=dev))
        pt = torch.exp(-ce)
        return ((1 - pt) ** FOCAL_GAMMA) * ce

    def step(item, train):
        bag_ref, organ_idx, nodes = item
        H = torch.from_numpy(load_bag(bag_ref)).to(dev)
        o_logits, z = model.organ(H)
        # detached organ SOFTMAX as the answer-head organ conditioning -> identical at train & eval
        # (no oracle GT one-hot leak, and the head sees organ uncertainty instead of a hard one-hot).
        org_vec = torch.softmax(o_logits, -1).detach()
        organ_loss = ORGAN_W * F.cross_entropy(o_logits.unsqueeze(0), torch.tensor([organ_idx], device=dev))
        ok_o = int(o_logits.argmax().item() == organ_idx)
        ans_loss = torch.zeros((), device=dev); ok_a = tot_a = 0
        for q_emb, cand_emb, gt in nodes:
            qa = torch.from_numpy(q_emb).to(dev)
            ca = torch.from_numpy(cand_emb).to(dev)
            a_logits, _ = model.answer(H, z, org_vec, qa, ca)      # z NOT detached -> answers train the encoder
            ans_loss = ans_loss + focal_ce(a_logits, gt)
            ok_a += int(a_logits.argmax().item() == gt); tot_a += 1
        # normalize answer loss by nodes-per-case so many-node breast cases don't dominate the update
        loss = organ_loss + (ans_loss / max(tot_a, 1))
        if train:
            (loss / ACCUM).backward()
        return ok_o, ok_a, tot_a, o_logits.argmax().item(), organ_idx

    import time
    best = -1.0
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        from collections import Counter
        t0 = time.time()
        for i, it in enumerate(tr):
            step(it, True)
            if (i + 1) % ACCUM == 0 or i + 1 == len(tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); opt.zero_grad(); sched.step()
            if (i + 1) % 1500 == 0:
                print(f"  ep{ep} {i+1}/{len(tr)}  ({(i+1)/(time.time()-t0):.0f} bags/s)", flush=True)
        model.eval()
        oo = aa = ta = 0; per_org = Counter(); per_org_ok = Counter()
        with torch.no_grad():
            for it in va:
                a, b, c, pred, gt = step(it, False); oo += a; aa += b; ta += c
                per_org[gt] += 1; per_org_ok[gt] += int(pred == gt)
        organ_acc = oo / max(len(va), 1)
        macro = np.mean([per_org_ok[o] / per_org[o] for o in per_org]) if per_org else 0.0
        ans_acc = aa / max(ta, 1)
        vscore = 0.5 * macro + 0.5 * ans_acc                       # model-selection proxy (macro-organ + answer)
        print(f"epoch {ep}: val organ_acc {organ_acc:.3f}  macro-organ {macro:.3f}  answer_acc {ans_acc:.3f}  "
              f"(nodes/case {ta/max(len(va),1):.1f})  vscore {vscore:.3f}")
        if args.out and not args.smoke and vscore > best:
            best = vscore; torch.save(model.state_dict(), args.out)
            print(f"  [saved best] {args.out}  (vscore {vscore:.3f})")
    if args.out and not args.smoke:
        print(f"[done] best vscore {best:.3f} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir"); ap.add_argument("--cot"); ap.add_argument("--text-emb")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default="gow/artifacts/gow_heads.pt")
    ap.add_argument("--split", default=DS.DEFAULT_SPLIT, help="shared patient-grouped split manifest")
    ap.add_argument("--fold-test", action="store_true",
                    help="fold the held-out test split into TRAINING (final submission model; keeps val for checkpointing)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    _, _, AV, _ = W.load_artifacts()
    text = TextEmb(args.text_emb)

    if args.smoke:
        rng = np.random.default_rng(0)
        centers = rng.normal(0, 1, (7, 2560)).astype(np.float32)
        # synthetic cases: organ + a few binary nodes whose GT depends on the bag
        cot = json.load(open(os.path.join(HERE, "..", "..", "data", "smoke_cot.json"))) \
            if os.path.exists(os.path.join(HERE, "..", "..", "data", "smoke_cot.json")) else None
        q = rng.normal(0, 1, 512).astype(np.float32)          # fixed per-question emb (as in real training)
        cand = rng.normal(0, 1, (2, 512)).astype(np.float32)  # fixed per-answer embs ("no"/"yes")
        examples = []
        for i in range(200):
            y = i % 7; n = int(rng.integers(120, 400))
            bag = (centers[y] + rng.normal(0, 3, (n, 2560))).astype(np.float32)
            sign = int(bag.mean(0)[0] > 0)                     # image -> answer signal
            examples.append((bag, y, [(q, cand, sign)]))
        import random
        random.Random(0).shuffle(examples)
        n = int(0.9 * len(examples))
        run(examples[:n], examples[n:], lambda b: b, args)
        print("\n[smoke] full head-training loop OK (organ_acc high, answer_acc >0.5).")
        return

    assert args.features_dir and args.cot, "need --features-dir + --cot (or --smoke)"
    labels = {os.path.splitext(c["id"])[0]: c for c in json.load(open(args.cot)) if "id" in c}
    # shared, patient-grouped, stratified split manifest (train_heads + eval_model read the SAME one)
    stem_split = DS.load(args.split)
    paths = {"train": [], "val": []}
    n_skip = 0
    for p in glob.glob(os.path.join(args.features_dir, "*.npz")):
        sid = os.path.splitext(os.path.basename(p))[0]
        if sid not in labels:                                  # stray bag (e.g. a test tiff) -> exclude
            n_skip += 1; continue
        sp = DS.split_of(sid, stem_split)                      # patient-stem -> train/val/test
        if sp == "test" and args.fold_test:                    # final model: train on the test split too
            sp = "train"
        if sp in ("train", "val"):
            paths[sp].append((p, sid))
        # (else) test split held out for the real scorer (eval_real.py)
    # Preload bags into RAM ONCE as fp16 (npz is DEFLATE-compressed -> np.load decompresses on EVERY
    # call; re-decompressing 6.8k bags x N epochs is the 100%-CPU bottleneck). Cast fp16->fp32 per step.
    import time
    tr, va = [], []
    for sp, dst in (("train", tr), ("val", va)):
        t0 = time.time()
        for j, (p, sid) in enumerate(paths[sp]):
            ex = build_examples(labels[sid], text, AV)
            if ex:
                dst.append((np.load(p)["X"], ex[0], ex[1]))    # keep fp16 in RAM
            if (j + 1) % 2000 == 0:
                print(f"  preloaded {sp} {j+1}/{len(paths[sp])} ({(j+1)/(time.time()-t0):.0f}/s)", flush=True)
    print(f"preloaded train {len(tr)} / val {len(va)} slides into RAM  (skipped {n_skip} not in CoT; test held out)")
    run(tr, va, lambda a: a.astype(np.float32), args)


if __name__ == "__main__":
    main()
