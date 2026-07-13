#!/usr/bin/env python3
"""
Per-question answer error analysis for the trained GOW heads.

Runs the model over the held-out split and, for EVERY answered node, records whether the predicted
answer is right - then breaks it down by question. Shows where the ~0.88 answer accuracy actually
leaks: which questions have the most errors, how far above the modal ("predict the most common
answer, ignore the image") baseline the model is, and the top confusions.

Uses GT organ for the candidate set (organ acc is 0.998) to isolate the answer head.
  CUDA_VISIBLE_DEVICES="" python gow/heads/answer_analysis.py --eval-split val --heads gow/artifacts/gow_heads_v2.pt
"""
import argparse, os, sys, json, glob
import numpy as np
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "walker"))
sys.path.insert(0, os.path.join(HERE, ".."))
import gow_model, gow_walker as W, data_split as DS
from train_heads import TextEmb, ORGANS, O2I, MAX_CAND


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="gow/artifacts/gow_heads_v2.pt")
    ap.add_argument("--text-emb", default="gow/artifacts/text_emb.npz")
    ap.add_argument("--cot", default="data/train_CoT_v01.json")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    import torch
    dev = args.device

    T, QSURF, AV, META = W.load_artifacts()
    text = TextEmb(args.text_emb)
    model = gow_model.build().to(dev)
    model.load_state_dict(torch.load(args.heads, map_location=dev)); model.eval()
    cot = {os.path.splitext(c["id"])[0]: c for c in json.load(open(args.cot)) if "id" in c}
    sm = DS.load()
    bags = [b for b in sorted(glob.glob("data/feats/*.npz"))
            if os.path.splitext(os.path.basename(b))[0] in cot
            and DS.split_of(b, sm) == args.eval_split][: args.n]
    print(f"[analysis] {len(bags)} {args.eval_split} bags, model {args.heads}\n")

    # per question: n, model-correct, modal-correct, confusions; also tag organ-conditioned questions
    Q = defaultdict(lambda: {"n": 0, "ok": 0, "modal": 0, "conf": Counter(), "organs": set()})
    tot = ok = 0
    for npz in bags:
        sid = os.path.splitext(os.path.basename(npz))[0]
        c = cot[sid]; organ = c["organ"]
        if organ not in O2I:
            continue
        H = torch.from_numpy(np.load(npz)["X"].astype(np.float32)).to(dev)
        with torch.inference_mode():
            o_logits, z = model.organ(H)
        org_vec = torch.softmax(o_logits, -1)
        ans = {}
        for s in c["chain-of-thought"]:
            cq = W.canon(s.get("question", ""))
            if cq and cq not in ans:
                ans[cq] = s.get("answer", "")
        for cq, gt in ans.items():
            if cq == W.REPORT_Q or organ not in AV or cq not in AV[organ]:
                continue
            d = AV[organ][cq]
            cands = [a for a, _ in sorted(d.items(), key=lambda kv: -kv[1])][:MAX_CAND]
            if gt not in cands:
                cands = [gt] + cands[:MAX_CAND - 1]
            cemb = torch.from_numpy(np.stack([text(x) for x in cands])).to(dev)
            with torch.inference_mode():
                logits, _ = model.answer(H, z, org_vec, torch.from_numpy(text(cq)).to(dev), cemb)
            pred = cands[int(logits.argmax())]
            modal = cands[0]                                   # most-common answer (image-blind baseline)
            r = Q[cq]
            r["n"] += 1; r["ok"] += int(pred == gt); r["modal"] += int(modal == gt); r["organs"].add(organ)
            if pred != gt:
                r["conf"][f"{gt}  ->  {pred}"] += 1
            tot += 1; ok += int(pred == gt)

    print(f"OVERALL micro answer accuracy: {ok/max(tot,1):.4f}  over {tot} answered nodes\n")
    rows = [(q, r) for q, r in Q.items() if r["n"] >= 5]
    # rank by number of ERRORS (n * (1-acc)) = where the misses concentrate
    rows.sort(key=lambda kv: -(kv[1]["n"] - kv[1]["ok"]))
    print(f"{'errors':>6} {'n':>5} {'acc':>6} {'modal':>6} {'uplift':>6}  question  (top confusion)")
    print("-" * 100)
    for q, r in rows[:22]:
        acc = r["ok"] / r["n"]; modal = r["modal"] / r["n"]; err = r["n"] - r["ok"]
        top = r["conf"].most_common(1)[0][0] if r["conf"] else ""
        nconf = r["conf"].most_common(1)[0][1] if r["conf"] else 0
        print(f"{err:6d} {r['n']:5d} {acc:6.3f} {modal:6.3f} {acc-modal:+6.3f}  {q[:44]:44}  [{top[:40]}] x{nconf}")

    # summary: how much of total error is in the top-5 questions
    total_err = sum(r["n"] - r["ok"] for _, r in rows)
    top5_err = sum(r["n"] - r["ok"] for _, r in rows[:5])
    print(f"\ntop-5 questions hold {top5_err}/{total_err} = {100*top5_err/max(total_err,1):.0f}% of all answer errors")
    # nodes where the model BEATS modal the most (learned something) and TRAILS (worse than guessing)
    trail = [(q, r) for q, r in rows if r["ok"]/r["n"] < r["modal"]/r["n"] - 0.02]
    if trail:
        print("\nquestions where the model is WORSE than the modal baseline (head hurting):")
        for q, r in sorted(trail, key=lambda kv: (kv[1]["ok"]-kv[1]["modal"]))[:8]:
            print(f"  acc {r['ok']/r['n']:.3f} vs modal {r['modal']/r['n']:.3f}  (n={r['n']})  {q[:50]}")


if __name__ == "__main__":
    main()
