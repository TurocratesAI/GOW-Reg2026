#!/usr/bin/env python3
"""
Model -> walker -> score: run the trained GOW heads as the walker's answer_fn, emit real
chain-of-thought, and score vs GT (EdgeF1/BPV structural + report byte-exact/token-Jaccard).

Held-out = the NEWEST bags (embedded after the training run launched) -> unseen by the model.
  python gow/walker/eval_model.py --n 300 --device cuda:1
"""
import argparse, os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # gow_walker
sys.path.insert(0, os.path.join(HERE, "..", "heads"))     # gow_model, train_heads
sys.path.insert(0, os.path.join(HERE, ".."))              # data_split
import gow_walker as W
import gow_model
import data_split as DS
from train_heads import TextEmb, ORGANS, MAX_CAND


def tok(s):
    import re
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--heads", default="gow/artifacts/gow_heads.pt")
    ap.add_argument("--text-emb", default="gow/artifacts/text_emb.npz")
    ap.add_argument("--cot", default="data/train_CoT_v01.json")
    ap.add_argument("--split", default=DS.DEFAULT_SPLIT, help="shared split manifest")
    ap.add_argument("--eval-split", default="test", choices=["val", "test"], help="which held-out split to score")
    args = ap.parse_args()
    import torch
    dev = args.device

    T, QSURF, AV, META = W.load_artifacts()
    text = TextEmb(args.text_emb)
    model = gow_model.build().to(dev)
    model.load_state_dict(torch.load(args.heads, map_location=dev))
    model.eval()
    cot = {os.path.splitext(c["id"])[0]: c for c in json.load(open(args.cot))}

    # held-out = the shared manifest's test (or val) stems -> disjoint from train BY CONSTRUCTION
    # (replaces the old non-reproducible os.path.getmtime "newest = unseen" heuristic).
    stem_split = DS.load(args.split)
    sample = [b for b in sorted(glob.glob("data/feats/*.npz"))
              if os.path.splitext(os.path.basename(b))[0] in cot
              and DS.split_of(b, stem_split) == args.eval_split][:args.n]

    tp = fp = fn = bpv = organ_ok = rex = 0
    jac = 0.0
    n = 0
    for npz in sample:
        sid = os.path.splitext(os.path.basename(npz))[0]
        c = cot[sid]
        H = torch.from_numpy(np.load(npz)["X"].astype(np.float32)).to(dev)
        with torch.inference_mode():
            o_logits, z = model.organ(H)
        oi = int(o_logits.argmax()); organ = ORGANS[oi]
        org_vec = torch.softmax(o_logits, -1)                  # match training's organ conditioning

        def afn(org, cq):
            d = AV.get(org, {}).get(cq, {})
            cands = [a for a, _ in sorted(d.items(), key=lambda kv: -kv[1])][:MAX_CAND]  # by count, not alphabetical
            if not cands:
                return ""
            cemb = torch.from_numpy(np.stack([text(x) for x in cands])).to(dev)
            qemb = torch.from_numpy(text(cq)).to(dev)
            with torch.inference_mode():
                logits, _ = model.answer(H, z, org_vec, qemb, cemb)
            return cands[int(logits.argmax())]

        first_q = META.get(organ, {}).get("first_question_canon", W.ORGAN_Q)
        co_roots = META.get(organ, {}).get("co_roots", [])
        chain, ans = W.walk(organ, afn, T, QSURF, first_q, use_renderer=True, co_roots=co_roots)

        gt_edges = W.edge_set(c["chain-of-thought"])
        pe = W.edge_set(chain)
        tp += len(gt_edges & pe); fp += len(pe - gt_edges); fn += len(gt_edges - pe)
        bpv += int(pe == gt_edges)
        organ_ok += int(organ == c["organ"])
        gt_rep = next((s["answer"] for s in c["chain-of-thought"]
                       if W.canon(s.get("question", "")) == W.REPORT_Q), "")
        pr_rep = W.render_report(ans)
        rex += int(pr_rep == gt_rep)
        gt_t, pr_t = tok(gt_rep.replace("\\n", " ")), tok(pr_rep.replace("\\n", " "))
        jac += (len(gt_t & pr_t) / len(gt_t | pr_t)) if (gt_t | pr_t) else 0.0
        n += 1

    p, r, f = W.prf(tp, fp, fn)
    print(f"held-out (newest, unseen) slides: {n}\n")
    print(f"  organ accuracy   : {organ_ok/n:.3f}")
    print(f"  EdgeF1           : {f:.3f}   (P {p:.3f} / R {r:.3f})")
    print(f"  BPV (exact set)  : {bpv/n:.3f}")
    print(f"  report byte-exact: {rex/n:.3f}")
    print(f"  report tok-Jaccard(proxy): {jac/n:.3f}   (real en_core_sci_lg-Jaccard = next step)")


if __name__ == "__main__":
    main()
