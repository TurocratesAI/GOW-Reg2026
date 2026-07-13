#!/usr/bin/env python3
"""
Test1 read-out: run the trained GOW heads + OOD gate over the 350 anonymized Test1 bags.

Test1 has NO ground-truth chains and NO per-slide organ labels (the IDs are anonymized). The manifest
only states the aggregate: 280 original + 70 uterus = 20 percent OOD. So the one quantitative check we
can do locally is the OOD FLAG RATE (it should land near 20 percent / 70 slides if the gate is
calibrated) plus the predicted-organ distribution and a sanity look at the emitted chains. A real
Metric-A on Test1 needs a platform submission (no ground truth here).

  python gow/walker/eval_test1.py --heads gow/artifacts/gow_heads_final.pt --device cuda:0
"""
import argparse, os, sys, json, glob
import numpy as np
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # gow_walker, eval_real
sys.path.insert(0, os.path.join(HERE, "..", "heads"))     # gow_model, train_heads, ood_route
sys.path.insert(0, os.path.join(HERE, ".."))              # data_split
import gow_walker as W
import gow_model
import ood_route as OR
from train_heads import TextEmb, ORGANS
from eval_real import predict_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="gow/artifacts/gow_heads_final.pt")
    ap.add_argument("--feats-dir", default="data/feats_test1")
    ap.add_argument("--text-emb", default="gow/artifacts/text_emb.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="gow/artifacts/eval_test1.json")
    ap.add_argument("--pred-out", default="gow/artifacts/test1_predictions.json")
    ap.add_argument("--expect-ood", type=int, default=70)
    args = ap.parse_args()
    import torch
    dev = args.device

    T, QSURF, AV, META = W.load_artifacts()
    text = TextEmb(args.text_emb)
    model = gow_model.build().to(dev)
    model.load_state_dict(torch.load(args.heads, map_location=dev)); model.eval()
    gate = OR.get_gate()
    if gate is None:
        print("[warn] OOD gate not found; flag-rate check disabled")

    bags = sorted(glob.glob(os.path.join(args.feats_dir, "*.npz")))
    print(f"[eval_test1] {len(bags)} Test1 bags | heads={args.heads} | dev={dev}")

    rows, preds, id2chain = [], [], {}
    for i, npz in enumerate(bags):
        sid = os.path.splitext(os.path.basename(npz))[0]
        H = torch.from_numpy(np.load(npz)["X"].astype(np.float32)).to(dev)
        mean_vec = H.float().mean(0).cpu().numpy()
        with torch.inference_mode():
            o_logits, _ = model.organ(H)
        org_vec = torch.softmax(o_logits, -1)
        raw_organ = ORGANS[int(o_logits.argmax())]
        conf = float(org_vec.max())
        maha = float(gate.score(mean_vec)) if gate else float("nan")
        in_cervix_bucket = raw_organ == OR.OOD_ROUTE_ORGAN
        flagged = bool(gate and gate.should_route_ood(raw_organ, mean_vec, org_vec))
        organ, chain = predict_chain(model, H, T, QSURF, AV, META, text, dev)  # same gate applied inside
        rows.append({"id": sid, "raw_organ": raw_organ, "conf": round(conf, 4),
                     "maha": round(maha, 3), "cervix_bucket": in_cervix_bucket,
                     "flagged_ood": flagged, "final_organ": organ,
                     "n_tiles": int(H.shape[0]), "chain_len": len(chain)})
        preds.append({"id": sid + ".tiff", "chain-of-thought": chain})
        id2chain[sid] = chain
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(bags)}  flagged so far: {sum(r['flagged_ood'] for r in rows)}")

    n = len(rows)
    n_flag = sum(r["flagged_ood"] for r in rows)
    n_cervix = sum(r["cervix_bucket"] for r in rows)
    tcerv = gate.cervix_thr if gate else float("nan")
    mahas = sorted(r["maha"] for r in rows)
    print("\n================ TEST1 OOD / ORGAN READ-OUT ================")
    print(f"  slides                               : {n}")
    print(f"  router-cervix bucket size            : {n_cervix}")
    print(f"  OOD-flagged (cervix bucket, score>T) : {n_flag}  ({100*n_flag/n:.1f}%)"
          f"   [expected ~{args.expect_ood} = {100*args.expect_ood/n:.0f}%]")
    print(f"    T_CERVIX                           : {tcerv:.1f}")
    print(f"  Mahalanobis min/median/p90/max       : {mahas[0]:.1f} / {mahas[n//2]:.1f} /"
          f" {mahas[int(0.9*n)]:.1f} / {mahas[-1]:.1f}")
    print("\n  RAW organ-router argmax distribution:")
    for o, c in Counter(r["raw_organ"] for r in rows).most_common():
        print(f"    {o:10} {c:4}")
    print("\n  FINAL organ distribution (after OOD routing to the gyn/cervix spine):")
    for o, c in Counter(r["final_organ"] for r in rows).most_common():
        print(f"    {o:10} {c:4}")

    for tag, pred in [("OOD-flagged", lambda r: r["flagged_ood"]),
                      ("in-dist   ", lambda r: not r["flagged_ood"])]:
        r = next((r for r in rows if pred(r)), None)
        if r:
            ch = id2chain[r["id"]]
            print(f"\n  example {tag} slide {r['id']}  (final_organ={r['final_organ']}, {len(ch)} steps):")
            for st in ch[:5]:
                print(f"    Q {str(st.get('question',''))[:58]!r}  ->  A {str(st.get('answer',''))[:38]!r}")

    json.dump({"rows": rows,
               "summary": {"n": n, "n_flagged_ood": n_flag, "n_cervix_bucket": n_cervix,
                           "expected_ood": args.expect_ood, "cervix_threshold": tcerv}},
              open(args.out, "w"), indent=1)
    json.dump(preds, open(args.pred_out, "w"), indent=1)
    print(f"\n[ok] per-slide -> {args.out}   predictions -> {args.pred_out}")


if __name__ == "__main__":
    main()
