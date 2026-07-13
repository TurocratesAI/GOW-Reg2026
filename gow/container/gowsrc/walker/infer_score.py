#!/usr/bin/env python3
"""
END-TO-END real score: trained GOW model -> fan-out walker -> Metric A on a held-out split.

For each held-out slide (embedded bag + CoT): predict organ, walk the mined DAG answering each
node with the trained CLIP heads (candidates = that (organ,q)'s answer set, scored in CONCH space),
render the CAP report, and score vs GT:
  EdgeF1 + BPV  (exact, canonicalized edge sets - the real scorer's definition)
  MESS          (edge-keyed; exact-match proxy = conservative lower bound of the embedding metric)
  final_report  (real REG25 scorer: en_core_sci_lg NER-Jaccard + ROUGE + BLEU [+ PubMedBERT cos if available])
  Metric A = 0.05*BPV + 0.30*EdgeF1 + 0.25*MESS + 0.40*final_report
Reports both PREDICTED-organ (true end-to-end) and GT-organ (answer-head ceiling).

  python gow/walker/infer_score.py --cot data/train_CoT_v01.json --features-dir data/feats --sample 120
"""
import argparse, json, os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                  # gow_walker
sys.path.insert(0, os.path.join(HERE, "..", "heads"))    # gow_model, train_heads (TextEmb, ORGANS)
sys.path.insert(0, os.path.join(HERE, "..", "scorer"))   # local_scorer (vendored REG25 evaluator)
import gow_walker as W
import gow_model
from train_heads import TextEmb, ORGANS, O2I, MAX_CAND


def load_report_evaluator():
    try:
        from local_scorer import get_report_evaluator
        ev = get_report_evaluator()
        print("[report] full REG25 scorer (en_core_sci_lg + PubMedBERT)", file=sys.stderr)
        return ev, "full"
    except Exception as e:                                 # PubMedBERT not cached -> keyword-only
        sys.path.insert(0, os.path.join(HERE, "..", "scorer"))
        from evaluate_metrics import KeywordEvaluator
        kw = KeywordEvaluator("en_core_sci_lg")
        print(f"[report] keyword-only (PubMedBERT unavailable: {str(e)[:60]})", file=sys.stderr)
        return kw, "keyword"


def report_score(ev, kind, gt, pred):
    if kind == "full":
        d = ev.evaluate_text(gt, pred)                    # keys: final_report_score, emb/key/bleu/rouge_score
        return float(d.get("final_report_score", d.get("ranking_score", 0.0)))
    return float(ev.get_score(gt, pred))                  # KeywordEvaluator -> Jaccard (the 0.40 sub-weight term)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", required=True); ap.add_argument("--features-dir", required=True)
    ap.add_argument("--weights", default="gow/artifacts/gow_heads.pt")
    ap.add_argument("--sample", type=int, default=120); ap.add_argument("--holdout-mod", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    import torch

    T, QSURF, AV, META = W.load_artifacts()
    text = TextEmb("gow/artifacts/text_emb.npz")
    model = gow_model.build().to(args.device)
    model.load_state_dict(torch.load(args.weights, map_location=args.device)); model.eval()
    ev, kind = load_report_evaluator()

    labels = {os.path.splitext(c["id"])[0]: c for c in json.load(open(args.cot)) if "id" in c}
    have = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(args.features_dir, "*.npz"))}
    hold = [labels[s] for i, s in enumerate(sorted(labels)) if i % args.holdout_mod == 0 and s in have][: args.sample]
    print(f"scoring {len(hold)} held-out slides (device={args.device}, report={kind})", file=sys.stderr)

    def answer_fn_for(H, z, organ_idx):
        oh = torch.zeros(7, device=args.device); oh[organ_idx] = 1
        organ = ORGANS[organ_idx]
        def f(_org, cq):
            cands = [a for a, _ in sorted(AV.get(organ, {}).get(cq, {}).items(), key=lambda kv: -kv[1])][:MAX_CAND]
            if not cands:
                return ""
            q = torch.from_numpy(text(cq)).to(args.device)
            ce = torch.from_numpy(np.stack([text(c) for c in cands])).to(args.device)
            with torch.no_grad():
                lg, _ = model.answer(H, z, oh, q, ce)
            return cands[int(lg.argmax())]
        return f

    agg = {m: [0, 0, 0, 0, 0, 0.0, 0.0, 0] for m in ("pred_organ", "gt_organ")}  # tp,fp,fn,bpv,n, mess_sum, rep_sum, organ_ok
    for c in hold:
        sid = os.path.splitext(c["id"])[0]
        H = torch.from_numpy(np.load(os.path.join(args.features_dir, sid + ".npz"))["X"].astype(np.float32)).to(args.device)
        with torch.no_grad():
            o_logits, z = model.organ(H)
        pred_oi = int(o_logits.argmax()); gt_oi = O2I.get(c["organ"], pred_oi)
        gt_chain = c["chain-of-thought"]
        gt_edges = W.edge_set(gt_chain)
        gt_report = next((s["answer"] for s in gt_chain if W.canon(s.get("question", "")) == W.REPORT_Q), "")
        # GT edge->answer for MESS (non-final)
        gt_ea = {(W.canon(s["question"]), W.canon_next(s["next_question"])): s.get("answer", "")
                 for s in gt_chain if W.canon(s.get("question", "")) and W.canon_next(s.get("next_question", "")) != "__END__"}
        for tag, oi in (("pred_organ", pred_oi), ("gt_organ", gt_oi)):
            organ = ORGANS[oi]
            first_q = META.get(organ, {}).get("first_question_canon", W.ORGAN_Q)
            co = META.get(organ, {}).get("co_roots", [])
            chain, ans = W.walk(organ, answer_fn_for(H, z, oi), T, QSURF, first_q, use_renderer=True, co_roots=co)
            pe = W.edge_set(chain)
            a = agg[tag]
            a[0] += len(gt_edges & pe); a[1] += len(pe - gt_edges); a[2] += len(gt_edges - pe)
            a[3] += int(pe == gt_edges); a[4] += 1; a[7] += int(oi == gt_oi)
            # MESS: for each GT non-final edge present in pred, exact-match answer (conservative)
            pred_ea = {(W.canon(s["question"]), W.canon_next(s["next_question"])): s.get("answer", "")
                       for s in chain if W.canon_next(s.get("next_question", "")) != "__END__"}
            if gt_ea:
                a[5] += sum(1.0 for k, v in gt_ea.items() if k in pred_ea and W.canon(pred_ea[k]) == W.canon(v)) / len(gt_ea)
            a[6] += report_score(ev, kind, gt_report, W.render_report(ans)) if gt_report else 0.0

    print(f"\n{'setting':11} {'organ':>6} {'EdgeF1':>7} {'BPV':>6} {'MESS*':>6} {'Report':>7} {'MetricA':>8}")
    for tag in ("gt_organ", "pred_organ"):
        tp, fp, fn, bpv, n, mess, rep, ook = agg[tag]
        _, _, ef1 = W.prf(tp, fp, fn)
        mess, rep, bpv, ook = mess / n, rep / n, bpv / n, ook / n
        A = 0.05 * bpv + 0.30 * ef1 + 0.25 * mess + 0.40 * rep
        print(f"{tag:11} {ook:6.3f} {ef1:7.4f} {bpv:6.3f} {mess:6.3f} {rep:7.4f} {A:8.4f}")
    print(f"\n* MESS = exact-match proxy (conservative lower bound); Report={kind}. Held-out (mod {args.holdout_mod}); "
          f"some overlap with the CPU model's train set -> indicative, not final.", file=sys.stderr)


if __name__ == "__main__":
    main()
