#!/usr/bin/env python3
"""
HONEST Metric-A: trained GOW heads -> walker -> chain-of-thought, scored by the VENDORED official
scorer (gow/scorer/evaluate_metrics.evaluate_workflow_dataset) so our number matches the leaderboard
grader exactly (BPV .05 / EdgeF1 .30 / MESS .25 / final_report .40; MESS+report = PubMedBERT, report
keyword-Jaccard = scispaCy en_core_sci_lg). Held-out = the shared manifest's test (or val) stems.

Runs everything on CPU by default (CUDA_VISIBLE_DEVICES="") so it never contends with the GPU
embedding/training jobs. Reports overall + per-organ + per-source (for the scanner-shift read).

  CUDA_VISIBLE_DEVICES="" python gow/walker/eval_real.py --heads gow/artifacts/gow_heads_v2.pt \
      --eval-split test --n 400 --out gow/artifacts/eval_v2.json
"""
import argparse, os, sys, json, glob, io, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # gow_walker
sys.path.insert(0, os.path.join(HERE, "..", "heads"))     # gow_model, train_heads
sys.path.insert(0, os.path.join(HERE, ".."))              # data_split
sys.path.insert(0, os.path.join(HERE, "..", "scorer"))    # evaluate_metrics
import gow_walker as W
import gow_model
import data_split as DS
from train_heads import TextEmb, ORGANS, MAX_CAND
EMB_MODEL = "NeuML/pubmedbert-base-embeddings"


def predict_chain(model, H, T, QSURF, AV, META, text, dev, use_ood=True):
    """One held-out bag -> (organ, predicted chain-of-thought) using the trained heads + walker.
    If the OOD gate flags a non-7-organ slide, route through the gyn spine + open-vocab naming instead
    of the force-argmax mis-route (edges stay in-ontology; names come from OOD_VOCAB)."""
    import torch
    import ood_route as OR
    with torch.inference_mode():
        o_logits, z = model.organ(H)
    oi = int(o_logits.argmax()); organ = ORGANS[oi]
    org_vec = torch.softmax(o_logits, -1)                  # matches training conditioning

    is_ood = False
    gate = OR.get_gate() if use_ood else None
    if gate is not None:
        # Generalized OOD routing (edge-safe, ALL organs): a slide whose per-organ (QDA) Mahalanobis distance
        # exceeds its routed organ's threshold is flagged as not-one-of-the-7. It KEEPS its nearest-trained
        # topology (so every emitted edge stays an in-ontology string), and only the organ + diagnosis NAMING
        # opens to the full CAP open-vocab library. A false flag can therefore only mis-name, never change an edge.
        if gate.should_route_ood(organ, H.float().mean(0).cpu().numpy(), org_vec):
            is_ood = True                                   # keep `organ` = the nearest trained topology

    ood_named = {"organ": None}                             # CAP organ chosen at the organ node -> its diagnoses

    def afn(org, cq):
        d = AV.get(org, {}).get(cq, {})
        cands = [a for a, _ in sorted(d.items(), key=lambda kv: -kv[1])][:MAX_CAND]
        if is_ood:                                          # open-vocab NAME the OOD organ + diagnosis from CAP,
            if cq == W.ORGAN_Q:                             # but KEEP the in-dist candidates LAST so a FALSE flag
                cands = list(dict.fromkeys(OR.OOD_ORGANS + OR.cap_organ_names() + cands))  # on a trained organ
            elif cq in W.DIAG_QS or "histologic type" in cq:                               # still names it right;
                cands = list(dict.fromkeys(OR.OOD_DX + OR.cap_diagnoses(ood_named["organ"]) + cands))  # a genuine
                #                                             OOD organ still has the full gyn+CAP open-vocab to win
        if not cands:
            return "Not specified"          # never emit an empty answer (GC schema rejects an empty question/answer)
        cemb = torch.from_numpy(np.stack([text(x) for x in cands])).to(dev)
        qemb = torch.from_numpy(text(cq)).to(dev)
        with torch.inference_mode():
            logits, _ = model.answer(H, z, org_vec, qemb, cemb)
        ans = cands[int(logits.argmax())]
        if is_ood and cq == W.ORGAN_Q:
            ood_named["organ"] = ans                        # scope the diagnosis candidates to this organ
        return ans

    first_q = META.get(organ, {}).get("first_question_canon", W.ORGAN_Q)
    co_roots = META.get(organ, {}).get("co_roots", [])
    chain, _ = W.walk(organ, afn, T, QSURF, first_q, use_renderer=True, co_roots=co_roots)
    return organ, chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="gow/artifacts/gow_heads_v2.pt")
    ap.add_argument("--text-emb", default="gow/artifacts/text_emb.npz")
    ap.add_argument("--cot", default="data/train_CoT_v01.json")
    ap.add_argument("--split", default=DS.DEFAULT_SPLIT)
    ap.add_argument("--eval-split", default="test", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=400, help="cap held-out cases (speed)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="gow/artifacts/eval_real.json")
    args = ap.parse_args()
    import torch
    dev = args.device

    T, QSURF, AV, META = W.load_artifacts()
    text = TextEmb(args.text_emb)
    model = gow_model.build().to(dev)
    model.load_state_dict(torch.load(args.heads, map_location=dev)); model.eval()
    cot = {os.path.splitext(c["id"])[0]: c for c in json.load(open(args.cot)) if "id" in c}
    stem_split = DS.load(args.split)

    bags = [b for b in sorted(glob.glob("data/feats/*.npz"))
            if os.path.splitext(os.path.basename(b))[0] in cot
            and DS.split_of(b, stem_split) == args.eval_split][: args.n]
    print(f"[eval_real] {len(bags)} held-out ({args.eval_split}) bags -> walking with {args.heads}")

    gt_cases, pred_cases, meta = [], [], {}
    for npz in bags:
        sid = os.path.splitext(os.path.basename(npz))[0]
        c = cot[sid]
        H = torch.from_numpy(np.load(npz)["X"].astype(np.float32)).to(dev)
        organ, chain = predict_chain(model, H, T, QSURF, AV, META, text, dev)
        gt_cases.append({"id": sid, "organ": c["organ"], "chain-of-thought": c["chain-of-thought"]})
        pred_cases.append({"id": sid, "chain-of-thought": chain})
        meta[sid] = {"gt_organ": c["organ"], "pred_organ": organ, "source": DS.source(sid)}

    print("[eval_real] loading vendored scorer (PubMedBERT + en_core_sci_lg)...")
    from evaluate_metrics import (SemanticScorer, REG25FinalReportEvaluator,
                                  evaluate_workflow_dataset)
    scorer = SemanticScorer(backend="embedding", embedding_model_name=EMB_MODEL)
    rep_eval = REG25FinalReportEvaluator(embedding_model=EMB_MODEL, spacy_model="en_core_sci_lg")

    print("[eval_real] scoring (this prints per-case internals -> suppressed)...")
    with contextlib.redirect_stdout(io.StringIO()):
        summary = evaluate_workflow_dataset(gt_cases, pred_cases, scorer, rep_eval)

    A = summary["final_ranking_score"]
    print("\n" + "=" * 64)
    print(f"  REAL METRIC-A (vendored scorer) on {summary['num_ground_truth_cases']} {args.eval_split} cases")
    print("=" * 64)
    print(f"  Metric-A (final_ranking_score) : {A:.4f}")
    print(f"    0.05 * BPV        {summary['average_binary_path_validity']:.4f}")
    print(f"    0.30 * EdgeF1     {summary['average_edge_f1']:.4f}   (P {summary['average_edge_precision']:.3f} / R {summary['average_edge_recall']:.3f})")
    print(f"    0.25 * MESS       {summary['average_mess_nonfinal']:.4f}")
    print(f"    0.40 * report     {summary['average_final_report_score']:.4f}")
    organ_ok = sum(meta[cs['case_id']]['pred_organ'] == meta[cs['case_id']]['gt_organ'] for cs in summary['per_case']) / max(len(summary['per_case']), 1)
    print(f"  organ accuracy                 : {organ_ok:.4f}")

    # per-organ + per-source Metric-A (group the per_case ranking scores)
    def group(keyfn, title):
        from collections import defaultdict
        g = defaultdict(list)
        for cs in summary["per_case"]:
            g[keyfn(cs["case_id"])].append(cs)
        print(f"\n  {title:10} {'n':>4} {'MetricA':>8} {'EdgeF1':>7} {'MESS':>6} {'report':>7} {'organ':>6}")
        for k in sorted(g):
            rows = g[k]; nn = len(rows)
            ma = np.mean([r["ranking_score"] for r in rows])
            ef = np.mean([r["edge_f1"] for r in rows])
            ms = np.mean([r["mess_nonfinal"] for r in rows])
            rp = np.mean([r["final_report_score"] for r in rows])
            oa = np.mean([meta[r["case_id"]]["pred_organ"] == meta[r["case_id"]]["gt_organ"] for r in rows])
            print(f"  {k:10} {nn:4d} {ma:8.4f} {ef:7.3f} {ms:6.3f} {rp:7.3f} {oa:6.3f}")

    group(lambda sid: meta[sid]["gt_organ"], "organ")
    group(lambda sid: meta[sid]["source"], "source")

    summary["_meta"] = meta; summary["_organ_accuracy"] = organ_ok
    json.dump({k: v for k, v in summary.items() if k != "per_case"} | {"per_case": summary["per_case"]},
              open(args.out, "w"), indent=1)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
