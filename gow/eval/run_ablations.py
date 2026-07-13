#!/usr/bin/env python3
"""
Component-contribution ablation on the clean held-out split, scored by the vendored REG scorer.

All rows are evaluated on the SAME held-out bags with the SAME scorer (loaded once), so they are directly
comparable. We keep the OOD module OFF here: the held-out split is in-distribution (no uterus), so the OOD
path can only cost there; its benefit is reported separately (leave-one-organ-out + the cervix-bucket figure).
Rows isolate: heads vs modal answers, report enrichment, co-root seeding, and the successor-set fan-out.

  python gow/eval/run_ablations.py --heads gow/artifacts/gow_heads_v3.pt --n 400 --device cuda:0
"""
import argparse, os, sys, io, contextlib, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "walker"))
sys.path.insert(0, os.path.join(HERE, "..", "heads"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "scorer"))
import gow_walker as W
import gow_model
import data_split as DS
from train_heads import TextEmb, ORGANS, MAX_CAND
EMB_MODEL = "NeuML/pubmedbert-base-embeddings"


def truncate_to_single(T):
    """Successor-set -> single majority successor (ablates the fan-out walker)."""
    out = {}
    for org, cqd in T.items():
        out[org] = {}
        for cq, cad in cqd.items():
            out[org][cq] = {ca: (list(succ)[:1] if succ else succ) for ca, succ in cad.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="gow/artifacts/gow_heads_v3.pt")
    ap.add_argument("--cot", default="data/train_CoT_v01.json")
    ap.add_argument("--features-dir", default="data/feats")
    ap.add_argument("--text-emb", default="gow/artifacts/text_emb.npz")
    ap.add_argument("--split", default="gow/artifacts/split.json")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--csv", default="gow/artifacts/ablations.csv")
    ap.add_argument("--tex", default="paper/tables/ablation.tex")
    args = ap.parse_args()
    import torch, json, glob

    dev = args.device
    T, QSURF, AV, META = W.load_artifacts()
    T_single = truncate_to_single(T)
    text = TextEmb(args.text_emb)
    model = gow_model.build().to(dev)
    model.load_state_dict(torch.load(args.heads, map_location=dev)); model.eval()
    cot = {os.path.splitext(c["id"])[0]: c for c in json.load(open(args.cot)) if "id" in c}
    stem_split = DS.load(args.split)
    bags = [b for b in sorted(glob.glob(os.path.join(args.features_dir, "*.npz")))
            if os.path.splitext(os.path.basename(b))[0] in cot
            and DS.split_of(b, stem_split) == args.eval_split][: args.n]
    print(f"[ablate] {len(bags)} held-out bags | heads={args.heads} | dev={dev}")

    # precompute per-bag: features, pooled z, organ softmax, predicted organ (config-independent)
    pre = []
    for npz in bags:
        sid = os.path.splitext(os.path.basename(npz))[0]
        H = torch.from_numpy(np.load(npz)["X"].astype(np.float32)).to(dev)
        with torch.inference_mode():
            o_logits, z = model.organ(H)
        org_vec = torch.softmax(o_logits, -1)
        pre.append((sid, cot[sid], H, z, org_vec, ORGANS[int(o_logits.argmax())]))

    def heads_afn(H, z, org_vec):
        def afn(org, cq):
            d = AV.get(org, {}).get(cq, {})
            cands = [a for a, _ in sorted(d.items(), key=lambda kv: -kv[1])][:MAX_CAND]
            if not cands:
                return ""
            cemb = torch.from_numpy(np.stack([text(x) for x in cands])).to(dev)
            qemb = torch.from_numpy(text(cq)).to(dev)
            with torch.inference_mode():
                logits, _ = model.answer(H, z, org_vec, qemb, cemb)
            return cands[int(logits.argmax())]
        return afn

    modal_afn = W.modal_answer_fn(AV)

    def walk_cases(answer, coroot, succ, enrich):
        if enrich:
            os.environ.pop("GOW_BARE_REPORT", None)          # enriched = env unset (must NOT be "0", which is truthy)
        else:
            os.environ["GOW_BARE_REPORT"] = "1"
        Tuse = T if succ == "set" else T_single
        gt_cases, pred_cases, organ_ok = [], [], []
        for sid, c, H, z, org_vec, organ in pre:
            afn = heads_afn(H, z, org_vec) if answer == "heads" else modal_afn
            first_q = META.get(organ, {}).get("first_question_canon", W.ORGAN_Q)
            co = META.get(organ, {}).get("co_roots", []) if coroot else []
            chain, _ = W.walk(organ, afn, Tuse, QSURF, first_q, use_renderer=True, co_roots=co)
            gt_cases.append({"id": sid, "organ": c["organ"], "chain-of-thought": c["chain-of-thought"]})
            pred_cases.append({"id": sid, "chain-of-thought": chain})
            organ_ok.append(organ == c["organ"])
        return gt_cases, pred_cases, float(np.mean(organ_ok))

    print("[ablate] loading vendored scorer (PubMedBERT + en_core_sci_lg)...")
    from evaluate_metrics import SemanticScorer, REG25FinalReportEvaluator, evaluate_workflow_dataset
    scorer = SemanticScorer(backend="embedding", embedding_model_name=EMB_MODEL)
    rep_eval = REG25FinalReportEvaluator(embedding_model=EMB_MODEL, spacy_model="en_core_sci_lg")

    CONFIGS = [
        ("Modal chain (predicted organ)", dict(answer="modal", coroot=True, succ="set", enrich=True)),
        ("GOW (full)",                    dict(answer="heads", coroot=True, succ="set", enrich=True)),
        ("GOW without report enrichment", dict(answer="heads", coroot=True, succ="set", enrich=False)),
        ("GOW without co-root seeding",   dict(answer="heads", coroot=False, succ="set", enrich=True)),
        ("GOW without successor-set walk", dict(answer="heads", coroot=True, succ="single", enrich=True)),
    ]
    rows = []
    for label, cfg in CONFIGS:
        gt, pred, oacc = walk_cases(**cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            s = evaluate_workflow_dataset(gt, pred, scorer, rep_eval)
        bpv, ef1 = s["average_binary_path_validity"], s["average_edge_f1"]
        mess, rep = s["average_mess_nonfinal"], s["average_final_report_score"]
        A = 0.05 * bpv + 0.30 * ef1 + 0.25 * mess + 0.40 * rep
        rows.append(dict(label=label, edge_f1=ef1, mess=mess, report=rep, bpv=bpv, metric_a=A, organ=oacc))
        print(f"  {label:34} EdgeF1 {ef1:.3f}  ans {mess:.3f}  report {rep:.3f}  MetricA {A:.3f}")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    def tex_row(r):
        return f"{r['label']} & {r['edge_f1']:.3f} & {r['mess']:.3f} & {r['report']:.3f} \\\\"
    tex = ["% Generated by gow/eval/run_ablations.py on the held-out split. Do not hand-edit.",
           "\\begin{table}[t]", "\\centering",
           f"\\caption{{Component contribution on the held-out split (n={len(bags)}), with the OOD module off "
           "(its effect is reported separately). Each row removes or replaces one component.}",
           "\\label{tab:ablation}", "\\begin{tabular}{lccc}", "\\toprule",
           "Configuration & Edge F1 & Answer sim. & Report \\\\", "\\midrule"]
    tex += [tex_row(r) for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    os.makedirs(os.path.dirname(args.tex), exist_ok=True)
    open(args.tex, "w").write("\n".join(tex))
    print(f"[saved] {args.csv}  and  {args.tex}")


if __name__ == "__main__":
    main()
