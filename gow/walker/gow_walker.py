#!/usr/bin/env python3
"""
GOW fan-out walker + verbatim-CAP report renderer + structural scorer (edges #1 & #2).

Walker: from the organ root, at each question answer via a pluggable answer_fn, look up the
mined (organ,q,answer) -> majority SUCCESSOR SET, and emit one {question,answer,next_question}
edge per successor (the flattened-tree representation the GT uses). No model in the loop.

Renderer: assemble the final report from the collected chain answers (organ, procedure, #N
diagnoses) as the verbatim CAP leaf -- zero padding, literal '\\n' separator + 2-space indent.

Scorer: canonicalized directed-edge SETS -> EdgeF1 + BPV, exactly per the official scorer
(canon = lower + collapse-ws + strip trailing .,;:!? + 4 typo aliases; terminal -> __END__).

Usage:
    python gow/walker/gow_walker.py --cot <train_CoT_v01.json> [--holdout-mod 10]
"""
import argparse, json, os, re, sys
from collections import defaultdict

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
TERMINAL = {"", "end", "stop", "finish", "finished", "none", "null",
            "no next question", "no further question"}
ALIASES = [("pridominant", "predominant"), ("dianoses", "diagnoses"),
           ("diagnosises", "diagnoses"), ("includes", "include")]

ORGAN_Q, PROC_Q = "what is the organ", "what is the procedure"
REPORT_Q = "what is the final pathology report"
DIAG_QS = [f"what is the #{i} diagnosis" for i in (1, 2, 3, 4)]


def canon(s):
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s.lower()).strip()
    s = re.sub(r"[.,;:!?]+$", "", s).strip()
    for a, b in ALIASES:
        s = s.replace(a, b)
    return s


def canon_next(s):
    c = canon(s)
    return "__END__" if c in TERMINAL else c


def load_artifacts():
    L = lambda n: json.load(open(os.path.join(ART, n)))
    return L("transitions.json"), L("questions.json"), L("answer_vocab.json"), L("organ_meta.json")


# ---------------------------------------------------------------- scorer
def edge_set(chain):
    return {(canon(s.get("question", "")), canon_next(s.get("next_question", "")))
            for s in chain if canon(s.get("question", ""))}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------------------------------------------------------------- renderer
# Canonical grade-node questions folded into the #1-diagnosis clause of the CAP report.
GS_Q, GG_Q = "what is the gleason score", "what is the grade group"
PCT4_Q, TV_Q = "what is the percentage of gleason pattern 4", "what is the tumor volume"
TUB_Q = "what is the score for tubular differentiation"
NUC_Q, MIT_Q = "what is the score for nuclear pleomorphism", "what is the score for mitotic rate"
DYS_Q = "what is the grade of dysplasia"                      # colon/stomach adenoma -> "with low grade dysplasia"
MUSC_Q = "is there any muscularis propria present"           # bladder -> "Note) The specimen (does not) include(s) muscle proper."


def _enrich_first_diag(diag, ans):
    """Append the organ-specific CAP detail the GT report carries but the bare #1-diagnosis leaf omits.
    Every field is one the model already answered; gated on presence (only the malignant branch walks
    them), so benign cases get no spurious tokens. Matches the GT surface format exactly."""
    gs = ans.get(GS_Q)
    if gs:                                                    # PROSTATE: Gleason's score / grade group / vol
        parts = [f"Gleason's score {gs}"]
        gg = ans.get(GG_Q)
        if gg:
            gg = gg[:1].lower() + gg[1:]                      # "Grade group 2" -> "grade group 2"
            pct = ans.get(PCT4_Q)
            parts.append(f"{gg} (Gleason pattern 4: {pct})" if pct else gg)
        tv = ans.get(TV_Q)
        if tv:
            parts.append(f"tumor volume: {tv}")
        return f"{diag}, " + ", ".join(parts)
    tub, nuc, mit = ans.get(TUB_Q), ans.get(NUC_Q), ans.get(MIT_Q)
    if tub and nuc and mit:                                   # BREAST invasive: Nottingham parenthetical
        return f"{diag} (Tubule formation: {tub}, Nuclear grade: {nuc}, Mitoses: {mit})"
    dys = ans.get(DYS_Q)                                      # COLON/STOMACH adenoma: "with low grade dysplasia"
    if dys and "dysplasia" not in diag.lower():
        return f"{diag} with {dys[:1].lower() + dys[1:]} dysplasia"
    return diag


def render_report(ans):
    organ, proc = ans.get(ORGAN_Q, ""), ans.get(PROC_Q, "")
    diags = [ans[q] for q in DIAG_QS if ans.get(q)]
    if not organ:
        return ans.get(REPORT_Q, "") or "Not specified"   # never emit an empty report answer (GC schema rejects it)
    head = f"{organ}, {proc[:1].lower() + proc[1:]};" if proc else f"{organ};"
    bare = os.environ.get("GOW_BARE_REPORT", "").strip().lower() in ("1", "true", "yes", "on")  # A/B toggle; only truthy strings disable enrichment ("0" must NOT count)
    if diags and not bare:
        diags = [_enrich_first_diag(diags[0], ans)] + diags[1:]  # enrich the #1 diagnosis only
    if len(diags) <= 1:
        body = f"\\n  {diags[0]}" if diags else ""
    else:
        body = "".join(f"\\n  {i + 1}. {d}" for i, d in enumerate(diags))
    musc = ans.get(MUSC_Q) if not bare else None              # bladder TUR: muscle-proper adequacy note
    if musc:
        inc = "includes" if "yes" in musc.lower() else "does not include"
        body += f"\\nNote) The specimen {inc} muscle proper."
    return head + body


# ---------------------------------------------------------------- walker
def walk(organ, answer_fn, T, QSURF, first_q, use_renderer=True, co_roots=()):
    """Traverse the organ sub-DAG emitting the fan-out successor set per node.

    The GT chains are a MULTI-ROOT DAG: besides `first_q` (the organ node), `what is the
    procedure` is an in-degree-0 co-root (present in ~100% of cases, never a successor) that
    BFS would never reach from `first_q` alone -> its edge + report clause would be dropped.
    Seed the queue with first_q + the mined per-organ `co_roots` (organ_meta.json)."""
    chain, ans, seen = [], {}, set()
    queue = [first_q] + [c for c in co_roots if c != first_q]
    Torg = T.get(organ, {})
    while queue:
        cq = queue.pop(0)
        if cq in seen:
            continue
        seen.add(cq)
        a = render_report(ans) if (cq == REPORT_Q and use_renderer) else answer_fn(organ, cq)
        ans[cq] = a
        adict = Torg.get(cq, {})
        succ = adict.get(canon(a))
        if succ is None:                                # answer matched no mined transition:
            if adict:                                   # fall back to this question's MAJORITY successor
                from collections import Counter         # SET (keep walking) instead of dead-ending at __END__
                succ = list(Counter(tuple(v) for v in adict.values()).most_common(1)[0][0])
            else:
                succ = ["__END__"]
        qsurf = QSURF.get(cq, {}).get("surface", cq)
        for s in succ:
            chain.append({"question": qsurf, "answer": a,
                          "next_question": "" if s == "__END__" else QSURF.get(s, {}).get("surface", s)})
            if s != "__END__" and s not in seen:
                queue.append(s)
    if REPORT_Q not in seen:                       # every case MUST emit a report (hard requirement)
        a = render_report(ans) if use_renderer else answer_fn(organ, REPORT_Q)
        ans[REPORT_Q] = a
        chain.append({"question": QSURF.get(REPORT_Q, {}).get("surface", "What is the final pathology report?"),
                      "answer": a, "next_question": ""})
    if use_renderer:                               # re-render from the COMPLETE ans (grade nodes can be
        final = render_report(ans)                 # dequeued AFTER REPORT_Q in BFS)
        ans[REPORT_Q] = final
        # submission requires the LAST step to have next_question=""; make the (single) report node that
        # step. Edge sets are order-invariant, so the score is unchanged; this only fixes the format.
        report_surf = QSURF.get(REPORT_Q, {}).get("surface", "What is the final pathology report?")
        chain = [st for st in chain if canon(st["question"]) != REPORT_Q]
        chain.append({"question": report_surf, "answer": final, "next_question": ""})
    return chain, ans


def modal_answer_fn(AV):
    def f(organ, cq):
        d = AV.get(organ, {}).get(cq, {})
        return max(d, key=d.get) if d else ""      # by COUNT (json is dumped sort_keys=True, i.e. alphabetical)
    return f


def gt_answer_fn(case):
    m = {}
    for s in case["chain-of-thought"]:
        cq = canon(s.get("question", ""))
        if cq and cq not in m:
            m[cq] = s.get("answer", "")
    return lambda organ, cq: m.get(cq, "")


# ---------------------------------------------------------------- harness
def tok(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", required=True)
    ap.add_argument("--holdout-mod", type=int, default=10, help="hold out every Nth case")
    args = ap.parse_args()
    T, QSURF, AV, META = load_artifacts()
    modal = modal_answer_fn(AV)
    data = json.load(open(args.cot))
    hold = [c for i, c in enumerate(data) if i % args.holdout_mod == 0]

    agg = {}  # name -> [tp,fp,fn,bpv_ok,n, rep_exact, jacc_sum, rep_n]
    for name in ("oracle", "modal"):
        agg[name] = [0, 0, 0, 0, 0, 0, 0.0, 0]

    for c in hold:
        organ = c["organ"]
        gt_chain = c["chain-of-thought"]
        gt_edges = edge_set(gt_chain)
        first_q = META.get(organ, {}).get("first_question_canon", ORGAN_Q)
        co_roots = META.get(organ, {}).get("co_roots", [])
        gt_report = next((s["answer"] for s in gt_chain
                          if canon(s.get("question", "")) == REPORT_Q), "")
        runs = {
            "oracle": walk(organ, gt_answer_fn(c), T, QSURF, first_q, use_renderer=False, co_roots=co_roots),
            "modal": walk(organ, modal, T, QSURF, first_q, use_renderer=True, co_roots=co_roots),
        }
        for name, (chain, ans) in runs.items():
            pe = edge_set(chain)
            tp = len(gt_edges & pe); fp = len(pe - gt_edges); fn = len(gt_edges - pe)
            a = agg[name]
            a[0] += tp; a[1] += fp; a[2] += fn
            a[3] += int(pe == gt_edges); a[4] += 1
            rep = render_report(ans) if name == "modal" else ans.get(REPORT_Q, "")
            if gt_report:
                a[5] += int(rep == gt_report)
                gt_t = tok(gt_report.replace("\\n", " "))
                pr_t = tok(rep.replace("\\n", " "))
                a[6] += (len(gt_t & pr_t) / len(gt_t | pr_t)) if (gt_t | pr_t) else 0.0
                a[7] += 1

    print(f"held-out cases: {len(hold)}  (every {args.holdout_mod}th)\n")
    print(f"{'run':7} {'EdgeF1':>7} {'prec':>6} {'rec':>6} {'BPV':>6} "
          f"{'rep_exact':>9} {'rep_tokJacc':>11}")
    for name in ("oracle", "modal"):
        tp, fp, fn, bpv, n, rex, jac, rn = agg[name]
        p, r, f = prf(tp, fp, fn)
        print(f"{name:7} {f:7.4f} {p:6.3f} {r:6.3f} {bpv / n:6.3f} "
              f"{rex / max(rn,1):9.3f} {jac / max(rn,1):11.3f}")
    print("\nnote: oracle=GT answers (validates walker code); modal=organ-conditioned modal answers "
          "with GT organ (the embedding-free floor). Report token-Jaccard is a proxy for the "
          "en_core_sci_lg NER-Jaccard; real scorer wiring is the next step.")


if __name__ == "__main__":
    main()
