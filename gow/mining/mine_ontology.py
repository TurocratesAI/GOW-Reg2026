#!/usr/bin/env python3
"""
Mine the REG2026 diagnostic ontology + transition table from train_CoT.

Produces the deterministic artifacts the Grounded Ontology Walker (interf1) rides:
  - questions.json        canonical_q -> {surface (modal raw string to EMIT), count, appears_as_source}
  - transitions.json      organ -> canon_q -> canon_answer -> [majority within-case successor SET]  (canonical keys)
  - answer_vocab.json     organ -> canon_q -> {answer_surface: count}  (label space per node; modal = to EMIT)
  - organ_meta.json       organ -> {first_question, question_set, n_cases}
  - mining_report.json    determinism + EdgeF1-ceiling self-checks

Edge/answer canonicalization mirrors the real scorer (submission_evaluation_code/evaluate_metrics.py):
  lower -> collapse whitespace -> strip trailing .,;:!? -> 4 typo aliases ; terminal next_question -> __END__.
No third-party deps (stdlib only).
"""
import json, os, re, sys, urllib.request
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.abspath(os.path.join(HERE, "..", "artifacts"))
os.makedirs(ART, exist_ok=True)

CANDIDATES = [
    os.path.join(HERE, "..", "..", "data", "train_CoT_v01.json"),
]
CDN = "https://d2ffc588b8gysg.cloudfront.net/train_CoT_v01.json"

TERMINAL = {"", "end", "stop", "finish", "finished", "none", "null",
            "no next question", "no further question"}
# 4 typo aliases applied by the scorer's canonicalize_text
ALIASES = [("pridominant", "predominant"), ("dianoses", "diagnoses"),
           ("diagnosises", "diagnoses"), ("includes", "include")]

def canon(s):
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[.,;:!?]+$", "", s).strip()
    for a, b in ALIASES:
        s = s.replace(a, b)
    return s

def canon_next(s):
    c = canon(s)
    return "__END__" if c in TERMINAL else c

def load_cot():
    for p in CANDIDATES:
        p = os.path.abspath(p)
        if os.path.exists(p):
            print(f"[load] {p}", file=sys.stderr)
            with open(p) as f:
                return json.load(f)
    dst = os.path.abspath(os.path.join(HERE, "..", "..", "data", "train_CoT_v01.json"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"[load] downloading {CDN} -> {dst}", file=sys.stderr)
    urllib.request.urlretrieve(CDN, dst)
    with open(dst) as f:
        return json.load(f)

def main():
    data = load_cot()
    n = len(data)

    q_surface = defaultdict(Counter)     # canon_q -> Counter(raw surface)
    q_as_source = Counter()              # canon_q -> #cases it is asked
    ans_surface = defaultdict(Counter)   # (organ, canon_q) -> Counter(raw answer surface)
    # transition: (organ, canon_q, canon_ans) -> Counter over FROZEN within-case successor SETS
    trans_setcounter = defaultdict(Counter)
    organ_first = defaultdict(Counter)   # organ -> Counter(first canon_q)
    organ_qset = defaultdict(set)
    organ_cases = Counter()
    skipped_empty_q = 0
    malformed = 0

    # scoring sanity structures
    tp = fp = fn = 0

    for case in data:
        organ = case.get("organ")
        chain = case.get("chain-of-thought") or []
        if not organ or not chain:
            malformed += 1
            continue
        organ_cases[organ] += 1

        # within-case: per (canon_q, canon_ans) node collect the SET of successors seen in THIS case
        node_succ = defaultdict(set)     # (canon_q, canon_ans) -> set(canon_next)
        node_ans_raw = {}                # canon_q -> raw answer surface (first seen)
        gt_edges = set()                 # (canon_q, canon_next) for this case
        first_q_raw = chain[0].get("question")
        organ_first[organ][canon(first_q_raw)] += 1

        for step in chain:
            q_raw = step.get("question", "")
            a_raw = step.get("answer", "")
            nq_raw = step.get("next_question", "")
            cq = canon(q_raw)
            if cq == "":
                skipped_empty_q += 1
                continue
            ca = canon(a_raw)
            cn = canon_next(nq_raw)
            q_surface[cq][q_raw.strip()] += 1
            q_as_source[cq] += 1
            organ_qset[organ].add(cq)
            ans_surface[(organ, cq)][a_raw.strip()] += 1
            node_succ[(cq, ca)].add(cn)
            node_ans_raw.setdefault(cq, a_raw.strip())
            gt_edges.add((cq, cn))

        # register the within-case successor SET for each node
        for (cq, ca), succset in node_succ.items():
            trans_setcounter[(organ, cq, ca)][frozenset(succset)] += 1

    # majority within-case successor SET per (organ,cq,ca)
    T = {}  # (organ,cq,ca) -> sorted list of successor canon keys
    identical_set_triples = 0
    for key, ctr in trans_setcounter.items():
        best_set, _ = ctr.most_common(1)[0]
        T[key] = sorted(best_set)
        if len(ctr) == 1:
            identical_set_triples += 1
    n_triples = len(trans_setcounter)

    # ---- EdgeF1 ceiling self-check: successor-SET walker given GT answers ----
    for case in data:
        organ = case.get("organ"); chain = case.get("chain-of-thought") or []
        if not organ or not chain:
            continue
        gt_edges = set()
        nodes = set()  # (cq, ca)
        for step in chain:
            cq = canon(step.get("question", ""))
            if cq == "":
                continue
            ca = canon(step.get("answer", ""))
            cn = canon_next(step.get("next_question", ""))
            gt_edges.add((cq, cn))
            nodes.add((cq, ca))
        pred_edges = set()
        for (cq, ca) in nodes:
            for s in T.get((organ, cq, ca), []):
                pred_edges.add((cq, s))
        tp += len(gt_edges & pred_edges)
        fp += len(pred_edges - gt_edges)
        fn += len(gt_edges - pred_edges)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # ---- serialize artifacts ----
    questions = {cq: {"surface": q_surface[cq].most_common(1)[0][0],
                      "count": q_as_source[cq],
                      "n_surface_variants": len(q_surface[cq])}
                 for cq in q_surface}
    # transitions: nested organ -> cq -> ca -> [successors]
    transitions = defaultdict(lambda: defaultdict(dict))
    for (organ, cq, ca), succ in T.items():
        transitions[organ][cq][ca] = succ
    answer_vocab = defaultdict(lambda: defaultdict(dict))
    for (organ, cq), ctr in ans_surface.items():
        answer_vocab[organ][cq] = dict(ctr.most_common())
    # co_roots: in-degree-0 questions (never a successor) present in ~all of the organ's cases,
    # e.g. `what is the procedure`. The walker must SEED these alongside the organ root, else BFS
    # never reaches them (their edge + report clause are dropped). Excludes low-support mining-noise
    # roots (e.g. an occasional `what is the grading system`). Support = sum of answer_vocab counts.
    def co_roots_for(org):
        root = organ_first[org].most_common(1)[0][0]
        succ = {s for adict in transitions[org].values() for ss in adict.values()
                for s in ss if s != "__END__"}
        out = []
        for cq in transitions[org]:
            if cq == root or cq in succ:
                continue
            if sum(answer_vocab[org][cq].values()) >= 0.9 * organ_cases[org]:
                out.append(cq)
        return sorted(out)

    organ_meta = {org: {"first_question_canon": organ_first[org].most_common(1)[0][0],
                        "first_question_surface": q_surface[organ_first[org].most_common(1)[0][0]].most_common(1)[0][0],
                        "n_cases": organ_cases[org],
                        "n_questions": len(organ_qset[org]),
                        "co_roots": co_roots_for(org),
                        "question_set": sorted(organ_qset[org])}
                  for org in organ_cases}
    report = {
        "n_cases": n,
        "n_question_types": len(questions),
        "n_transition_triples": n_triples,
        "identical_successor_set_fraction": round(identical_set_triples / n_triples, 4) if n_triples else None,
        "skipped_empty_question_steps": skipped_empty_q,
        "malformed_cases": malformed,
        "edgeF1_ceiling_successor_set_walker": {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)},
        "organ_case_counts": dict(organ_cases.most_common()),
    }

    def dump(name, obj):
        with open(os.path.join(ART, name), "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)

    dump("questions.json", questions)
    dump("transitions.json", {k: dict(v) for k, v in transitions.items()})
    dump("answer_vocab.json", {k: dict(v) for k, v in answer_vocab.items()})
    dump("organ_meta.json", organ_meta)
    dump("mining_report.json", report)

    print(json.dumps(report, indent=2))
    print(f"\n[ok] artifacts -> {ART}", file=sys.stderr)

if __name__ == "__main__":
    main()
