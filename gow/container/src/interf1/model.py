"""
Interface 1 - Workflow Reasoning. Our GOW pipeline:
  WSI -> GrandQC tissue tiles @20x -> Virchow2 bag [N,2560] -> organ router + answer heads
      -> deterministic walker over the mined ontology -> enriched CAP chain-of-thought.
No language model in the loop; answers use the precomputed CONCH text embeddings (text_emb.npz),
so interf1 needs NO CONCH at runtime.
"""
from __future__ import annotations
from pathlib import Path
from typing import TypedDict

import gowcfg                                                # sets sys.path + weight env (import FIRST)

# The per-case wall-clock bound lives cooperatively inside extract_slide/fast_thumbnail (a monotonic deadline
# checked at Python loop boundaries -- see gowsrc/extract). No signals in this process: a SIGALRM handler only
# runs at a main-thread bytecode boundary, so it fires tens of seconds LATE behind the long native CUDA/openslide
# calls and can land past the 5-min kill (no output -> whole submission fails). The cooperative deadline fires
# promptly and always leaves a valid chain: a real partial bag on a slow slide, or a fast modal-walk fallback.


class ChainOfThoughtStep(TypedDict):
    question: str
    answer: str
    next_question: str


_M: dict = {}


def _load():
    if _M:
        return _M
    import torch
    from types import SimpleNamespace
    import extract_features as EF
    import gow_model, gow_walker as W
    from train_heads import TextEmb
    import eval_real as ER
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = gow_model.build().to(dev)
    model.load_state_dict(torch.load(gowcfg.GOW_HEADS, map_location=dev))
    model.eval()
    T, QSURF, AV, META = W.load_artifacts()
    text = TextEmb(gowcfg.TEXT_EMB)
    virchow2 = EF.load_virchow2(gowcfg.VIRCHOW2, dev)
    args = SimpleNamespace(
        qc="grandqc", grandqc_no_artifact=True,              # tissue-only QC (train/inference consistency)
        grandqc_tissue=gowcfg.GRANDQC_TISSUE, grandqc_artifact=gowcfg.GRANDQC_ARTIFACT,
        # tiles + read parallelism EXACTLY as the 36-success image (max_tiles=6000, readers=8): identical
        # bag-mean / OOD gate / MIL, identical peak RAM and I/O concurrency. readers=24 gave no speedup (the
        # slow step, fast_thumbnail, hardcodes readers=12) and only added handle/thread contention on GC storage.
        tissue_frac=0.25, max_tiles=6000, batch_size=64, readers=8,
        tmpdir="/tmp/gow", dry_run=False,
    )
    _M.update(dev=dev, model=model, T=T, QSURF=QSURF, AV=AV, META=META,
              text=text, virchow2=virchow2, args=args, EF=EF, ER=ER, W=W)
    return _M


# Last-resort chain used ONLY if the model itself cannot load or every walker path fails. It is a valid,
# parseable chain (correct step keys + the canonical final-report question) so a single bad case can never
# crash the whole submission; it scores low but keeps the run alive for all the other cases.
SAFE_CHAIN: list = [
    {"question": "What is the organ?", "answer": "Prostate", "next_question": "What is the procedure?"},
    {"question": "What is the procedure?", "answer": "Needle core biopsy",
     "next_question": "Is there any abnormality present?"},
    {"question": "Is there any abnormality present?", "answer": "No, there is no abnormality.",
     "next_question": "What is the final pathology report?"},
    {"question": "What is the final pathology report?",
     "answer": "Prostate, needle core biopsy;\\n  Benign prostatic tissue.", "next_question": ""},
]


def _normalize_chain(chain):
    """GUARANTEE the exact submission structure before the chain leaves our code: a NON-EMPTY JSON
    array in which EVERY step is an object carrying EXACTLY the three string keys question / answer /
    next_question, AND with question + answer NON-EMPTY. Grand Challenge validates chain-of-thought.json
    against a schema that fails a case when a step is missing a key, holds a non-string (e.g. null), OR
    holds an EMPTY / whitespace-only question or answer (organizer-confirmed: those fields are not allowed
    to be empty). next_question is left untouched -- the terminal step legitimately carries "". Any empty
    question/answer is FILLED (never dropped, so the next_question linkage is preserved) with a valid
    non-empty placeholder, so an odd step scores low on one edge instead of failing the whole submission;
    edge sets are order-invariant for scoring, so this never lowers the score of a well-formed chain, on
    which it is a no-op. Every coercion is logged so a failing case's log names the exact culprit. Any
    non-dict element is dropped; an empty result falls back to SAFE_CHAIN."""
    def _s(v):
        return v if isinstance(v, str) else ("" if v is None else str(v))
    steps = []
    if isinstance(chain, list):
        for i, s in enumerate(chain):
            if not isinstance(s, dict):
                print(f"[interf1] normalize: dropping non-dict step {i}: {s!r}", flush=True)
                continue
            q  = _s(s.get("question", ""))
            a  = _s(s.get("answer", ""))
            nq = _s(s.get("next_question", ""))
            if not q.strip():
                print(f"[interf1] normalize: step {i} EMPTY question (orig={s.get('question')!r}) -> placeholder", flush=True)
                q = "What is the finding?"
            if not a.strip():
                print(f"[interf1] normalize: step {i} EMPTY answer (orig={s.get('answer')!r}) -> placeholder", flush=True)
                a = "Not specified"
            steps.append({"question": q, "answer": a, "next_question": nq})
    if not steps:
        print("[interf1] normalize: empty/invalid chain -> SAFE_CHAIN", flush=True)
        return [dict(s) for s in SAFE_CHAIN]
    return steps


def predict_chain_of_thought(*, wsi_path: Path) -> list[ChainOfThoughtStep]:
    """Public entry point (imported by inference.py). Runs inference, then passes the result through
    _normalize_chain so the written chain-of-thought.json ALWAYS has the exact 3-key step structure
    Grand Challenge validates -- a malformed edge case can never raise its "invalid structure" error."""
    return _normalize_chain(_predict_chain_raw(wsi_path=wsi_path))


def _fallback_chain(m):
    """Truly-blank slide (no tissue even after Otsu): emit a valid modal-organ chain so output is never empty."""
    W = m["W"]
    organ = "prostate"                                       # global modal organ
    first_q = m["META"].get(organ, {}).get("first_question_canon", W.ORGAN_Q)
    co = m["META"].get(organ, {}).get("co_roots", [])
    chain, _ = W.walk(organ, W.modal_answer_fn(m["AV"]), m["T"], m["QSURF"], first_q, use_renderer=True, co_roots=co)
    return chain or SAFE_CHAIN


def _predict_chain_raw(*, wsi_path: Path) -> list[ChainOfThoughtStep]:
    """Bulletproof: ANY failure (unreadable slide, OOM, walker/OOD edge case) degrades to a valid chain
    instead of crashing the case (Grand Challenge fails the whole submission if one case raises). Huge slides
    are bounded INSIDE extract_slide by a cooperative wall-clock deadline (no signals), so a slow slide returns
    a real partial bag or a fast fallback well under the 5-min platform kill."""
    import torch, traceback
    try:
        m = _load()
    except Exception as e:                                   # model/artifacts could not load -> stub chain
        print(f"[interf1] MODEL LOAD FAILED: {e!r} -> SAFE_CHAIN", flush=True); traceback.print_exc()
        return SAFE_CHAIN
    try:
        X, coords, meta = m["EF"].extract_slide(str(wsi_path), m["virchow2"], m["dev"], m["args"])
        print(f"[interf1] tiles={0 if X is None else len(X)} meta={meta}", flush=True)
        if X is None or len(X) == 0:
            return _fallback_chain(m)
        H = torch.from_numpy(X.astype("float32")).to(m["dev"])
        organ, chain = m["ER"].predict_chain(m["model"], H, m["T"], m["QSURF"], m["AV"], m["META"], m["text"], m["dev"])
        print(f"[interf1] organ={organ} steps={len(chain)}", flush=True)
        return chain if chain else _fallback_chain(m)
    except Exception as e:                                   # any slide/inference failure -> valid fallback chain
        print(f"[interf1] FAILED on {wsi_path}: {e!r} -> fast fallback", flush=True); traceback.print_exc()
        try:
            return _fallback_chain(m)
        except Exception as e2:
            print(f"[interf1] fallback also failed: {e2!r} -> SAFE_CHAIN", flush=True)
            return SAFE_CHAIN
