#!/usr/bin/env python3
"""
CONCH zero-shot organ naming with the official recipe (measured best, ultracode-validated): per organ, build a
prototype by averaging (organ name + its real CAP diagnoses) x the 22 official CONCH templates, L2-normalize,
mean-center the prototype matrix; score a slide by mean-pooled tile cosine, argmax. Iterates on the cached tile
embeddings (gow/eval/cache_conch_tiles.py). Reports open-vocab (all CLEAN_SITES) and constrained accuracy.

  python gow/eval/mizero_name.py --dir data/tcga_ood --device cuda:1
"""
import argparse, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "heads"))
import conch_image_id as CI
import ood_route as OR


def build_protos(cmodel, tok, dev, sites, use_dx=True, center=True):
    """Prototype per site = mean over (site name + its CAP diagnoses) x 22 CONCH templates; then mean-center."""
    protos = []
    for s in sites:
        prompts = [s] + (OR.cap_diagnoses(s)[:40] if use_dx else [])
        e = CI.encode_text(cmodel, tok, prompts, dev, CI.TEMPLATES)   # [P,512], each template-averaged + normed
        v = e.mean(0); protos.append(v / (np.linalg.norm(v) + 1e-8))
    P = np.stack(protos)
    if center:
        P = P - P.mean(0); P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    return P


def score(tile_emb, protos, topk=0):
    S = tile_emb @ protos.T
    if topk and topk < S.shape[0]:
        return np.sort(S, axis=0)[-topk:].mean(0)
    return S.mean(0)                                                  # mean-pool (best on single-organ slides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/tcga_ood")
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev = a.device
    print("[mizero] loading CONCH + building CAP-grounded prototypes (22-template ensemble) ...")
    cmodel, _, tok = CI.load_conch(dev)

    caches = sorted(glob.glob(os.path.join(a.dir, "conch_cache", "*.npz")))
    data = [(np.load(c, allow_pickle=True)) for c in caches]
    embs = [d["emb"].astype("float32") for d in data]
    embs = [e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8) for e in embs]
    known = [str(d["organ"]) for d in data]

    # constrained candidate set = the true organs present (title-cased to match CLEAN_SITES style)
    def to_site(o):
        for s in CI.CLEAN_SITES:
            if o.split()[0].lower() in s.lower() or s.lower() in o.lower():
                return s
        return o.title()
    constrained = sorted({to_site(o) for o in known})
    open_sites = CI.CLEAN_SITES

    for label, sites in [("open-vocab (47 sites)", open_sites), (f"constrained ({len(constrained)} organs)", constrained)]:
        P = build_protos(cmodel, tok, dev, sites)
        hit = 0
        rows = []
        for e, o in zip(embs, known):
            s = score(e, P, a.topk)
            pred = sites[int(s.argmax())]
            ok = o.split()[0].lower() in pred.lower() or pred.lower() in o.lower()
            hit += int(ok); rows.append((o, pred, ok))
        print(f"\n[{label}]  accuracy = {hit}/{len(known)}")
        for o, pred, ok in rows:
            print(f"    {o:12} -> {pred:20} {'<-hit' if ok else ''}")


if __name__ == "__main__":
    main()
