#!/usr/bin/env python3
"""
Validate the OOD gate + CONCH-image naming on REAL novel organs (public slides, organs outside our 7).
For each slide: Virchow2 embed -> organ router (the mis-route) -> QDA OOD gate (flagged?) -> CONCH-image organ+dx.
COMPLIANT: validation only, no training. Reports OOD recall + naming accuracy vs the known organ.

  python gow/eval/bench_ood.py --dir data/tcga_ood --device cuda:0
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for s in ("extract", "walker", "heads"):
    sys.path.insert(0, os.path.join(ROOT, s))
import extract_features as EF, wsi_io, gow_model, ood_route as OR, conch_image_id as CI
from train_heads import TextEmb, ORGANS

TW = "/home/swapnil/master/qc/wsiqc/models/tissue_detection_mpp10.pth"
AW = "/home/swapnil/master/qc/wsiqc/models/grandqc_artifact_mpp15_turoquant.pth"


def main():
    import torch, openslide
    from types import SimpleNamespace
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/tcga_ood")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    os.makedirs("/tmp/gowbench", exist_ok=True)
    eargs = SimpleNamespace(qc="grandqc", grandqc_no_artifact=True, grandqc_tissue=TW, grandqc_artifact=AW,
                            tissue_frac=0.25, max_tiles=4000, batch_size=64, readers=8, tmpdir="/tmp/gowbench", dry_run=False)
    print("[bench] loading Virchow2 + heads + OOD gate + CONCH ...")
    virchow2 = EF.load_virchow2(None, dev)
    model = gow_model.build().to(dev); model.load_state_dict(torch.load(os.path.join(ROOT, "artifacts/gow_heads_final.pt"), map_location=dev)); model.eval()
    gate = OR.get_gate()
    text = TextEmb(os.path.join(ROOT, "artifacts/text_emb.npz"))
    cmodel, preprocess, tok = CI.load_conch(dev)

    picks = {p["organ"].replace(" ", "_"): p for p in json.load(open(os.path.join(a.dir, "manifest.json")))}
    print(f"\n{'known organ':12} {'router':9} {'OOD?':5} {'score':>6}  {'CONCH-img organ':16} {'CONCH dx':28}")
    n_flag = n_name = n = 0
    for key, p in picks.items():
        slide = os.path.join(a.dir, key + ".svs")
        if not os.path.exists(slide):
            continue
        X, coords, meta = EF.extract_slide(slide, virchow2, dev, eargs)
        if X is None or len(X) == 0:
            print(f"{p['organ']:12} (no tissue)"); continue
        mean = X.mean(0)
        H = torch.from_numpy(X.astype("float32")).to(dev)
        with torch.inference_mode():
            o_logits, _ = model.organ(H)
        raw_organ = ORGANS[int(o_logits.argmax())]
        org_vec = torch.softmax(o_logits, -1)
        score = gate.score(mean)
        flagged = gate.should_route_ood(raw_organ, mean, org_vec)
        resolved, tmp = wsi_io.resolve_tiled(slide, "/tmp/gowbench")
        s = openslide.OpenSlide(resolved)
        try:
            mpp = float(s.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.5) or 0.5)
        except Exception:
            mpp = 0.5
        s.close()
        v = CI.slide_embedding(resolved, np.asarray(coords), mpp, cmodel, preprocess, dev)
        site, dx, _ = CI.identify(v, cmodel, tok, dev)
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
        # naming hit if the known organ keyword is in the CONCH site (loose)
        hit = p["organ"].split()[0].lower() in site.lower() or site.lower() in p["organ"].lower()
        n += 1; n_flag += int(flagged); n_name += int(hit)
        print(f"{p['organ']:12} {raw_organ:9} {str(flagged):5} {score:6.0f}  {site:16} {dx[:28]:28} {'<-hit' if hit else ''}")
    print(f"\n[bench] OOD recall = {n_flag}/{n} flagged as OOD  |  CONCH-image organ hit = {n_name}/{n}")


if __name__ == "__main__":
    main()
