#!/usr/bin/env python3
"""
Fetch a small PUBLIC TCGA diagnostic-slide sample (open access, no token) spanning organs OUTSIDE the 7 trained,
for the paper's reproducible external OOD validation. Inference/validation only; never used for training.
Picks the smallest diagnostic SVS per organ to keep downloads light. Writes data/tcga_ood/<organ>.svs + manifest.

  python gow/eval/tcga_fetch.py
"""
import json, os
import requests

# organs OUTSIDE our 7 (prostate/breast/colon/stomach/bladder/lung/cervix) -> a TCGA project each
PROJECTS = {
    "brain": "TCGA-GBM", "kidney": "TCGA-KIRC", "thyroid": "TCGA-THCA", "liver": "TCGA-LIHC",
    "pancreas": "TCGA-PAAD", "skin": "TCGA-SKCM", "ovary": "TCGA-OV", "head and neck": "TCGA-HNSC",
    "adrenal": "TCGA-ACC", "esophagus": "TCGA-ESCA",
}
GDC = "https://api.gdc.cancer.gov"
OUT = "data/tcga_ood"
os.makedirs(OUT, exist_ok=True)


def smallest_slide(project):
    filt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
        {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}}]}
    r = requests.get(f"{GDC}/files", params={"filters": json.dumps(filt), "fields": "file_id,file_name,file_size",
                                             "size": "1", "sort": "file_size:asc", "format": "json"}, timeout=60)
    hits = r.json()["data"]["hits"]
    return hits[0] if hits else None


def main():
    manifest = []
    for organ, proj in PROJECTS.items():
        h = smallest_slide(proj)
        if not h:
            print(f"  {organ}: no diagnostic slide"); continue
        manifest.append({"organ": organ, "project": proj, "file_id": h["file_id"],
                         "file_name": h["file_name"], "size_mb": int(h["file_size"]) // 10**6})
        print(f"  {organ:14} {proj:12} {h['file_id']}  {int(h['file_size'])//10**6} MB")
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print(f"\n[tcga] downloading {len(manifest)} slides (open access) ...")
    for m in manifest:
        dst = os.path.join(OUT, m["organ"].replace(" ", "_") + ".svs")
        if os.path.exists(dst) and os.path.getsize(dst) > 1e6:
            print(f"  have {dst}"); continue
        with requests.get(f"{GDC}/data/{m['file_id']}", stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
        print(f"  {m['organ']}: {os.path.getsize(dst)//10**6} MB -> {dst}")
    print("[tcga] done")


if __name__ == "__main__":
    main()
