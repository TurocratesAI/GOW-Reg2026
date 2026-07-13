#!/usr/bin/env python3
"""
Build gow/artifacts/cap_organs.json - a per-organ CAP vocabulary library for the OOD path.

Source: the official CAP cancer-protocol JSONs (public, used as a FIXED REFERENCE vocabulary only -
NOT training data; rules-clean, same footing as CONCH's shipped label lists). For each protocol we
extract the organ surface, procedures, diagnoses (histologic types/subtypes), and grade system+values.

These feed the OOD/unseen-organ path in two rules-safe ways:
  (1) CONCH open-vocab CANDIDATE SETS (organ id + diagnosis naming) -> answers (MESS) + report entities.
  (2) the CAP report line for a flagged organ.
Never the question/edge set.

  python gow/extract/build_cap_library.py --json-dir data/cap/json --out gow/artifacts/cap_organs.json
"""
import argparse, glob, json, os, re

DX_RE = re.compile(r"carcinoma|sarcoma|adenoma|blastoma|melanoma|lymphoma|neoplasm|tumou?r|"
                   r"mesothelioma|glioma|adenocarcinoma|leiomyoma|myoma|teratoma|seminoma|"
                   r"germ cell|hyperplasia|dysplasia|adenosis|papilloma|nevus|cyst\b|polyp", re.I)
GRADE_RE = re.compile(r"grade|gleason|figo|nottingham|differentiat|who/isup|g[1-4]\b|well |moderately |poorly ", re.I)
SKIP_LABEL = re.compile(r"reference|comment|explanatory|note|ajcc|amin |staging|"
                        r"^p[tnm] |category|margin|dimension|size|specify|laterality|"
                        r"lymph node|site|invasion|necrosis|finding|configuration|focality|"
                        r"treatment|biomarker|method|block|accession|specimen integrity", re.I)


def clean_organ(s):
    if not s:
        return ""
    s = re.split(r"[:(]", s)[0].strip()                      # "KIDNEY: Biopsy" / "UTERUS (SARCOMA" -> head
    s = re.sub(r"\s+", " ", s).strip()
    small = {"and", "or", "of", "the"}
    return " ".join(w if w in small else w.capitalize() for w in s.lower().split())


def opt_labels(fld):
    out = []
    for o in fld.get("options", []) or []:
        lab = (o.get("label") or o.get("value") or "") if isinstance(o, dict) else str(o)
        lab = re.sub(r":\s*_+.*$", "", lab).strip().rstrip(":").strip()   # drop "(specify): ____"
        if lab and lab.lower() not in ("other", "not specified", "not identified", "cannot be assessed",
                                       "not applicable", "none identified", "not known", "present"):
            out.append(lab)
    return out


def extract(d):
    organ = clean_organ(d.get("organ") or d.get("template_id", "").split("_")[0].split(".")[0])
    procedures, diagnoses, grades, grade_sys = [], [], [], None
    for sec in d.get("sections", []):
        for fld in sec.get("fields", []):
            if not isinstance(fld, dict):
                continue
            lab = (fld.get("label") or "").strip()
            if not lab or SKIP_LABEL.search(lab):
                continue
            labs = opt_labels(fld)
            if lab.lower().startswith("procedure"):
                procedures += labs
            elif GRADE_RE.search(lab):
                for x in labs:
                    grades.append(x)
                m = re.search(r"gleason|figo|nottingham|who/isup", lab, re.I)
                if m and not grade_sys:
                    grade_sys = m.group(0)
            else:
                diagnoses += [x for x in labs if DX_RE.search(x)]
    # de-dup, keep order
    dd = lambda xs: list(dict.fromkeys(xs))
    return organ, {"surface": organ, "procedures": dd(procedures),
                   "diagnoses": dd(diagnoses), "grade_system": grade_sys, "grades": dd(grades)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default="data/cap/json")
    ap.add_argument("--out", default="gow/artifacts/cap_organs.json")
    args = ap.parse_args()
    lib = {}
    for f in sorted(glob.glob(os.path.join(args.json_dir, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        organ, rec = extract(d)
        if not organ:
            continue
        if organ in lib:                                     # merge multiple protocols for one organ (Bx + resection)
            for k in ("procedures", "diagnoses", "grades"):
                lib[organ][k] = list(dict.fromkeys(lib[organ][k] + rec[k]))
            lib[organ]["grade_system"] = lib[organ]["grade_system"] or rec["grade_system"]
        else:
            lib[organ] = rec
    json.dump(lib, open(args.out, "w"), indent=1)
    print(f"[cap] {len(lib)} organs -> {args.out}")
    for o in sorted(lib):
        r = lib[o]
        print(f"  {o:26} dx={len(r['diagnoses']):3d}  proc={len(r['procedures']):2d}  "
              f"grades={len(r['grades']):2d}  sys={r['grade_system']}")


if __name__ == "__main__":
    main()
