#!/usr/bin/env bash
# Stage the GOW source + weights into the container dir so `docker build` (context = this dir) is
# self-contained, and model/ holds the tarball-upload weights (mounted to /opt/ml/model at run time).
set -e
cd "$( dirname -- "${BASH_SOURCE[0]}" )"           # gow/container
GOW=".."                                            # gow/

echo "== bundle GOW source -> gowsrc/ =="
rm -rf gowsrc && mkdir -p gowsrc
cp -r "$GOW/extract" "$GOW/heads" "$GOW/walker" "$GOW/interf0" gowsrc/
cp "$GOW/data_split.py" gowsrc/
cp -r "$GOW/artifacts" gowsrc/                      # transitions/questions/answer_vocab/organ_meta/text_emb (small)
rm -f gowsrc/artifacts/*.pt gowsrc/artifacts/gow_heads*.pt 2>/dev/null || true   # weights go in model/, not the image
# Keep ONLY the runtime artifacts. Drop research/eval caches -- ood_feats_cache.npz and the enriched eval JSONs
# contain BIOBANK-derived features and must NEVER be shipped in the image or published. Whitelist (not blacklist)
# so a new eval/cache file can never sneak in.
find gowsrc/artifacts -type f \
  ! -name text_emb.npz ! -name ood_gate.npz ! -name transitions.json ! -name questions.json \
  ! -name answer_vocab.json ! -name organ_meta.json ! -name cap_organs.json ! -name split.json \
  -delete 2>/dev/null || true
find gowsrc -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "== stage weights -> model/ =="
mkdir -p model/conch model/grandqc model/virchow2
cp /home/swapnil/master/screening_v1_container/weights/conch/conch_v1.pt        model/conch/conch_v1.pt
cp /home/swapnil/master/qc/wsiqc/models/tissue_detection_mpp10.pth              model/grandqc/tissue_detection_mpp10.pth
cp /home/swapnil/master/qc/wsiqc/models/grandqc_artifact_mpp15_turoquant.pth    model/grandqc/grandqc_artifact_mpp15.pth
cp "$GOW/artifacts/gow_heads_final.pt"                                          model/gow_heads.pt

# Virchow2 loads via timm `hf-hub:paige-ai/Virchow2` which needs config.json + model.safetensors from the
# HF cache. Bundle the whole cache entry -> model/hf_cache/hub/ ; the container sets HF_HOME there so
# timm.create_model(pretrained=True) resolves fully offline (HF_HUB_OFFLINE=1).
V2DIR="$HOME/.cache/huggingface/hub/models--paige-ai--Virchow2"
if [ -d "$V2DIR" ]; then
    mkdir -p model/hf_cache/hub
    rm -rf model/hf_cache/hub/models--paige-ai--Virchow2      # idempotent: allow re-staging over an existing tree
    # -L dereferences symlinks: the source cache (and its snapshot->blob links) are symlinks to host paths that
    # do NOT exist inside the container, so we must copy REAL files or the offline Virchow2 load fails.
    cp -rL "$V2DIR" model/hf_cache/hub/
    rmdir model/virchow2 2>/dev/null || true
    echo "  Virchow2 HF cache -> model/hf_cache/hub/models--paige-ai--Virchow2/"
else
    echo "  !! Virchow2 cache NOT found - resolve before offline build"
fi

echo "== staged =="
du -sh gowsrc model 2>/dev/null
ls -la model/*/ model/*.pt 2>/dev/null | awk '{print "  ",$5,$NF}'
