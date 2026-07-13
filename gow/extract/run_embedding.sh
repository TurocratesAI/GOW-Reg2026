#!/usr/bin/env bash
# Sharded GrandQC TISSUE-ONLY + Virchow2 embedding: 3 workers per GPU (6 total), streaming from S3.
# Tissue-only (no artifact stage) = one thumbnail read -> ~15% faster AND matches the shipped
# container's QC (train/inference consistency; the artifact stage is too slow for the 5-min/case A10G).
# Rare-histologic-type-first via data/embed_order.txt. Unbuffered (-u) so logs flush.
cd /home/swapnil/master/reg2026
PY=/home/swapnil/master/gleason/.venv/bin/python
mkdir -p logs data/feats data/_tmp0 data/_tmp1 data/_tmp2 data/_tmp3 data/_tmp4 data/_tmp5
C="gow/extract/extract_features.py --wsi-list data/embed_order.txt --out-dir data/feats \
   --qc grandqc --grandqc-no-artifact --readers 16 --batch-size 128 --max-tiles 6000"
for s in 0 2 4; do nohup $PY -u $C --shard $s/6 --device cuda:0 --tmpdir data/_tmp$s > logs/embed$s.log 2>&1 & echo "w$s cuda:0 pid $!"; done
for s in 1 3 5; do nohup $PY -u $C --shard $s/6 --device cuda:1 --tmpdir data/_tmp$s > logs/embed$s.log 2>&1 & echo "w$s cuda:1 pid $!"; done
echo "progress: ls data/feats/*.npz | wc -l   |   fallbacks: grep -c FB logs/embed*.log"
