#!/usr/bin/env bash
# Auto-start head training once enough bags are embedded. Polls every 5 min; launches once; exits.
#   nohup bash gow/train_when_ready.sh 1500 cuda:1 >/dev/null 2>&1 &
cd /home/swapnil/master/reg2026
PY=/home/swapnil/master/gleason/.venv/bin/python
COT=data/train_CoT_v01.json
THRESHOLD=${1:-1500}
DEV=${2:-cuda:1}
LOG=logs/train_watch.log
mkdir -p logs
echo "$(date) watcher up: threshold=$THRESHOLD dev=$DEV" >> "$LOG"
while true; do
  N=$(ls data/feats/*.npz 2>/dev/null | wc -l)
  echo "$(date) bags=$N/$THRESHOLD" >> "$LOG"
  if [ "$N" -ge "$THRESHOLD" ]; then
    echo "$(date) THRESHOLD reached -> launching train_heads on $DEV" >> "$LOG"
    nohup $PY gow/heads/train_heads.py --features-dir data/feats --cot "$COT" \
      --text-emb gow/artifacts/text_emb.npz --device "$DEV" --epochs 12 --out gow/artifacts/gow_heads.pt > logs/train_heads.log 2>&1 &
    echo "$(date) train_heads launched, PID $!  (see logs/train_heads.log)" >> "$LOG"
    break
  fi
  sleep 300
done
