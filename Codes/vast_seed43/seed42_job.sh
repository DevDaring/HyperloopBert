#!/usr/bin/env bash
# FIRE 2026 revision: capability checks on the existing seed-42 band-2.2 snapshots
# (all four encoders), on a cheap GPU. No training.
#   - GLUE SST-2 and MRPC fine-tuning (the planned capability leg that never ran)
# Env required: HF_TOKEN, VAST_KEY, HARD_LIMIT_H
set -uo pipefail
W=/root/HyperloopBert
C=$W/Codes
LOG=$W/job.log
mkdir -p $W
stage() { echo "=== $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
FAILS=0
run()   { stage "RUN $*"; "$@" >>"$LOG" 2>&1; local rc=$?; stage "rc=$rc $1"; [ $rc -ne 0 ] && FAILS=$((FAILS+1)); return $rc; }
selfstop() {
  local key="${CONTAINER_API_KEY:-$VAST_KEY}"; local Y=""; [ "$1" = destroy ] && Y="-y"
  stage "self-$1 instance ${CONTAINER_ID:-?}"
  vastai "$1" instance "$CONTAINER_ID" $Y --api-key "$key" >>"$LOG" 2>&1 \
    || vastai "$1" instance "$CONTAINER_ID" $Y --api-key "$VAST_KEY" >>"$LOG" 2>&1
}
export RESULT_PATH=results/stage3_seed42_capability_revision
crit()  { run "$@" || { stage "CRITICAL FAILURE: $1 -- uploading log and stopping"; \
            bash $C/vast_seed43/upload.sh >>"$LOG" 2>&1; selfstop stop; exit 1; }; }
( sleep $(( ${HARD_LIMIT_H:-2} * 3600 )); stage "WATCHDOG fired"; \
  bash $C/vast_seed43/upload.sh >>"$LOG" 2>&1; selfstop stop ) &

stage "start"
(apt-get update -qq && apt-get install -y -qq wget curl) >>"$LOG" 2>&1
nvidia-smi >>"$LOG" 2>&1
crit python3 -m pip install -q "numpy<2.0" transformers==4.46.0 accelerate==0.34.0 \
    datasets==2.16.0 "pyarrow>=14,<17" pandas==2.2.2 "tokenizers>=0.20,<0.21" "huggingface_hub>=0.24,<0.27" \
    "scipy>=1.10,<1.14" "scikit-learn>=1.3,<1.6" python-dotenv==1.0.0 vastai
cat > $C/.env <<EOF
HF_KEY=$HF_TOKEN
HF_TOKEN=$HF_TOKEN
HUGGINGFACE_TOKEN=$HF_TOKEN
EOF
export HF_TOKEN HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
crit python3 -c "import datasets, pyarrow, numpy; print(datasets.__version__, pyarrow.__version__, numpy.__version__)"
cd $C
crit python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Debk/HyperloopBERT', local_dir='.', allow_patterns=['tokenizer/*','models/stage3/iso_band_models/*seed42*/band_2p20/*'])
"
mkdir -p data/tokenizer && cp tokenizer/* data/tokenizer/

GLUE_TASKS=sst2,mrpc TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 run python3 Stage3/eval_glue_stage3.py --bands 2.2
stage "done"
stage "failed steps: $FAILS"
if bash $C/vast_seed43/upload.sh >>"$LOG" 2>&1 && [ "$FAILS" -eq 0 ]; then selfstop destroy; else selfstop stop; fi
