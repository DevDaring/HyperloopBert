#!/usr/bin/env bash
# FIRE 2026 revision: second training seed for VanillaBERT and LoopedBERT, plus
# the capability checks the reviewers asked about. Runs on one Vast.ai H100.
#
#   1. rebuild the corpus with the original pipeline (first 10k FineWeb-Edu
#      documents = the validation set; the rest filtered and pre-tokenized)
#   2. train VanillaBERT and LoopedBERT with seed 43 on the original 7B-token
#      learning-rate schedule, stopping once every validation-loss band is crossed
#   3. score every seed-43 snapshot with the original bias evaluation
#   4. official CrowS-Pairs rule at band 2.2 (seed 43)
#   5. masked-pronoun probes (seed 43)
#   6. WinoBias antecedent test (types 1, 2) and GLUE SST-2 / MRPC fine-tuning
#      for the two seed-43 band-2.2 encoders
#   7. upload everything to the private HF repo, then destroy the instance
#
# Env required: HF_TOKEN, VAST_KEY (fallback for self-stop), HARD_LIMIT_H
# Seed-42 capability checks run separately (seed42_job.sh) on a cheaper GPU.
set -uo pipefail
W=${ROOTDIR:-/root/HyperloopBert}
BAND=${BAND:-2.2}; BTAG=${BTAG:-2p20}; export ROOTDIR=$W BTAG
C=$W/Codes
LOG=$W/job.log
mkdir -p $W
stage() { echo "=== $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
FAILS=0
run()   { stage "RUN $*"; "$@" >>"$LOG" 2>&1; local rc=$?; stage "rc=$rc $1"; [ $rc -ne 0 ] && FAILS=$((FAILS+1)); return $rc; }

selfstop() {   # $1 = stop | destroy
  [ "${SMOKE:-0}" = 1 ] && { stage "SMOKE: would self-$1"; return 0; }
  # Newer vastai CLIs prompt on destroy and need -y; older ones reject -y.
  # Try both forms with each key, answering any prompt with "y" on stdin.
  local k Y
  stage "self-$1 instance ${CONTAINER_ID:-?}"
  for k in "${CONTAINER_API_KEY:-}" "$VAST_KEY"; do
    [ -n "$k" ] || continue
    for Y in "-y" ""; do
      [ "$1" = stop ] && [ -n "$Y" ] && continue
      echo y | vastai "$1" instance "$CONTAINER_ID" $Y --api-key "$k" >>"$LOG" 2>&1 && return 0
    done
  done
  stage "self-$1 FAILED"
}

upload() { if [ "${SMOKE:-0}" = 1 ]; then stage "SMOKE: upload skipped"; else bash $C/vast_seed43/upload.sh >>"$LOG" 2>&1; fi; }
# ---- watchdog: hard wall-clock cap so a hang can never drain the credit ----
crit()  { run "$@" || { stage "CRITICAL FAILURE: $1 -- uploading log and stopping"; \
            upload; selfstop stop; exit 1; }; }
( sleep $(( ${HARD_LIMIT_H:-7} * 3600 )); stage "WATCHDOG fired"; \
  upload; selfstop stop ) &

stage "start SMOKE=${SMOKE:-0} BAND=$BAND"
(apt-get update -qq && apt-get install -y -qq wget curl) >>"$LOG" 2>&1
nvidia-smi >>"$LOG" 2>&1

# ---- 0. environment ---------------------------------------------------------
crit python3 -m pip install -q "numpy<2.0" transformers==4.46.0 accelerate==0.34.0 \
    datasets==2.16.0 "pyarrow>=14,<17" pandas==2.2.2 tqdm==4.65.0 python-dotenv==1.0.0 requests==2.31.0 \
    sentencepiece==0.2.0 protobuf==4.25.0 "tokenizers>=0.20,<0.21" "huggingface_hub>=0.24,<0.27" \
    "scipy>=1.10,<1.14" "scikit-learn>=1.3,<1.6" pyahocorasick vastai
crit python3 -c "import datasets, pyarrow, numpy; print(datasets.__version__, pyarrow.__version__, numpy.__version__)"
PYTAG="cp$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
FA="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-${PYTAG}-${PYTAG}-linux_x86_64.whl"
( cd /tmp && wget -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${FA}" \
  && python3 -m pip install -q --no-deps "/tmp/${FA}" ) >>"$LOG" 2>&1 || stage "flash-attn unavailable, SDPA fallback"
python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))" >>"$LOG" 2>&1

cat > $C/.env <<EOF
HF_KEY=$HF_TOKEN
HF_TOKEN=$HF_TOKEN
HUGGINGFACE_TOKEN=$HF_TOKEN
EOF
export HF_TOKEN HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# ---- 1. data ----------------------------------------------------------------
cd $C
mkdir -p data/tokenizer
crit python3 -c "
from huggingface_hub import hf_hub_download
for f in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
    hf_hub_download('Debk/HyperloopBERT', 'tokenizer/'+f, local_dir='data_hf')
"
cp data_hf/tokenizer/* data/tokenizer/
# a 3.5M-document prefix of the original 5M-document stream: same validation set,
# same document order, and about 3B training tokens (training stops near 2.1B)
crit python3 -c "
import sys; sys.path.insert(0,'.')
from Dataset.download_training_corpus import download_fineweb_edu
download_fineweb_edu('data/fineweb-edu/train_raw.jsonl', target_docs=${CORPUS_DOCS:-3500000})
"
# The project HF mirror changed its schema and nyu-mll/crows_pairs no longer loads with
# datasets 2.16, so the exact 1508 CrowS-Pairs items used for seed 42 are shipped here
# (rebuilt from the seed-42 per-item files: same sentences, order and categories).
mkdir -p data/datasets_eval/multicrows
cp $C/vast_seed43/crows_pair_english.csv data/datasets_eval/multicrows/crows_pair_english.csv
crit python3 Dataset/download_eval_datasets.py
crit python3 Dataset/contamination_filter.py
grep -q "Simply copying raw" "$LOG" && { stage "CRITICAL: filter had no n-grams"; upload; selfstop stop; exit 1; }
rm -f data/fineweb-edu/train_raw.jsonl
rm -rf data/datasets_eval/indian_bias          # not used by the paper
crit python3 Dataset/pretokenize_corpus.py --workers ${PRETOK_WORKERS:-27}
ls -la data/fineweb-edu >>"$LOG"
curl -sfL -o /tmp/crows_pairs_anonymized.csv \
  https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv

# ---- 2. training ------------------------------------------------------------
export STAGE3_ARCHS=VanillaBERT,LoopedBERT STAGE3_SEEDS=43 STOP_WHEN_BANDS_DONE=1
crit python3 Stage3/train_stage3.py
ls models/stage3/iso_band_models/*seed43*/band_$BTAG/pytorch_model.bin >>"$LOG" 2>&1 || { stage "CRITICAL: no band-2.2 snapshot"; upload; selfstop stop; exit 1; }
upload      # checkpoint the expensive part first

# ---- 3-5. seed-43 evaluation --------------------------------------------------
run python3 Stage3/eval_bias_stage3.py
run python3 $C/vast_seed43/official_seed43.py
run python3 Stage3/qualitative_stage3.py --all-bands
upload

# ---- 6. seed-43 band-2.2 capability: antecedent test and GLUE -----------------
M=models/stage3/iso_band_models
CK="VanillaBERT=$M/VanillaBERT_base_seed43/band_$BTAG/pytorch_model.bin:VanillaBERT_base_seed43 LoopedBERT=$M/LoopedBERT_base_seed43/band_$BTAG/pytorch_model.bin:LoopedBERT_base_seed43"
for t in 1 2; do
  run python3 $W/analysis/fire2026/winobias_antecedent.py --type $t --raw $C/data_wino_raw \
      --tokenizer data/tokenizer --ckpt $CK \
      --out $C/results/stage3/capability/winobias_type${t}_antecedent_seed43.csv
done
upload
# HyperloopBERT seed 42 was skipped by the seed-42 GLUE run; add it here
run python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Debk/HyperloopBERT', local_dir='.', allow_patterns=['models/stage3/iso_band_models/HyperloopBERT_base_seed42_streams4/band_2p20/*'])
"
GLUE_TASKS=sst2,mrpc TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 run python3 Stage3/eval_glue_stage3.py --bands "$BAND,2.2"

# ---- 7. upload and destroy ------------------------------------------------------
stage "done"
stage "failed steps: $FAILS"
if upload && [ "$FAILS" -eq 0 ]; then selfstop destroy; else selfstop stop; fi
