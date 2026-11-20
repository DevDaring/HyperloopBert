#!/usr/bin/env bash
# Upload seed-43 snapshots, all results and the job log to the private HF repo.
# Idempotent: upload_folder skips files that are already identical.
set -uo pipefail
cd /root/HyperloopBert/Codes
cp /root/HyperloopBert/job.log results/stage3/vast_seed43_job.log 2>/dev/null || true
python3 - <<'EOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "Debk/HyperloopBERT"
api.upload_folder(repo_id=repo, folder_path="results/stage3",
                  path_in_repo=os.environ.get("RESULT_PATH", "results/stage3_seed43_revision"),
                  commit_message="FIRE revision: seed-43 replication and capability results")
if os.path.isdir("models/stage3/iso_band_models"):
    api.upload_folder(repo_id=repo, folder_path="models/stage3/iso_band_models",
                      path_in_repo="models/stage3/iso_band_models",
                      allow_patterns=["*seed43*/*"],
                      commit_message="FIRE revision: seed-43 band snapshots")
print("upload ok")
EOF
