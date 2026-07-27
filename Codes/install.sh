#!/usr/bin/env bash
# Canonical install recipe (BUILD SPECIFICATION section 3.1).
# Global Python environment only -- NO venv, NO conda.
# Target: Linux x86-64, Python 3.12, NVIDIA L4 (sm_89), CUDA 12.x driver.
set -euo pipefail

python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
 && python3 -m pip install "numpy<2.0" transformers==4.46.0 accelerate==0.34.0 datasets==2.16.0 \
      bitsandbytes==0.46.1 pandas==2.2.2 tqdm==4.65.0 python-dotenv==1.0.0 requests==2.31.0 \
      sentencepiece==0.2.0 protobuf==4.25.0 \
 && python3 -m pip install "tokenizers>=0.20,<0.21" "huggingface_hub>=0.24" "scipy>=1.10,<1.14" \
      "scikit-learn>=1.3,<1.6" "matplotlib>=3.7,<3.10" "seaborn>=0.13,<0.14" \
 && wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl -O /tmp/flash_attn.whl \
 && python3 -m pip install --no-deps /tmp/flash_attn.whl

# Verification probe: every line must import cleanly.
python3 - <<'EOF'
import sys
failures = []

import torch
print(f"torch            {torch.__version__}")
print(f"torch CUDA       {torch.version.cuda}")
if not torch.__version__.startswith("2.5"):
    failures.append("torch: expected 2.5.x -- reinstall with the cu124 index URL")

try:
    import bitsandbytes
    print(f"bitsandbytes     {bitsandbytes.__version__}")
except Exception as e:
    failures.append(f"bitsandbytes failed to import ({e}) -- expected 0.46.1")

try:
    import flash_attn
    print(f"flash_attn       {flash_attn.__version__}")
except Exception as e:
    failures.append(f"flash_attn failed to import ({e}) -- expected 2.8.3 wheel "
                    f"(cp312 / cu12 / torch2.5 / cxx11abiFALSE). Check Python==3.12 "
                    f"and the torch C++ ABI.")

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU              {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    if cap not in [(8, 0), (8, 6), (8, 9), (9, 0)]:
        failures.append(f"GPU sm_{cap[0]}{cap[1]} is outside FlashAttention-2 support "
                        f"(sm_80/86/89/90); the pipeline will fall back to SDPA.")
else:
    print("WARNING: CUDA not available at install time.")

if failures:
    print("\nINSTALL VERIFICATION FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("\nInstall verification PASSED.")
EOF
