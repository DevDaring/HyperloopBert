"""Official CrowS-Pairs metric on the seed-43 band-2.2 snapshots (FIRE revision).

Writes raw per-item sums (score_more, score_less), as phase5_official_crows.py did
for seed 42, so the direction convention is applied only at analysis time.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

W = os.environ.get("ROOTDIR", "/root/HyperloopBert")
C = os.path.join(W, "Codes")
sys.path.insert(0, C)
sys.path.insert(0, os.path.join(W, "analysis/fire2026"))
from official_crows import official_crows_pair          # noqa: E402
from common.architectures import build_model            # noqa: E402
from transformers import PreTrainedTokenizerFast        # noqa: E402

OUT = os.path.join(C, "results/stage3/official")
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv("/tmp/crows_pairs_anonymized.csv")
if int(os.environ.get("EVAL_LIMIT", "0") or 0):   # smoke test only
    df = df.head(int(os.environ["EVAL_LIMIT"]))
BTAG = os.environ.get("BTAG", "2p20")
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(C, "data/tokenizer/tokenizer.json"),
                              mask_token="[MASK]", cls_token="[CLS]", sep_token="[SEP]",
                              pad_token="[PAD]", unk_token="[UNK]")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def get_logits(o):
    if isinstance(o, dict):
        for k in ("mlm_logits", "logits", "prediction_logits"):
            if k in o:
                return o[k]
    return o[0]


class OnDevice(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids.to(dev), attention_mask=attention_mask.to(dev))


for arch in ("VanillaBERT", "LoopedBERT"):
    run = f"{arch}_base_seed43"
    p = os.path.join(C, "models/stage3/iso_band_models", run, f"band_{BTAG}/pytorch_model.bin")
    if not os.path.exists(p):
        print("missing", p)
        sys.exit(1)
    m = build_model(arch, "base")
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=False)
    m = OnDevice(m.eval().float().to(dev))
    rows = []
    for i in range(len(df)):
        r = df.iloc[i]
        o = official_crows_pair(m, tok, r.sent_more, r.sent_less, get_logits)
        rows.append(dict(idx=i, bias_type=r.bias_type, direction=r.stereo_antistereo,
                         **(o or dict(score_more=np.nan, score_less=np.nan, n_shared_more=0,
                                      n_shared_less=0, norm_more=np.nan, norm_less=np.nan))))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{arch}_seed43_{BTAG}.csv"), index=False)
    print("wrote", arch)
