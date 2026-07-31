"""
Score every available snapshot with the faithful official CrowS-Pairs metric.

Order of work, so that partial results are still usable if this is interrupted:
  1. bert-base-uncased on all 1508 items  -> validation against the published 60.5
  2. the four architectures at the deepest matched point -> the headline table
  3. the remaining distinct snapshots      -> the across-capability trajectory

Each (model, snapshot) writes its own CSV and is skipped if already present.
CPU only; no training.
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch

ROOT = "/home/Debz/Research/HyperloopBert"
sys.path.insert(0, os.path.join(ROOT, "Codes"))
sys.path.insert(0, os.path.join(ROOT, "analysis/fire2026"))
from official_crows import official_crows_pair

CKPT = "/tmp/ss_ckpt"
OUT = os.path.join(ROOT, "analysis/fire2026/out/official")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(max(1, os.cpu_count() - 4))

CSV = "/tmp/crows_pairs_anonymized.csv"
BANDS = ["2p20", "2p40", "2p70", "3p00", "3p40", "4p00", "5p00"]
RUNS = {
    "Vanilla":   ("VanillaBERT",      "VanillaBERT_base_seed42",            {}),
    "Looped":    ("LoopedBERT",       "LoopedBERT_base_seed42",             {}),
    "ALBERT":    ("ALBERTLoopedBERT", "ALBERTLoopedBERT_base_seed42",       {}),
    "Hyperloop": ("HyperloopBERT",    "HyperloopBERT_base_seed42_streams4", {"num_streams": 4}),
}


def get_logits(o):
    if isinstance(o, dict):
        for k in ("mlm_logits", "logits", "prediction_logits"):
            if k in o:
                return o[k]
        raise KeyError(list(o.keys()))
    return o[0]


def score_all(model, tok, df, tag):
    dest = os.path.join(OUT, f"{tag}.csv")
    if os.path.exists(dest):
        print(f"    {tag}: present, skipped")
        return
    rows, t0 = [], time.time()
    for i in range(len(df)):
        r = df.iloc[i]
        o = official_crows_pair(model, tok, r.sent_more, r.sent_less, get_logits)
        rows.append(dict(idx=i, bias_type=r.bias_type,
                         direction=r.stereo_antistereo,
                         **(o or dict(score_more=np.nan, score_less=np.nan,
                                      n_shared_more=0, n_shared_less=0,
                                      norm_more=np.nan, norm_less=np.nan))))
        if (i + 1) % 400 == 0:
            el = time.time() - t0
            print(f"      {i+1}/{len(df)}  {el:.0f}s, ~{el/(i+1)*(len(df)-i-1):.0f}s left",
                  flush=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"    {tag}: wrote {dest} ({time.time()-t0:.0f}s)", flush=True)


def main():
    df = pd.read_csv(CSV)
    print(f"official CrowS-Pairs CSV: {len(df)} items, "
          f"{torch.get_num_threads()} threads\n")

    # ---- 1. validation -----------------------------------------------------
    if not os.path.exists(os.path.join(OUT, "bert-base-uncased.csv")):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        print("  [1/3] validating on bert-base-uncased (published score 60.5)")
        tk = AutoTokenizer.from_pretrained("bert-base-uncased")
        md = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").eval()
        score_all(md, tk, df, "bert-base-uncased")
        del md
    else:
        print("  [1/3] bert-base-uncased already scored")

    # ---- our models --------------------------------------------------------
    from transformers import PreTrainedTokenizerFast
    from common.architectures import build_model
    tok = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(CKPT, "tokenizer/tokenizer.json"),
        mask_token="[MASK]", cls_token="[CLS]", sep_token="[SEP]",
        pad_token="[PAD]", unk_token="[UNK]")

    def load(arch, run, kw, band):
        p = os.path.join(CKPT, "models/stage3/iso_band_models", run,
                         f"band_{band}/pytorch_model.bin")
        if not os.path.exists(p):
            return None
        m = build_model(arch, "base", **kw)
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True),
                          strict=False)
        return m.eval().float()

    print("\n  [2/3] four architectures at the deepest matched point (band 2p20)")
    for short, (arch, run, kw) in RUNS.items():
        m = load(arch, run, kw, "2p20")
        if m is None:
            print(f"    {short} band 2p20 missing"); continue
        score_all(m, tok, df, f"{short}_2p20"); del m

    print("\n  [3/3] remaining distinct snapshots (across-capability trajectory)")
    for band in BANDS[1:]:
        for short, (arch, run, kw) in RUNS.items():
            tag = f"{short}_{band}"
            if os.path.exists(os.path.join(OUT, f"{tag}.csv")):
                continue
            m = load(arch, run, kw, band)
            if m is None:
                continue
            print(f"    {tag}")
            score_all(m, tok, df, tag); del m


if __name__ == "__main__":
    main()
