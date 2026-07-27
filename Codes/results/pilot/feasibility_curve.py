"""
FEASIBILITY CURVE -- at what model scale/quality does CrowS-Pairs stereotype
bias become statistically detectable with OUR scorer?

The pilot gave two points: our tiny @100M tokens (loss 6.77) -> NOT detectable,
and bert-base-uncased (fully trained) -> detectable. Everything between is
unknown, and that gap is exactly what decides whether an H100 run can work.

Google's BERT miniatures (Turc et al. 2019) are properly-trained models at the
SAME hidden sizes as our arms, so they isolate TRAINING ADEQUACY from
ARCHITECTURE SIZE:

    bert_uncased_L-2_H-128   ~4.4M     (smaller than our tiny)
    bert_uncased_L-4_H-256   ~11.3M    <-- our 'tiny' hidden size (256)
    bert_uncased_L-8_H-512   ~41.4M    <-- our 'small' hidden size (512)
    bert-base-uncased        ~110M     <-- our 'base'  hidden size (768)

If the H=256 miniature shows DETECTABLE bias, then a hidden-256 model CAN
encode measurable stereotype association when adequately trained -- meaning our
tiny arm failed on TOKEN BUDGET, not capacity, and more compute fixes it.
If even H=512 is not detectable, small models cannot support this measurement
at all and no amount of GPU time rescues the design.
"""
import os, sys, math
sys.path.insert(0, os.path.abspath('.'))
import torch
import pandas as pd

from common.bias_metrics import score_bias_pair
from common.stats_engine import bootstrap_ci

device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_PAIRS = int(os.environ.get('FC_PAIRS', '400'))

MODELS = [
    ("google/bert_uncased_L-2_H-128_A-2", "H=128  L=2   ~4.4M"),
    ("google/bert_uncased_L-4_H-256_A-4", "H=256  L=4  ~11.3M  <- our tiny width"),
    ("google/bert_uncased_L-8_H-512_A-8", "H=512  L=8  ~41.4M  <- our small width"),
    ("distilbert-base-uncased",           "H=768  L=6   ~66M"),
    ("bert-base-uncased",                 "H=768  L=12 ~110M  <- our base width"),
]

df = pd.read_csv('data/datasets_eval/multicrows/crows_pair_english.csv')
if len(df) > N_PAIRS:
    df = df.sample(n=N_PAIRS, random_state=42)

print("=" * 92)
print(f"FEASIBILITY CURVE -- CrowS-Pairs preference, {len(df)} pairs, our FP32 PLL scorer")
print("detectable == bootstrap 95% CI excludes 0.500 from above")
print("=" * 92)
print(f"{'model':38s} {'geometry':26s} {'pref':>7} {'CI':>18} {'verdict':>14}")

from transformers import AutoModelForMaskedLM, AutoTokenizer
for name, geom in MODELS:
    try:
        tk = AutoTokenizer.from_pretrained(name)
        md = AutoModelForMaskedLM.from_pretrained(name).to(device).eval()
    except Exception as e:
        print(f"{name:38s} {geom:26s}  load failed: {str(e)[:30]}")
        continue
    prefs = []
    for _, r in df.iterrows():
        s = score_bias_pair(md, tk, r['stereo'], r['anti'], device, compute_ss=False)
        if s['Stereotype_Preferred'] is not None:
            prefs.append(s['Stereotype_Preferred'])
    if prefs:
        rate, lo, hi = bootstrap_ci(prefs)
        det = "DETECTABLE" if lo > 0.5 else "not detectable"
        print(f"{name:38s} {geom:26s} {rate:7.4f} [{lo:.4f},{hi:.4f}] {det:>14}")
    del md
    torch.cuda.empty_cache()

print("=" * 92)
print("Our VanillaBERT @100M tokens (H=256 L=12, loss 6.77): 0.4525 [0.4050,0.5025]  not detectable")
print("=" * 92)
