"""
Phase 8 (FIRE revision): second training seed for VanillaBERT and LoopedBERT.

Same tests as phase2b_allband.py (paired sign-flip permutation, Phipson-Smyth
p-value, item-bootstrap percentile interval, 10^4 draws each):

  A. VanillaBERT minus LoopedBERT at every validation-loss band, seed 43
     (the seed-42 values are in out/allband_contrasts.csv)
  B. the same architecture across seeds, seed 42 minus seed 43, at every band:
     the run-to-run difference that one training run cannot show
  C. official CrowS-Pairs rule at band 2.2, seed 43, in the official orientation
     (every pair scored as sent_more minus sent_less)

Usage: python phase8_seed43.py SEED43_RESULTS_DIR
  SEED43_RESULTS_DIR holds bias/*seed43*_progress.csv and official/*seed43_2p20.csv
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/home/Debz/Research/HyperloopBert"
BIAS42 = os.path.join(ROOT, "Codes/results/stage3/bias")
OFF42 = os.path.join(ROOT, "analysis/fire2026/out/official")
OUT = os.path.join(ROOT, "analysis/fire2026/out")
R43 = sys.argv[1]

SEED = 20260729
N_PERM = 10000
N_BOOT = 10000
BANDS = [5.0, 4.0, 3.4, 3.0, 2.7, 2.4, 2.2]


def load(arch, seed, band):
    base = BIAS42 if seed == 42 else os.path.join(R43, "bias")
    p = os.path.join(base, f"multicrows_{arch}_base_seed{seed}_band{band}_progress.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p).sort_values("Row_Index").reset_index(drop=True)


def perm_test(d, rng, m=N_PERM):
    d = np.asarray(d, float)
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(m, d.size))
    null = (flips * d).mean(axis=1)
    return obs, (int(np.sum(np.abs(null) >= abs(obs) - 1e-15)) + 1) / (m + 1)


def boot_ci(x, rng, n=N_BOOT):
    x = np.asarray(x, float)
    idx = rng.integers(0, x.size, size=(n, x.size))
    return np.percentile(x[idx].mean(axis=1), [2.5, 97.5])


def contrast(a, b, rng):
    d = a.Effect_Size.values - b.Effect_Size.values
    obs, p = perm_test(d, rng)
    lo, hi = boot_ci(d, rng)
    return obs, lo, hi, p


rng = np.random.default_rng(SEED)
rows = []
for band in BANDS:
    v42, l42 = load("VanillaBERT", 42, band), load("LoopedBERT", 42, band)
    v43, l43 = load("VanillaBERT", 43, band), load("LoopedBERT", 43, band)
    if v43 is None or l43 is None:
        print(f"band {band}: seed-43 files missing, skipped")
        continue
    for name, a, b in [("seed43 Vanilla vs Looped", v43, l43),
                       ("Vanilla seed42 vs seed43", v42, v43),
                       ("Looped seed42 vs seed43", l42, l43)]:
        obs, lo, hi, p = contrast(a, b, rng)
        rows.append(dict(Band=band, Contrast=name, Delta=obs, CI_Low=lo, CI_High=hi, P_Raw=p,
                         Loss_A=float(a.Validation_Loss.iloc[0]),
                         Loss_B=float(b.Validation_Loss.iloc[0]),
                         Effect_A=float(a.Effect_Size.mean()), Effect_B=float(b.Effect_Size.mean())))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "seed43_contrasts.csv"), index=False)
pd.set_option("display.width", 200)
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# C. official rule, band 2.2, official orientation (sent_more minus sent_less on every pair)
def off_effect(path):
    d = pd.read_csv(path).sort_values("idx").reset_index(drop=True)
    return (d.score_more - d.score_less).values

pv = os.path.join(R43, "official", "VanillaBERT_seed43_2p20.csv")
pl = os.path.join(R43, "official", "LoopedBERT_seed43_2p20.csv")
if os.path.exists(pv) and os.path.exists(pl):
    ev, el = off_effect(pv), off_effect(pl)
    ok = ~(np.isnan(ev) | np.isnan(el))
    d = ev[ok] - el[ok]
    obs, p = perm_test(d, rng)
    lo, hi = boot_ci(d, rng)
    print(f"\nofficial rule, band 2.2, seed 43, Vanilla vs Looped: delta {obs:+.4f} "
          f"CI [{lo:.4f}, {hi:.4f}] p {p:.4f} (n={ok.sum()})")
    pd.DataFrame([dict(Contrast="seed43 Vanilla vs Looped", Rule="official sum", Delta=obs,
                       CI_Low=lo, CI_High=hi, P_Raw=p, N=int(ok.sum()))]).to_csv(
        os.path.join(OUT, "seed43_official_2p20.csv"), index=False)
