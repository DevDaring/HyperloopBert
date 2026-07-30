"""
Phase 2.5 / 2.6c / 2.6d — the loss-adjusted analysis and the remaining
uncertainty statistics.

2.5 is the most important item in Phase 2. Table 2 of the paper compares four
snapshots whose realised losses span 2.163-2.196 nats, and the encoders with the
LOWER effect size sit at HIGHER loss. Since effect size grows as loss falls, part
of the headline contrast could be residual training progress rather than
architecture -- the exact confound the iso-loss protocol claims to remove.

The test: recompute every Vanilla-vs-shared contrast at EVERY band that all four
encoders reached, not only the headline band. If the contrast is an artefact of
the 0.033-nat residual gap it should wander with band; if it is architectural it
should persist.

Inputs are the per-item files already on disk. No new inference.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/home/Debz/Research/HyperloopBert"
BIAS = os.path.join(ROOT, "Codes/results/stage3/bias")
OUT = os.path.join(ROOT, "analysis/fire2026/out")
os.makedirs(OUT, exist_ok=True)

SEED = 20260729
N_PERM = 10000
N_BOOT = 10000
BANDS = [5.0, 4.0, 3.4, 3.0, 2.7, 2.4, 2.2]
ARCHS = ["VanillaBERT", "LoopedBERT", "ALBERTLoopedBERT", "HyperloopBERT"]
SHORT = {"VanillaBERT": "Vanilla", "LoopedBERT": "Looped",
         "ALBERTLoopedBERT": "ALBERT", "HyperloopBERT": "Hyperloop"}


def load(arch, band):
    tag = "HyperloopBERT_base_seed42_streams4" if arch == "HyperloopBERT" \
        else f"{arch}_base_seed42"
    p = os.path.join(BIAS, f"multicrows_{tag}_band{band}_progress.csv")
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
    mu = x[idx].mean(axis=1)
    return np.percentile(mu, [2.5, 97.5])


rng = np.random.default_rng(SEED)

# ------------------------------------------------------- 2.5 all-band contrasts
rows = []
traj = []
for band in BANDS:
    d = {a: load(a, band) for a in ARCHS}
    if any(v is None for v in d.values()):
        print(f"  band {band}: incomplete, skipped")
        continue
    # realised loss actually differs from the nominal band; record the spread
    losses = {a: float(d[a].Validation_Loss.iloc[0]) for a in ARCHS}
    spread = max(losses.values()) - min(losses.values())
    for a in ARCHS:
        traj.append(dict(Band=band, Architecture=SHORT[a],
                         Val_Loss=losses[a], Effect_Size=float(d[a].Effect_Size.mean())))
    for B in ["LoopedBERT", "ALBERTLoopedBERT", "HyperloopBERT"]:
        diff = d["VanillaBERT"].Effect_Size.values - d[B].Effect_Size.values
        obs, p = perm_test(diff, rng)
        lo, hi = boot_ci(diff, rng)
        rows.append(dict(Band=band, Contrast=f"Vanilla vs {SHORT[B]}",
                         Delta=obs, CI_Low=lo, CI_High=hi, P_Raw=p,
                         Loss_Vanilla=losses["VanillaBERT"], Loss_Other=losses[B],
                         Loss_Gap=losses["VanillaBERT"] - losses[B],
                         Band_Spread=spread))
ab = pd.DataFrame(rows)
ab.to_csv(os.path.join(OUT, "allband_contrasts.csv"), index=False)
tj = pd.DataFrame(traj)
tj.to_csv(os.path.join(OUT, "allband_trajectories.csv"), index=False)

print("=== 2.5 Vanilla-vs-shared contrast at EVERY common band ===")
print(ab[["Band", "Contrast", "Delta", "CI_Low", "CI_High", "P_Raw", "Loss_Gap"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print("--- how often is each contrast positive, and how often significant? ---")
for c, g in ab.groupby("Contrast"):
    pos = int((g.Delta > 0).sum())
    sig = int((g.P_Raw < 0.05).sum())
    print(f"  {c:24s} positive at {pos}/{len(g)} bands, p<0.05 at {sig}/{len(g)}, "
          f"mean Delta {g.Delta.mean():+.4f}")
print()

# Does the contrast track the residual loss gap? If the effect were driven by the
# 0.033-nat mismatch, Delta and Loss_Gap would correlate.
print("--- is the contrast explained by the residual loss gap? ---")
for c, g in ab.groupby("Contrast"):
    if len(g) >= 3:
        r, p = stats.spearmanr(g.Loss_Gap, g.Delta)
        print(f"  {c:24s} Spearman(loss gap, Delta) = {r:+.3f}, p = {p:.3f}, n = {len(g)}")
print()

# ---------------------------------------------------- 2.6c effect-size trend
print("=== 2.6c does effect size really grow monotonically with training? ===")
tr = []
for a in ARCHS:
    g = tj[tj.Architecture == SHORT[a]].drop_duplicates("Val_Loss").sort_values("Val_Loss")
    if len(g) < 3:
        continue
    # loss decreases with training, so a NEGATIVE rho means effect size rises
    r, p = stats.spearmanr(g.Val_Loss, g.Effect_Size)
    dips = int((np.diff(g.sort_values("Val_Loss", ascending=False).Effect_Size.values) < 0).sum())
    tr.append(dict(Architecture=SHORT[a], N_points=len(g), Spearman_rho=r,
                   P=p, Dips=dips))
trd = pd.DataFrame(tr)
trd.to_csv(os.path.join(OUT, "effect_size_trend.csv"), index=False)
print(trd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("  (rho < 0 means effect size rises as loss falls; Dips counts "
      "band-to-band decreases)")
print()

# ------------------------------------------------- 2.6d coreference binomial CI
print("=== 2.6d coreference: is 'at chance' defensible? ===")
w = pd.read_csv(os.path.join(BIAS, "winobias_summary.csv"))
w = w[w.Band == 2.2]
res = []
for _, r in w.iterrows():
    for split in ["Pro", "Anti"]:
        acc = float(r[f"{split}_Stereotype_Accuracy"])
        # WinoBias dev split size used by the scorer
        n = 374
        k = int(round(acc * n))
        lo, hi = stats.beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)  # Jeffreys
        res.append(dict(Architecture=r.Architecture, Split=split, Accuracy=acc,
                        N=n, CI_Low=lo, CI_High=hi,
                        Includes_Chance=bool(lo <= 0.5 <= hi)))
wb = pd.DataFrame(res)
wb.to_csv(os.path.join(OUT, "winobias_binomial_ci.csv"), index=False)
print(wb.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"  all intervals include 0.5: {bool(wb.Includes_Chance.all())}")
print()
print(f"outputs written to {OUT}")
