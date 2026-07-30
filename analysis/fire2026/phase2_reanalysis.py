"""
Phase 2 — free re-analysis of the existing per-item CrowS-Pairs scores.

No new training and no new inference: every number below comes from the per-item
score files already produced by the Stage 3 run
(Codes/results/stage3/bias/multicrows_*_progress.csv).

Covers instruction items 2.1 through 2.7. Outputs go to analysis/fire2026/out/
as CSV so the LaTeX numbers can be checked against a committed file.

Statistical conventions, stated once and used throughout:
  * per-item effect size  e_i = (PLL_stereo - PLL_anti) / T   (already stored)
  * architecture contrast d_i = e_i(A) - e_i(B), paired on the same item
  * permutation test      sign-flip of d_i, m = 10000 draws, two-sided
  * p-value               (b + 1) / (m + 1)  [Phipson and Smyth, 2010]
  * Cohen's d             mean(d) / sd(d), i.e. the SD of the PAIRED DIFFERENCES
  * bootstrap CI          10000 item-level resamples, percentile interval
Seeds are fixed so the outputs are reproducible.
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

ARCHS = ["VanillaBERT", "LoopedBERT", "ALBERTLoopedBERT", "HyperloopBERT"]
SHORT = {"VanillaBERT": "Vanilla", "LoopedBERT": "Looped",
         "ALBERTLoopedBERT": "ALBERT", "HyperloopBERT": "Hyperloop"}

# Integer category codes verified two ways: the nine group sizes match the
# published CrowS-Pairs bias-type counts exactly (516/262/172/159/105/87/84/63/60),
# and sampled sentences in each group carry the expected content.
CATEGORY = {0: "race-colour", 1: "socioeconomic", 2: "gender", 3: "disability",
            4: "nationality", 5: "sexual-orientation", 6: "physical-appearance",
            7: "religion", 8: "age"}


def load(arch, band):
    """Per-item scores for one architecture at one iso-loss band."""
    tag = "HyperloopBERT_base_seed42_streams4" if arch == "HyperloopBERT" \
        else f"{arch}_base_seed42"
    p = os.path.join(BIAS, f"multicrows_{tag}_band{band}_progress.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p).sort_values("Row_Index").reset_index(drop=True)
    return d


def perm_test(d, rng, m=N_PERM):
    """Two-sided sign-flip permutation test on paired differences."""
    d = np.asarray(d, float)
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(m, d.size))
    null = (flips * d).mean(axis=1)
    b = int(np.sum(np.abs(null) >= abs(obs) - 1e-15))
    return obs, (b + 1) / (m + 1)


def boot_ci(x, rng, n=N_BOOT, alpha=0.05):
    x = np.asarray(x, float)
    idx = rng.integers(0, x.size, size=(n, x.size))
    means = x[idx].mean(axis=1)
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def holm(pvals):
    """Holm step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        val = (len(p) - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


rng = np.random.default_rng(SEED)
BAND = 2.2
data = {a: load(a, BAND) for a in ARCHS}
missing = [a for a, v in data.items() if v is None]
if missing:
    raise SystemExit(f"missing per-item files for {missing}")
n_items = len(data["VanillaBERT"])
print(f"loaded {n_items} items x {len(ARCHS)} encoders at band {BAND}\n")

# item alignment is required for any paired test
for a in ARCHS[1:]:
    assert (data[a].Row_Index.values == data["VanillaBERT"].Row_Index.values).all(), \
        f"{a} item order differs from VanillaBERT"
print("item alignment verified across all four encoders\n")

# ---------------------------------------------------------------- 2.1 + 2.6a/b
# All Vanilla-vs-X contrasts, including the previously untested ALBERT one.
rows = []
contrasts = [("VanillaBERT", "LoopedBERT"), ("VanillaBERT", "HyperloopBERT"),
             ("VanillaBERT", "ALBERTLoopedBERT"), ("LoopedBERT", "HyperloopBERT")]
for A, B in contrasts:
    d = data[A].Effect_Size.values - data[B].Effect_Size.values
    obs, p = perm_test(d, rng)
    lo, hi = boot_ci(d, rng)
    sd = d.std(ddof=1)
    rows.append(dict(Contrast=f"{SHORT[A]} vs {SHORT[B]}", Delta=obs,
                     CI_Low=lo, CI_High=hi, P_Raw=p,
                     Cohens_d=obs / sd, SD_of_paired_diff=sd, N=len(d)))
con = pd.DataFrame(rows)
# Holm across the three Vanilla-vs-shared contrasts (the family of interest);
# Looped-vs-Hyperloop is the separate null test and is reported unadjusted.
fam = con.Contrast.str.startswith("Vanilla")
con.loc[fam, "P_Holm"] = holm(con.loc[fam, "P_Raw"].values)
con.loc[~fam, "P_Holm"] = con.loc[~fam, "P_Raw"]
con["Significant"] = con.P_Holm < 0.05
con.to_csv(os.path.join(OUT, "contrasts_band2.2.csv"), index=False)
print("=== 2.1/2.6 contrasts at band 2.2 (Holm over the three Vanilla contrasts) ===")
print(con.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()

# ------------------------------------------------------------------- 2.2 + 2.3
# Per-bias-type effect size, and the literature-standard stereotype score.
per_cat, overall = [], []
for a in ARCHS:
    d = data[a]
    e_all = d.Effect_Size.values
    lo, hi = boot_ci(e_all, rng)
    # standard CrowS-Pairs metric: percentage of pairs where the stereotypical
    # member receives the higher score
    ss = 100.0 * (d.PLL_Stereotypical.values > d.PLL_AntiStereotypical.values).mean()
    ss_ci = boot_ci((d.PLL_Stereotypical.values > d.PLL_AntiStereotypical.values) * 100.0, rng)
    overall.append(dict(Architecture=SHORT[a], Effect_Size=e_all.mean(),
                        CI_Low=lo, CI_High=hi,
                        Stereotype_Score=ss, SS_CI_Low=ss_ci[0], SS_CI_High=ss_ci[1]))
    for c, name in CATEGORY.items():
        m = d.Category.values == c
        e = e_all[m]
        clo, chi = boot_ci(e, rng)
        per_cat.append(dict(Architecture=SHORT[a], Category=name, N=int(m.sum()),
                            Effect_Size=e.mean(), CI_Low=clo, CI_High=chi))
ov = pd.DataFrame(overall)
pc = pd.DataFrame(per_cat)
ov.to_csv(os.path.join(OUT, "overall_band2.2.csv"), index=False)
pc.to_csv(os.path.join(OUT, "per_category_band2.2.csv"), index=False)
print("=== 2.3 overall + literature-standard stereotype score ===")
print(ov.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print("=== 2.2 per-bias-type effect size (pivot) ===")
print(pc.pivot(index="Category", columns="Architecture", values="Effect_Size")
        .reindex(columns=[SHORT[a] for a in ARCHS])
        .to_string(float_format=lambda v: f"{v:.4f}"))
print()

# ----------------------------------------------------------------------- 2.4
print("=== 2.4 sharing ratio vs effect size (non-monotonicity) ===")
mono = pd.DataFrame([
    dict(Architecture=SHORT[a],
         Shared_Ratio=float(data[a].Shared_Ratio.iloc[0]),
         Val_Loss=float(data[a].Validation_Loss.iloc[0]),
         Effect_Size=float(data[a].Effect_Size.mean()))
    for a in ARCHS]).sort_values("Shared_Ratio")
mono.to_csv(os.path.join(OUT, "monotonicity.csv"), index=False)
print(mono.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
rho, prho = stats.spearmanr(mono.Shared_Ratio, mono.Effect_Size)
print(f"  Spearman(shared ratio, effect size) = {rho:.3f}, p = {prho:.3f}  (n=4)")
print()

# ----------------------------------------------------------------------- 2.6e
# Leave-one-category-out: does any single bias type drive the headline contrasts?
loo = []
for A, B in contrasts[:3]:
    full = (data[A].Effect_Size.values - data[B].Effect_Size.values).mean()
    for c, name in CATEGORY.items():
        m = data[A].Category.values != c
        d = data[A].Effect_Size.values[m] - data[B].Effect_Size.values[m]
        loo.append(dict(Contrast=f"{SHORT[A]} vs {SHORT[B]}", Dropped=name,
                        Delta=d.mean(), Full_Delta=full,
                        Pct_Change=100 * (d.mean() - full) / full))
lo_df = pd.DataFrame(loo)
lo_df.to_csv(os.path.join(OUT, "leave_one_category_out.csv"), index=False)
print("=== 2.6e leave-one-category-out: largest shift per contrast ===")
for c, g in lo_df.groupby("Contrast"):
    w = g.iloc[g.Pct_Change.abs().argmax()]
    print(f"  {c}: worst drop = {w.Dropped}, Delta {w.Full_Delta:.4f} -> {w.Delta:.4f} "
          f"({w.Pct_Change:+.1f}%), sign preserved: {np.sign(w.Delta)==np.sign(w.Full_Delta)}")
print()

# ----------------------------------------------------------------------- 2.7
# TOST equivalence for the Looped-vs-Hyperloop null. SESOI is pre-stated as half
# the Vanilla-vs-Looped difference: the smallest architecture effect this study
# would call meaningful.
d_lh = data["LoopedBERT"].Effect_Size.values - data["HyperloopBERT"].Effect_Size.values
sesoi = 0.5 * abs(con.loc[con.Contrast == "Vanilla vs Looped", "Delta"].iloc[0])
se = d_lh.std(ddof=1) / np.sqrt(d_lh.size)
t_lo = (d_lh.mean() + sesoi) / se
t_hi = (d_lh.mean() - sesoi) / se
df = d_lh.size - 1
p_lo = stats.t.sf(t_lo, df)      # H0: diff <= -SESOI
p_hi = stats.t.cdf(t_hi, df)     # H0: diff >= +SESOI
p_tost = max(p_lo, p_hi)
ci90 = stats.t.interval(0.90, df, loc=d_lh.mean(), scale=se)
tost = pd.DataFrame([dict(Contrast="Looped vs Hyperloop", Delta=d_lh.mean(),
                          SESOI=sesoi, CI90_Low=ci90[0], CI90_High=ci90[1],
                          P_TOST=p_tost,
                          Equivalent=bool(p_tost < 0.05))])
tost.to_csv(os.path.join(OUT, "tost_equivalence.csv"), index=False)
print("=== 2.7 TOST equivalence (Looped vs Hyperloop) ===")
print(f"  SESOI (half the Vanilla-Looped difference) = +/-{sesoi:.4f}")
print(f"  Delta = {d_lh.mean():.5f}, 90% CI [{ci90[0]:.4f}, {ci90[1]:.4f}]")
print(f"  TOST p = {p_tost:.5f} -> statistically equivalent: {p_tost < 0.05}")
print()

# ----------------------------------------------------------------------- 3.4
# Scorer validation, available for free: SS_PLL is the changed-token-only variant
# that the official CrowS-Pairs script uses.
print("=== 3.4 scorer agreement: full-sentence PLL vs changed-token-only ===")
# The changed-token scorer failed to align on 8 of 1508 pairs (its stored values
# are NaN there). Those items are excluded from the changed-token analysis only,
# and the same reduced item set is used for every encoder so the comparison stays
# paired. The full-sentence analysis above uses all 1508 items.
ok = np.ones(n_items, bool)
for a in ARCHS:
    ok &= data[a].SS_PLL_Stereotypical.notna().values
    ok &= data[a].SS_PLL_AntiStereotypical.notna().values
print(f"  changed-token scorer usable on {ok.sum()} of {n_items} items "
      f"({n_items - ok.sum()} unaligned pairs dropped)")
agree = []
for a in ARCHS:
    d = data[a]
    e_full = d.Effect_Size.values[ok]
    e_ss = (d.SS_PLL_Stereotypical.values - d.SS_PLL_AntiStereotypical.values)[ok]
    r, _ = stats.spearmanr(e_full, e_ss)
    ss_std = 100.0 * (d.SS_PLL_Stereotypical.values[ok] > d.SS_PLL_AntiStereotypical.values[ok]).mean()
    agree.append(dict(Architecture=SHORT[a], N_used=int(ok.sum()), Spearman_r=r,
                      Stereotype_Score_full=100.0 * (d.PLL_Stereotypical.values[ok] >
                                                     d.PLL_AntiStereotypical.values[ok]).mean(),
                      Stereotype_Score_changed_only=ss_std))
ag = pd.DataFrame(agree)
ag.to_csv(os.path.join(OUT, "scorer_agreement.csv"), index=False)
print(ag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()

# do the headline contrasts survive under the changed-token-only scorer?
alt = []
for A, B in contrasts[:3]:
    dA = (data[A].SS_PLL_Stereotypical.values - data[A].SS_PLL_AntiStereotypical.values)[ok]
    dB = (data[B].SS_PLL_Stereotypical.values - data[B].SS_PLL_AntiStereotypical.values)[ok]
    d = dA - dB
    assert np.isfinite(d).all(), "non-finite value survived the completeness mask"
    obs, p = perm_test(d, rng)
    lo, hi = boot_ci(d, rng)
    alt.append(dict(Contrast=f"{SHORT[A]} vs {SHORT[B]}", Delta=obs,
                    CI_Low=lo, CI_High=hi, P_Raw=p, N=int(ok.sum())))
al = pd.DataFrame(alt)
al["P_Holm"] = holm(al.P_Raw.values)
al.to_csv(os.path.join(OUT, "contrasts_changed_token_scorer.csv"), index=False)
print("=== 3.4 same contrasts under the changed-token-only scorer ===")
print(al.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print(f"outputs written to {OUT}")
