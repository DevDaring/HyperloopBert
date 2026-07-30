"""
Rules 4, 5 and 6 of the revision plan.

RULE 4 — continuous adjustment for realised validation loss.
    The manuscript asserts that matching within ~0.034 nats removes
    training-progress confounding. That is an assumption, not a result. Every item
    is scored under every architecture at several checkpoints, so the data support
    a model that adjusts for realised loss CONTINUOUSLY instead of relying on the
    matching being tight enough.

        effect_i,a,c = b0 + b_arch[a] + b_loss * loss_a,c
                          + b_int * (arch x loss) + e

    Observations are not independent: the same 1508 benchmark items reappear at
    every checkpoint. Standard errors are therefore clustered on item. This is a
    cluster-robust OLS rather than a mixed model; with 4 architectures and 5-7
    checkpoints each there is not enough between-group structure to identify
    random slopes reliably, and that limitation is reported rather than hidden.

RULE 5 — the 8 items the changed-token formulation could not align.
RULE 6 — per-category estimates with multiplicity control, kept descriptive.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = "/home/Debz/Research/HyperloopBert"
BIAS = os.path.join(ROOT, "Codes/results/stage3/bias")
OUT = os.path.join(ROOT, "analysis/fire2026/out")
os.makedirs(OUT, exist_ok=True)

BANDS = [5.0, 4.0, 3.4, 3.0, 2.7, 2.4, 2.2]
ARCHS = ["VanillaBERT", "LoopedBERT", "ALBERTLoopedBERT", "HyperloopBERT"]
SHORT = {"VanillaBERT": "Vanilla", "LoopedBERT": "Looped",
         "ALBERTLoopedBERT": "ALBERT", "HyperloopBERT": "Hyperloop"}
CATEGORY = {0: "race-colour", 1: "socioeconomic", 2: "gender", 3: "disability",
            4: "nationality", 5: "sexual-orientation", 6: "physical-appearance",
            7: "religion", 8: "age"}
SEED = 20260729


def load(arch, band):
    tag = "HyperloopBERT_base_seed42_streams4" if arch == "HyperloopBERT" \
        else f"{arch}_base_seed42"
    p = os.path.join(BIAS, f"multicrows_{tag}_band{band}_progress.csv")
    return pd.read_csv(p).sort_values("Row_Index").reset_index(drop=True) \
        if os.path.exists(p) else None


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


rng = np.random.default_rng(SEED)

# ======================================================================= RULE 4
# Stack every distinct (architecture, checkpoint) into one long frame. Bands that
# resolve to the same snapshot are collapsed so a snapshot is not counted twice.
recs = []
for a in ARCHS:
    seen = set()
    for b in BANDS:
        d = load(a, b)
        if d is None:
            continue
        loss = round(float(d.Validation_Loss.iloc[0]), 6)
        if loss in seen:          # same snapshot reached several nominal bands
            continue
        seen.add(loss)
        recs.append(pd.DataFrame(dict(
            item=d.Row_Index.values, arch=SHORT[a], loss=loss,
            category=[CATEGORY[c] for c in d.Category.values],
            effect=d.Effect_Size.values)))
long = pd.concat(recs, ignore_index=True)
long["arch"] = pd.Categorical(long["arch"],
                              categories=["Vanilla", "Looped", "ALBERT", "Hyperloop"])
n_snap = long.groupby("arch", observed=True).loss.nunique()
print("=== RULE 4: distinct snapshots entering the model ===")
print(n_snap.to_string())
print(f"  total observations {len(long)}, items {long.item.nunique()}, "
      f"snapshots {long.groupby(['arch'], observed=True).loss.nunique().sum()}")
print()

# centre loss so the architecture coefficients are interpretable at the mean loss
long["loss_c"] = long.loss - long.loss.mean()

m_main = smf.ols("effect ~ C(arch) + loss_c", data=long).fit(
    cov_type="cluster", cov_kwds={"groups": long.item})
m_int = smf.ols("effect ~ C(arch) * loss_c", data=long).fit(
    cov_type="cluster", cov_kwds={"groups": long.item})

def tidy(m, name):
    t = pd.DataFrame(dict(term=m.params.index, coef=m.params.values,
                          se=m.bse.values, p=m.pvalues.values,
                          ci_low=m.conf_int()[0].values,
                          ci_high=m.conf_int()[1].values))
    t.insert(0, "model", name)
    return t

tab = pd.concat([tidy(m_main, "main effects"), tidy(m_int, "with interaction")],
                ignore_index=True)
tab.to_csv(os.path.join(OUT, "loss_adjusted_model.csv"), index=False)
print("=== RULE 4: architecture effect adjusting continuously for realised loss ===")
print("    (reference level = Vanilla; SEs clustered on benchmark item)")
print(tab.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
print()
lr = m_int.compare_lr_test(m_main)
print(f"  architecture x loss interaction, LR test: stat={lr[0]:.3f}, p={lr[1]:.4f}, df={lr[2]:.0f}")
print(f"  main-effects model R^2 = {m_main.rsquared:.5f}  (n={int(m_main.nobs)})")
print()

# ======================================================================= RULE 5
print("=== RULE 5: the items the changed-token formulation could not align ===")
ref = {a: load(a, 2.2) for a in ARCHS}
bad = np.zeros(len(ref["VanillaBERT"]), bool)
for a in ARCHS:
    bad |= ref[a].SS_PLL_Stereotypical.isna().values
    bad |= ref[a].SS_PLL_AntiStereotypical.isna().values
idx = np.where(bad)[0]
v = ref["VanillaBERT"]
det = v.loc[idx, ["Row_Index", "Category", "Sentence_Stereotypical",
                  "Sentence_AntiStereotypical"]].copy()
det["Category"] = det.Category.map(CATEGORY)


def diff_tokens(a, b):
    """Word-level symmetric difference — what the changed-token scorer needs."""
    sa, sb = a.split(), b.split()
    return len(set(sa) ^ set(sb)), len(sa), len(sb)


det[["n_diff_words", "len_stereo", "len_anti"]] = [
    diff_tokens(r.Sentence_Stereotypical, r.Sentence_AntiStereotypical)
    for r in det.itertuples()]
det.to_csv(os.path.join(OUT, "unaligned_items.csv"), index=False)
print(det[["Row_Index", "Category", "n_diff_words", "len_stereo", "len_anti"]]
      .to_string(index=False))
print("  sample pair:")
print("    S :", det.Sentence_Stereotypical.iloc[0][:90])
print("    A :", det.Sentence_AntiStereotypical.iloc[0][:90])
print(f"  length differs in {(det.len_stereo != det.len_anti).sum()} of {len(det)} pairs")
print()

# sensitivity: does keeping the 8 items (scored by the full-sentence formulation)
# change the changed-token conclusion?
print("  sensitivity — changed-token contrasts, dropped vs imputed-from-full-PLL:")
ok = ~bad
for B in ["LoopedBERT", "ALBERTLoopedBERT", "HyperloopBERT"]:
    def ct(a, mask):
        return (ref[a].SS_PLL_Stereotypical.values - ref[a].SS_PLL_AntiStereotypical.values)[mask]
    d_drop = ct("VanillaBERT", ok) - ct(B, ok)
    # imputation: substitute the full-sentence difference for the 8 unalignable pairs
    def ct_imp(a):
        x = (ref[a].SS_PLL_Stereotypical.values - ref[a].SS_PLL_AntiStereotypical.values)
        f = (ref[a].PLL_Stereotypical.values - ref[a].PLL_AntiStereotypical.values)
        return np.where(np.isnan(x), f, x)
    d_imp = ct_imp("VanillaBERT") - ct_imp(B)
    print(f"    Vanilla vs {SHORT[B]:10s} dropped Delta={d_drop.mean():+.5f}  "
          f"imputed Delta={d_imp.mean():+.5f}  (n {ok.sum()} vs {len(d_imp)})")
print()

# ======================================================================= RULE 6
print("=== RULE 6: per-category paired contrasts, Holm-corrected within contrast ===")
rows = []
for B in ["LoopedBERT", "ALBERTLoopedBERT", "HyperloopBERT"]:
    ps, recs2 = [], []
    for c, name in CATEGORY.items():
        m = ref["VanillaBERT"].Category.values == c
        d = ref["VanillaBERT"].Effect_Size.values[m] - ref[B].Effect_Size.values[m]
        obs = d.mean()
        flips = rng.choice([-1.0, 1.0], size=(10000, d.size))
        null = (flips * d).mean(axis=1)
        p = (int(np.sum(np.abs(null) >= abs(obs) - 1e-15)) + 1) / 10001
        bs = d[rng.integers(0, d.size, size=(10000, d.size))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ps.append(p)
        recs2.append(dict(Contrast=f"Vanilla vs {SHORT[B]}", Category=name,
                          N=int(m.sum()), Delta=obs, CI_Low=lo, CI_High=hi, P_Raw=p))
    adj = holm(ps)
    for r, a in zip(recs2, adj):
        r["P_Holm"] = a
        r["Sig_Holm"] = a < 0.05
        rows.append(r)
cat = pd.DataFrame(rows)
cat.to_csv(os.path.join(OUT, "per_category_contrasts.csv"), index=False)
print(cat.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print(f"  categories significant after Holm: {int(cat.Sig_Holm.sum())} of {len(cat)}")
sign_flip = (cat.groupby("Category").Delta.apply(lambda s: (s > 0).any() and (s < 0).any()))
print(f"  categories where the contrast changes sign across architectures: "
      f"{int(sign_flip.sum())} of {sign_flip.size}")
print()
print(f"outputs written to {OUT}")
