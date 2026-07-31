"""
Issue 5 — is the loss-adjusted conclusion robust to how dependence is modelled?

The previous analysis pooled 31 668 architecture-snapshot-item rows and clustered
standard errors on benchmark item only. Architecture and realised loss, however,
vary at the SNAPSHOT level: the 1508 items inside one snapshot all share a single
model checkpoint. Clustering on item alone treats those 1508 rows as carrying
1508 units of information about the architecture coefficient when they carry
closer to one.

There are 21 distinct snapshots and one training seed. Four specifications are
compared rather than defending the original one.

  M1  cluster on item                     (the previous specification)
  M2  two-way cluster on item and snapshot
  M3  snapshot-level regression: collapse to one mean effect per snapshot,
      n = 21, which is the number of units that actually identify the
      architecture and loss terms
  M4  wild cluster bootstrap over snapshots (Rademacher, 9999 draws),
      appropriate when the number of clusters is small

The reported conclusion is whatever survives all four.
"""
import os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = "/home/Debz/Research/HyperloopBert"
BIAS = os.path.join(ROOT, "Codes/results/stage3/bias")
OUT = os.path.join(ROOT, "analysis/fire2026/out")
BANDS = [5.0, 4.0, 3.4, 3.0, 2.7, 2.4, 2.2]
ARCHS = ["VanillaBERT", "LoopedBERT", "ALBERTLoopedBERT", "HyperloopBERT"]
SHORT = {"VanillaBERT": "Vanilla", "LoopedBERT": "Looped",
         "ALBERTLoopedBERT": "ALBERT", "HyperloopBERT": "Hyperloop"}
SEED = 20260729


def load(a, b):
    tag = "HyperloopBERT_base_seed42_streams4" if a == "HyperloopBERT" else f"{a}_base_seed42"
    p = os.path.join(BIAS, f"multicrows_{tag}_band{b}_progress.csv")
    return pd.read_csv(p).sort_values("Row_Index").reset_index(drop=True) \
        if os.path.exists(p) else None


recs = []
for a in ARCHS:
    seen = set()
    for b in BANDS:
        d = load(a, b)
        if d is None:
            continue
        loss = round(float(d.Validation_Loss.iloc[0]), 6)
        if loss in seen:
            continue
        seen.add(loss)
        recs.append(pd.DataFrame(dict(item=d.Row_Index.values, arch=SHORT[a],
                                      loss=loss, effect=d.Effect_Size.values,
                                      snap=f"{SHORT[a]}@{loss:.4f}")))
long = pd.concat(recs, ignore_index=True)
long["arch"] = pd.Categorical(long["arch"], ["Vanilla", "Looped", "ALBERT", "Hyperloop"])
long["loss_c"] = long.loss - long.loss.mean()
n_snap = long.snap.nunique()

print("=== design ===")
print(f"  item-level rows            : {len(long)}")
print(f"  benchmark items            : {long.item.nunique()}")
print(f"  DISTINCT MODEL SNAPSHOTS   : {n_snap}   <- units identifying architecture")
print(f"  training seeds             : 1")
print(f"  snapshots per architecture : "
      f"{long.groupby('arch', observed=True).snap.nunique().to_dict()}\n")

F = "effect ~ C(arch) + loss_c"
terms = ["C(arch)[T.Looped]", "C(arch)[T.ALBERT]", "C(arch)[T.Hyperloop]", "loss_c"]
rows = []

m1 = smf.ols(F, long).fit(cov_type="cluster", cov_kwds={"groups": long.item})
for t in terms:
    rows.append(dict(model="M1 cluster: item", term=t, coef=m1.params[t],
                     se=m1.bse[t], p=m1.pvalues[t]))

m2 = smf.ols(F, long).fit(cov_type="cluster",
                          cov_kwds={"groups": np.column_stack(
                              [long.item.factorize()[0], long.snap.factorize()[0]])})
for t in terms:
    rows.append(dict(model="M2 cluster: item+snapshot", term=t, coef=m2.params[t],
                     se=m2.bse[t], p=m2.pvalues[t]))

# M3 — collapse to one observation per snapshot
snap = (long.groupby(["snap", "arch", "loss"], observed=True)
        .effect.mean().reset_index())
snap["loss_c"] = snap.loss - snap.loss.mean()
m3 = smf.ols(F, snap).fit()
for t in terms:
    rows.append(dict(model=f"M3 snapshot-level (n={len(snap)})", term=t,
                     coef=m3.params[t], se=m3.bse[t], p=m3.pvalues[t]))

# M4 — wild cluster bootstrap over snapshots, imposing the null on each term
rng = np.random.default_rng(SEED)
X = pd.get_dummies(long[["arch"]], drop_first=True).astype(float)
X["loss_c"] = long.loss_c.values
X.insert(0, "const", 1.0)
y = long.effect.values
codes = long.snap.factorize()[0]
Xv = X.values


def ols(Xm, yv):
    return np.linalg.lstsq(Xm, yv, rcond=None)[0]


names = list(X.columns)
b_full = ols(Xv, y)
for j, nm in enumerate(names):
    if nm == "const":
        continue
    keep = [k for k in range(Xv.shape[1]) if k != j]
    b0 = ols(Xv[:, keep], y)
    resid0 = y - Xv[:, keep] @ b0            # residuals under H0: beta_j = 0
    fitted0 = Xv[:, keep] @ b0
    t_obs = abs(b_full[j])
    cnt = 0
    B = 9999
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], size=n_snap)[codes]
        yb = fitted0 + resid0 * w
        bb = ols(Xv, yb)
        cnt += int(abs(bb[j]) >= t_obs)
    rows.append(dict(model=f"M4 wild bootstrap ({n_snap} snapshot clusters)",
                     term=nm, coef=b_full[j], se=np.nan, p=(cnt + 1) / (B + 1)))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "regression_audit.csv"), index=False)
pretty = {"C(arch)[T.Looped]": "Looped", "C(arch)[T.ALBERT]": "ALBERT",
          "C(arch)[T.Hyperloop]": "Hyperloop", "loss_c": "realised loss",
          "arch_Looped": "Looped", "arch_ALBERT": "ALBERT",
          "arch_Hyperloop": "Hyperloop"}
res["term"] = res.term.map(lambda t: pretty.get(t, t))
print("=== architecture and loss coefficients under four dependence models ===")
print(res.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
print()
arch_terms = res[res.term.isin(["Looped", "ALBERT", "Hyperloop"])]
loss_terms = res[res.term == "realised loss"]
print("=== does the conclusion survive? ===")
for m, g in arch_terms.groupby("model"):
    print(f"  {m:38s} architecture terms with p<0.05: {int((g.p < 0.05).sum())}/3")
for m, g in loss_terms.groupby("model"):
    print(f"  {m:38s} loss term p = {g.p.iloc[0]:.4f}")
print(f"\noutputs written to {OUT}")
