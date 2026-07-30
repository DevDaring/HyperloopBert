"""
Analyse the StereoSet replication and decide the direction question.

Pre-committed interpretation rule (stated before the numbers were seen, in the
revision plan): if the DIRECTION of the shared-versus-unshared difference matches
CrowS-Pairs, the finding replicates across two instruments of the same
correlation type; if it does not, that is reported and the claim is confined to
CrowS-Pairs. Either outcome is reportable; concealing a mismatch is not.

Same statistics as everywhere else in this revision: paired sign-flip permutation
with m = 10^4 and (b+1)/(m+1) p-values, 10^4-resample item bootstraps, Holm over
the three baseline-versus-shared contrasts.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/home/Debz/Research/HyperloopBert"
OUT = os.path.join(ROOT, "analysis/fire2026/out")
SEED, M, B = 20260729, 10000, 10000
ORDER = ["Vanilla", "Looped", "ALBERT", "Hyperloop"]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); a[i] = min(run, 1.0)
    return a


def perm(d, rng, m=M):
    d = np.asarray(d, float); obs = d.mean()
    null = (rng.choice([-1.0, 1.0], size=(m, d.size)) * d).mean(axis=1)
    return obs, (int(np.sum(np.abs(null) >= abs(obs) - 1e-15)) + 1) / (m + 1)


def boot(x, rng, n=B):
    x = np.asarray(x, float)
    return np.percentile(x[rng.integers(0, x.size, size=(n, x.size))].mean(axis=1), [2.5, 97.5])


rng = np.random.default_rng(SEED)
data = {a: pd.read_csv(os.path.join(OUT, f"stereoset_{a}.csv")) for a in ORDER}

# keep only items every encoder scored, so all contrasts are paired on one set
ok = np.ones(len(data["Vanilla"]), bool)
for a in ORDER:
    ok &= data[a].effect.notna().values
n_use = int(ok.sum())
print(f"=== StereoSet intrasentence: {n_use} of {len(ok)} items scored by all four "
      f"encoders ===\n")

# ---------------------------------------------------------------- per-encoder
rows = []
for a in ORDER:
    d = data[a][ok]
    e = d.effect.values
    lo, hi = boot(e, rng)
    ss = 100.0 * (d.PLL_stereo.values > d.PLL_anti.values).mean()
    m_ = d.PLL_unrelated.notna().values
    best = np.maximum(d.PLL_stereo.values, d.PLL_anti.values)
    lms = 100.0 * (best[m_] > d.PLL_unrelated.values[m_]).mean()
    icat = lms * min(ss, 100 - ss) / 50.0
    rows.append(dict(Architecture=a, Effect=e.mean(), CI_Low=lo, CI_High=hi,
                     SS=ss, LMS=lms, ICAT=icat))
per = pd.DataFrame(rows)
per.to_csv(os.path.join(OUT, "stereoset_summary.csv"), index=False)
print("--- per-encoder (SS = stereotype score, LMS = language modelling score) ---")
print(per.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()

# ----------------------------------------------------------------- contrasts
rows = []
for B_ in ["Looped", "ALBERT", "Hyperloop"]:
    d = data["Vanilla"][ok].effect.values - data[B_][ok].effect.values
    obs, p = perm(d, rng)
    lo, hi = boot(d, rng)
    rows.append(dict(Contrast=f"Vanilla vs {B_}", Delta=obs, CI_Low=lo, CI_High=hi,
                     P_Raw=p, Cohens_d=obs / d.std(ddof=1), N=n_use))
con = pd.DataFrame(rows)
con["P_Holm"] = holm(con.P_Raw.values)
con["Significant"] = con.P_Holm < 0.05
con.to_csv(os.path.join(OUT, "stereoset_contrasts.csv"), index=False)
print("--- baseline-versus-shared contrasts (Holm over the three) ---")
print(con.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()

# ------------------------------------------------------- direction comparison
cp = pd.read_csv(os.path.join(OUT, "contrasts_band2.2.csv")).set_index("Contrast")
print("--- DIRECTION CHECK against CrowS-Pairs (the pre-committed rule) ---")
agree = 0
for _, r in con.iterrows():
    c = r.Contrast
    if c not in cp.index:
        continue
    cd, sd = cp.loc[c, "Delta"], r.Delta
    same = np.sign(cd) == np.sign(sd)
    agree += int(same)
    print(f"  {c:22s} CrowS {cd:+.4f} (p_Holm {cp.loc[c,'P_Holm']:.4f})  |  "
          f"StereoSet {sd:+.4f} (p_Holm {r.P_Holm:.4f})  |  "
          f"same sign: {'YES' if same else 'NO'}")
print(f"\n  directions agreeing: {agree}/3")
print(f"  significant on StereoSet after Holm: {int(con.Significant.sum())}/3")
if agree == 3 and con.Significant.sum() >= 2:
    verdict = "REPLICATES: same direction and significant on both instruments"
elif agree >= 2:
    verdict = ("PARTIAL: directions mostly agree but significance does not carry "
               "across instruments")
else:
    verdict = ("DOES NOT REPLICATE: direction differs between instruments; the "
               "claim stays confined to CrowS-Pairs")
print(f"  VERDICT: {verdict}")
pd.DataFrame([dict(directions_agreeing=agree,
                   significant_stereoset=int(con.Significant.sum()),
                   verdict=verdict)]).to_csv(
    os.path.join(OUT, "stereoset_direction_verdict.csv"), index=False)
print()

# ------------------------------------------------------------- by bias type
rows = []
for a in ORDER:
    d = data[a][ok]
    for bt, g in d.groupby("bias_type"):
        rows.append(dict(Architecture=a, Bias_Type=bt, N=len(g),
                         Effect=g.effect.mean()))
bt = pd.DataFrame(rows)
bt.to_csv(os.path.join(OUT, "stereoset_by_bias_type.csv"), index=False)
print("--- per bias type ---")
print(bt.pivot(index="Bias_Type", columns="Architecture", values="Effect")
        .reindex(columns=ORDER).to_string(float_format=lambda v: f"{v:.4f}"))
print(f"\noutputs written to {OUT}")
