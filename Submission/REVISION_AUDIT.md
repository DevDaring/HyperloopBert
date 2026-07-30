# REVISION_AUDIT — claim-by-claim reproduction

Every number in the original manuscript was recomputed from the per-item score
files before any text was changed. Nothing below is taken on trust from the
previous PDF or from `CRITICAL_FINDING.md`.

**Sources.** `analysis/fire2026/phase2_reanalysis.py`,
`phase2b_allband.py`, `phase3_robustness.py`, reading
`Codes/results/stage3/bias/multicrows_*_progress.csv` (37 files, 1508 items each)
and `winobias_summary.csv`. Outputs in `analysis/fire2026/out/`.
Seeds fixed (20260729); re-running produced **12/12 byte-identical CSVs**.
Python 3.10.12, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, statsmodels 0.14.6.

| # | Claim | Original statement | Reproduced evidence | Verdict | Required revision | Type | Source |
|---|---|---|---|---|---|---|---|
| 1 | Vanilla vs Looped at matched point | Δ 0.0237, p_Holm 0.0003 | Δ 0.02366, CI [0.0134, 0.0340], p_Holm 0.00030, d 0.115 | **Valid** | keep, add CI | confirmatory | contrasts_band2.2.csv |
| 2 | Vanilla vs Hyperloop | Δ 0.0241, p_Holm 0.0004 | Δ 0.02411, CI [0.0119, 0.0367], p_Holm 0.00030 | **Valid** (p_Holm differs: 3-contrast family) | keep, correct p_Holm | confirmatory | contrasts_band2.2.csv |
| 3 | "both weight-shared encoders record a lower effect" | abstract, contrib 2, §6, §8 | there are **three** shared encoders; ALBERT Δ 0.0116, CI [−0.0009, 0.0240], p 0.065 | **Invalid** | withdraw; name the contrasts | confirmatory | contrasts_band2.2.csv |
| 4 | "differences cannot be attributed to one model being better trained" | §5.2 | across 5 distinct matched points Vanilla-vs-Looped is positive at **1/5**, mean Δ −0.0063; at 2.997 vs 2.968 nats Δ −0.0222, p 0.0002 | **Invalid** | withdraw; report the reversal | exploratory | allband_contrasts.csv |
| 5 | iso-loss removes the confound | §3.2, §5.2 | clustered OLS over 31 668 obs: no architecture term reaches p<0.05; loss term −0.0184, p 0.0008 | **Too strong** | weaken to "reduces, does not eliminate" | exploratory | loss_adjusted_model.csv |
| 6 | scorer choice | not addressed | changed-token: Looped p_Holm 0.735, ALBERT 0.735, Hyperloop 0.063; item-level ρ 0.46–0.49 | **New — contradicts** | add as central experiment | exploratory | contrasts_changed_token_scorer.csv |
| 7 | "for every encoder the effect size grows as training proceeds" | §5.2 | ρ: ALBERT −0.90 (p 0.037), Vanilla −0.80 (p 0.20), Hyperloop −0.71 (p 0.071), **Looped +0.30** | **Too strong** | say "for three of four" | exploratory | effect_size_trend.csv |
| 8 | Looped vs Hyperloop "adds nothing" | contrib 3, §6 | Δ 0.00045, p 0.934; TOST vs ±0.0118 gives p 0.0233 | **Valid but mislabelled** | "no detected difference" + exploratory equivalence | exploratory | tost_equivalence.csv |
| 9 | equivalence margin pre-specified? | implied | **no** mention of TOST/equivalence/SESOI in pre-registration or Stage 3 code; script written 2026-07-29 06:26, after the result | **Invalid if claimed** | label exploratory | — | grep + file mtime |
| 10 | coreference "at chance" | §5.3 | Jeffreys intervals over n=374 include 0.5 for **all 8** architecture×split cells; pro-accuracy 0.489–0.503 | **Valid** | keep, add intervals and n | confirmatory | winobias_binomial_ci.csv |
| 11 | GLUE leg | "did not yield usable numbers" | results dir empty; nothing recoverable | **Valid** | state the gate cannot pass regardless | — | filesystem |
| 12 | "reduction follows from sharing weights" | §5.5 | not supported once loss is adjusted for | **Invalid** | delete causal claim | — | loss_adjusted_model.csv |
| 13 | "weight reuse makes optimisation more delicate" | §5.6 | one Hyperloop divergence ×2 LRs, one ALBERT spike, **one seed** | **Too strong** | restrict to "in these runs" | descriptive | training log |
| 14 | aggregate hides category structure | not addressed | 4/27 category tests survive Holm; **4/9 categories change sign** across architectures | **New** | add | exploratory | per_category_contrasts.csv |
| 15 | scientist probe strongest | §5.4 | ALBERT secretary +3.80 > scientist +3.65 | **Invalid** | table dropped in revision | — | mlm_targeted_contrast.csv |
| 16 | "same 7 billion tokens" vs "quarter fewer tokens" | §4 / §5.1 | same corpus; matched point reached at 1.53–2.07 B | **Internally inconsistent** | say "same corpus", give token range | — | summary_table.csv |
| 17 | 8 unalignable items | not reported | 4–9 words differ per pair, 5/8 differ in length; drop-vs-impute changes Δ by <0.0006 | **New** | report as benchmark-quality evidence | exploratory | unaligned_items.csv |

## Summary

Two of the original confirmatory contrasts reproduce exactly. The generalisation
built on them does not: the untested third shared encoder is null, the contrast
reverses across matched points, it vanishes under continuous loss adjustment, and
it does not survive the benchmark's own scoring formulation. The revision keeps
the reproduced numbers and withdraws the generalisation.
