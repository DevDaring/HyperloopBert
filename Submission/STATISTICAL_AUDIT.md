# STATISTICAL_AUDIT

## Conventions used throughout

| Quantity | Definition |
|---|---|
| per-item effect | `e_i = (PLL_stereo − PLL_anti) / T` (full-sentence) or the same difference restricted to differing tokens (changed-token) |
| architecture contrast | `d_i = e_i(A) − e_i(B)`, paired on the item |
| test | two-sided sign-flip permutation, m = 10 000 |
| p-value | `(b+1)/(m+1)` — Phipson & Smyth, never an exact zero |
| interval | 10 000-resample percentile bootstrap **over benchmark items** |
| Cohen's d | `mean(d) / sd(d)`, i.e. SD of the **paired differences** |
| correction | Holm within each contrast family, family stated at each use |
| clustered model | OLS with cluster-robust SEs, clusters = benchmark items |

## What the intervals do and do not cover

Every interval quantifies **benchmark-item sampling**. With one training seed per
architecture, none of them covers training-run variation. This is stated in the
paper and is its largest limitation.

## Multiplicity

- Three baseline-versus-shared contrasts: Holm within that family.
- Looped-vs-Hyperloop: reported unadjusted; it is a separate question, not a
  fourth test of the same hypothesis.
- Nine per-category tests: Holm **within each contrast**, and labelled
  descriptive rather than confirmatory.

## Confirmatory vs exploratory

**Confirmatory** (specified before analysis of the comparison point): the three
baseline-versus-shared contrasts at the deepest matched point; the three-leg
capability gate.

**Exploratory** (run after the single-point instability was found): all-band
recomputation, continuous loss adjustment, scoring-formulation comparison,
per-category analysis, leave-one-category-out, the StereoSet replication, and
TOST equivalence with its ±0.0118 margin. The paper says so in the Limitations.
The StereoSet direction rule was, however, committed to in writing before the
numbers were computed, and is reported under that rule whichever way it fell.

## Known statistical limitations

1. **No mixed model.** With four architectures and 4–7 snapshots each there is
   too little between-group structure to identify random slopes. A cluster-robust
   OLS is reported instead, and the substitution is stated rather than hidden.
2. **Interaction test.** `compare_lr_test` is not valid under a robust covariance
   and statsmodels warns accordingly; the block p-value (0.39) is therefore
   reported as indicative only, alongside the individual coefficient.
3. **R² ≈ 0.0006.** Item-level variance dominates. The model identifies a mean
   shift, not a predictive relationship, and is not presented as the latter.
4. **Non-independence across snapshots.** The same items recur at every
   checkpoint; clustering addresses the standard errors but the design remains
   repeated-measures with few groups.
5. **n = 4 for the sharing-ratio trend.** Spearman on four architectures
   (ρ = −0.32, p = 0.68) is descriptive only.

## Reproducibility of the analysis

Seeds fixed at 20260729. Re-running both analysis scripts produced 12/12
byte-identical output files. Software: Python 3.10.12, NumPy 2.2.6, pandas 2.3.3,
SciPy 1.15.3, statsmodels 0.14.6.
