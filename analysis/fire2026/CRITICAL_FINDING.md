# CRITICAL: the headline claim does not survive Phase 2

**Date:** 2026-07-29 · **Source:** `phase2_reanalysis.py`, `phase2b_allband.py`
**Status:** blocking — Phase 4 (GPU spend) held until this is resolved.

The paper currently claims that weight-shared encoders record a lower stereotype
effect size than the unshared baseline at matched validation loss. Three
independent checks from Phase 2 each weaken that claim, and together they
withdraw its support.

## 1. The contrast reverses across matched-loss points (item 2.5)

Because several nominal bands share one snapshot, the seven bands collapse to
five distinct Vanilla-vs-Looped comparisons. The claimed direction appears at
exactly one of them — the band the paper reports.

| Comparison (Vanilla loss vs Looped loss) | Δ | p |
|---|---|---|
| 2.997 vs 3.141 | −0.0094 | 0.0787 |
| 2.997 vs 2.968 | **−0.0222** | **0.0002** |
| 2.587 vs 2.688 | −0.0092 | 0.0683 |
| 2.353 vs 2.396 | −0.0077 | 0.1881 |
| **2.183 vs 2.192 (headline band)** | **+0.0237** | **0.0001** |

Positive in **1 of 5**. Mean Δ across bands is **−0.0063**, i.e. the shared
encoder is on average *higher*, not lower. At the 2.997-vs-2.968 comparison — a
tighter loss match (0.029 nats) than the headline band — the effect is
significant in the **opposite** direction.

Vanilla vs Hyperloop is positive at 5/7 but significant at only 2/7.
Vanilla vs ALBERT tracks the residual loss gap closely
(Spearman = −0.889, p = 0.007), which is the confound the protocol was meant to
remove.

## 2. The untested contrast is not significant (item 2.1)

Vanilla vs ALBERT at the headline band: Δ = 0.0116, p = 0.0653,
95% CI [−0.0008, 0.0240]. The interval includes zero. The most heavily shared
encoder does **not** differ significantly from the unshared baseline, so the
abstract's "both weight-shared encoders" was already wrong on two counts: there
are three shared encoders, and one of them shows no effect.

## 3. The contrast does not survive the official scorer (item 3.4)

The paper scores a length-normalised full-sentence pseudo-log-likelihood. The
official CrowS-Pairs script scores only the tokens that differ between the pair.
Both variants were already stored per item, so this cost nothing to check.

| Contrast | Δ (changed-token) | p | p_Holm |
|---|---|---|---|
| Vanilla vs Looped | 0.0054 | 0.368 | 0.735 |
| Vanilla vs Hyperloop | 0.0143 | 0.021 | 0.063 |
| Vanilla vs ALBERT | 0.0062 | 0.393 | 0.735 |

None survives correction. Per-item rank agreement between the two scorers is only
ρ ≈ 0.46–0.49, so they are not measuring the same thing at item level.
(8 of 1508 pairs could not be aligned by the changed-token scorer and were
dropped from this comparison only.)

## 4. What still holds

- **The equivalence result strengthens.** TOST gives p = 0.0233 against a
  pre-stated bound of ±0.0118, so Looped and Hyperloop are *statistically
  equivalent*, not merely "not distinguishable". This is a positive claim.
- **The capability-gate failure is confirmed rigorously.** Jeffreys intervals on
  374 WinoBias items include 0.5 for all eight architecture-split combinations.
  "At chance" is now backed by an interval, not an eyeball.
- **Effect size rises with training** for three of four encoders
  (ALBERT ρ = −0.90, p = 0.037; Hyperloop ρ = −0.71, p = 0.071;
  Vanilla ρ = −0.80, p = 0.200), though LoopedBERT does not follow it
  (ρ = +0.30). The paper's "for every encoder" is too strong.
- **Per-bias-type structure is real and large.** Effect size ranges from about
  −0.08 (gender) to +0.47 (sexual orientation) — the aggregate hides sign
  reversals across categories.

## 5. Consequence

The evidence does not support "weight sharing reduces stereotype association".
It supports a narrower and still publishable claim:

> Applying an iso-loss protocol across the full band range shows that an
> architecture contrast which looks significant at a single matched-loss point
> does not persist across other matched-loss points, and does not survive a
> change of scorer variant. Single-band, single-scorer architecture comparisons
> of stereotype association are therefore fragile.

That is a methodology-and-negative-result paper. It is honest, it is supported by
the data already collected, and it makes the same experiments worth reporting.
The alternative — reporting the band-2.2 number alone — would not survive a
reviewer who asks for the other bands.
