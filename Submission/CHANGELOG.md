# CHANGELOG — original → revised manuscript

| # | OLD CLAIM | NEW CLAIM | WHY CHANGED | EVIDENCE |
|---|---|---|---|---|
| 1 | Title: *Does Weight Sharing Reduce Stereotype Association?* | *Single-Point Bias Comparisons Are Fragile: An Iso-Loss Sensitivity Study…* | the answer to the old question is "not reliably measurable this way"; the title now names what the paper establishes | whole revision |
| 2 | "both weight-shared encoders record a lower stereotype effect size" | contrasts named individually; ALBERT reported as null | there are **three** shared encoders and the third shows no effect (Δ 0.0116, p 0.065) | `contrasts_band2.2.csv` |
| 3 | "a difference … cannot be explained by one model having learned more language" | "iso-loss matching reduces training-progress confounding but leaves residual capability differences" | with realised loss in a clustered model, no architecture term reaches p<0.05 while the loss term does (−0.0184, p 0.0008) | `loss_adjusted_model.csv` |
| 4 | single matched point reported as the result | contrast reported at **all** distinct matched points | Vanilla-vs-Looped is positive at 1 of 5 points; at a tighter match it is −0.0222, p 0.0002 | `allband_contrasts.csv`, Fig. 2 |
| 5 | one scoring formulation | both formulations reported side by side | under the benchmark's changed-token rule no contrast survives Holm; item ρ ≈ 0.47 | `contrasts_changed_token_scorer.csv`, Fig. 3a |
| 6 | "the reduction follows from sharing weights across depth" | causal wording removed | the design compares bundles differing in more than sharing, and the effect does not survive adjustment | Rule 15 audit |
| 7 | "Hyperloop adds nothing" | "no detected additional reduction"; exploratory equivalence within ±0.0118 | absence of evidence was being reported as evidence of absence; TOST now supplies the positive statement, labelled exploratory | `tost_equivalence.csv` |
| 8 | equivalence margin implied as principled | explicitly **exploratory** | no mention of TOST/equivalence/SESOI in the pre-registration or Stage 3 code; margin chosen after seeing the contrast | grep + file timestamps |
| 9 | "reusing weights across depth makes optimisation more delicate" | "in these runs HyperloopBERT was markedly harder to optimise" | one seed; two divergences and one spike cannot establish an architecture property | Rule 14 |
| 10 | "for every encoder the effect size grows as training proceeds" | "for three of the four" | LoopedBERT trends the other way (ρ +0.30) | `effect_size_trend.csv` |
| 11 | coreference "at chance" asserted | Jeffreys intervals, n = 374, all 8 cells include 0.5 | the claim needed an interval, not an eyeball | `winobias_binomial_ci.csv` |
| 12 | aggregate score only | per-category contrasts added | 4 of 9 categories change sign across architectures; aggregate is carried by race-colour and disability | `per_category_contrasts.csv`, Fig. 3b |
| 13 | scorer disagreement unexamined | 8 unalignable pairs identified and explained | they differ in 4–9 words, not only the demographic term — an instance of documented benchmark item-quality problems | `unaligned_items.csv` |
| 14 | "released snapshots" (no location) | wording removed; anonymised-archive statement + camera-ready comment | double-blind: naming the location would identify the authors | ANONYMITY_AUDIT.md |
| 15 | "contrasts fixed in advance" | "specified before analysis of the comparison point" | the timestamped artifact exists but cannot be cited without deanonymising | ANONYMITY_AUDIT.md |
| 16 | no AI disclosure | Generative AI Use Disclosure section added | required by the FIRE/ACM policy | Rule 18 |
| 17 | no reproducibility section | Reproducibility section added | FIRE requests verification information | Rule 17 |
| 18 | qualitative probe table with an incorrect reading | table removed | the text claimed the scientist probe was strongest; ALBERT's secretary probe is higher | `mlm_targeted_contrast.csv` |
| 19 | no IR framing | one Discussion paragraph + two verified IR-fairness citations | venue fit without overclaiming deployment | Rule 24 |
| 20 | single benchmark | StereoSet intrasentence added (2106 items, same protocol, CPU, $0) | answers the "one benchmark" objection; direction agrees 3/3, significance 1/3 | `stereoset_contrasts.csv`, Table 4 |
| 21 | PLL defined as a raw sum with the difference divided by *T* | PLL defined as the **mean** log-probability per scoreable token; effect is the plain difference | the stated equation did not match the code, and the two differ for unequal-length pairs | `bias_metrics.py:113`; verified numerically to 1.3e-15 |

