# CLAUDE CODE INSTRUCTION PROMPT — FIRE 2026 submission improvement
# Paper: "Does Weight Sharing Reduce Stereotype Association? An Iso-Loss Comparison of Looped Transformer Encoders"
# Generated: 2026-07-29. Venue deadline: 2026-08-15 (~17 days). Treat this file as the single source of truth for the task.

---

## 0. MISSION

You (Claude Code) are improving an existing, essentially complete paper for **FIRE 2026** (Regular Paper, double-blind).
The paper is scientifically sound but has fixable weaknesses that a reviewer will attack. Your job is to execute the phased plan below: run a small set of **budget-capped experiments and free analyses**, then rewrite the paper so every claim matches the evidence.

**Primary files**
- Paper source: `FIRE_HyperloopBERT.tex`
- Bibliography: `reference.bib`
- Figures: `images/` (you may add new figures here)
- Prior artifacts: all four pre-trained encoder snapshots + training code + per-item result files are on Hugging Face and GitHub (links/credentials in the repo and `.env`). **Read these first** — most new numbers below come from re-analysing existing artifacts, not new training.

**Hard venue constraints (non-negotiable)**
| Item | Requirement |
|---|---|
| Deadline | 2026-08-15 (plan backwards: all experiments done by 2026-08-06, text frozen by 2026-08-10) |
| Class | `\documentclass[sigconf,natbib=true,anonymous=true]{acmart}` |
| Length | **max 9 pages of content, references excluded** — every addition must fit; cut before you overflow |
| Review | **DOUBLE-BLIND** — see Phase 1, this is the most common desk-reject cause |
| Metadata | ACM CCS concepts + keywords required |

**Budget (hard cap, log everything)**
- Total GPU spend for this task: **≤ $15 USD**. CPU work is free and always preferred.
- `PHD_VAST_AI_KEY` is in `.env`. Use it ONLY in Phase 4 (GLUE leg) and only on a **spot/interruptible** instance (RTX 4090/A5000/A40 class, ≤ $0.45/hr). Destroy the VM immediately after use.
- Keep a running `COST_LOG.md`: instance type, hourly price, start/stop time, total. If the estimate for any step exceeds its cap, STOP and report instead of spending.
- **Never fabricate a number.** Every new number in the paper must trace to an executed run or a re-analysis script whose output and logs you commit.

---

## 1. PHASE 1 — COMPLIANCE & ANONYMISATION AUDIT (CPU-FREE, do first, gate everything else)

FIRE Regular Papers are double-blind. The submitted PDF must not identify the author.

1.1. Grep the entire `.tex` (and any included files) for: `Debk`, `DevDaring`, `huggingface.co/Debk`, `github.com/DevDaring`, `http`, `acknowledg`, `grant`, `funded`, institution names, and "our previous work". The known exposures are the Hugging Face namespace `Debk/HyperloopBERT`, the GitHub repo `DevDaring/HyperloopBert`, and any `--dataset-namespace Debk/...` strings.
1.2. Replace every exposure with an **anonymised mirror reference**: create/verify an `anonymous.4open.science` mirror (or a fresh anonymous HF account) and cite that. Leave a `% CAMERA-READY: restore real links` LaTeX comment at each site (comments must not render).
1.3. Confirm `anonymous=true` is set, no author block renders, no acknowledgements section exists, and any self-citation is written in third person ("Dey [x] showed…", never "we previously").
1.4. Verify compiled PDF shows ≤ 9 content pages (references excluded), CCS concepts + keywords render, and figures stay inside margins. NOTE: the current draft is only ~6 content pages — you have ~3 pages of budget for the new tables/figures below. Use them; do not pad prose.
1.5. **"Released snapshots" compliance.** §4 says snapshots are "released" but names no location. Either point to the anonymised mirror (1.2) in a short Data-Availability statement, or delete the word "released". Never name `Debk/...` or `DevDaring/...`.
1.6. **"Fixed in advance" provenance.** The paper repeatedly says targets/contrasts were "fixed in advance". The project pre-registration spec (§7.2, supports `--dataset-namespace` anonymised mirror) is the evidence. Cite the anonymised pre-registration/manifest artifact where these phrases appear; if no timestamped anonymisable artifact exists, soften to "specified before analysis of the comparison point".
**Acceptance:** a grep for the strings above returns zero hits in the rendered PDF text; `pdfinfo`/page count confirms the limit. Report the audit table before moving on.

---

## 2. PHASE 2 — FREE RE-ANALYSIS OF EXISTING RESULTS (CPU, no new training)

All per-item CrowS-Pairs scores for the four snapshots are already on GitHub/HF. Do these analyses with small scripts and commit outputs to `analysis/fire2026/`.

2.1. **Fix the untested contrast (highest-priority science fix).** The abstract and Contribution (2) say "both weight-shared encoders record a lower effect size", but there are THREE shared encoders and Table 3 never tests **VanillaBERT vs ALBERTLoopedBERT** (0.071 vs 0.059). Run the paired permutation test for Vanilla-vs-ALBERT (same item-level protocol, Phipson–Smyth p-values), label it **post-hoc**, and add it to Table 3 with Holm correction across all four Vanilla contrasts (or keep the three pre-registered contrasts and report this one separately, clearly marked post-hoc — choose one scheme and state it in §3.3).
2.2. **Per-bias-type breakdown.** CrowS-Pairs spans 9 bias types (gender, race, religion, age, nationality, disability, sexual orientation, socioeconomic, physical appearance). Compute per-type effect sizes with bootstrap intervals for all four encoders at the comparison point. Add a compact table or heatmap figure. This answers "where does the reduction come from?" at zero cost and is genuine new analysis, not padding.
2.3. **Stereotype-score metric.** Recompute the standard CrowS-Pairs stereotype score (% of pairs where the stereotypical member scores higher) per encoder from the same per-item scores, alongside the existing length-normalised effect size. Add one column to Table 2. This makes the numbers comparable to the published literature (needed in Phase 5.3).
2.4. **Non-monotonicity numbers.** Shared ratio is Vanilla 0.00 → Looped 0.50 → Hyperloop 0.50 → ALBERT 0.92, yet effect sizes are 0.071 → 0.047 → 0.047 → 0.059 (NOT monotone in sharing). Extract the exact numbers needed for the Phase 5 discussion of this.
2.5. **Loss-adjusted, all-band reanalysis (closes the biggest methodological hole).** Table 2's snapshots span 2.163–2.196 nats and the lower-effect encoders sit at HIGHER loss; since effect size grows as loss falls, part of the contrast could still be residual training progress — the exact confound the protocol claims to remove. Using ALL saved snapshots on GitHub/HF: (a) build a common realised-loss grid, interpolate (or locally regress) each encoder's effect-size trajectory, and recompute the Vanilla-vs-shared contrasts at EVERY common band, not just the headline band; (b) report the contrast-with-CI as a function of loss (small table or figure). If per-item PLLs were saved only for the headline band, re-score the other saved snapshots inference-only (CPU). Replace §5.2's "cannot be attributed to one model being better trained" with whatever this analysis supports.
2.6. **Uncertainty and trend statistics.** (a) Add bootstrap CIs for the paired Δ in Table 3 (currently only p and d). (b) State Cohen's d denominator (SD of paired differences). (c) §5.2's "for every encoder the effect size grows as training proceeds" conflicts with visible dips in Figure 2 — replace with "increases overall" plus a Spearman trend per encoder with CI. (d) §5.3's "at chance" coreference claim needs a binomial CI on pro-stereotype accuracy with N stated. (e) Leave-one-category-out rerun of the headline contrasts to show no single bias type drives them.
2.7. **Equivalence test for the null.** Looped-vs-Hyperloop (Δ=0.0005, p=0.47) is "no detected difference", not "adds nothing". Run a TOST equivalence test with a pre-stated smallest-effect-size-of-interest (e.g., half the Vanilla–Looped Δ) and report the equivalence bound; reword Contribution (3) and §6 to "no detected additional reduction beyond looping" and note the stability evidence comes from one implementation/configuration.
**Acceptance:** every new statistic has a script + committed output; numbers in the tex match the committed outputs exactly.

---

## 3. PHASE 3 — SECOND INSTRUMENT: StereoSet replication (CPU, inference-only)

The paper currently rests on ONE instrument (CrowS-Pairs, 1508 items). Reviewer objection: "single benchmark, known item-quality issues (Blodgett et al.)".

3.1. Score all four comparison-point snapshots on **StereoSet (intrasentence split)** using the SAME pseudo-log-likelihood protocol as §3.3 (32-bit scoring, same masking, same effect-size definition and bootstrap). This is inference-only — CPU is acceptable (est. a few hours); do NOT rent a GPU for this.
3.2. Report: StereoSet language-modelling score, stereotype score, and the paired Vanilla-vs-{Looped, ALBERT, Hyperloop} contrasts with Holm correction, in a new compact table in §5.
3.3. Interpretation rule (pre-commit to this): if the DIRECTION of the shared-vs-unshared difference matches CrowS-Pairs, state that the finding replicates across two instruments of the same correlation type; if it does not, report that honestly and soften the claim in the abstract/conclusion to CrowS-Pairs specifically. Either outcome is publishable; hiding a mismatch is not.
3.4. **Scorer validation.** The paper uses a custom length-normalised full-sentence PLL. Validate it: reproduce the official CrowS-Pairs scoring (changed-token-only variant) on the same four snapshots and report the agreement between variants (rank correlation of per-item effects; whether the headline contrasts hold under both). One short paragraph + one number set. If the contrast survives only under one scorer variant, say so.
3.5. **External reference calibration (clearly labelled).** Score two PUBLIC checkpoints — `bert-base-uncased` and `albert-base-v2` — on CrowS-Pairs with the identical pipeline, inference-only (CPU). Report them in §6 as EXTERNAL references at full training, explicitly NOT matched controls. This empirically anchors the literature point in Phase 5.2 (ALBERT at full training is among the most stereotype-preferring) without any claim of comparability to your snapshots.
3.6. Name the benchmark precisely in §3.3 (CrowS-Pairs, version/source, licence, 1508 items, category metadata) — it is currently identifiable only via citation.
**Acceptance:** new tables compile; direction decision recorded; scorer-agreement and external-reference numbers committed; abstract updated to match the evidence.

---

## 4. PHASE 4 — COMPLETE THE GLUE LEG OF THE CAPABILITY GATE (GPU-CHEAP, hard cap $8)

§3.4/§5.3 currently say "the GLUE leg did not yield usable numbers within the available compute". That is the weakest sentence in the paper: the gate framework has three legs and two are unmet. Fine-tuning the EXISTING snapshots is cheap — no pretraining.

4.1. Provision ONE vast.ai spot instance (RTX 4090/A5000/A40, ≤ $0.45/hr) using `PHD_VAST_AI_KEY`. Estimated total need: 4 snapshots × 3 tasks × ~20–40 min ≈ 4–8 GPU-hours ≈ $2–4. **Cap: $8.** If the spot price would exceed the cap, abort and keep the current wording.
4.2. Fine-tune each of the four comparison-point snapshots on **SST-2, MRPC, and RTE** (add CoLA only if time/budget remain). Identical fine-tuning recipe for all four (same epochs, LR sweep over the same small grid, selection on dev). Report dev-set accuracy (Matthews for CoLA).
4.3. Update §5.3: state which GLUE tasks each encoder passes above the chance/majority baseline, and state the gate outcome precisely (e.g., "legs 1–2 met, leg 3 unmet"). Then tighten §6's methodological point — it becomes STRONGER: "association instruments (PLL), GLUE-type understanding, and coreference behaviour become usable at different points in training; at this scale the first two are usable and the third is not."
4.4. Destroy the VM, record final spend in `COST_LOG.md`, push fine-tune configs/logs to GitHub.
**Acceptance:** Table (GLUE dev scores × 4 encoders) in §5.3; gate outcome text matches the table; cost log ≤ $8.

---

## 5. PHASE 5 — WRITING UPGRADES (CPU-FREE)

5.1. **Non-monotonicity discussion (mandatory).** A reviewer WILL ask why the most-shared encoder (ALBERT, 0.92 shared) does not show the lowest effect size. Add a paragraph in §6 using Phase 2.4 numbers. Candidate explanations to weigh (state as hypotheses, not findings): (a) with one block reused 12×, item-level associations must live inside that single block and may be re-encoded at every pass, whereas a 6-block loop distributes them; (b) ALBERTLoopedBERT is under visibly greater capacity pressure at this scale (its loss trajectory, Figure 1); (c) single seed — the ordering of the three shared encoders may not be stable. Connect to the limitation honestly.
5.2. **Literature anchor table (zero compute, big credibility win).** Add a short table/paragraph in §6 with PUBLISHED CrowS-Pairs stereotype scores for public models from Nangia et al. (2020): BERT ≈ 60.5, RoBERTa ≈ 67.3, ALBERT ≈ 66.8 (VERIFY the exact numbers from the paper before writing them). Key point: at FULL training, ALBERT-family models score among the MOST stereotype-preferring, which does not contradict the present result at ~2 nats — it is consistent with §5.2's finding that the effect size grows with training and with the limitation that longer training may widen/close/reverse the gap. This converts a weakness (undertrained models) into an articulated scientific question.
5.3. **FIRE topical fit.** FIRE is an IR venue listing "explainability and fairness" among topics. Add: (a) one paragraph in §1 (or §6) on why this matters for retrieval/ranking pipelines that sit on pre-trained encoders (association inherited by downstream rankers/classifiers; architecture choice as a pre-training-time fairness lever); (b) 3–5 IR-fairness citations in §2, e.g. Ekstrand et al., "Fairness in Information Access Systems" (FnTIR 2022); Singh & Joachims, "Fairness of Exposure in Rankings" (KDD 2018); plus one recent neural-IR bias study you verify exists. Do not overclaim IR relevance — one honest paragraph.
5.4. **Clarify the HyperloopBERT contribution vs Zeitoun et al. [22].** Add one sentence in §3.1 stating precisely how this implementation relates to Hyperloop Transformers (which components are shared/differ), so the negative result is read as about the mechanism, not an implementation artifact.
5.5. **Ethics statement + reproducibility statement** (short, ~100–150 words each, before references): benchmark limitations (Blodgett et al.), association-vs-harm distinction (Wang et al.), no new human data; reproducibility: corpus, tokenizer manifest/checksum, seed, hardware, token counts, anonymised artifact link.
5.6. **Tighten the abstract** after Phases 2–4: fix the "both weight-shared encoders" wording to match the final contrast set; keep the honest scope sentence (association in the MLM head, not downstream behaviour); ≤ 250 words.
5.7. **India-centric limitation**: keep and slightly strengthen §7's statement (dropped Indian-BhED/caste-religion instrument due to English-only tokenizer; first future extension). Given FIRE's regional focus this candour helps; do not quietly omit it.
5.8. **Narrow the claims to the evidence (mandatory re-tone).** The design compares four architecture bundles that differ in sharing AND parameter count AND optimisation path AND (for Hyperloop) learning rate and tokens consumed; iso-loss matches one scalar. Therefore: Contribution (1) "attributes a difference … to architecture" → "compares architecture bundles at matched validation loss"; avoid causal verbs ("reduces", "suppresses") for the headline claim — prefer "records a lower effect size at matched loss in this single-seed regime"; frame the paper explicitly as a **single-seed, early-training, PLL-level methodology study with suggestive evidence**. Keep the title, but ensure abstract/intro/conclusion carry this scoping.
5.9. **Fix every internal inconsistency below (verified against the current PDF):**
   - Abstract/Contrib (2)/§6/§8 say "both/two weight-shared encoders" — there are THREE (Looped, ALBERT, Hyperloop). Reword to name the focal contrasts or update per Phase 2.1 results.
   - §5.4 says the association "is strongest on the scientist probe for every encoder" — Table 4 shows ALBERTLoopedBERT's secretary log-odds (+3.80) exceed scientist (+3.65). Correct the sentence.
   - §4 "pre-trained on the same 7 billion tokens" vs §5.1 "HyperloopBERT reaches the comparison point with about a quarter fewer tokens" — write "same corpus" and add per-snapshot token/step counts to a reproducibility table.
   - Figure 1 shows one HyperloopBERT trajectory and one divergence marker but §5.6 describes TWO failed runs at different LRs — label which LR is plotted and mark both failure events (or state why only one is shown).
   - §5.1 "LoopedBERT is not under real capacity pressure" rests on loss alone while §5.3 reports chance coreference — qualify: "not under capacity pressure as measured by validation loss at this scale; capability evidence is limited (§5.3)".
   - §5.1 "quality falls as weight reuse rises" is not true at the comparison point (ALBERTLoopedBERT has the LOWEST loss, 2.163) — scope the sentence to final stable trajectories.
   - Table 2 rounds Looped and Hyperloop effect sizes both to 0.047 while Table 3 reports Δ=0.0005 — use ≥3 decimals consistently.
   - §5.5's "disagreement" metric: define it (formula), give units, and add a CI for the 0.40 value and the representational-similarity claim.
5.10. **Reproducibility table (use the spare page budget).** One compact table or appendix-style subsection: nominal target-loss list and which bands shared snapshots; validation interval and validation-set size; snapshot token/step counts; bootstrap/permutation draw counts and seeds; masking rate, LR schedule, weight decay, Adam betas, dropout, heads, FFN size; whether layer norms are shared; the exact Table 4 probe prompts; PLL treatment of special tokens and definition of T for unequal pair lengths; WinoBias adaptation details and split size.

---

## 6. PHASE 6 — BIBLIOGRAPHY HYGIENE (CPU-FREE)

6.1. For every arXiv entry in `reference.bib` (currently at least [5], [6], [7], [17], [20], [21], [22], [25]), check DBLP/Semantic Scholar for a published peer-reviewed version; upgrade where one exists (venue, year, pages), keep arXiv otherwise. **Never invent a venue** — if you cannot verify, keep the arXiv entry.
6.2. Add the new citations from Phases 3 (StereoSet is already [11]) and 5.3 with complete, verified metadata.
6.3. Compile with natbib and confirm no missing/dangling references and consistent style.

---

## 7. PHASE 7 — FIGURES, RENDERING, PAGE BUDGET

7.1. Produce at most TWO new figures/tables beyond existing ones (suggested: per-bias-type heatmap from 2.2; GLUE/gate summary table from 4.3; StereoSet table from 3.2 — pick what fits). Keep visual style consistent with Figures 1–2 (low-saturation, clean).
7.2. Recompile; enforce the 9-page content limit by tightening prose (do NOT shrink fonts or margins).
7.3. Render final PDF; run the Phase 1 anonymisation grep on the RENDERED pdf text again.

---

## 8. OPTIONAL STRETCH — only if ALL of the following hold (otherwise skip silently)

Gates: (a) Phases 1–7 complete; (b) ≥ 8 days left before 2026-08-15; (c) total spend so far ≤ $8; (d) you can stay within an ADDITIONAL $7.
Task: a **small-scale seed-sensitivity check** of the headline contrast only: Vanilla vs Looped, reduced config (e.g., hidden 384, 8 effective depths, ≤ 1.5B training tokens), TWO seeds each, iso-loss snapshot at one matched band, CrowS-Pairs scoring. Report in §7 as "a small-scale two-seed check shows the contrast direction is consistent / is not yet conclusive", whichever the data say. This pre-empts the #1 reviewer objection (single seed) at minimal cost. If any gate fails, instead strengthen the single-seed limitation sentence and move on.
**Abort rule:** any divergence, spot termination twice, or projection > $15 total → stop, destroy VM, keep existing text.

---

## 9. GLOBAL EXECUTION RULES

- Work order: Phase 1 → 2 → 3 → 4 → 5 → 6 → 7 (Phase 8 optional). Do not start writing (Phase 5) until experiment numbers exist.
- Commit as you go: analysis scripts, run configs, logs, COST_LOG.md, and a short `FIRE2026_CHANGES.md` summarising what changed and why.
- Every number in the tex must match a committed output. If a number in the current pdf turns out to be unreproducible from the artifacts, FLAG it — do not silently patch.
- Do not rename architectures, do not change the pre-registered protocol retroactively, do not add experiments beyond this list.
- If any instruction conflicts with the double-blind requirement, the double-blind requirement wins.

**Final QA gate (must all be YES before you declare done):**
1. Rendered PDF: no author-identifying strings (Phase 1 grep on PDF text); "released snapshots" backed by an anonymous artifact or removed; "fixed in advance" backed by the anonymised pre-registration or softened.
2. ≤ 9 content pages, references excluded; CCS + keywords present.
3. Abstract claims == Table 2/3/5 evidence (including the fixed ALBERT contrast); causal verbs removed from headline claims.
4. Capability-gate section matches the new GLUE table; coreference "at chance" backed by a binomial CI.
5. StereoSet section present with direction statement; scorer-validation agreement reported.
6. Loss-adjusted/all-band analysis reported; "cannot be attributed to better training" wording matches what that analysis supports.
7. All §5.9 internal inconsistencies fixed; reproducibility table present.
8. Every arXiv ref verified; no fabricated venues; every figure/table referenced; ≥3-decimal consistency in Tables 2–3.
9. COST_LOG.md ≤ $15 total; VM destroyed.
10. CAMERA-READY comments present for restoring real HF/GitHub links.
