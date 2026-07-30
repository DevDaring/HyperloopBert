
You are acting as the lead research scientist, statistical reviewer, LaTeX editor,
and reproducibility auditor for my FIRE 2026 submission.

This is a high-stakes revision. Do NOT optimise for preserving the existing story.
Optimise for scientific correctness, reviewer defensibility, reproducibility,
FIRE 2026 fit, and acceptance probability.

You have access to the entire repository, including:

- FIRE_HyperloopBERT.tex and all LaTeX source
- reference.bib
- code
- checkpoints
- cached predictions
- training logs
- experimental results
- figures
- phase2_reanalysis.py
- phase2b_allband.py
- CRITICAL_FINDING.md
- FIRE2026_REQUIREMENTS.md
- any preregistration/specification documents
- GPU access through the environment variable PHD_VAST_AI_KEY

IMPORTANT SECURITY RULE:
PHD_VAST_AI_KEY is a secret.
Never print it, echo it, write it to a file, place it in commands shown in reports,
commit it, put it into LaTeX, or expose it in logs.
Read it only from the environment if GPU access is absolutely required.

============================================================
PRIMARY OBJECTIVE
=================

Transform the existing manuscript into the strongest scientifically honest FIRE
2026 submission supported by the actual evidence.

The current manuscript's headline claim appears to have failed subsequent
robustness analysis.

DO NOT attempt to rescue the statement:

    "weight sharing reduces stereotype association"

unless new analysis genuinely demonstrates that it is robust.

The new candidate thesis is:

    Architecture-level conclusions about stereotype association are fragile when
    drawn from a single capability-matched checkpoint and a single benchmark
    scoring definition. Across the full set of distinct matched-loss comparisons,
    the apparent reduction associated with looping is inconsistent, and the
    architecture contrast largely disappears under the changed-token CrowS-Pairs
    scoring rule. Therefore bias comparisons across neural architectures should
    test robustness across capability levels and scoring formulations rather than
    report a single matched operating point.

Treat that as a hypothesis to verify, NOT as text that must be preserved.

============================================================
RULE 1 — REPRODUCE EVERYTHING FIRST
====================================

Do not trust either the current paper or CRITICAL_FINDING.md merely because they
contain numerical values.

Before editing the paper:

1. Identify every script/data/checkpoint producing the reported numbers.
2. Re-run all CPU-feasible analyses.
3. Reproduce every critical statistic independently from raw/cached outputs.
4. Record the exact scripts, input files, seeds, item counts, excluded items and
   software versions.
5. Compare reproduced numbers against:

   - FIRE_HyperloopBERT.tex
   - CRITICAL_FINDING.md
6. Create:

   REVISION_AUDIT.md

with columns:

Claim
Current paper statement
Reproduced evidence
Valid / Invalid / Too strong
Required revision
Confirmatory or exploratory
Source script/file

Do not modify the scientific narrative until this audit is complete.

============================================================
RULE 2 — WITHDRAW UNSUPPORTED CLAIMS
=====================================

Explicitly search the complete LaTeX source for every occurrence or paraphrase of:

- weight sharing reduces stereotype association
- weight sharing lowers bias
- shared encoders record lower stereotype effects
- both weight-shared encoders
- direction is consistent
- for every encoder the effect increases
- differences cannot be explained by training quality
- reduction follows from sharing weights
- Hyperloop proves...
- fairness improvement
- debiasing
- downstream fairness

Check title, abstract, introduction, contributions, Results, figure captions,
Discussion, Limitations and Conclusion.

Delete or rewrite every unsupported occurrence.

There are THREE architectures with cross-depth weight sharing:
LoopedBERT, HyperloopBERT and ALBERTLoopedBERT.

Never refer to LoopedBERT and HyperloopBERT as "both weight-shared encoders" when
ALBERTLoopedBERT is also part of the experiment.

============================================================
RULE 3 — REBUILD THE PRIMARY ANALYSIS ACROSS ALL ISO-LOSS POINTS
=================================================================

Do not use the 2.183-vs-2.192 comparison as the sole headline result.

Bands that map to the same snapshot must NOT be treated as independent
observations.

Build a canonical table of DISTINCT model snapshots / matched comparisons.

For every distinct matched-loss comparison calculate:

- realised validation loss for each architecture
- absolute loss gap
- mean stereotype effect
- bootstrap 95% CI
- paired architecture Δ
- paired permutation p-value
- corrected p-value where inferentially appropriate
- Cohen's d or the existing paired standardised effect
- number of items

At minimum compare:

Vanilla vs Looped
Vanilla vs Hyperloop
Vanilla vs ALBERT

Do not omit ALBERT merely because its result is inconvenient.

Also analyse Looped vs Hyperloop separately as an equivalence question.

Produce a compact figure showing architecture Δ across realised validation loss.

Zero must be clearly visible.

The figure should make sign reversals obvious.

============================================================
RULE 4 — MODEL VALIDATION LOSS CONTINUOUSLY
============================================

The existing manuscript makes the claim that matching within approximately
0.034 nats removes training-progress confounding.

Do not assume this.

Use the repeated measurements across distinct snapshots to test whether residual
validation-loss differences are associated with architecture effect differences.

Use an appropriate repeated-measure / clustered analysis, considering:

- mixed-effects modelling,
- cluster-robust regression,
- GEE,
- or another statistically defensible model.

Items are repeatedly measured across checkpoints, so do not treat all
checkpoint-item observations as independent.

Estimate architecture effects while adjusting continuously for realised validation
loss.

Test architecture × validation-loss interaction where justified.

Report diagnostics and uncertainty.

If the data are insufficient for a reliable mixed model, explicitly state that and
use a simpler sensitivity analysis rather than pretending otherwise.

Rewrite "iso-loss removes the confound" to the weaker scientifically justified
claim:

    iso-loss matching reduces training-progress confounding but approximate
    checkpoint matching may leave residual capability differences.

============================================================
RULE 5 — SCORER ROBUSTNESS IS A CENTRAL EXPERIMENT
===================================================

Evaluate BOTH scoring formulations:

A. length-normalised full-sentence pseudo-log-likelihood used in the original paper
B. changed-token scoring corresponding to the CrowS-Pairs evaluation implementation

Verify exactly what each scorer computes and document the difference.

For each architecture comparison report:

- Δ
- raw p
- corrected p
- CI
- effect size
- scorer

Report item-level rank agreement between scorers.

Investigate the 8 items reported as unalignable:

- identify them
- explain why alignment failed
- verify that dropping them does not create the conclusion
- give a sensitivity result with an appropriate treatment if possible

Do not call either scorer "the correct scorer" unless justified by the benchmark
documentation.

Use terminology such as:
"full-sentence PLL formulation" and "changed-token CrowS-Pairs formulation."

The scientific point is scorer sensitivity, not accusing one implementation of
being wrong.

============================================================
RULE 6 — CATEGORY HETEROGENEITY
================================

The aggregate score appears to hide substantial category heterogeneity.

Generate per-bias-category analyses for all sufficiently populated CrowS-Pairs
categories.

For each category report:

- item count
- architecture effect estimates
- bootstrap CIs
- paired differences where meaningful

Create one compact figure showing category-level effects and sign reversals.

Do not overinterpret very small categories.

Apply correction if category-wise significance tests are reported.

Distinguish descriptive heterogeneity from confirmatory hypothesis testing.

Discuss how an aggregate bias number can hide opposing category patterns.

============================================================
RULE 7 — LOOPED vs HYPERLOOP EQUIVALENCE
=========================================

Reproduce the TOST result between LoopedBERT and HyperloopBERT.

A reported result is:

equivalence bound ±0.0118
TOST p ≈ 0.0233

Verify it from source data.

CRITICAL:
Search timestamped preregistration/specification/git history to determine whether
±0.0118 was defined BEFORE examining this outcome.

If genuinely pre-specified:
call it a pre-specified equivalence margin.

If it was defined after seeing the result:
call it an exploratory equivalence analysis.

Never falsely use "pre-registered", "pre-specified", or "fixed in advance."

Also test whether the equivalence conclusion is sensitive to:

- matched-loss point
- scoring formulation

if this can be done from existing cached results.

Do not generalise equivalence beyond the conditions actually supported.

============================================================
RULE 8 — CAPABILITY GATE
=========================

Reproduce the WinoBias/coreference capability-gate analysis.

Calculate Jeffreys or another justified binomial interval for each
architecture × split combination.

If all intervals include 0.5, clearly state that the coreference capability gate
fails.

Do NOT spend substantial GPU time trying to make the gate pass.

The correct conclusion is that the CrowS-Pairs result concerns association in the
masked-language-model head and cannot be interpreted as demonstrated downstream
behavioural fairness.

Check whether any usable GLUE results already exist in logs/checkpoints.

If existing results can be recovered, report them.

If GLUE requires new expensive training, do NOT run it merely to make the paper
look complete because the coreference gate already fails.

Do not quietly delete a preregistered gate.

============================================================
RULE 9 — GPU BUDGET
====================

DEFAULT DECISION: NO NEW GPU TRAINING.

First exhaust:

- cached predictions
- saved logits
- existing checkpoints
- training logs
- saved evaluation outputs
- CPU statistical analysis

Do NOT repeat four 7B-token pretraining runs.

Do NOT train new architectures simply to save the original hypothesis.

Before ANY new GPU experiment:

1. Write GPU_REQUIRED.md containing:

   - exact reviewer problem addressed
   - why existing evidence cannot answer it
   - experiment
   - models/checkpoints
   - dataset
   - estimated GPU type
   - estimated runtime
   - current Vast.ai hourly price
   - total expected monetary cost
   - expected scientific value
   - what conclusion changes depending on the result
2. Inspect existing project notes/logs to determine compute already consumed.
3. Respect the overall low-compute budget already specified for this project.
   Do not assume unused budget exists merely because a GPU key exists.
4. Query Vast.ai prices before launching anything.
5. Prefer evaluation/rescoring over training.
6. Use the cheapest GPU that can safely execute the task.
7. Terminate the instance immediately after results are copied and verified.
8. Never leave a Vast.ai instance running unattended.

Only run a GPU experiment when it addresses a likely reviewer-blocking problem
and fits the remaining documented budget.

If an experiment would merely provide a nicer figure or another weak robustness
check, do not spend GPU money.

============================================================
RULE 10 — SEEDS
================

Search thoroughly for checkpoints/results from additional training seeds.

If additional seeds already exist, analyse them.

If they do not exist, do not manufacture pseudo-replication by treating
checkpoints/bands/items as training seeds.

State clearly that training uses one seed.

Bootstrap CIs across CrowS-Pairs items measure benchmark-item uncertainty, NOT
training-run uncertainty.

Make that distinction explicit in the paper.

Do not undertake several new full pretraining runs under the current deadline
unless an existing budget document shows that they are trivially affordable.

============================================================
RULE 11 — REFRAME THE PAPER
============================

Consider replacing the title with:

"Single-Point Bias Comparisons Are Fragile:
An Iso-Loss Sensitivity Study of Weight-Shared Transformer Encoders"

Alternative:

"When Does Weight Sharing Appear to Reduce Stereotype Association?
A Robustness Study Across Capability Levels and Scorers"

Choose the title that most accurately matches the reproduced results.

The revised contributions should approximately become:

1. An iso-loss robustness framework for separating architecture comparisons from
   gross training-progress differences while explicitly testing sensitivity to
   the selected matched capability level.
2. Empirical evidence that the apparent architecture effect on CrowS-Pairs is
   sensitive to matched-loss point and scoring formulation, making a simple claim
   that weight sharing reduces stereotype association unsupported.
3. Evidence that LoopedBERT and HyperloopBERT have equivalent association within
   a practically defined margin under the conditions for which equivalence is
   actually supported, suggesting that the extra routing mechanism does not yield
   a measurable association benefit there.
4. Evidence that aggregate stereotype scores conceal substantial bias-category
   heterogeneity and sign changes.
5. A capability-gate result showing that these association measurements should
   not be interpreted as downstream behavioural fairness at the present training
   scale.

Modify these depending on reproduced evidence.

============================================================
RULE 12 — DECIDE REGULAR vs PERSPECTIVE PAPER
==============================================

Read FIRE2026_REQUIREMENTS.md and the current FIRE 2026 CFP carefully.

Assess both:

A. Regular Paper
B. Perspective Paper

Given the revised methodological/negative-result story, specifically evaluate
whether Perspective Paper is the stronger fit.

A Perspective framing could be:

"Fairness comparisons between neural architectures should not be based on a
single capability-matched checkpoint and a single scorer."

If selecting Perspective Paper, create a short explicit section such as:

"Perspective: Bias Evaluation as a Robustness Problem"

Explain:

- conventional practice being challenged
- why single operating-point comparisons are vulnerable
- evidence from this study
- recommended evaluation protocol
- boundaries of the argument

Do not artificially call it a Perspective Paper merely for lower perceived
reviewer expectations. It must genuinely satisfy the track definition.

Write TRACK_DECISION.md with:

- recommended track
- reasoning
- risks
- changes needed for that track

============================================================
RULE 13 — REDUCE REPETITION
============================

The existing paper unnecessarily repeats several messages.

Search for repetition of:

A. weight sharing lowers stereotype association
B. Hyperloop adds no reduction
C. Hyperloop is harder to optimise
D. capability gate is not met
E. association does not imply downstream fairness

Use this structure:

Abstract:
one concise statement.

Introduction:
motivation and contribution only.

Results:
full evidence.

Discussion:
interpretation, not repetition of numbers.

Limitations:
scope restrictions only.

Conclusion:
2–3 key findings without re-explaining all experiments.

Reduce rhetorical repetition aggressively while preserving necessary scientific
signposting.

============================================================
RULE 14 — TRAINING STABILITY CLAIM
===================================

The existing manuscript risks generalising from individual runs that:

"weight reuse makes optimisation more delicate."

One Hyperloop run diverging and one ALBERT transient spike under a single seed is
not sufficient to establish a general architecture property.

Use wording such as:

"In these runs, HyperloopBERT exhibited substantially greater optimisation
instability..."

Do not claim a general causal relationship without multi-seed evidence.

Separate observed fact from mechanistic interpretation.

============================================================
RULE 15 — CAUSAL LANGUAGE
==========================

Audit the entire manuscript for:

causes
reduces
improves fairness
due to
results from
follows from
suppresses
makes

Replace causal language with associational language unless the experimental design
supports causality.

Examples:

BAD:
"the stereotype reduction follows from sharing weights"

BETTER:
"the observed difference at this operating point is associated with the
weight-sharing configuration."

But after the robustness analysis, even that statement may need to be removed.

============================================================
RULE 16 — REFERENCES
=====================

Audit EVERY bibliography entry.

For each citation verify:

- paper exists
- author names
- title
- year
- venue
- pages
- DOI where available

Prefer published conference/journal versions over arXiv when available.

Pay special attention to recent 2025/2026 references.

Never invent a citation.

Create REFERENCES_AUDIT.md.

If you cannot verify a reference, flag it rather than guessing.

============================================================
RULE 17 — REPRODUCIBILITY
==========================

FIRE requests enough information for verification and encourages a reproducibility
statement.

Add a compact Reproducibility section containing, as appropriate:

- data
- tokenizer
- architecture configuration
- training budget
- seeds
- checkpoint selection
- scoring implementation
- statistical tests
- software versions
- hardware
- code availability

For double-blind review, all repository URLs must be anonymous.

Search the source, bibliography, PDF metadata, comments and supplementary files
for identifiers including:

Debk
DevDaring
author names
institution names
personal usernames
email addresses
grant identifiers

Do not expose:
Debk/HyperloopBERT
DevDaring/HyperloopBert
or any identifying dataset namespace.

Prepare ANONYMITY_AUDIT.md.

============================================================
RULE 18 — GENERATIVE AI DISCLOSURE
===================================

FIRE 2026 explicitly requires compliance with its GenAI/ACM policy.

This research/revision has used Claude.

Determine exactly how Claude was used:

- writing/editing only?
- statistical analysis?
- code generation?
- experiment design?
- result interpretation?
- figure generation?

AI use relevant to the actual research methodology/results must be disclosed
accurately.

FIRE explicitly asks for disclosure of AI-generated/AI-assisted content.

Do not hide the disclosure because the manuscript is double-blind.

Create a short non-identifying "Generative AI Use Disclosure" before References,
or another FIRE/ACM-compliant location that remains visible in anonymous mode.

Do not use an Acknowledgements environment if the anonymous ACM template suppresses
it.

Never describe Claude as an author.

The human authors remain responsible for every number and claim.

============================================================
RULE 19 — ACCESSIBILITY AND FIRE FORMAT
========================================

Maintain:

\documentclass[sigconf,natbib=true,anonymous=true]{acmart}

Regular/Perspective submissions are double blind.

Maximum:
9 pages CONTENT
references excluded.

ACM CCS concepts and keywords are mandatory.

Add meaningful \Description{} text to figures.

Ensure all figures remain interpretable when printed and for readers with
colour-vision deficiencies.

Do not depend only on colour to distinguish architectures:
use markers, line styles, labels, or combinations.

The present manuscript is substantially under the 9-page content allowance,
so use extra space for ROBUSTNESS ANALYSIS, not filler.

============================================================
RULE 20 — FIGURES AND TABLES
=============================

Reconsider all current figures/tables.

Likely final visual structure:

Figure 1:
training trajectories / validation loss, only if necessary to explain matching.

Figure 2:
architecture association effect across DISTINCT validation-loss checkpoints.

Figure 3:
architecture contrast Δ across capability levels and both scoring formulations,
or a compact multi-panel robustness figure.

Figure 4 if space permits:
category-level heterogeneity.

Table 1:
architecture specifications.

Table 2:
primary robust statistical findings.

Table 3:
equivalence/capability/scorer robustness summary if necessary.

Do not retain a table merely because it existed in the old manuscript.

Every figure should answer a reviewer question.

============================================================
RULE 21 — ABSTRACT
===================

Completely rewrite the abstract after the analyses are frozen.

Suggested logical structure:

1 sentence:
why architecture/bias comparisons are confounded by training capability.

1 sentence:
iso-loss experiment with four encoders and CrowS-Pairs.

2–3 sentences:
actual robustness findings across matched-loss points and scorers.

1 sentence:
Looped-Hyperloop equivalence if verified.

1 sentence:
coreference gate limitation.

Final sentence:
methodological implication.

Do not use "debiasing", "fairer", "bias reduction", or "weight sharing reduces
stereotypes" unless supported by the final analysis.

============================================================
RULE 22 — DISCUSSION
=====================

The Discussion should answer:

1. Why did the original single-point analysis look convincing?
2. Why does all-band analysis change that conclusion?
3. Why can two plausible PLL scoring formulations disagree?
4. What does approximate iso-loss matching solve, and what does it not solve?
5. What does Looped-Hyperloop equivalence tell us?
6. Why do category-specific sign changes matter?
7. What can researchers designing architecture fairness comparisons learn from
   this?

Do not speculate about a biological/representational mechanism without evidence.

============================================================
RULE 23 — LIMITATIONS
======================

Explicitly retain:

- one training seed
- English only
- CrowS-Pairs benchmark quality limitations
- capability gate failure
- incompletely trained encoders
- approximate rather than exact loss matching
- sensitivity to scorer
- exploratory nature of analyses added after discovering the headline instability

Do not bury the post-hoc nature of the robustness analyses.

Reviewer trust is more important than preserving a dramatic result.

============================================================
RULE 24 — FIRE RELEVANCE
=========================

Strengthen the connection to FIRE without artificial claims.

Explain why the result matters for IR/NLP systems using pretrained encoders:
architecture comparisons used in ranking, classification, retrieval and related
pipelines can produce apparently different fairness conclusions depending on
evaluation operating point and scorer.

Do not claim that the models were evaluated in a deployed IR system unless they
actually were.

Keep the India-centric omission as an explicit limitation/future direction.

Do not add a rushed India-bias experiment simply for venue cosmetics unless there
is already a valid compatible dataset and it can be evaluated methodologically
correctly at negligible cost.

============================================================
FINAL DELIVERABLES
==================

After completing the work, produce:

1. FIRE_HyperloopBERT_REVISED.tex
2. FIRE_HyperloopBERT_REVISED.pdf
3. REVISION_AUDIT.md
4. STATISTICAL_AUDIT.md
5. REFERENCES_AUDIT.md
6. ANONYMITY_AUDIT.md
7. TRACK_DECISION.md
8. GPU_REQUIRED.md, only if GPU experimentation was genuinely necessary
9. CHANGELOG.md

CHANGELOG.md must contain:

OLD CLAIM
NEW CLAIM
WHY CHANGED
EVIDENCE

Then perform a hostile reviewer pass.

Pretend you are Reviewer #2 attempting to reject the paper.

List every reason for rejection under:

- novelty
- statistics
- confounding
- benchmark choice
- architecture fairness attribution
- scorer choice
- one seed
- capability
- FIRE relevance
- reproducibility
- anonymity
- references
- AI disclosure
- writing

Fix every issue that can be fixed honestly.

Finally output FINAL_REVIEW.md containing:

Overall recommendation:
READY / NOT READY FOR FIRE

Recommended FIRE track

Five strongest contributions

Five remaining weaknesses

Every claim that must NOT be made

Any indispensable work before submission

Do not mark READY unless the PDF and all reported numbers are internally
consistent.
