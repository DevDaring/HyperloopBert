
I need a final scientific audit and correction of my FIRE 2026 paper.

You have access to:

- the current LaTeX manuscript and PDF
- all source code
- all checkpoints
- all cached logits / scores / result files
- training logs
- statistical-analysis scripts
- CPU/GPU environment
- Vast.ai access through PHD_VAST_AI_KEY

DO NOT rewrite the paper cosmetically first.

There are several scientific and consistency issues that must be resolved.
Some are potentially blocking for submission.

=========================================================
BLOCKING ISSUE 1 — CROWS-PAIRS OFFICIAL SCORER IS MISDESCRIBED
===============================================================

The current paper says that the CrowS-Pairs reference implementation uses a
"changed-token" scorer, i.e. it scores the tokens that DIFFER between the
stereotypical and anti-stereotypical sentences.

This appears to be incorrect.

Verify this yourself from BOTH:

1. Nangia et al. 2020 CrowS-Pairs paper
2. official repository:
   https://github.com/nyu-mll/crows-pairs
   especially metric.py

The official implementation appears to:

- use difflib.SequenceMatcher
- identify spans with operation == "equal"
- treat these as shared/non-changing tokens
- mask and score the NON-CHANGING / UNMODIFIED tokens
- condition them on the modified demographic tokens

It also appears to SUM the relevant log probabilities rather than use the
length-normalised mean currently described in Eq. 2.

Do not trust this instruction blindly.
Inspect the official paper and source code directly and document exactly what
the official metric computes.

Create:

CROWS_SCORER_AUDIT.md

with:

- original CrowS-Pairs mathematical definition
- official metric.py behaviour
- our current full-sentence PLL definition
- our current changed-token definition
- differences in:
  * scored token set
  * normalisation
  * handling of unequal tokenisation
  * sentence alignment
  * resulting stereotype score

=========================================================
RECOMPUTE THE OFFICIAL CROWS-PAIRS RESULT
=========================================

Implement a scorer reproducing the official CrowS-Pairs metric as faithfully as
possible for our custom BERT architectures.

Validate the implementation first.

If possible:

- run official BERT-base through both the original metric.py and our
  reimplementation on a subset/full benchmark;
- confirm numerical or preference-level agreement.

Then score our architectures.

At minimum evaluate the four models at the deepest common matched point:

VanillaBERT
LoopedBERT
ALBERTLoopedBERT
HyperloopBERT

Preferably evaluate ALL existing distinct snapshots because no retraining is
needed.

Use existing per-token outputs if they contain enough information.
Otherwise perform inference from saved checkpoints.

NO MODEL PRETRAINING IS REQUIRED.

If GPU is required, use the cheapest suitable instance and terminate it
immediately when inference is complete. Never expose PHD_VAST_AI_KEY.

For the official metric report:

- conventional CrowS-Pairs stereotype score
- continuous per-item statistic if a defensible continuous form exists
- paired architecture contrasts
- bootstrap intervals
- permutation tests
- appropriate multiple-comparison correction

Do not manipulate the official metric merely to make it compatible with the
old results.

=========================================================
KEEP THE CHANGED-TOKEN ANALYSIS ONLY IF USEFUL
==============================================

Our existing changed-token scorer can remain as an ADDITIONAL sensitivity
analysis if scientifically meaningful.

But NEVER call it:

- official CrowS-Pairs scorer
- reference implementation scorer
- benchmark's own scoring rule
- conventional CrowS-Pairs metric

Rename it accurately, for example:

"attribute-token-only PLL"
or
"modified-token-only PLL"

depending on exactly what it computes.

The final paper could compare three scoring formulations:

A. full-sentence length-normalised PLL
B. official CrowS-Pairs unmodified/shared-token scoring
C. modified-token-only scoring

if all three are methodologically defensible.

The central scorer-sensitivity claim must be based on the actual reproduced
results, not the desired story.

Update every affected occurrence in:

- Abstract
- Contributions
- §3.3
- §5.5
- figures
- Discussion
- Limitations
- Conclusion

Search globally for:
"changed-token"
"reference implementation"
"benchmark's own"
"benchmark implementation"
"official"
and audit every occurrence.

=========================================================
BLOCKING ISSUE 2 — STEREOSET TEXT CONTRADICTS TABLE 2
======================================================

The current §5.6 says approximately:

"the comparisons that were significant on CrowS-Pairs, against LoopedBERT and
ALBERTLoopedBERT, are not..."

This contradicts Table 2.

At the deepest CrowS-Pairs point Table 2 reports approximately:

Vanilla vs Looped:
pHolm = 0.0003 — significant

Vanilla vs ALBERT:
pHolm = 0.0653 — NOT significant

Vanilla vs Hyperloop:
pHolm = 0.0003 — significant

Therefore the significant CrowS-Pairs contrasts were Looped and Hyperloop,
not Looped and ALBERT.

Correct §5.6 and search the entire manuscript for similar stale interpretation
errors.

Create a programmatic consistency table mapping every prose statement about
significance to the corresponding result table.

No prose significance statement may disagree with a table.

=========================================================
ISSUE 3 — TEST STEREOSET ACROSS MORE THAN ONE CHECKPOINT
=========================================================

The paper's central argument is that single-point architecture conclusions are
fragile.

However, the StereoSet replication currently appears to evaluate only the four
snapshots at the deepest matched point.

That creates an obvious reviewer question:

"Why criticise single-point evaluation and then validate on the second
benchmark using another single point?"

StereoSet evaluation is CPU/inference only and requires no new training.

If computationally practical, score StereoSet across all available distinct
snapshots for all architectures.

Then determine:

- whether direction is stable across capability
- whether architecture ordering changes across capability
- whether significance is checkpoint-dependent
- whether the Hyperloop result persists
- whether benchmark × capability interactions exist

Do not force StereoSet into the same narrative.

If results differ from CrowS-Pairs, report that honestly.

Add a compact supplementary/main-paper analysis only if it improves the paper.

=========================================================
ISSUE 4 — DEFINE EXACTLY WHAT "STEREOSET EFFECT" MEANS
=======================================================

The paper reports:

Effect
SS
LMS

for StereoSet, but the method section does not adequately explain how "Effect"
is constructed from StereoSet's stereotype, anti-stereotype and unrelated
continuations.

Document precisely:

- dataset split
- number of examples
- tokenisation
- masking procedure
- stereotype candidate
- anti-stereotype candidate
- unrelated candidate
- formula for our continuous "Effect"
- formula for standard SS
- formula for LMS
- whether these reproduce the official StereoSet implementation

Do not say "identical protocol" if the benchmark structure requires a different
scoring procedure.

=========================================================
ISSUE 5 — REVISIT THE CONTINUOUS CAPABILITY REGRESSION
=======================================================

The current paper pools:

31,668 item × snapshot observations
1508 items
21 snapshots

and uses OLS with standard errors clustered only on benchmark item.

This needs a more careful statistical audit.

Architecture and realised validation loss vary at the SNAPSHOT level, while the
1508 item measurements within each snapshot share the same model checkpoint.

Clustering only by item may fail to capture checkpoint-level dependence and
may make the effective amount of information look much larger than the 21
model snapshots that actually identify architecture/loss effects.

Do NOT simply defend the current model.

Compare statistically appropriate alternatives, such as:

1. two-way clustering by:

   - item
   - model snapshot
2. small-cluster correction / wild cluster bootstrap where appropriate
3. analysis of snapshot-level mean effects:

   - one effect estimate per snapshot
   - uncertainty/weights obtained from the item-level data
4. hierarchical/mixed modelling only if identifiable from this design
5. sensitivity analysis using architecture-level trajectories

There are only 21 distinct snapshots and one training seed, so be conservative.

Explicitly distinguish:

number of item-level observations = 31,668

from

number of distinct trained model snapshots = 21.

Do not imply that architecture inference has 31,668 independent experimental
units.

Create:

REGRESSION_AUDIT.md

Report whether the conclusion

"architecture terms are not distinguishable from zero after adjusting for
realised loss"

survives reasonable dependence structures.

If it does, retain it with appropriately cautious language.

If it does not, rewrite the manuscript.

=========================================================
ISSUE 6 — REMOVE CAUSAL OVERSTATEMENT IN DISCUSSION
====================================================

The current Discussion says approximately:

"capability is doing the work that the architecture appeared to be doing."

That is stronger than the evidence supports.

A non-significant architecture coefficient after loss adjustment does NOT prove
that capability caused the apparent architecture difference.

It may also reflect:

- limited snapshot-level sample size
- single training seed
- residual confounding
- architecture × training-stage variation
- uncertainty in the loss adjustment

Replace causal statements with language such as:

"The adjusted analysis is consistent with training capability accounting for
part of the single-point architecture contrast."

or, if justified by the final analysis:

"After adjustment, the data do not provide clear evidence for an architecture
effect independent of realised validation loss."

Audit words including:

cause
drives
doing the work
explains
removes
isolates
results from
due to

Use causal language only where the design permits it.

=========================================================
ISSUE 7 — MATCHED POINTS ARE NOT A "DISTRIBUTION"
==================================================

The Discussion currently says that repeating the analysis across matched points
"converts a point estimate into a distribution".

These matched checkpoints are not independent draws from a sampling
distribution.

They form a sensitivity trajectory / set of estimates across capability.

Replace "distribution" with:

"trajectory"
"set of estimates"
"sensitivity curve"
or similar accurate language.

Do not interpret variation across checkpoints as statistical sampling
variation.

=========================================================
ISSUE 8 — MULTIPLE TESTING ACROSS ALL MATCHED POINTS
=====================================================

§5.3 reports individual p-values across several matched checkpoints, including
p = 0.0002 in the opposite direction, and discusses how many points reach
p < 0.05.

These are exploratory repeated tests.

Do not call individual pointwise values "significant" without specifying the
multiplicity treatment.

Count all tests actually performed.

At minimum there appear to be multiple baseline-vs-shared tests across
multiple matched points.

Choose and justify an inferential family.

Possible approaches:

- Holm correction across all pointwise exploratory architecture contrasts;
- correction separately within architecture contrast, with explicit
  justification;
- or remove pointwise significance language and present estimates + confidence
  intervals as a sensitivity analysis.

Prefer the last approach if inferential families are arbitrary.

When an uncorrected exploratory p-value is shown, label it "nominal p".

Keep the deepest-point pre-specified three-contrast Holm family separate from
the post-hoc all-band analyses.

=========================================================
ISSUE 9 — CATEGORY MULTIPLICITY
================================

The paper performs 27 category-by-contrast tests:

9 categories × 3 architecture contrasts

and currently performs Holm correction "within each contrast".

Audit whether this family definition is justified.

Since the category analysis is explicitly exploratory, consider:

A. correcting all 27 together;

OR

B. retaining three families of 9 with an explicit rationale;

OR preferably

C. making category analysis primarily descriptive:
   effect estimates + uncertainty + sign structure, while de-emphasising
   thresholded significance.

Do not make "four significant categories" a major contribution if the number
depends strongly on the correction family.

=========================================================
ISSUE 10 — POST-HOC EQUIVALENCE MARGIN
=======================================

The current manuscript says LoopedBERT and HyperloopBERT are "equivalent within
an exploratory margin".

The ±0.0118 margin was chosen AFTER looking at the result and is defined as
half of the observed baseline-vs-looped difference.

This makes the equivalence conclusion substantially weaker than a
pre-specified SESOI.

Do not promote this as strong equivalence evidence.

Investigate whether a principled equivalence margin can be justified using:

- prior published work
- benchmark measurement reliability
- an independently defined practically meaningful effect
- preregistration/specification written before seeing the contrast

If no independent margin exists:

retain the TOST only as explicitly post-hoc sensitivity evidence and use
language such as:

"Under a post-hoc illustrative margin of ±..., the interval falls within the
equivalence region."

Do NOT simply say:

"the two architectures are equivalent."

Consider removing equivalence from the Abstract and main Contributions if it
cannot be supported by an independently justified margin.

=========================================================
ISSUE 11 — ARCHITECTURES DO NOT "DIFFER ONLY" IN WEIGHT REUSE
==============================================================

Search for statements such as:

"the four architectures differ only in how weights are reused across depth"

This is not literally true.

HyperloopBERT additionally includes:

- four residual streams
- learned mixing/hyper-connections
- ~19M additional parameters over LoopedBERT

Parameter counts also differ substantially:

Vanilla ≈ 110.1M
Looped ≈ 67.6M
Hyperloop ≈ 86.5M
ALBERT ≈ 32.1M

Hyperloop also uses a different learning rate after instability.

Rewrite architecture descriptions accurately, e.g.:

"All models share tokenizer, hidden width, effective depth, objective and
training corpus, but differ in parameter-sharing scheme; Hyperloop additionally
introduces parallel residual streams and learned mixing."

Do not attribute observed differences uniquely to weight sharing.

=========================================================
ISSUE 12 — INTRODUCTION OVERCLAIMS DOWNSTREAM INHERITANCE
==========================================================

The Introduction currently says approximately:

"a stereotypical association acquired during pre-training is inherited by every
component built on it."

This is too deterministic.

Fine-tuning and downstream training can alter representations and behaviour.

Use:

"can propagate into downstream components"
"may influence downstream systems"
or similarly cautious wording.

Also audit claims like:

"a better-trained model gives sharper answers on every probe."

Replace "every" unless directly demonstrated.

=========================================================
ISSUE 13 — SECOND-BENCHMARK INTERPRETATION
===========================================

The Discussion says that direction agreement across CrowS-Pairs and StereoSet
is "mild evidence that some small difference exists rather than none."

This may conflict with:

- all-band sign reversals
- scorer sensitivity
- continuous adjustment
- single seed
- different significant architecture across benchmarks

Use more neutral language unless statistically justified.

For example:

"Directional agreement at the selected operating point is suggestive, but does
not establish a stable architecture effect because the significance pattern,
aggregate benchmark behaviour, and capability sensitivity differ."

=========================================================
ISSUE 14 — REFERENCE AUDIT: CORRECT VERIFIED ERRORS
====================================================

Audit every bibliography record against the official publication/arXiv/OpenReview
entry.

I have identified likely errors that MUST be checked:

[22] mHC: Manifold-Constrained Hyper-Connections

The current manuscript lists:
"Zhijian Xie et al."

The arXiv entry appears to list:
"Zhenda Xie et al."

Verify and correct the complete author metadata.

[24] Hyperloop Transformers

The current manuscript lists:
"Abdelrahman Zeitoun"

The official arXiv entry appears to list:
"Abbas Zeitoun"

Verify and correct.

[27]

The manuscript gives the title approximately:

"Ouro: Scaling Latent Reasoning via Looped Language Models"

The official title appears to be:

"Scaling Latent Reasoning via Looped Language Models"

"Ouro" is the model/family name, not part of the paper title.

Verify and correct.

Also check whether:

"Adaptive Loops and Memory in Transformers: Think Harder or Know More?"

has a published ICLR 2026 workshop version that should replace the arXiv-only
citation.

Never invent venue metadata.

Create REFERENCES_FINAL_AUDIT.md with:
reference number,
current record,
verified source,
correction.

=========================================================
ISSUE 15 — AI DISCLOSURE SHOULD BE FULLY SPECIFIC
==================================================

FIRE requires compliance with ACM generative-AI policy.

The current disclosure says only:

"a large language model based coding assistant"

That is unnecessarily vague for a disclosure intended to be transparent.

Determine the actual tools used.

If Anthropic Claude was used, say Anthropic Claude.
If OpenAI ChatGPT was also materially used in writing, analysis, code, figures,
or interpretation, disclose it too.

Include model/version where known and stable.

Accurately describe the role:

- code writing/debugging
- statistical-analysis assistance
- experimental auditing
- figure generation
- prose editing
- identification of robustness weaknesses

Do not imply the AI independently made experimental decisions if humans made
them.

Keep:

"All numerical claims and interpretations were verified by the authors."

Do not list an AI system as an author.

=========================================================
ISSUE 16 — FIRE TRACK CHECK
============================

Check the CURRENT official FIRE 2026 CFP.

The paper now has a substantial empirical robustness analysis and could be
submitted as a Regular Paper.

However, its thesis also challenges a common evaluation practice and could fit
Perspective.

Do not change track automatically.

Write TRACK_FINAL_DECISION.md comparing:

Regular Paper
Perspective Paper

If Perspective is selected, FIRE specifically expects a short section explaining
the perspective offered.

In that case add a clearly labelled section such as:

"Perspective: Architecture Bias Evaluation as a Robustness Problem"

If Regular is selected, do not add artificial Perspective framing.

=========================================================
ISSUE 17 — CCS SCOPE ACCURACY
==============================

The current CCS concepts include:

Information systems -> Retrieval models and ranking

but the paper does not evaluate a retrieval or ranking model directly.

Check whether this CCS code accurately represents the contribution.

Do not use a retrieval CCS label merely to look more FIRE-specific.

Prefer CCS concepts matching what is actually studied:

- NLP
- fairness / user characteristics
- evaluation / information retrieval only where technically justified

FIRE relevance can be explained in Introduction/Discussion without falsely
implying a retrieval experiment.

=========================================================
FINAL PAPER-WIDE CONSISTENCY AUDIT
==================================

After all analyses are frozen, create a machine-readable table containing every
numerical claim in:

Abstract
Introduction
Results
Discussion
Limitations
Conclusion

For every number record:

- value
- table/figure/script source
- corrected or raw p-value
- confirmatory vs exploratory
- benchmark
- scorer
- checkpoint/band
- item count

Then check that no two parts of the paper describe the same result
inconsistently.

Pay special attention to:

Looped significance
ALBERT significance
Hyperloop significance
which scorer is official
number of matched points
StereoSet interpretation
number of items
correction families
equivalence language

=========================================================
GPU RULE
========

Do NOT retrain any encoder.

The only potentially required new computation is evaluation of existing
checkpoints, especially the genuine official CrowS-Pairs metric and perhaps
StereoSet across snapshots.

First use:

- cached logits
- stored per-token probabilities
- existing evaluation results
- CPU inference

Only use GPU when necessary for checkpoint inference.

Use the cheapest suitable Vast.ai GPU.
Estimate cost before launching.
Terminate immediately afterward.
Never expose PHD_VAST_AI_KEY.

=========================================================
FINAL DELIVERABLES
==================

Produce:

1. FIRE_HyperloopBERT_FINAL.tex
2. FIRE_HyperloopBERT_FINAL.pdf
3. CROWS_SCORER_AUDIT.md
4. REGRESSION_AUDIT.md
5. REFERENCES_FINAL_AUDIT.md
6. CONSISTENCY_AUDIT.md
7. TRACK_FINAL_DECISION.md
8. FINAL_REVIEW.md

FINAL_REVIEW.md must provide:

BLOCKING ISSUES REMAINING:
NONE or list

SCIENTIFIC CLAIM NOW SUPPORTED:
one paragraph

CLAIMS THAT MUST NOT BE MADE:
list

CONFIRMATORY ANALYSES:
list

EXPLORATORY ANALYSES:
list

RECOMMENDED FIRE TRACK:
Regular / Perspective

READY TO SUBMIT:
YES / NO

Do not mark READY TO SUBMIT = YES until the genuine official CrowS-Pairs
scoring implementation has been verified and all prose/table inconsistencies
have been eliminated.
