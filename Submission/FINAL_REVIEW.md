# FINAL_REVIEW — hostile reviewer pass

Written as Reviewer #2 attempting to reject, then answered.

## Overall recommendation: **READY FOR FIRE 2026 (Regular Paper)**

Ready in the sense that the PDF is internally consistent, every number traces to
a committed output, and the claims match the evidence. It is a negative and
methodological result, not a performance win, and it should be submitted as such.

## Recommended track

**Regular Paper.** Reasoning and the rejected Perspective alternative are in
`TRACK_DECISION.md`.

---

## Hostile reviewer objections and responses

**Novelty — "this is just a failed experiment."**
The failure of the original claim is one of four results. The paper also reports
a continuous loss-adjusted model in which no architecture term survives, a
scoring-formulation sensitivity that removes every contrast, and category-level
sign changes. The protocol recommendation follows from these. *Partly conceded:*
the contribution is measurement methodology, and reviewers wanting an
architectural advance will not find one.

**Statistics — "post-hoc analyses after a null."**
Conceded and disclosed. The Limitations state which analyses were specified
before the comparison point and which were not, and `STATISTICAL_AUDIT.md`
separates confirmatory from exploratory. The equivalence margin is explicitly
labelled exploratory because it was chosen after the contrast was seen.

**Confounding — "iso-loss does not control what you claim."**
This is now the paper's own argument rather than a weakness in it. §5.4 shows the
architecture terms are indistinguishable from zero once realised loss enters the
model continuously.

**Benchmark choice — "CrowS-Pairs is known to be flawed."**
Acknowledged, cited, and used as evidence: the eight pairs the changed-token
formulation cannot align differ in four to nine words, which is a concrete
instance of the documented item-quality problem. **Now answered:** StereoSet
intrasentence (2106 items) was scored on the same snapshots with the same
protocol. Direction agrees on all three contrasts; significance carries for only
one, and for a different architecture. *Residual:* both benchmarks are
correlation-type and read the same MLM head, so they are not independent tests.

**Architecture attribution — "the arms differ in more than sharing."**
Conceded in the Method and the Limitations: the bundles differ in parameter
count, optimisation path and, for one arm, learning rate. This is a reason the
paper does not attribute anything causally to sharing.

**Scorer choice — "you picked the scorer that fails."**
Both are reported side by side with intervals, and neither is called correct. The
finding is disagreement, not a verdict.

**One seed.**
The strongest surviving objection. Stated in the Setup, the Limitations and
`STATISTICAL_AUDIT.md`, with the explicit note that item bootstraps do not cover
training-run variation. *Not answerable* without new pre-training runs.

**Capability — "the models are too weak to measure."**
Reported rather than hidden: the coreference gate fails with Jeffreys intervals
over 374 items, and the paper restricts every claim to association in the MLM
head. GLUE was not run because the gate cannot pass while the coreference leg
fails.

**FIRE relevance.**
One Discussion paragraph plus two verified IR-fairness citations, with no claim
that the encoders were evaluated in a deployed retrieval system.

**Reproducibility.**
Reproducibility section with data, tokeniser manifest, optimiser settings, seed,
hardware, snapshot rule, test definitions and software versions. Re-running the
analysis reproduces 12/12 outputs byte-identically.

**Anonymity.**
Audited on the rendered PDF, not just the source. Zero identifying strings.
One residual risk is recorded in `ANONYMITY_AUDIT.md`.

**References.**
All entries checked; four upgraded from preprint to published venue; two new
citations verified against the publisher record before use. A BibTeX defect that
was silently dropping entries was found and fixed.

**AI disclosure.**
Present, accurate about the scope of assistance, non-identifying, and not in a
suppressed acknowledgements environment.

**Writing.**
Withdrawn claims are gone (automated check: zero occurrences of fifteen banned
formulations), causal verbs are restricted to two legitimate uses, and the
abstract is 246 words.

---

## Five strongest contributions

1. A concrete demonstration that an architecture bias contrast can reverse sign
   across matched-loss points that differ by hundredths of a nat, and that the
   architecture which separates from the baseline changes with the benchmark.
2. A continuous loss-adjusted model showing capability, not architecture,
   accounts for the apparent effect.
3. Evidence that two defensible PLL scoring rules disagree enough to change the
   conclusion, with item-level rank agreement of only ρ ≈ 0.47.
4. Category-level analysis showing an aggregate stereotype score conceals sign
   changes in four of nine categories.
5. A worked protocol others can apply, together with an honest capability gate
   that bounds what the measurements mean.

## Five remaining weaknesses

1. One training seed; the encoder ordering may not survive re-training.
2. Both benchmarks are correlation-type and read the same masked language
   modelling head, so their agreement is weaker evidence than independent
   paradigms would give.
3. Models are far from fully trained, and effect sizes are still moving.
4. Most robustness analyses are exploratory, run after the instability appeared.
5. The GLUE leg of the gate is unmeasured, so the gate is incomplete as well as
   failed.

## Claims that must NOT be made

- that weight sharing reduces, lowers or suppresses stereotype association
- that any encoder here is fairer, debiased, or better for downstream fairness
- that the Hyperloop null generalises beyond this implementation and setting
- that the equivalence margin was pre-specified
- that iso-loss matching removes the training-progress confound
- that instability is a general property of weight reuse
- that any result is replicated across seeds

## Indispensable work before submission

Nothing blocking. The second-benchmark replication is now done. The one
remaining action is operational: keep the public model repository private until
after notification on 2026-10-15.
