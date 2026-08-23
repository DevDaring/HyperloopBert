# Claude Code Instructions — Plainer vocabulary and shorter paper, `Submission/FIRE_HyperloopBERT.tex`

## 0. Context and working method

Two goals this pass, both from the angle of a strict human reviewer (plain language, tight,
no redundancy, honesty kept):

- **Part A — plainer vocabulary.** Replace elevated, "academic-sounding" words with plain
  ones, WITHOUT touching genuine technical terms.
- **Part B — shorter paper.** Cut tangential analyses and repeated explanation so only the
  significant contributions remain.

The register conversion (formal, non-conversational) and the structural fixes (roadmap,
verb-first contributions, table columns, figure labels) are **already done in this draft.
Do not redo them, and do not reintroduce a casual tone while "simplifying" — see the
guardrail in A4.**

Edit `Submission/FIRE_HyperloopBERT.tex` in place. Keep it compilable. Keep a
`CHANGELOG.md`. Change no numeric value, no result, no citation content, and no symbol
definition. When compressing, move or delete whole sentences — never alter a number to make
text shorter.

---

# PART A — Plainer vocabulary

## A1. The "confound" family (the word flagged by the author)

Honest note to apply, not to print: "confound/confounded" is a legitimate statistics term,
not a buzzword, but the paper uses it about eight times, and that repetition is what reads
as mechanical. The abstract already contains the plain form: *"reflects both the
architecture and the amount of training."*

**Action.** Replace almost every instance with a plain paraphrase, varying by context:

- "This comparison is confounded with model capability." → "This comparison is affected by
  differences in model capability." / "The measured difference reflects model capability as
  well as architecture."
- "confounds the architecture with how far each model has trained" → "mixes the effect of
  the architecture with the effect of training."
- "None of them controls this confound explicitly." → "None of them separates the two
  explicitly."
- "inherits the capability confound" → "still mixes architecture with capability."
- "confounds the two" (secretary probe) → "mixes the two."
- "removes this confound" / "confound the paired construction removes" → "removes this
  problem" / "the difference the paired construction cancels."
- "confounding the capacity change with the recovery dynamics" → "mixing the capacity
  change with the recovery dynamics."

Optionally keep the noun "confound" a single time where the phenomenon is first named, if a
precise anchor is wanted; defaulting to the plain paraphrase everywhere is fine and is the
author's stated preference.

## A2. Other elevated words to make plain (replace throughout)

Apply wherever each appears, not only at the example location. Keep the meaning exact.

- "attenuate or amplify" → "weaken or strengthen".
- "propagate to / propagate into" (of an association spreading) → "carry into" / "appear
  in" / "spread to".
- "consequential for scoring" → "affects scoring" / "matters for scoring".
- "corroborates the direction but not the inference" → "agrees on the direction but not on
  significance".
- "markedly uneven" → "very uneven" (or drop the adverb; the 516-vs-60 numbers show it).
- "markedly harder to optimise" → "much harder to optimise".
- "exhibiting a stereotypical preference" → "showing a stereotypical preference".
- "in places opposing directions" → "sometimes opposing directions".
- "is suggestive but does not establish" → "points in that direction but does not
  establish".
- "indistinguishable from zero" → "cannot be distinguished from zero" (optional; the
  original is acceptable).
- "invariant under four dependence models" → "holds under all four dependence models"
  (optional).

If any other word feels like it was chosen to sound formal rather than to be precise,
prefer the everyday word with the same meaning. Do not introduce buzzwords (leverage,
harness, delve, robust, comprehensive, seamless, novel, etc.) while doing this.

## A3. DO NOT simplify these — they are real technical terms (keep exactly)

Simplifying any of these would make the paper wrong and would look less rigorous to the
reviewer, not more. Leave them as they are:

pseudo-log-likelihood (PLL); (paired sign-flip) permutation test; Holm correction /
step-down procedure / family-wise error rate; percentile bootstrap; wild cluster bootstrap;
Rademacher weights; Cohen's d; Phipson–Smyth estimate; cluster-robust OLS; covariate;
categorical factor; mean-centred; degrees of freedom; intrasentence split; coreference;
Jeffreys binomial interval; masked language modelling / masked-position probing; validation
loss; realised loss; matched point; direction convention. Domain nouns such as
weight-sharing, looped, hyper-connections, and residual stream also stay.

## A4. Guardrail — plain but precise, and still formal

The target is plain **and** formal, the register of the current draft. Do not swing back to
the conversational tone that an earlier version had (no "harder than it looks", "does not
hold up", "flips its sign", rhetorical fragments). Plain means the everyday word for the
same idea, in a complete declarative sentence, not a casual one.

---

# PART B — Shorter paper (keep only the significant contributions)

## B0. What is significant, and must stay

The three contributions and the results that support them are the paper. Keep and, at most,
tighten:

- the core finding that an architecture bias comparison reverses under evaluation choices;
- §6.4 (across matched points), §6.5 (continuous adjustment), §6.6 (scoring rule and the
  direction-convention decomposition), §6.7 (second benchmark), §6.8 (category
  heterogeneity), and §6.10 (capability gate not met);
- the Limitations section and all calibrated hedging.

Reduce length by cutting peripheral analysis and repeated explanation, NOT by cutting the
limitations, the capability gate, or the honesty. Approximate target: remove on the order
of two pages of body text through the steps below.

## B1. §6.9 "The two looped variants" → compress to two or three sentences `[largest cut]`

The paper itself says this equivalence result is "carried neither into the abstract nor the
contributions", so it is not a significant contribution. Reduce the whole subsection to the
single point that survives: the four hyper-connected streams gave no detected reduction in
association over plain looping at this point, while adding 18.9 million parameters (28 per
cent more than LoopedBERT). Drop the equivalence-margin machinery (the ±0.0118 margin, the
two one-sided tests, the 90% interval, the p = 0.023) or reduce it to half a sentence noted
as a post-hoc sensitivity check. Consider merging the remaining sentences into §6.8 or the
Discussion and removing the subsection heading.

## B2. §6.11 "Optimisation behaviour in these runs" → compress to two or three sentences

This is a caveat about the runs, explicitly "not evidence of a general optimisation
property", not a finding. Reduce to: HyperloopBERT was much harder to optimise and diverged
at the shared learning rate (hence the lower rate); ALBERTLoopedBERT had one transient spike
that recovered; with one seed these are properties of the runs, not the architectures, and
the analysed snapshots predate the divergences. Fold this into the end of §6.1 (where the
divergence is first mentioned) or into Limitations, and remove the standalone subsection.

## B3. §6.3 "What the encoders predict" and Table 5 → compress heavily

This is a diagnostic on hand-built probes drawn from neither benchmark, not one of the three
contributions. Reduce the subsection to three or four sentences that keep only the load-
bearing point: an unpaired single-mask probe mixes pronoun frequency with the occupational
association (the secretary probe prefers *he* although the stereotype is *she*), which is the
reason the paired per-item difference is used. Move Table 5 to an appendix, or cut it and
keep the secretary example in prose. If Table 5 is removed, also remove the "masked-position
predictions behind Table 5" clause from §9.

## B4. §7 Discussion → cut what merely repeats Section 6 `[large cut]`

The Discussion currently re-explains results that Section 6 already reports. Keep only:

1. the synthesis that the single-point result is one of several equally defensible analyses
   and the others do not reproduce it;
2. one brief statement of why the scoring rules disagree (token set, then the decisive
   direction convention) — but only if this is not already fully made in §6.6; if it is,
   cut it here and cite §6.6;
3. the practical implication for practitioners choosing an encoder (the comparison can
   reverse under the matched level, the scoring rule, and the direction convention).
   Delete the paragraphs that restate the matched-points distribution (§6.4), the StereoSet
   disagreement (§6.7), the matching-on-loss caveat (§6.5 / Limitations), and the aggregate-
   hides-categories point (§6.8). Target roughly half the current Discussion length.

## B5. Say the direction-convention result once

The sign reversal from the 218 antistereo pairs is currently explained in §4.3, §6.6, and
§7. Keep the full explanation in one place (§6.6, where the decomposition table is), give
§4.3 only the one-line definition of the direction label, and in §7 refer to §6.6 rather
than re-deriving it.

## B6. Tighten the §4.3 worked example

The single-item walkthrough ("The three rules can be read off a single item … 19 scoreable
positions …") can be reduced to one or two sentences, or dropped if Table 3 plus the rule
definitions already make the point. Keep the definitions of the three rules and Table 3.

## B7. Small reviewer-caught fix while editing: reference [27]

Reference [27] is malformed — it currently reads "Rui-Jie Zhu et al. [n. d.]. Scaling Latent
Reasoning via Looped Language Models. ([n. d.])." Restore the full entry: authors, year
2025, the title with its "Ouro:" prefix, and the arXiv identifier arXiv:2510.25741. Match
the formatting of the other arXiv entries. Verify the fields against the bibliography source
before writing; do not invent details.

---

# PART C — Do NOT cut or weaken

- §6.10 (capability gate not met) and the sentence bounding what the measurements
  demonstrate — this is central to the paper's credibility.
- The Limitations section, the single-seed caveat, and every calibrated qualifier
  ("absence of evidence rather than evidence of absence", "weaker than a causal claim").
- Any numeric result, table cell, or citation (beyond fixing [27]'s formatting).
- The structure, roadmap, contributions list, figures, and table columns already fixed.
- British/Indian spelling; the Generative AI Use Disclosure section.

---

# Verification checklist (run before reporting done)

- [ ] LaTeX compiles; no number, result, symbol, or citation content changed (only [27]
  formatting restored).
- [ ] "confound/confounded" reduced to at most one instance; the rest replaced with the
  plain paraphrases in A1.
- [ ] The A2 elevated words replaced throughout; no technical term from A3 altered.
- [ ] Tone still formal and plain, not casual (A4); no buzzwords introduced.
- [ ] §6.9 and §6.11 reduced to a few sentences each; their headings removed or merged.
- [ ] §6.3 compressed; Table 5 moved to appendix or cut (and §9 clause updated if cut).
- [ ] §7 Discussion cut to synthesis + one mechanism statement + practical implication.
- [ ] Direction-convention result explained once (§6.6), referenced elsewhere.
- [ ] §6.10, Limitations, and all hedging left intact.
- [ ] Reference [27] restored to a complete, correctly formatted entry.
- [ ] `CHANGELOG.md` lists every replacement and every cut by section.

```

```
