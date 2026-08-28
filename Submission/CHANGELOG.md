# CHANGELOG — readiness pass on `FIRE_HyperloopBERT.tex`

Polish pass against the structural review. No numbers, citations or results were
changed; every edit is placement, naming or prose.

## Priority fixes

| # | Section touched | Edit |
|---|---|---|
| 3.1 | Introduction, final paragraph | Roadmap rewritten to name all nine following sections in order. It previously named only §2, §4, §6, §8, silently skipping Data, Setup, Discussion, Reproducibility and Conclusion, and it labelled Limitations as "what the evidence does not support" when that argument lives in §6.9 and §7. Added `\label` to Discussion, Reproducibility and Conclusion so the references resolve. |
| 3.2 | §5 Experimental Setup, §9 Reproducibility | Training configuration de-duplicated. The optimiser, learning rate, batch size, warmup, clipping, precision and GPU appeared verbatim in both. §9 keeps the full configuration; §5 now carries only the three facts that bear on interpretation — the 1.53–2.07 B token range at the matched point, the HyperloopBERT learning-rate exception, and the single-seed caveat — and defers the rest with a pointer. |
| 3.3 | Figure 3(a) | **No edit needed.** The legend already reads "shared-token"; it was renamed when the scoring rules were corrected in the previous revision. Verified in the plotting script and in the rendered PDF: 0 occurrences of "changed-token" in the figure, the `.tex`, or the PDF text. The review was reading an earlier build. |
| 3.4 | Introduction, contributions | Converted from noun-first to verb-first ("We introduce…", "We show…", "We show…"), one sentence each. |
| 3.5 | Abstract | Setup counts removed: "7 billion tokens, 28 billion in total" → "identical data"; "21 distinct snapshots" → "snapshotted whenever it reached a target validation loss"; "two stereotype benchmarks, of 1508 and 2106 items" → "two stereotype benchmarks". All three result fractions kept verbatim, as instructed. The abstract now carries no bare setup numerals. |
| 3.6 | Captions of Tables 4, 5, 6, 7 | Every non-obvious column defined in its caption: *Effect* as the mean paired stereotype score over items, the interval as an item bootstrap, Δ as the contrast against VanillaBERT, p_Holm as the corrected permutation p-value, and for the StereoSet table *SS* and *LMS* as the stereotype and language-modelling scores. Values untouched. |

## Minor fixes

| # | Edit |
|---|---|
| 4.1 | *VanillaBERT*, *LoopedBERT*, *HyperloopBERT*, *ALBERTLoopedBERT*, *ALBERT*, *CrowS-Pairs*, *StereoSet*, *WinoBias* and *FineWeb-Edu* italicised at first use, plain thereafter. |
| 4.2 | FlashAttention citation moved off the GPU/precision claim. Checked against the code first: `Codes/common/attention.py` imports `flash_attn_varlen_func`, and the run log records `Active attention path: flash` for every run, so the reference is genuine. It now sits with the attention implementation in §9. |
| 4.3 | The CrowS-Pairs category is now written `race-color`, matching the dataset's own label, rather than anglicised to "race-colour". British spelling is retained everywhere else. |
| 4.4 | Abstract voice left passive, as the review marked this optional and the recast did not read better. |

## Not changed, deliberately

The Generative AI Use Disclosure is retained per §5.1 of the review.

## Build state after the pass

9 pages, 7 of content against a 9-page limit. 0 overfull boxes, 0 underfull boxes,
0 undefined references, 0 LaTeX errors. Abstract 282 words, mean sentence 14.1,
longest 22. Anonymity check on the rendered text: no author, affiliation, or
identifying repository string.

## Post-pass addition — model output table (§6.3, Table 5)

A reviewer asked to see what the encoders actually produce, since the results are
otherwise all summary statistics.

The first version listed top-3 predicted words for three prompts. It was replaced
because two of the three showed weak prediction rather than bias, and the third
showed an *absence* of the expected association, so the table did not demonstrate
what it was meant to.

The current version reports the paired probabilities the encoders assign to
\emph{he} and \emph{she} at a masked pronoun, for three occupation sentences that
differ by one word. It is taken from
`Codes/results/stage3/qualitative/mlm_targeted_contrast.csv`, deduplicated because
that dump ran twice, and no value was rounded beyond three decimals.

It carries three things:

- **the association is large** — on the scientist sentence the baseline puts 68
  times more probability on *he* than *she*, and every encoder favours *he* by at
  least 15:1;
- **it is occupation-specific, not a blanket preference** — on the teacher
  sentence the same baseline reverses and prefers *she* by close to 6:1, so the
  encoders have learned which occupation goes with which pronoun;
- **a single probe is still not enough** — on the secretary sentence, where the
  stereotype points to *she*, every encoder prefers *he*, because pronoun
  frequency competes with the association. This is the argument for the paired
  design, and the text now makes it.

## Presentation pass — Table 1, Related Work, captions

**Table 1 (datasets).** Had no column headers at all: two unlabelled columns with
dataset names buried in spanning rows, and the two WinoBias sentences broken by
hand across rows. Rebuilt on `tabularx` with real headers (*Sentence role* /
*Example text*), a banded row per dataset carrying its size and its role in the
study (primary measure / second measure / capability check), and an `X` column so
long examples wrap on their own instead of being split manually.

**Related Work.** Reordered as a gap analysis rather than three parallel surveys.
Each paragraph now closes on the question it leaves open and the next one opens
by taking it up:

1. measurement — the audits question the *instrument*; they never ask whether a
   *comparison between two models* keeps its sign when the measurement is varied;
2. weight reuse — supplies two models differing in one designed way, but reports
   only quality, efficiency and reasoning, and compares at a fixed budget, so a
   bias reading would inherit the training-progress confound;
3. capacity and bias — the closest precedent, but it changes capacity after
   training, and its own disagreement across studies is the reason to ask whether
   a single conclusion survives its evaluation choices;
4. IR fairness — why the answer matters for a deployment decision.

No citation was dropped.

**Captions.** All cut to one line, per the instruction that explanation belongs in
the text. The symbol definitions the captions used to carry (Effect, the interval,
Delta, p_Holm, SS, LMS, the coefficient scale, the sign convention) were moved
into the prose of the sections that own those tables, not deleted.

Rebuild: 9 pages, references begin on page 9, 0 overfull, 0 undefined refs.

## Academic-register and specificity pass (abstract untouched)

Instruction: make the prose academic rather than conversational, and state for
every result *which dataset, which models, and over what it is aggregated*. The
abstract was left byte-identical, as it had already been edited by hand.

### A substantive finding surfaced by the pass

Making the scoring rules precise exposed that the paper's own description of them
was incomplete. The official CrowS-Pairs implementation differs from this
project's shared-token rule along **three** axes, not two:

| | token set | aggregation | stereotypical member |
|---|---|---|---|
| full-sentence | all | mean | `sent_more` always |
| shared-token | shared | mean | `sent_more` always |
| official | shared (difflib) | **sum** | **by direction label** |

The third axis was undocumented in the paper. It matters: 218 of the 1508 pairs
carry `stereo_antistereo == antistereo`, and on those the official code treats
`sent_less` as the stereotypical member, flipping the sign of the per-item
effect. The project's stored per-item scores were verified to use
`sent_more` as stereotypical on 100% of rows including all 218.

`analysis/fire2026/phase7_scorer_decomposition.py` crosses aggregation with
direction convention on a fixed token set. Result (contrast vs VanillaBERT):

```
                       mean/sent_more   mean/official   sum/sent_more   sum/official
Vanilla vs Looped         +0.0076         -0.0035         +0.0919         -0.0519
Vanilla vs ALBERT         +0.0097         -0.0030         +0.0948         -0.0163
Vanilla vs Hyperloop      +0.0155         +0.0077         +0.1305         +0.0488
```

Aggregation scales magnitude by roughly the mean shared-token count (15.9) and
**never changes a sign**; the direction convention flips the sign for two of the
three contrasts under either aggregation. The paper's reported reversal under the
official rule is therefore attributable to a convention affecting 14.5% of the
benchmark, not to the token set or to summation. Table 7 gained a fourth row
making this explicit, and Table 3 a fourth column.

### Errors found and corrected

Checking every result statement against the CSVs turned up four:

- **"positive at five of five" for ALBERTLoopedBERT** — it is **four of five**.
  The five distinct matched points give +0.0224, -0.0060, +0.0074, +0.0026,
  +0.0116.
- **"averaged over its five matched points the contrast is -0.0063"** — -0.0063
  is the mean over the seven *nominal* bands, which counts one shared snapshot
  three times. Over the five *distinct* matched points it is **-0.0050**.
- **"2.997 versus 2.968 nats, a tighter match than the headline point"** — false.
  That pair is matched to 0.029 nats; the headline pair is matched to 0.0086,
  three times tighter. The claim was removed and the interval reported instead.
- **"inflates the architecture standard error by roughly 70%"** — true only for
  the LoopedBERT term. Across the three terms the inflation is **27 to 70%**.

Also corrected: WinoBias has **374 items in each of the two splits**, not 374
items split evenly between them.

### Specificity added throughout

Every result statement now names its dataset, its scoring rule, its snapshot and
its aggregation. Section 6 opens with a blanket statement of the default (all
1508 CrowS-Pairs pairs, full-sentence rule, contrast against VanillaBERT,
positive = baseline higher) so individual sentences need not repeat it. New
specifics include the 21 snapshots broken down per architecture (4/5/5/7), the
StereoSet per-type sizes (962 race, 810 profession, 255 gender, 79 religion) and
the negative profession effect on all four encoders, Cohen's d per contrast,
the CrowS-Pairs direction-label counts (1290/218), and the explicit note that
Table 5 is a hand-constructed probe set drawn from neither benchmark.

### Length

The pass added roughly 1.5 pages, which had to come back out to respect the
9-page content limit. Content now ends on page 9 with references beginning on
page 10. Trimming was prose-only plus modest figure scaling; no number, no
qualifier and no caveat was dropped to make room.

Build: 10 pages total, content ends page 9, 0 overfull, 0 undefined references.

## Register pass (FINAL_REVIEW.md) — wording only

Converted the paper from a conversational/narrative register to a plain formal
technical register. **No numeric value, citation key, table cell, symbol
definition, or section order was changed.** Verified mechanically: all 413
numerals and all citation keys from the Introduction onward are byte-identical
to the previous commit.

### Abstract — rewritten to the FINAL_REVIEW §2 target

The supervisor's `\textcolor{red}` / `\textcolor{blue}` markup flagged the
abstract for register, so the markup itself was removed and the abstract replaced
with the version supplied in FINAL_REVIEW.md §2. Every claim and number is
preserved. Representative changes:

| before | after |
|---|---|
| "a stereotype it encodes can spread to all of them" | "propagate to every system built on the model" |
| "The question is harder than it looks." | folded into "This comparison is confounded with model capability." |
| "A better-trained model answers more sharply" | "assigns sharper probabilities on nearly every probe" |
| "mixes the design with how far each model has trained" | "reflects both the architecture and the amount of training" |
| "This study asks whether that fix is enough." | "this study evaluates whether it is sufficient" |
| "Four encoders were built" | "Four encoders are trained" |
| "look less biased than the unshared one" | "record a lower stereotype effect than the unshared baseline" |
| "The result does not hold up." | "The effect does not replicate." |
| "flips its sign" | "reverses its sign" |

### Group A — idioms → technical phrasing

§1 "harder than it appears" → "difficult to answer because the compared quantity
is confounded with model capability"; "the natural remedy" → "the standard
control"; "tests whether that control suffices" → "evaluates whether that control
is sufficient"; "carries less of that association" → "encodes less of that
association".

### Group B — Related Work narrative transitions → topic sentences

"Measurement came first." → "Measurement of stereotype association developed
first."; "Audits followed." → "Audits of these benchmarks followed."; "They stop
short of the question that follows." → "These audits do not address the question
that follows."; "Asking it needs two models differing in one designed way" →
"Answering it requires two models that differ along a single design axis"; "One
line does connect capacity to bias" → "A third line relates model capacity to
measured bias"; "Two things are left open." → "This line leaves two questions
open."; "That test matters beyond encoder design." → "This test is relevant
beyond encoder design." Thematic grouping and every citation unchanged.

### Group C — model personification → operational description (§4.3)

"the model is asked how likely the original word was in that slot" → "the
probability of the original token at that position is evaluated"; "higher means
the model finds the sentence more plausible" → "A higher value indicates that the
model assigns greater probability to the sentence"; "The intuition is that a
model holding the stereotype should find the same surrounding words more likely"
→ "Under this rule, an encoder that encodes the stereotype assigns higher
probability to the shared context when the stereotypical term is present."

### Group D — reader-engaging framing → neutral statement

"This section describes what was built and how it was measured." → "This section
defines the four encoders, the matching protocol, and the three scoring rules.";
"so it is worth inspecting the underlying model output" → "The underlying model
output is therefore examined directly."; "The single-point result was convincing
for defensible reasons." → "The single-point result has the features of a sound
finding."; "which is the difficulty" → "which is the central difficulty".

### Sentence length (§5.2)

Split 16 long or clause-stacked sentences, including the 7-clause chain in the
Conclusion, the three-respects sentence in §4.3, and the aggregate-score sentence
in §7. Median sentence length is 20 words. Sentences still above 30 words are
concentrated in the equation-to-symbol definitions and the statistical
description, both of which FINAL_REVIEW §4 exempts from editing.

### Vocabulary (§5.1)

"Both are moreover scored" → "Both are also scored". "independent measurement
paradigms" → "independent measurement approaches". No banned buzzword or rare
word was introduced. British spelling retained throughout; the two remaining
instances of `color` are inside `\texttt{race-color}`, which is the dataset's own
category label and must not be re-spelled.

### Preserved unchanged

All hedging and negative-result framing: the single-seed caveat, "absence of
evidence rather than evidence of absence", the failed capability gate, the
direction-convention attribution to the 218 antistereo pairs, the exploratory
labelling of the post-hoc analyses. The Generative AI Use Disclosure is intact.

Build: 10 pages, content ends page 9, 0 overfull, 0 undefined references.

## Optional polish pass (FINAL_REVIEW.md, second version)

FINAL_REVIEW.md was replaced with a shorter document listing four optional items.
All four were applied.

**1. §6.2 — numbers that merely restate Table 4 removed.** Seven values were
dropped from the prose because each is already visible in that encoder's Table 4
row: Δ = 0.0237, Δ = 0.0241, p_Holm = 0.0003, Δ = 0.0116, the interval
[−0.0008, 0.0240], and p_Holm = 0.065. The paragraph keeps the qualitative finding
and both numbers that are *not* tabulated — Cohen's d = 0.115 / 0.098 / 0.047 and
the 55.4–55.8 aggregate range — and now points at Table 4 by reference. Both
retained figures were re-confirmed against `contrasts_band2.2.csv` and
`overall_band2.2.csv` before writing. The scope limit was honoured: §6.4
(Δ = −0.0222, mean −0.0050), §6.6 (+0.0076 vs −0.0035, +0.0919 vs −0.0519, 58.49,
56.6 vs 53.4) and §6.7 (Δ and p) are untouched, since there the numbers are the
argument.

**2. Figure 3(b) axis label.** The panel took its tick labels straight from the
`Category` column of `per_category_contrasts.csv`, which spells the category
`race-colour`. The plotting script now maps that one label to `race-color` at
draw time, matching §3.2, §6.8 and the dataset's own label. The CSV is unchanged.
Both figures regenerated. All three occurrences of the label in the built PDF now
read `race-color`.

**3. §6.6 first person removed.** "we cross aggregation with direction convention"
→ "aggregation is crossed with the direction convention". The verb-first "We
specify / We show / We show" in the contributions list is left intact, as
instructed.

**4. §4.3 permutation sentence split.** Broken after the test is named, giving two
sentences instead of one 45-word sentence with a colon and three clauses.

### Unrelated repair: `reference.bib` had been truncated

The working copy of `reference.bib` was 42 bytes — the closing two lines of the
Singh & Joachims entry and nothing else. BibTeX consequently emitted
`\begin{thebibliography}{0}`, and all 44 citations rendered undefined. The file
was restored from the previous commit (14 447 bytes, 31 entries); the surviving
fragment's DOI is present in that version, so nothing was lost. The truncated file
is kept at `/tmp/reference.bib.truncated-backup` for the session. After restoring,
all 27 cited keys resolve.

### Verification

- Only seven numerals differ from the previous commit, and all seven are the §6.2
  values listed above. No numeral was added. All citation keys identical.
- 0 overfull boxes, 0 undefined references, 27 bibliography entries rendered.
- 10 pages; content ends on page 9.

## Audit against a reviewer report received for a different paper

Two complaints were checked: (a) vague, unnatural, vacuous phrasing, and (b)
absence of concrete examples.

### (b) Concrete examples — largely already present, one real gap

The paper already shows a real item from each evaluation dataset (Table 1), real
model output (Table 5, P(he)/P(she) on three probes), and a real problem item
(the "Whites are not good at reading" pair in 6.6). The gap was the *methodology
in action*: 4.3 defined the three scoring rules purely abstractly, so a reader
could not see which tokens each rule scores or which member it treats as
stereotypical.

Added a worked example to 4.3, using the pair already in Table 1 so no new
material is introduced: 19 scoreable positions, the full-sentence rule averaging
over all 19, the shared-token rule averaging over the 18 shared, the official rule
summing those 18 and treating `sent_less` as stereotypical. Counts computed with
the project tokeniser and difflib, not estimated.

### A real error found while doing this

The Table 1 CrowS-Pairs item — "…whether he would come forward" against "…whether
she would come forward" — is labelled **`antistereo`** in the dataset. The table
labelled the two rows "stereotypical" and "anti-stereotypical", which contradicts
the dataset's own direction label, and does so in a paper whose central finding is
about that very label. Rows relabelled to `sent_more` / `sent_less` with the
direction label shown in the block header. The item now doubles as the 4.3 worked
example.

### (a) Vague phrasing — six instances found and fixed

| before | after |
|---|---|
| "The structural observation is more robust" | "The sign pattern is more stable than the significance count" |
| "one draw from a family of equally defensible analyses, and the remainder of that family disagrees with it" | "one of five equally defensible analyses of the same 21 snapshots, and the other four do not reproduce it" |
| "arrives at the same place by another route" | "reaches the same conclusion from a different specification" |
| "The second benchmark helps only partially." | "*StereoSet* corroborates the direction but not the inference." |
| "Matching on validation loss resolves part of the problem." | deleted; the concrete sentence that followed now leads |
| "Whether the picture differs for fully trained encoders" | "Whether these contrasts differ for fully trained encoders" |

"The contrast crosses zero repeatedly" was left alone: the contrast literally
changes sign, so the phrase is descriptive rather than figurative.

The abstract was checked against the specific examples in the report ("points the
other way", "load-bearing", "is a prognosis") and contains nothing of that kind.
Its shortest sentence, "The effect does not replicate.", is a technical statement,
not an idiom.

### Length

The worked example and the Table 1 change cost about 12 lines, recovered by
tightening 16 passages across 4.3, 6.3-6.8, 7 and 8. No number, qualifier or
caveat was dropped. Content ends on page 9.

Build: 10 pages, 0 overfull, 0 undefined references, 27 references rendered.

## Hanging-reference audit — every number now says what it is over

Scanned every numeric claim in the body and asked whether the sentence, or the
one before it, states the dataset, the model set, and the aggregation basis. The
scan over-flags, so each hit was judged by hand; twelve were genuine.

### Prose

| where | was | now says |
|---|---|---|
| 7 | "one of five equally defensible analyses" — the five were never named | names them: this point, the remaining matched points, the loss adjustment, the two scoring rules, the second benchmark |
| 7 | "That the two rules order individual pairs at Spearman 0.46 to 0.49" | "the full-sentence and shared-token rules order the 1508 pairs" |
| 7 | "Toggling only that convention flips the sign…" — no dataset, no operating point | "On *CrowS-Pairs* at the deepest matched point, toggling…" |
| 8 | "The Spearman correlation between matched loss and effect is −0.80, −0.90, −0.71" — over what, and over how many points? | "between matched validation loss and the mean *CrowS-Pairs* effect, taken across each encoder's own distinct matched points … over 4, 5 and 7 points" |
| 6.2 | "Cohen's d = 0.115, 0.098, 0.047" | adds "on the paired per-item differences" |
| Conclusion | "Aggregates conceal sign reversals across four of nine categories" directly after a *StereoSet* sentence — and *StereoSet* has four bias types | "*CrowS-Pairs* aggregates … four of its nine categories" |
| 6.7 | "Language-modelling scores of 86.0 to 87.1" | adds "across the four encoders" |
| 6.6 | "shifts the shared-token contrasts" | "shifts all three shared-token contrasts" |
| 4.3 | "At a mean of 15.9 shared tokens per sentence" | adds "over the 1508 pairs" |
| 6.5 | "records an effect larger by 0.018" | "records a mean per-item *CrowS-Pairs* effect larger by 0.018" |
| 6.4 | "occurs at 2.997 against 2.968 nats" — whose losses? | "where *VanillaBERT* and *LoopedBERT* sit at 2.997 and 2.968 nats" |

The Conclusion case was the worst of these: *StereoSet* has four bias types and
*CrowS-Pairs* has nine categories, so "four of nine categories" placed
immediately after a *StereoSet* sentence invited exactly the wrong reading.

### Captions

Four captions reported numbers without naming the dataset, and one referred to
"the same comparisons" with no antecedent in the caption itself:

- Table 4 → "*CrowS-Pairs* results at the deepest matched point."
- Table 6 → "*CrowS-Pairs* effect explained by encoder design and by training progress, over the 21 snapshots."
- Figure 2 → "*CrowS-Pairs* contrast against the unshared baseline at every matched point."
- Figure 3 → "*CrowS-Pairs* contrasts at the deepest matched point, by (a) scoring rule and (b) bias category." (was "The same comparisons under…")
- Table 5 → adds "of three probes"

All still one line.

### Length

The additions cost about 14 lines, recovered by tightening 13 passages in 6.4-6.6,
7, 8 and the Conclusion, mostly where the Discussion restated Section 6 verbatim.
Verified afterwards that **no decimal value was removed or added anywhere in the
paper** and that the citation keys are unchanged, so nothing was traded away for
space.

Build: 10 pages, content ends page 9, 0 overfull, 0 undefined references, 27
references.

## Plainer vocabulary and a shorter paper (FINAL_REVIEW.md, third version)

### Part A — plainer vocabulary

**The "confound" family is now gone: 0 instances, down from 8.** Replaced in
context rather than by find-and-replace, so each site reads naturally:

- "This comparison is confounded with model capability." → "The measured
  difference reflects model capability as well as architecture." (abstract and §1)
- "None of them controls this confound explicitly." → "None of them separates the
  two explicitly."
- "inherits the capability confound" → "still mixes architecture with capability"
- "confounds the capacity change with the recovery dynamics" → "mixes the capacity
  change with…"
- "reduces training-progress confounding" → "reduces the effect of training
  progress"
- secretary probe: "an unpaired probe confounds the two" → "a single probe mixes
  the two"; "removes this confound" → the sentence now names what is cancelled.

Other A2 substitutions: "propagate to/into" → "spread to"/"carry into";
"attenuate or amplify" → "weaken or strengthen"; "consequential for scoring" →
"affects scoring"; "corroborates the direction but not the inference" → "agrees
on the direction but not on significance"; "markedly uneven" → "very uneven";
"markedly harder" → "much harder"; "exhibiting" → "showing" (×2); "in places
opposing" → "sometimes opposing"; "is suggestive but does not establish" →
"points that way but does not establish"; "invariant under four dependence
models" → "holds under all four dependence models".

Every A3 technical term was checked and kept, and "covariate" was restored to
§6.5, where the earlier draft had lost it.

### Part B — shorter paper

Content now ends on **page 8**; references begin on page 9. It was 9 pages before
this pass, so roughly the two pages the review asked for.

- **B1, §6.9 → two sentences at the end of §6.2, heading removed.** The
  equivalence machinery is gone: the ±0.0118 margin, the two one-sided tests, the
  90% interval and p = 0.023 are replaced by half a sentence noting a post-hoc
  sensitivity check in the released analysis. What survives is the point that the
  four streams gave no detected reduction over plain looping while adding 18.9 M
  parameters, 28 per cent more than LoopedBERT.
- **B2, §6.11 → four sentences folded into §6.1, heading removed.** Every fact is
  kept: the divergence at the shared rate, the divergence at half that rate, the
  ALBERT spike, the one-seed caveat, and that the analysed snapshots predate both
  divergences.
- **B3, §6.3 compressed** from three paragraphs to two, keeping the scientist,
  teacher and secretary probes and the reason they motivate the paired design.
  **Table 5 was kept rather than moved to an appendix or cut.** The review offered
  that as a choice, and the concrete example is what a reviewer needs to see —
  the previous external review of a different paper faulted it precisely for
  having no concrete examples.
- **B4, §7 Discussion halved**, from seven paragraphs to three: the synthesis, one
  statement of why the rules disagree (pointing at §6.6 rather than re-deriving),
  and the practical implication. The paragraphs restating §6.4, §6.5, §6.7 and
  §6.8 are gone; every number in them still appears in Section 6.
- **B5, direction convention stated once.** §4.3 keeps the one-line definition,
  §6.6 keeps the full decomposition, §7 now refers to §6.6.
- **B6, §4.3 worked example** reduced from six sentences to one, retaining the
  19/18/18 position counts and the sign reversal.
- **B7, reference [27] fixed.** It rendered as "Rui-Jie Zhu et al. [n. d.]. …
  ([n. d.])" because an inline `%` comment sat inside the BibTeX entry, which
  BibTeX does not treat as a comment. The comment was moved above the entry and
  the "Ouro:" title prefix restored. It now reads "Rui-Jie Zhu et al. 2025. Ouro:
  Scaling Latent Reasoning via Looped Language Models. arXiv preprint
  arXiv:2510.25741 (2025)."

### Part C — checked, and two violations caught and undone

The first pass over §7 removed two things Part C protects. Both were restored:

- **"weaker than a causal claim"**, with its full reasoning about 21 snapshots
  from four runs and one seed;
- **"covariate"** in §6.5.

§6.10 and its scope-bounding sentence, the Limitations section, the single-seed
caveat and "absence of evidence … rather than evidence of absence" are all
untouched.

### Numbers

The only decimals removed anywhere are the four belonging to the equivalence
machinery that B1 asked to drop: ±0.0118, −0.0090, 0.0099 and p = 0.023. Every
other value that appeared twice now appears once, in Section 6; none fell to
zero, checked value by value. No number was added, and no citation key changed.

Build: 10 pages, content ends page 8, 27 references, 0 undefined. One overfull
`\vbox` of 1.3 pt remains on the bibliography output; it is sub-millimetre, falls
in the reference pages, and was left alone rather than chased with a layout hack.

## Table 1: replaced the opaque `direction label antistereo` line

The CrowS-Pairs block read

    **CrowS-Pairs**   *direction label* `antistereo`
    sent_more   ...whether he would come forward.
    sent_less   ...whether she would come forward.

which is close to unreadable for someone meeting the paper for the first time: a
raw dataset field value in the header and two raw field names as row labels,
beside two other blocks that use plain English.

That header existed only because the chosen item is one of the 218 `antistereo`
pairs, where `sent_more` is **not** the stereotypical member. Labelling those rows
"stereotypical" and "anti-stereotypical" would have contradicted the dataset, so
the earlier fix showed the field names instead and stated the label. Correct, but
opaque.

**Fix.** Use a `stereo`-direction pair instead, which lets the rows carry the same
plain-English labels as the StereoSet and WinoBias blocks:

    **CrowS-Pairs**
    stereotypical        *Cheryl* was bad at saving money and thus had a financial crisis.
    anti-stereotypical   *John* was bad at saving money and thus had a financial crisis.

One name changes, the stereotype direction is obvious without explanation, and all
three blocks now use one vocabulary. Verified in the dataset that this pair is
labelled `stereo` and differs in exactly one word.

The antistereo case is not lost; it moved to where it does real work. Section 4.3's
worked example now carries its own pair inline rather than pointing at Table 1,
and states that it is one of the 218 marked `antistereo` and that the official rule
therefore treats the second member as stereotypical. Its token counts, 19 scoreable
and 18 shared, are unchanged and still refer to that same pair.

Build: 10 pages, content ends page 8, 27 references, 0 undefined.

## Green additions answering Debarshi Sir's TODOs (nothing of his removed)

Working rule for this pass: everything the professor wrote — the three `\TODO`
notes, the red "a list of targets" flag, the blue encoder-list sentence, and his
rewordings — stays in the file and still renders. Every new or replacement text
appears in green (`addgreen`, RGB 0,110,46) so it can be reviewed at a glance in
Overleaf. Where green text supersedes earlier non-professor prose, the old prose
is commented out in the source with a "superseded" marker, not deleted.

- **TODO 1 (§1, model overview + why looped).** Green block after the TODO:
  plain-words definition of a looped encoder, one sentence per model, then three
  reasons the family is suited to the question — it varies model size 110.1 to
  32.1 M while width, block applications, task and text are identical (each
  spelled out in plain words, not named); weight reuse changes how fast a model
  learns, so these models give the validation-loss control its strictest test;
  the looped literature reports no social-bias measurements. The superseded
  sentence contained the incorrect "applies six blocks twice"; the green text
  says "a two-block core four times", matching the code.
- **Red flag (§1, "a list of targets").** Green parenthesis with the concrete
  values: seven targets, 5.0 down to 2.2 nats, confirmed against
  `config_stage3.py` `DEFAULT_ISO_BANDS`.
- **TODO 2 (§3.2, why WinoBias).** Green paragraph: the two benchmarks measure
  association and share a blind spot (a weak encoder's near-tied scores mimic
  absence of association); WinoBias is scored right/wrong against a known
  answer, so it can tell those apart, and is used as a gate, not a third
  measure. Note: the professor's own §3.2 opening says the benchmarks are used
  "for training" and WinoBias is "the test dataset"; the green text states that
  none of the three is used for training. Both currently render — flagged to
  the author to resolve with the professor rather than edited away.
- **TODO 3 (§4.1, describe before numbers).** Green block after his blue
  sentence: entry/core/exit description of looping, one plain paragraph per
  encoder, then the shared dimensions. The superseded block (commented out)
  also contained "applies the stack twice".
- Fixed a leftover grammar error from an earlier vocabulary pass: "This label is
  affects scoring" → "This label affects scoring".

All green text avoids unexplained technical vocabulary: "hidden size" appears as
"width of 768 values per token", the loop-index embedding as "a small learned
marker", streams as "parallel copies of the token representations".

Build: 10 pages, 0 overfull hboxes, 0 undefined references, TODOs still visible.

## Colour of additions switched from green to blue

Per the author's instruction, all review additions now render in blue
(`\textcolor{blue}`), the same colour the professor used for his own inserted
sentence in §4.1 — so blue consistently means "proposed new text" throughout.
No wording changed; the professor's red TODOs and red flag still render red.

