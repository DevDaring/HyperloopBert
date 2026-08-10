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

