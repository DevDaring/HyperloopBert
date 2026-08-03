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

