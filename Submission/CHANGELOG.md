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
