# TRACK_DECISION — FIRE 2026

## Recommendation: **Regular Paper**

## Reasoning

The revised contribution is methodological, which superficially suggests the
Perspective track. It should still go to Regular, for three reasons.

1. **The evidence base is original empirical work, not argument.** Four encoders
   were pre-trained from scratch on identical data (7B tokens each), snapshotted
   at matched validation loss, and scored across 21 distinct snapshots and 1508
   items, giving 31 668 architecture-snapshot-item observations. A Perspective
   paper carrying this much new experimentation would be miscategorised.
2. **The central claims are statistical results, not positions.** "The contrast
   reverses across matched points" and "no architecture term survives continuous
   loss adjustment" are measurements with intervals and tests. A Perspective
   paper argues from existing literature; this argues from new data.
3. **The recommendation follows from the data rather than motivating it.** The
   protocol advice in the Discussion is a consequence of the experiments, which
   is the Regular-paper shape.

## Risks of this choice

- A reviewer may read a negative result as a thin contribution. Mitigated by the
  fact that the paper reports four distinct robustness analyses, three of which
  produce new positive findings (loss-adjusted null, scorer sensitivity, category
  sign changes) rather than only the failure of the original claim.
- Regular Papers attract stricter methodological review. That is acceptable here:
  the revision was driven by exactly the objections such review would raise, and
  the answers are already in the paper.

## What would change under Perspective

A Perspective version would cut the architecture specification and training
details to a short paragraph, drop Figure 1 and Table 1, and expand the
recommended-protocol argument with evidence drawn from published comparisons
beyond this study. That is a different paper, and it would discard the strongest
asset — controlled from-scratch pre-training with matched checkpoints. Not
recommended.

## Track requirements either way

Both Regular and Perspective are double-blind at FIRE 2026, both cap content at
9 pages excluding references, and both require CCS concepts and keywords. The
submission satisfies all of these (see ANONYMITY_AUDIT.md).
