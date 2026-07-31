# REGRESSION_AUDIT

Issue 5 asked whether the loss-adjusted conclusion survives a more careful
treatment of dependence. It does, but the original standard errors were too
small, and the paper now reports the conservative specification.

## The design problem

Architecture and realised validation loss vary at the **snapshot** level. The
1508 items inside one snapshot all share a single model checkpoint. Clustering on
item alone therefore treats those rows as carrying far more information about the
architecture coefficient than they do.

| quantity | value |
|---|---|
| item-level rows | 31,668 |
| benchmark items | 1,508 |
| **distinct model snapshots** | **21** |
| training seeds | **1** |
| snapshots per architecture | Vanilla 4, Looped 5, ALBERT 5, Hyperloop 7 |

The 21 snapshots, not the 31,668 rows, are the units that identify the
architecture and loss terms.

## Four specifications

| Model | Looped | ALBERT | Hyperloop | realised loss |
|---|---|---|---|---|
| M1 cluster on item *(original)* | 0.0047 (p 0.164) | −0.0065 (p 0.137) | −0.0066 (p 0.141) | −0.0184 (**p 0.0008**) |
| M2 two-way cluster item + snapshot | 0.0047 (se undefined) | −0.0065 (se undefined) | −0.0066 (se undefined) | −0.0184 (**p < 0.0001**) |
| **M3 snapshot-level, n = 21** | 0.0047 (p 0.424) | −0.0065 (p 0.280) | −0.0066 (p 0.265) | −0.0184 (**p 0.0002**) |
| M4 wild cluster bootstrap, 21 clusters, 9999 draws | 0.0047 (p 0.488) | −0.0065 (p 0.169) | −0.0066 (p 0.191) | −0.0184 (**p 0.0049**) |

Point estimates are identical across all four; only the uncertainty differs.

**Architecture terms reaching p < 0.05: 0 of 3 under every specification.**
**The realised-loss term is non-zero under every specification.**

## What changed, and what did not

The review's concern about the standard errors is correct. Moving from item
clustering to the snapshot-level analysis inflates the architecture standard
error by roughly 70% (0.0034 → 0.0058), which is the expected consequence of
counting 21 units instead of 1508. The **conclusion** is unchanged because the
architecture coefficients were far from significance to begin with.

M2 is reported for completeness but its architecture standard errors are
undefined: architecture is constant within a snapshot, so the snapshot dimension
nests it and the two-way estimator returns a non-positive-definite covariance for
those terms. This is a property of the design, not a numerical failure, and it is
itself evidence that snapshot-level variation is what carries the architecture
information. The loss term, which varies within architecture, is estimable.

## What the paper now reports

Table 3 carries the **M3 snapshot-level** estimates, with the item-level rows
stated as the input and 21 given as the number of identifying units. The text
says the data do not provide clear evidence of an architecture effect independent
of realised validation loss, and does not claim that capability *causes* the
single-point contrast.

## Remaining limitations

With 21 snapshots and one seed, none of these procedures has much power to detect
a small architecture effect. The honest reading is an absence of evidence at this
sample size, not evidence of absence. A mixed model with random slopes was not
attempted: with four architectures and 4–7 snapshots each, the variance
components are not identifiable.
