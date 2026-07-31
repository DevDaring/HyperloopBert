# Testing the Robustness of Architecture Bias Comparisons

Anonymised artifact for a FIRE 2026 submission. It contains the training and
evaluation code, the per-item scores for every model snapshot, and the analysis
scripts that produce every number in the paper.

## What this study asked, and what it found

Comparing two model designs for social bias is confounded by capability: a
better-trained model answers more sharply on nearly every probe, bias probes
included. The usual remedy is to compare checkpoints that have reached the same
validation loss. This work tested whether that remedy is sufficient.

Four encoders were pre-trained from scratch on identical data, differing in how
far they reuse the same weights across depth. Each saw the same 7 billion tokens
(28 billion in total) with **one random seed**, producing **21 distinct
snapshots**. Those snapshots were scored on two stereotype benchmarks under three
scoring rules.

**At a single matched point the answer looked clear. It did not survive.**

| Check | Result |
|---|---|
| Single deepest matched point | 2 of 3 weight-reusing encoders record a lower stereotype effect, both surviving Holm correction (p = 0.0003) |
| Repeated at every matched point | the contrast **changes sign**; it favours the reusing encoder at **1 of 5** points, mean −0.0063 |
| Adjusting continuously for realised loss | **no** architecture term distinguishable from zero; the loss term is (p = 0.0002) |
| Shared-token scoring rule | 0 of 3 contrasts survive correction |
| Benchmark's official scoring rule | 0 of 3 survive, and the looped contrast **reverses direction** |
| Second benchmark (StereoSet) | direction agrees 3 of 3, significance carries 1 of 3, for a *different* encoder |
| Capability check (WinoBias) | all four encoders at chance (0.489–0.503), so these numbers index word association, not task behaviour |

The conclusion is methodological: a bias comparison between architectures that
rests on one matched checkpoint and one scoring rule can reverse under either
choice.

### Two things worth knowing before reusing this code

**The three scoring rules are not interchangeable.** The benchmark's released
implementation scores the tokens the two sentences have *in common* — not the
tokens that differ — and it *sums* log probabilities rather than averaging them.
An earlier version of this analysis described that rule backwards. See
`analysis/fire2026/official_crows.py` for a faithful reimplementation.

**The reimplementation does not exactly reproduce the published number.** On
`bert-base-uncased` it returns 58.49 where Nangia et al. (2020) report 60.5. The
two-point gap was not closed and is most likely library drift. Because all
architectures are scored with identical code the offset is common-mode and
cancels in the contrasts, but absolute scores here should not be compared against
other papers.

## Repository layout

```
Codes/
  common/          architectures, training loop, bias metrics, schemas, integrity
  Dataset/         corpus download, pre-tokenisation, benchmark download
  Stage1..Stage3/  staged training and evaluation entry points
  results/         per-item scores and summaries for every snapshot
analysis/fire2026/ the analysis behind the paper (see below)
```

### The analysis scripts

| Script | Produces |
|---|---|
| `phase2_reanalysis.py` | contrasts at the matched point, per-category effects, stereotype scores, equivalence test |
| `phase2b_allband.py` | the same contrasts recomputed at every distinct matched point |
| `phase3_robustness.py` | continuous loss adjustment, unalignable-item forensics, per-category correction |
| `phase4_stereoset.py`, `phase4b_stereoset_analysis.py` | second-benchmark replication |
| `official_crows.py`, `phase5_official_crows.py` | faithful official-metric scorer and its application |
| `phase6_regression_audit.py` | four dependence models for the loss-adjusted regression |
| `make_revision_figures.py` | the figures, from committed CSVs only |

Outputs land in `analysis/fire2026/out/`. Every number in the paper traces to a
file there. Seeds are fixed, and re-running reproduces the outputs
byte-identically.

## Reproducing the analysis

The analysis runs on CPU. It needs no GPU and no retraining:

```bash
python -m pip install numpy pandas scipy statsmodels matplotlib
python analysis/fire2026/phase2_reanalysis.py
python analysis/fire2026/phase2b_allband.py
python analysis/fire2026/phase3_robustness.py
python analysis/fire2026/phase6_regression_audit.py
```

Rescoring from checkpoints, also CPU and a few hours, additionally needs `torch`,
`transformers` and `datasets`:

```bash
python analysis/fire2026/phase4_stereoset.py        # second benchmark
python analysis/fire2026/phase5_official_crows.py   # official metric
```

Software used: Python 3.10, NumPy 2.2, pandas 2.3, SciPy 1.15, statsmodels 0.14.

## Reproducing the training

Pre-training needs one H100-class GPU and roughly seven hours per encoder.

```bash
python Codes/Dataset/download_training_corpus.py
python Codes/Dataset/pretokenize_corpus.py
python Codes/Dataset/download_eval_datasets.py
python Codes/Stage3/train_stage3.py --seeds 1
```

Configuration lives in `Codes/Stage3/config_stage3.py`: AdamW, peak learning rate
3e-4 (1.5e-4 for the stream variant, which diverged at the higher rate), batch
512, 10% warmup, gradient clipping 1.0, mixed precision, sequence length 128.
Snapshots are written the first time validation loss crosses each target in
`DEFAULT_ISO_BANDS`.

## Credentials

No credentials are included and none are needed to reproduce the analysis.
Scripts that upload artifacts read tokens from a local `.env` that is not part of
this repository. Dataset namespaces are set to `ANONYMOUS` for review.

## Limitations carried by this artifact

- **One training seed.** Every interval in the results is a bootstrap over
  benchmark items and describes item sampling, not variation between training
  runs. The ordering of the encoders may not be stable under retraining.
- **Undertrained models.** The deepest matched point is around two nats, well
  short of a fully trained encoder, and the measured bias is still moving there.
- **Correlation-type measurement only.** Both benchmarks read the same masked
  language modelling head, and the coreference capability check fails, so nothing
  here demonstrates downstream behavioural fairness.
- **Post-hoc analyses.** Only the three contrasts at the deepest matched point
  were specified before that point was analysed. Everything else is a robustness
  check and is reported as exploratory.
- **English only.** A caste-and-religion instrument was planned, but the
  available mirror was unusable with an English-only tokeniser.

## Benchmarks used

- CrowS-Pairs (Nangia et al., EMNLP 2020) — 1508 pairs, 9 bias categories
- StereoSet (Nadeem et al., ACL 2021) — intrasentence split, 2106 items
- WinoBias (Zhao et al., NAACL 2018) — 374 items, capability check only
- FineWeb-Edu (Penedo et al., NeurIPS 2024 Datasets and Benchmarks) — corpus

All carry documented item-quality caveats; see Blodgett et al. (ACL 2021).
