# HyperloopBERT — Pre-Registration Amendment (2026-07-21)

This document amends the original pre-registered analysis plan (STAGE1/2/3
instructions + `Codes/HYPERLOOP_BIAS_BUILD_PROMPT.md`). It records every
deviation from, and addition to, the registered plan made during the Round-2
re-audit, with the reasoning. Reviewers respect disclosed amendments; they do
not respect silent deviations. Cite this file in the paper's methods section
and release it in the artifact repository alongside the original
pre-registration.

Each amendment lists **what changed**, **why**, and **where in the code**.

---

## A1. Bias scoring runs in FP32, not BF16 autocast

**What changed.** All PLL, SS-PLL, WinoBias masked-pronoun, and loop-trajectory
scoring forwards now run in full FP32 (autocast disabled; attention routed off
the BF16-only FlashAttention kernel). Training is unchanged (still BF16
autocast). Only *measurement* precision changed.

**Why.** BF16 rounding is the same order of magnitude as the PLL gaps that
decide borderline pairs, so scoring under autocast injected metric noise
exactly where the paired contrasts are decided. FP32 scoring removes that noise
without affecting the trained weights.

**Where.** `common/bias_metrics.py` (`_fp32_forward`, applied in `compute_pll`,
`compute_ss_pll`, `score_winobias_masked_pronoun`);
`common/attention.py` (`force_full_precision_attention` context manager);
`Stage2/loop_trajectory_stage2.py` (depth-probe forwards).

---

## A2. Stage 1 capability gate expanded from one leg to three

**What changed.** The undertrained-model guard is now a three-leg gate,
evaluated on VanillaBERT (base) at the primary iso-loss band before ANY
contrast is interpreted:

- **Leg 1 (new, pulled forward from Stage 2):** SST-2 + RTE fine-tuning
  accuracy above chance (one-sided exact binomial test, α = 0.05).
- **Leg 2 (original):** baseline stereotype bias detectable — item-level
  preference-rate bootstrap CI excludes 0.5 from above.
- **Leg 3 (new):** WinoBias masked-pronoun accuracy on the pro-stereotypical
  splits above chance (exact binomial). On pro items, stereotype knowledge and
  coreference ability point the same way, so this is the most sensitive
  available detector that *any* gendered-pronoun signal was learned; at chance
  means Stage 2 WinoBias would measure noise.

Gate status is now `PASS` / `FAIL` / `INCOMPLETE` (evidence missing → contrasts
must not be interpreted until produced).

**Why.** A single baseline-bias leg cannot distinguish "the architecture
mitigated bias" from "the model is too undertrained to exhibit or resolve the
constructs at all". Legs 1 and 3 establish the model has the general capability
(GLUE) and the specific gendered-coreference signal (WinoBias) that the bias
instruments presuppose.

**Where.** `Stage1/eval_capability_stage1.py` (new; produces the evidence);
`Stage1/analyze_stage1.py` (`capability_gate`, `_glue_capability_leg`,
`_coreference_capability_leg`); `Stage1/config_stage1.py` (`GLUE_TASKS`,
`CAPABILITY_ALPHA`); `common/stats_engine.py` (`binomial_p_above_chance`).

---

## A3. Corpus-level stereotype statistics added

**What changed.** A new one-pass corpus script counts, for every benchmark
pair, the co-occurrence of the pair's stereotype/anti lexemes with its shared
context lexemes in the training corpus, and reports per-category summaries
(mean/median co-occurrence, zero-co-occurrence fraction).

**Why.** FineWeb-Edu is heavily filtered. If the tested stereotype associations
barely occur in the corpus, a null bias result means "never learned", not
"architecturally mitigated", and a near-chance baseline is *expected* rather
than suspicious. This table is the pre-registered interpretation aid for null
or near-chance results, and the defense if baseline bias lands near chance.

**Where.** `Dataset/corpus_stereotype_stats.py` (new); schemas
`CORPUS_PAIR_STATS_COLUMNS` / `CORPUS_CATEGORY_STATS_COLUMNS` in
`common/io_schemas.py`.

---

## A4. Indian-context deliverables completed to spec

**What changed.** The Indian-context instrument now writes the four
per-category CSVs (`english/{Caste,Gender,India_Religious,Race}.csv`) and the
`provenance/provenance_report.json` (base = Indian-BhED, extension source and
relationship, validation status SUPPLEMENTARY, IAA note, and the
kappa < 0.6 → demote-to-supplementary recommendation). The merged
`indian_bias_english.csv` is still written and remains the file the evaluators
consume (per-category rows carry their `Category`, so per-category preference
rates are recovered from the single eval pass).

**Why.** Spec 7.2/7.3 deliverables; the India-centric extension must be
defensible and its provenance auditable.

**Where.** `Dataset/download_eval_datasets.py` (`download_indian_bias`,
`_write_indian_provenance`).

---

## A5. Stage 3 sequence-length adaptation tail implemented (kept, not amended away)

**What changed.** Stage 3 primary runs now execute the registered two-phase
budget: 400M tokens @ seq=128, then a 100M-token adaptation tail @ seq=256,
continuing from the phase-1 weights (500M total). The stream-count ablation
arms remain seq=128 only, so the dose-response varies stream count alone.

**Why.** Spec 2.2 requires the tail (sequence-length adaptation). It was
missing from the config; rather than amend the spec to drop it, it is now
implemented. The paper's limitations section no longer needs to disclose an
omission here.

**Where.** `Stage3/config_stage3.py` (`SEQ_LENGTH_TAIL`, `TAIL_TOKENS`,
`TAIL_TOKEN_MARKERS`, `TAIL_ISO_BANDS`); `Stage3/train_stage3.py` (phase-2
driver); `common/train_loop.py` (`run_mlm_training` now accepts `seq_length`).

---

## A6. Stage 2 external-calibration gate — deviation from spec (DISCLOSED)

**What deviates.** The registered plan asked the calibration gate to reproduce
the published anchor *ordering* on CrowS-Pairs. The implemented gate instead
requires each published anchor (bert-base-uncased, albert-base-v2) to show
**above-chance** stereotype preference on the canonical shared-token (Nangia et
al. 2020) metric, and records the observed ordering descriptively without
gating on it.

**Why.** Our internal primary metric is full-sentence PLL, which differs from
the published shared-token scoring. Demanding the *published ordering* under a
*different metric* is an invalid check: any ordering difference could come from
the metric change rather than a pipeline fault. Above-chance-on-the-canonical-
metric is the valid sanity condition. The observed ordering is still reported
for transparency.

**Additional hardening.** Missing a *required* published anchor now hard-fails
the calibration step (rather than logging an error and proceeding on an
unevaluable gate). roberta-base is treated as an auxiliary anchor and does not
hard-fail.

**Where.** `Stage2/analyze_stage2.py` (`calibration_check`);
`Stage2/external_calibration_stage2.py` (hard-fail on required anchors).

---

## A7. Hyper-connection matrix stability check added (no retraining)

**What changed.** The Stage 3 mechanistic suite now reports, for the trained
4-stream HyperloopBERT, per-stream Frobenius norms of the learned depth/width
projection blocks, their deviation from the structured initialization, and each
projection's spectral norm per loop.

**Why.** Xie et al. (2025, MHC, arXiv:2512.24880) show unconstrained
hyper-connections can destabilize at scale. These statistics let the paper
report whether our scale exhibits that instability (cite MHC as validation) or
not, at zero additional training cost.

**Where.** `Stage3/stream_analysis_stage3.py` (`analyze_hyperconnection_stats`);
schema `HYPERCONNECTION_STATS_COLUMNS` in `common/io_schemas.py`.

---

## A8. Operational fixes (no analysis-plan impact)

- **Failed eval rows are retried on resume.** Previously a `max(Row_Index)+1`
  cursor froze any failed row forever. Resume now retries rows with a null
  `Effect_Size`; summaries deduplicate per `Row_Index` (keep last) so
  failed/scored counts stay exact. (`Stage1/eval_bias_stage1.py`,
  `Stage2/eval_bias_stage2.py`.)
- **Dead `use_bf16` flag removed** throughout `common/architectures.py` and
  `common/attention.py` (attention dtype is governed by the autocast context,
  not this flag).
- **Stale Stage 1 seed comment corrected**; seeds now come from an explicit
  ordered `SEED_POOL` (`Stage1/config_stage1.py`, `Stage1/train_stage1.py`).

---

## A11. Empirical calibration episode: learning rate, token budget, and the
##      iso-loss bands (MATERIAL — changes registered hyperparameters)

This section documents a full pre-run calibration episode carried out on
2026-07-27 on an NVIDIA L4. It changes two registered hyperparameters and
invalidates the registered token budget. Everything here is evidence-driven and
every claim below is backed by a logged experiment; the raw logs are archived in
`results/pilot/`.

### A11.1 The registered learning rate collapses the model

The first full Stage 1 run was launched with the registered
`LEARNING_RATE = 5e-4`. After **47M tokens** the model had learned the unigram
distribution and nothing else:

- validation loss pinned at the unigram-entropy plateau (~7.2) and *rising*
  (7.2042 → 7.1875 → 7.2589);
- the model emitted an **identical** distribution for every input
  (cosine similarity between three unrelated contexts = **1.0000**, i.e. LESS
  context-sensitive than random initialisation at 0.9865);
- top predictions were pure function-word frequencies (`the . , of and to a in`).

Ruled out by direct test, not assumption: MLM masking was correct (14.8%
supervised / 11.6% `[MASK]` / 86.9% unchanged), and FlashAttention numerics
matched the eager reference to 7.9e-3 at real token positions (an apparent 1.40
discrepancy was located entirely at PADDED positions, which the loss ignores).

**Cause: the LR is far too high for a post-norm BERT at an 8192-token effective
batch — roughly 80x more aggressive per token than the original BERT recipe.**

A 5-configuration sweep (6M tokens each, seed 42, identical data order)
isolated it. Validation loss is uninformative this early (every config sits at
the unigram floor); **context sensitivity is the discriminating metric**:

| config | end loss | ctx_cos | reading |
|---|---|---|---|
| lr=5e-4 batch=64 (REGISTERED) | 7.20 | **1.0000** | context ignored |
| **lr=1e-4 batch=64** | 7.20 | **0.9103** | **learning context** |
| lr=3e-5 batch=64 | 7.58 | 0.9953 | too slow |
| lr=5e-4 batch=256 | 7.20 | 1.0000 | batch is not the lever |
| lr=2e-4 batch=256 | 7.23 | 1.0000 | still too high |

**AMENDMENT: `LEARNING_RATE` is changed from 5e-4 to 1e-4.** A 30M-token
confirmation run reproduced steady improvement (ctx_cos 0.9993 → 0.9390 while
loss fell 9.605 → 6.961), confirming the fix.

### A11.2 The registered token budget cannot reach the registered iso-loss bands

With the corrected LR, the measured loss curve decays approximately
logarithmically: a 4x token increase (7.5M → 30M) bought only 0.36 nats.
Extrapolated to the registered `MAX_TOKENS = 200M`, the predicted loss is
**~6.4–6.5 (pseudo-perplexity ~600–650)**.

The registered protocol requires:
- iso-loss bands `[4.0, 3.7, 3.4, 3.1]` — **never crossed**, so **zero** iso-band
  snapshots would be written;
- quality screen PP <= 60 (loss <= 4.09) — **every** snapshot skipped.

**A Stage 1 run at the registered budget would therefore complete "successfully"
and produce no data at all.** This is a calibration error in the registered
design, independent of the learning rate: the bands are appropriate to a
multi-billion-token budget, not 200M.

### A11.3 Pilot study: the capability gate fails at this budget

A pre-registered-style pilot trained the Stage 1 contrast pair (VanillaBERT vs
LoopedBERT, `tiny`, lr=1e-4, 100M tokens each) and applied the real capability
gate and primary test:

| | val loss | PP | mask acc | ctx_cos |
|---|---|---|---|---|
| VanillaBERT | 6.766 | 867.9 | 0.066 | 0.880 |
| LoopedBERT | 6.769 | 870.3 | 0.068 | 0.866 |

- **Capability gate (leg 2): FAIL.** Vanilla stereotype preference
  **0.4525, 95% CI [0.4050, 0.5025]** — the CI includes chance, so there is no
  statistically detectable baseline bias to reduce.
- **Signal: absent.** Vanilla and Looped preference rates were identical
  (0.4525 vs 0.4525); item-level delta = -0.0051 [-0.0150, +0.0054],
  p = 0.8345, d = -0.048, n = 400. This is noise, **not** evidence against SCH.

### A11.4 The instrument is sound; the models were undertrained

To distinguish "our metric is broken" from "our models are too weak", the same
FP32 PLL scorer was applied to properly-trained public models on the same 400
pairs:

| model | geometry | preference | 95% CI | detectable |
|---|---|---|---|---|
| google/bert_uncased_L-2_H-128 | 4.4M | 0.5825 | [0.5349, 0.6350] | YES |
| google/bert_uncased_L-4_H-256 | 11.3M (our `tiny` width) | 0.5875 | [0.5375, 0.6400] | YES |
| google/bert_uncased_L-8_H-512 | 41.4M | 0.5500 | [0.5000, 0.6000] | borderline (n=400) |
| bert-base-uncased | 110M | 0.5975 | [0.5500, 0.6425] | YES |
| **ours (H=256, L=12, 100M tokens)** | 17.6M | **0.4525** | [0.4050, 0.5025] | **NO** |

**A 4.4M-parameter model — smaller than our `tiny` — shows detectable
stereotype bias when adequately trained.** Therefore model capacity is NOT the
limiting factor and the measurement instrument is valid; the binding constraint
is the **token budget**. This is the decisive justification for re-running at a
substantially larger budget rather than abandoning the design or reporting a
spurious null.

### A11.5 Consequent amendments

1. **`LEARNING_RATE`: 5e-4 -> 1e-4** (A11.1).
2. **Token budget: raised by ~1-2 orders of magnitude.** The registered 200M is
   demonstrably insufficient for the capability gate to pass.
3. **Iso-loss bands: to be re-derived from a measured trajectory** at the new
   budget rather than assumed, so that all architectures actually cross a common
   band (the iso-loss protocol is meaningless otherwise).
4. **Data pipeline: corpus pre-tokenised to a memory-mapped binary**
   (`Dataset/pretokenize_corpus.py`). The on-the-fly tokenizer was measured at
   ~4% model-FLOPs utilisation (2.4 of ~60 TFLOPS on an L4). The pre-tokenised
   path was verified to emit **byte-identical** sequences to the original
   `data_generator`, so this is a pure throughput change with no effect on the
   science.
5. **A null result at an inadequate budget will NOT be reported as evidence
   about SCH.** Per A11.3/A11.4 such a null reflects undertrained models, and
   the capability gate correctly refuses to interpret it. This is recorded here
   so the distinction cannot be blurred after the fact.

### A11.6 What this episode demonstrates about the protocol

The pre-registered capability gate did exactly what it was designed to do: it
refused to convert an undertrained model into a publishable bias claim. The
pilot cost ~1.5 GPU-hours and prevented a ~28 GPU-hour run whose output would
have been uninterpretable. This is reportable as a methods contribution in its
own right.

---

## A10. Construct scoping against difference awareness (NEW — framing, no code change)

**What changed.** The paper now states explicitly *which kind* of fairness construct
it measures, using the taxonomy of Wang et al. (2025), "Fairness through
Difference Awareness: Measuring *Desired* Group Discrimination in LLMs"
(ACL 2025 Best Paper, arXiv:2502.01926). That paper distinguishes:

- **descriptive** benchmarks (fact-based group representation),
- **normative** benchmarks (value-based judgements about group treatment),
- **correlation** benchmarks (association-based).

Our primary instrument — CrowS-Pairs-style PLL stereotype *preference rate* —
is a **correlation** benchmark. The paper now says so, and scopes its claim
accordingly: we measure stereotype **association**, NOT difference-aware
fairness.

**Why.** Wang et al. show that difference-*unaware* treatment (colour-blindness)
is not universally the correct fairness target, and that **existing bias-mitigation
strategies can backfire** on difference-aware tasks. Our pre-registered decision
rule encodes the assumption "lower stereotype preference rate is better". That
assumption is defensible for correlation-type bias but must not be silently
generalised to fairness overall. Stating the scope is more honest than implying
a broader claim, and pre-empts an obvious reviewer objection at a venue that
just awarded this paper Best Paper.

**Consequences for the claims (all framing, no analysis change):**
1. **Scoping:** the finding is about correlation-type stereotype association at
   matched quality, not about difference-aware fairness.
2. **Limitation:** a reduced group-association rate is not automatically
   desirable; in principle the same mechanism could erode *legitimate* group
   distinctions.
3. **Sharpened prediction (future work):** SCH says weight sharing reduces
   *total* stereotype storage at matched quality. If that mechanism is
   non-selective, it should ALSO erode legitimate **descriptive** group
   knowledge. This is a stated failure mode of the hypothesis, and it
   complements the manipulation-vs-storage framing from Ouro/Frey (A9).

**Not run as an experiment, and why.** The difference-awareness benchmark is
16k 3-option multiple-choice items targeting instruction-tuned generative LLMs,
and its descriptive items require factual knowledge (asylum law, occupations,
religions) while its normative items require value judgements. Our models are
17M–110M-parameter MLM encoders trained from scratch on 200M tokens, which the
Stage 1 capability gate already screens as barely above chance on SST-2/RTE.
Every architecture and size would score at chance (~33%), producing a uniform
null with no discriminative power for SCH. Running it would add cost and noise,
not evidence. It is therefore cited and engaged with, not executed — and named
as future work at instruction-tuned scale.

**Where.** `Codes/README.md` (citations), `research_explained_technical.html`
(SCH framing + scope), `STAGE1_INSTRUCTIONS.md` (citations),
`Codes/common/bias_metrics.py` (construct note next to the Blodgett caveat),
`Codes/Stage1/analyze_stage1.py` and `Codes/Stage2/analyze_stage2.py`
(generated paper-outline limitations).

---

## A9. Literature/citation corrections (see paper related-work)

Citation errors identified in the Round-2 review are corrected in the docs
(see `improvement_suggestion.md` Part B.0): the Ouro looped-LM work is
attributed to Zhu et al. (2025), arXiv:2510.25741; arXiv:2603.08391 is
correctly attributed to Frey et al. (2026); and the Frey/Ouro findings are
reframed as *support* for (not counter-evidence against) the Stereotype
Consolidation Hypothesis, with SCH sharpened to a claim about *total* capacity
at matched quality (per-parameter memorization is preserved under looping).
