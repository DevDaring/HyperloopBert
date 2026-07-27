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

## A9. Literature/citation corrections (see paper related-work)

Citation errors identified in the Round-2 review are corrected in the docs
(see `improvement_suggestion.md` Part B.0): the Ouro looped-LM work is
attributed to Zhu et al. (2025), arXiv:2510.25741; arXiv:2603.08391 is
correctly attributed to Frey et al. (2026); and the Frey/Ouro findings are
reframed as *support* for (not counter-evidence against) the Stereotype
Consolidation Hypothesis, with SCH sharpened to a claim about *total* capacity
at matched quality (per-parameter memorization is preserved under looping).
