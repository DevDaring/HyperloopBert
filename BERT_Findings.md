# HyperloopBERT — Findings, Evidence, and Honest Assessment

**Status:** LIVE DOCUMENT — updated during the experiment, finalised on completion.
**Last updated:** 2026-07-28 08:15 UTC / 13:45 IST
(VanillaBERT ✅ · LoopedBERT ✅ · ALBERTLoopedBERT training · HyperloopBERT queued)

This document records what was actually found, with the data behind each claim, and an
honest accounting of strengths and weaknesses. It is written so a reviewer (or future-you)
can distinguish **what is established**, **what is provisional**, and **what is unknown**.

---

## 0. Executive summary

| | |
|---|---|
| **Strongest result so far** | A measurement-validity finding (§3) that holds regardless of whether SCH is true |
| **Core hypothesis (SCH)** | Not yet testable — pending completion of all four architectures |
| **Biggest risk resolved** | Models now reach **PP 4.39** (bert-base class). The pilot's models sat at PP 868 and could measure nothing. |
| **Biggest risk remaining** | Architectures are tracking so closely in quality that the bias contrast may be null (§6.2) |
| **Bugs found and fixed** | **16**, six of which would have produced no data or silently wrong data (§2) |

---

## 1. What the experiment is

**Stereotype Consolidation Hypothesis (SCH):** cross-layer weight sharing in looped
transformer encoders reduces the encoding of stereotypical associations **at matched model
quality**.

The critical design feature is **iso-loss matching**: architectures are compared at equal
validation loss — not equal parameters, not equal token budget — so "this model is less
biased" cannot be confounded with "this model is simply worse".

Four architectures at effective depth 12, `base` size (hidden 768), seed 42, 7B tokens each:

| Architecture | Unique layers | Shared ratio | Role |
|---|---|---|---|
| VanillaBERT | 12 | 0.00 | baseline |
| LoopedBERT | 6 | 0.50 | core SCH contrast |
| ALBERTLoopedBERT | 1 | 0.92 | maximum sharing (extreme point) |
| HyperloopBERT | 6 + streams | 0.50 | multi-stream + CWSA (novel architecture) |

---

## 2. Engineering findings — 16 real bugs

These matter scientifically, not merely operationally: **six would have produced either no
data or silently wrong data.**

### 2.1 Critical — would have invalidated results

| # | Bug | Consequence if unfixed |
|---|---|---|
| 1 | `get_attention_path` imported from `common.architectures` but defined only in `common.attention` | **Every training run in every stage crashes on import.** Present since the initial commit — the pipeline had never been run end-to-end. |
| 2 | Dry-run harness wrote into the **same** `results/`, `models/`, `checkpoints/` dirs as real runs, and `tiny` is a real size in Stages 1–3 | Dry-run garbage (50k-token models, artificial iso-bands) appended to the real `summary_table.csv`, silently corrupting every downstream contrast |
| 3 | `masked_pll_at_depths` derived architecture from `type(model).__name__`; `VanillaBERT6` **is** `VanillaBERT(num_layers=6)` — the same class | 12-layer hook layout applied to the 6-layer model → `IndexError`. Which arm crashed was decided by dict iteration order, so it looked non-deterministic. |
| 4 | FP32 input to the flash path raised, hit the runtime-failure handler, and **permanently demoted `ATTENTION_PATH` globally** | A single FP32 eval call silently downgrades every later forward **and** makes the logged "flash" provenance false in the paper |
| 5 | `Stage3/qualitative_stage3.py` was referenced by the run script but **did not exist** | The reviewer-facing MLM-output artifact would have failed "non-fatally" and produced nothing |
| 6 | Iso-loss bands unreachable at the registered token budget (§4.2) | Zero snapshots written; the run completes "successfully" with no data |

### 2.2 Correctness and performance

| # | Bug | Fix |
|---|---|---|
| 7 | Contamination filter matched 36,288 phrases per document — O(docs × phrases), ~13 docs/s → **~20 h** for 1M docs | Aho-Corasick automaton → minutes |
| 8 | `install.sh` renamed the flash-attn wheel to `flash_attn.whl`; modern pip parses version/ABI from the filename and rejects it | preserve the original filename |
| 9 | flash-attn wheel hardcoded `cp312`; the H100 image ships **Python 3.11** | detect the ABI tag at runtime (this is why Vast.ai provisioning worked first try) |
| 10 | CrowS-Pairs fallback mirror `HuggingFaceM4/Multi-lingual-crows-pairs` **no longer exists** | use canonical `nyu-mll/crows_pairs` |
| 11 | Indian-bias loader hard-crashed the whole dataset stage; the `Debk` mirror is **Bengali**, unusable by an English-only model | non-fatal + `Target/MASK` schema parser + English-only guard |
| 12 | `loop_trajectory_stage2` never populated identity columns → `KeyError` on write | populate from `get_model_info` |
| 13 | LLM-judge schema validation checked the JSON-Schema's own top-level keys instead of `required`/`properties` | validate the right keys |
| 14 | Stage 3 dry run would have triggered a **real 100M-token seq=256 phase** | `ENABLE_SEQ_TAIL` + dry-run overrides |
| 15 | Autopush swept the `dry_run/` QA sandbox into the results history | exclude + gitignore |
| 16 | HF upload would have published 4 byte-identical checkpoints as if they were distinct models | content-hash dedup + `duplicate_checkpoints.json` |

**Interpretation.** The frequency and severity here is itself a finding: a research pipeline
can look complete, read well, and still be structurally incapable of producing data. The
dry-run gate and the capability gate are what surfaced this.

---

## 3. FINDING 1 — Bias measurability depends on training adequacy, not model capacity

**Currently the strongest and most transferable result. It holds regardless of whether SCH
is true.**

### 3.1 The pilot failed, and the failure was informative

VanillaBERT and LoopedBERT, `tiny`, 100M tokens each:

| | val loss | PP | mask acc | ctx_cos |
|---|---|---|---|---|
| VanillaBERT | 6.766 | 867.9 | 0.066 | 0.880 |
| LoopedBERT | 6.769 | 870.3 | 0.068 | 0.866 |

Capability gate — CrowS-Pairs stereotype preference (n = 400 pairs, FP32 PLL):

- **VanillaBERT: 0.4525, 95% CI [0.4050, 0.5025] → CI includes chance → FAIL**
- Vanilla vs Looped: preference **identical to 4 d.p.** (0.4525 vs 0.4525);
  item-level Δ = −0.0051 [−0.0150, +0.0054], p = 0.8345, d = −0.048

This is **noise, not evidence against SCH**.

### 3.2 The instrument was fine — the models were not

Same scorer, same 400 pairs, properly-trained public models:

| Model | Geometry | Preference | 95% CI | Detectable |
|---|---|---|---|---|
| google/bert_uncased_L-2_H-128 | **4.4M** | 0.5825 | [0.5349, 0.6350] | **YES** |
| google/bert_uncased_L-4_H-256 | 11.3M (= our `tiny` width) | 0.5875 | [0.5375, 0.6400] | **YES** |
| google/bert_uncased_L-8_H-512 | 41.4M | 0.5500 | [0.5000, 0.6000] | borderline (n=400) |
| bert-base-uncased | 110M | 0.5975 | [0.5500, 0.6425] | **YES** |
| **ours (H=256, L=12, 100M tokens)** | 17.6M | **0.4525** | [0.4050, 0.5025] | **NO** |

**A 4.4M-parameter model — four times smaller than our `tiny` — shows clearly detectable
stereotype bias when adequately trained.**

### 3.3 Why this matters beyond this paper

Capacity is **not** the barrier to measuring stereotype association; **training adequacy is**.
Published claims of the form *"our small/efficient model shows no measurable bias"* may
therefore be measuring undertraining rather than fairness, unless the authors first
demonstrate that their baseline exhibits detectable bias. That is precisely what a
capability gate does, and it is rarely reported.

---

## 4. FINDING 2 — Calibration: what a from-scratch bias study actually costs

### 4.1 The registered learning rate destroys the model

`LEARNING_RATE = 5e-4` (registered) produced, after **47M tokens**:

- validation loss pinned at the unigram-entropy plateau (~7.2) and *rising*
- **cosine similarity between three unrelated contexts = 1.0000** — identical output
  regardless of input. A randomly-initialised model scores 0.9865, so training made it
  *worse* at using context.
- top predictions were pure function-word frequencies: `the . , of and to a in`

Ruled out by direct test rather than assumption: MLM masking correct (14.8% supervised /
11.6% `[MASK]` / 86.9% unchanged); FlashAttention numerically correct (matches eager to
7.9×10⁻³ at **real** token positions — an apparent 1.40 discrepancy lay entirely at
**padded** positions, which the loss ignores).

Cause: too-high LR for a post-norm BERT at an 8192-token effective batch (~80× more
aggressive per token than the original BERT recipe).

**Key methodological point: validation loss did not distinguish the configurations.** All
candidates sat at the unigram floor early on. **Context sensitivity (`ctx_cos`) did:**

| batch 64, 6M tokens | end loss | ctx_cos | reading |
|---|---|---|---|
| lr=5e-4 (registered) | 7.20 | **1.0000** | context ignored |
| **lr=1e-4** | 7.20 | **0.9103** | learning context |
| lr=3e-5 | 7.58 | 0.9953 | too slow |
| lr=5e-4, batch 256 | 7.20 | 1.0000 | batch is not the lever |
| lr=2e-4, batch 256 | 7.23 | 1.0000 | still too high |

Re-validated at the final batch (512), `base`, 250M tokens each on the real schedule shape:

| lr | final loss | ctx_cos | |
|---|---|---|---|
| 1e-4 | 6.286 | 0.7442 | |
| **3e-4** | 6.088 | **0.6969** | **CHOSEN** — best loss *and* best context use |
| 6e-4 | 6.052 | 0.7764 | 0.6% better loss, clearly worse context use |

An automated rule selecting on loss alone would have chosen 6e-4. It was rejected because
stereotype associations are inherently contextual, so context use is the property that must
be protected.

### 4.2 The registered token budget could not reach the registered bands

At 200M tokens the extrapolated final loss was **~6.4–6.5 (PP ~600)**, while the registered
iso-loss bands were `[4.0, 3.7, 3.4, 3.1]` and the quality screen was PP ≤ 60.

Consequence: **zero bands crossed → zero snapshots → zero bias evaluations.** The run would
have consumed ~28 GPU-hours and completed "successfully" with no data.

### 4.3 Loss curves have a breakthrough; early extrapolation is dangerous

Measured on the H100 at the 7B budget:

```
 876M → 5.7708  (pp 320)      pre-breakthrough plateau
1377M → 5.2008  (pp 181)
1502M → 2.9966  (pp  20.0)    breakthrough: 2.2 nats in ONE validation interval
2034M → 2.1831  (pp   8.9)
7000M → 1.4800  (pp   4.39)   final
```

Two extrapolations made *before* the breakthrough were both badly wrong (predicting ~5.2 and
~4.0 final loss). **Log-linear extrapolation from the plateau phase of an MLM run is
unreliable — the plateau is not the asymptote.** Practical consequence: iso-loss bands must
be set from a completed run, or spread wide enough to survive being wrong.

### 4.4 Compute requirements (measured, not estimated)

| Configuration | Throughput | MFU |
|---|---|---|
| L4, on-the-fly tokenisation, micro=16 | 42k tok/s | **~4%** |
| H100, micro=16, grad-ckpt ON | 102,860 tok/s | 5.3% |
| **H100, micro=512, grad-ckpt OFF** | **278,103 tok/s** | 14.3% (≈18% counting the MLM head) |
| H100 in production | ~298,000 tok/s | |

Two reportable findings:

1. **Pre-tokenising the corpus to a memory-mapped binary was worth ~8×.** The
   single-threaded tokenizer starved the GPU (4% MFU). The pre-tokenised path was verified
   to emit **byte-identical** sequences, so this is pure throughput with no scientific
   effect.
2. **Gradient checkpointing cost ~45% for nothing** — enabled by an `'auto'` rule for `base`
   while only 5.4 GB of 80 GB was in use.

Combined, the naive configuration would have needed ~171 h for work the tuned configuration
completes in ~26 h.

---

## 5. Empirical results so far

### 5.1 VanillaBERT (architecture 1 of 4) — COMPLETE

7.000B tokens, seed 42, lr 3e-4, batch 512.

| Checkpoint | Tokens | Val loss | PP |
|---|---|---|---|
| bands 5.0 / 4.0 / 3.4 / 3.0 † | 1.502B | 2.9966 | 20.02 |
| band 2.7 | 1.627B | 2.5867 | 13.29 |
| band 2.4 | 1.815B | 2.3530 | 10.52 |
| marker 2B | 2.000B | 2.2162 | 9.17 |
| band 2.2 | 2.034B | 2.1831 | 8.87 |
| marker 4B | 4.000B | 1.7199 | 5.58 |
| **marker 7B (final)** | **7.000B** | **1.4800** | **4.393** |

† These four bands were crossed inside a single validation interval and therefore reference
the **same** checkpoint. Recorded in `duplicate_checkpoints.json` on the HF repo.

Mask accuracy at the 4B checkpoint: **0.6419** (the pilot model that could measure nothing
scored 0.0657).

**Quality verdict: PP 4.39 is at or better than bert-base-uncased on this validation set.**
This is the single most important development — it moves the capability gate from "probably
fails" to "probably passes".

### 5.2 LoopedBERT (architecture 2 of 4) — COMPLETE

7.000B tokens. All **7 bands crossed**, identical to VanillaBERT — so the iso-loss
comparison rests on genuinely matched checkpoints rather than extrapolation.

| | Unique layers | Final loss | Final PP | Bands | Checkpoints |
|---|---|---|---|---|---|
| VanillaBERT | 12 | **1.4800** | **4.393** | 7 | 7 |
| LoopedBERT | 6 | **1.5158** | **4.553** | 7 | 8 |

LoopedBERT retains one extra *distinct* checkpoint (band 3.0) because its descent through
the breakthrough was slightly more gradual, so a validation interval landed between bands
3.4 and 3.0 where Vanilla's collapsed. The primary contrast uses the deepest **common**
band (2.2), which both possess.

### 5.2.1 FINDING 3 — Weight sharing costs ~0.036 nats for half the unique parameters

Measured at six matched budgets across the full run:

| Budget | VanillaBERT | LoopedBERT | Δ (nats) |
|---|---|---|---|
| 2.0B | 2.2162 | 2.1917 | −0.025 (Looped *ahead*) |
| 3.0B | 1.8753 | 1.8828 | +0.008 |
| 4.0B | 1.7199 | 1.7373 | +0.017 |
| 6.38B | 1.5100 | 1.5419 | +0.032 |
| **7.0B (final)** | **1.4800** | **1.5158** | **+0.036** |

The gap is near zero early and widens slowly as both converge, ending at **0.036 nats
(PP 4.393 → 4.553, ~3.6% relative)** for a **50% reduction in unique parameters** at
identical effective depth, identical data order, identical schedule and identical hardware.

This is a clean, controlled parameter-efficiency result and it is **independent of the bias
question**. It is consistent with ALBERT's original claim but measured here under far
tighter control (same seed, same data ordering, same step budget, iso-loss checkpoints).

**Caveat that cuts against SCH:** if weight sharing genuinely squeezed out memorised
content, a larger quality cost might be expected. A 3.6% relative penalty suggests the
shared model is *not* meaningfully capacity-starved at this budget — which weakens the
mechanism SCH proposes, and is a reason my estimate of a significant bias effect stays
modest (§6.3).

### 5.3 ALBERTLoopedBERT (architecture 3 of 4) — IN PROGRESS

Started 07:44 UTC / 13:14 IST. Early trajectory is visibly worse at matched steps, as
expected for a single shared layer applied 12 times:

| Step | Tokens | ALBERT loss | (Vanilla at same step) |
|---|---|---|---|
| 2000 | 125M | 6.9712 | 6.4649 |
| 4000 | 250M | 6.4925 | 6.0744 |
| 6000 | 375M | 6.1357 | 5.9649 |
| 8000 | 501M | 5.9969 | 5.8844 |

This is the arm most at risk of failing to reach the deepest common band, and the arm where
SCH predicts the **largest** effect. Both make it the most informative of the four.

### 5.3 Data integrity (verified computationally every hour)

- `Validation_Loss`: no NaN, all > 0
- `Pseudo_Perplexity` vs `exp(Validation_Loss)`: max relative error **6.0×10⁻¹⁶**
  (floating-point exact — confirms both metrics derive from the same tensor)
- Monotone descent: VanillaBERT 7 checkpoints **0 upticks**; LoopedBERT 6 checkpoints **0 upticks**
- Zero errors, zero OOM across the entire run

---

## 6. Honest assessment

### 6.1 Pros — what is genuinely strong

1. **The measurement-validity finding (§3) stands on its own.** It does not depend on SCH,
   it is quantitative, and it implies a concrete methodological correction for other work.
2. **The protocol demonstrably works.** Iso-loss matching plus a three-leg capability gate
   *refused to convert an undertrained model into a publishable bias claim*. Demonstrated
   guardrails are stronger evidence than proposed ones. The pilot cost ~1.5 GPU-hours and
   prevented a ~28 GPU-hour run that could not have produced interpretable output.
3. **Model quality is real.** PP 4.39 from scratch means bias is measured on genuine
   language models, not toys — the central weakness of the pilot.
4. **The artifact suite is reusable.** Four architectures spanning the weight-sharing
   spectrum, trained under *identical* conditions (same data order, seed, schedule,
   hardware), with iso-loss-matched checkpoints and full provenance, released publicly.
   Controlled suites like this are rare and support questions beyond this paper.
5. **Full reproducibility.** Every hyperparameter change is evidence-backed and documented in
   `PRE_REGISTRATION_AMENDMENT.md`; raw calibration logs are archived in
   `Codes/results/pilot/`.

### 6.2 Cons — what is genuinely weak

1. **Single seed.** Only seed 42. The item-level permutation test (n ≈ 1508 pairs) is the
   pre-registered primary test and does not require many seeds, but no seed-level variance
   estimate exists. Disclosed limitation; a reviewer will note it.
2. **The architectures may be too similar to separate.** §5.2 cuts both ways: if Vanilla and
   Looped are near-identical in quality, they may be near-identical in bias. SCH proposes
   that fewer unique parameters squeeze out memorised associations; if that effect were
   strong, some trace might be expected in the loss curve, and little is visible.
3. **One stage, not the full program.** Running all four architectures covers much of
   Stages 2–3, but the stream-count dose-response ablation (n = 1/2/4) and the seq=256
   adaptation tail were **dropped** for budget. Disclosed in the amendment.
4. **Scale generalisation is unproven.** These are 110M-parameter encoders on 7B tokens.
   Transfer to modern LM scale is untested — though §3 shows the phenomenon exists even at
   4.4M parameters, which partially addresses it.
5. **Construct scope is narrow.** The endpoint is a **correlation**-type measure in the
   taxonomy of Wang et al. (2025). It measures stereotype *association*, **not**
   difference-aware fairness. A lower preference rate is not automatically "more fair".
6. **Iso-loss bands are coarse in places.** Four bands were crossed within one validation
   interval and alias to a single checkpoint, so the effective number of distinct comparison
   points is smaller than the band list implies.

### 6.3 Predictions (recorded *before* the result, for honesty)

| Outcome | Estimate |
|---|---|
| Capability gate passes (baseline bias detectable) | **~65%** |
| Clearly significant Vanilla-vs-Looped effect | **~35–40%** |
| Most likely single outcome | **A well-controlled null** |

A null is worth reporting *because the models are good*: "we tested this properly at
bert-base quality and found no effect" is a real contribution; "we found no effect in a
broken model" is not. If any arm shows an effect, **ALBERTLoopedBERT** (1 shared layer × 12,
the extreme of the sharing spectrum) is where it should be largest.

---

## 7. Still unknown / pending

- [ ] LoopedBERT, ALBERTLoopedBERT, HyperloopBERT final quality
- [ ] **Capability gate verdict** — does VanillaBERT show detectable bias at PP 4.39?
- [ ] Primary contrast: Vanilla vs Looped at the deepest common iso-loss band
- [ ] Sharing dose-response: Vanilla → Looped → ALBERT
- [ ] Hyperloop + CWSA contribution, and the mechanistic suite (loop trajectory, CKA,
      stream disagreement, early-merge, hyper-connection/MHC stability)
- [ ] Qualitative MLM outputs across the spectrum at matched quality
- [ ] GLUE capability evidence
- [ ] Final GO / NO-GO verdict

---

## 8. Recommended framing for the paper

Do **not** frame this as *"does HyperloopBERT reduce bias"* — that framing only wins in the
~35% branch. Frame it as:

> *"What does it take to measure stereotype bias in from-scratch encoders at all — and does
> weight sharing matter once model quality is controlled?"*

This is true in **every** branch:

- effect found → headline result **plus** the method
- clean null → pre-registered null **plus** the method
- gate fails → the measurement-validity paper, with §3 as the centrepiece

---

## 9. The future of this research

Written while the experiment is still running, so it reflects what the data *so far*
actually supports rather than what would be convenient. Ordered by expected value per unit
of effort.

### 9.1 Immediate — cheap, high value, finishes the current story

| # | Work | Cost | Why |
|---|---|---|---|
| 1 | **Seeds 43, 44** for the two headline arms | ~13 GPU-h | Only real weakness of the current design. The item-level test (n≈1508) is primary, but zero seed-variance is the first thing a reviewer flags. |
| 2 | **Stream-count dose-response** (n = 1/2/4 Hyperloop) | ~20 GPU-h | Dropped for budget. It is the *causal* arm of the program — only stream count varies — and currently the weakest-supported claim about Hyperloop. |
| 3 | **Readout blind-spot control** | ~0 (eval-time) | *Dense Supervision Is Not Enough* (arXiv:2606.24898) directly attacks loop-wise trajectory claims. The eval-time early-merge intervention is already implemented; running it converts a vulnerability into evidence of rigour. |
| 4 | **INLP / projection debiasing on VanillaBERT** | ~1 GPU-h | Answers the most predictable reviewer question — *"why architecture-level mitigation when post-hoc debiasing exists?"* — with data instead of prose. |

### 9.2 The measurement-validity thread — the most promising standalone direction

§3 is the result that does not depend on SCH, and it is the seed of a larger contribution.

The natural next paper is a **minimum-viable-quality standard for intrinsic bias
benchmarking**: a calibration curve mapping model quality (validation loss / perplexity) to
the *statistical detectability* of stereotype bias, across model families, benchmarks
(CrowS-Pairs, StereoSet, WinoBias) and metrics (PLL, SS-PLL, embedding association).

The deliverable is concrete and reusable:

> *"Below quality X on benchmark Y with metric Z, a null result is uninformative; report the
> capability gate alongside the bias number."*

We already have four points on that curve (4.4M, 11.3M, 41.4M, 110M properly-trained, plus
our own undertrained models). Extending it is cheap because it requires **no pretraining** —
only scoring existing public checkpoints. This is the highest value-per-GPU-hour direction
available, and it would give the community a checklist item that currently does not exist.

### 9.3 Scientific extensions of SCH itself

1. **Where on the sharing spectrum does the effect appear?** The current design has three
   points (12 / 6 / 1 unique layers). A finer sweep (12/8/6/4/2/1) at fixed effective depth
   would turn a comparison into a **dose-response curve** — a far stronger causal claim.
2. **Sharing at pretraining vs compression after training.** This work varies sharing *at
   pretraining under iso-loss*; the compression–fairness literature
   (Hooker et al. 2020; Xu et al. 2022; Ramesh et al. 2023) removes parameters from an
   already-trained model. The `VanillaBERT6` parameter-matched control is the bridge, and a
   direct head-to-head (share-then-train vs train-then-prune, matched on final loss) is a
   clean, well-posed question nobody has answered under quality control.
3. **The difference-awareness failure mode.** SCH sharpened (per Wang et al. 2025 and the
   Ouro/Frey manipulation-vs-storage results) predicts that if sharing suppresses group
   associations *non-selectively*, it should also erode **legitimate descriptive** group
   knowledge. That is a falsifiable prediction and it needs instruction-tuned-scale models
   plus the difference-awareness benchmark. This is the most intellectually interesting
   open question the project has generated.
4. **Does it survive scale?** Everything here is 110M encoders at 7B tokens. Whether the
   mechanism holds for billion-parameter decoder LMs is untested. §3 partially reassures
   (the phenomenon exists at 4.4M), but extrapolation upward is not established.

### 9.4 Regional / multilingual extension (relevant to the FIRE venue)

The India-centric arm was **dropped**: the available mirror is Bengali, unusable by an
English-only WordPiece tokenizer (§2.2, bug 11). Restoring it requires either an
English-language Indian-BhED source or a multilingual tokenizer and corpus. Given FIRE's
regional focus this is the most venue-relevant extension, and IndiBias (NAACL 2024),
IndiCASA and IndRegBias are natural instruments. Note this changes the experiment
materially — a multilingual tokenizer alters the whole pretraining setup — so it is a
follow-up study, not a patch.

### 9.5 Engineering artifacts worth releasing separately

The infrastructure findings (§2, §4.4) have standalone value:

- the **capability gate** as a reusable component for bias studies
- the **iso-loss snapshotting** protocol
- the observation that a research pipeline can pass review and still be structurally
  incapable of producing data (16 bugs, 6 fatal)
- concrete throughput numbers: pre-tokenisation ~8×, gradient checkpointing costing 45%
  when memory is abundant, and 4% → 18% MFU

A short reproducibility/tooling note would be low-effort and genuinely useful to others
running from-scratch bias studies on a budget.

### 9.6 What I would do with the next 100 GPU-hours

In priority order, and this is a real recommendation rather than a wish list:

1. **Seeds 43/44** on the four architectures (~26 h) — removes the single-seed weakness
2. **Stream-count dose-response** (~20 h) — makes the Hyperloop claim causal
3. **Finer sharing sweep** 12/8/6/4/2/1 (~40 h) — converts the core result into a curve
4. **Extend the feasibility curve** (~2 h, no pretraining) — the cheapest real contribution
5. Hold ~12 h in reserve; this project has consistently needed it

---

*Sections 5, 6.3 and 7 will be revised when all four architectures and the full evaluation
suite complete.*
