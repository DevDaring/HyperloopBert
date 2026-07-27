# HyperloopBERT — Improvement Suggestions (Round 2: Re-Audit + 2025–2026 Literature Gap Review)

> **Round 1** (previous version of this file) was a full code audit with TIER 1/2/3 items.
> **Round 2 (this version, 2026-07-21)** re-verified every Round-1 item against the current
> working tree and adds what Round 1 lacked: a systematic check of the 2025–2026 literature
> your paper must engage with to survive journal review.
>
> **Headline.** All TIER-1 bugs are fixed. The pipeline is now technically sound enough to
> spend GPU-hours. The biggest remaining risks are no longer in the code — they are
> **citation errors, missing 2025–2026 comparisons, and one methodological caveat**
> (intermediate-loop readouts) that a 2026 paper directly attacks. Fix Part B before
> writing the paper; fix the small Part A remainders before the final runs.

---

## PART A — Code status

### A.1 Round-1 items verified FIXED (do not re-open)

- 1.1 Permutation resampling unit → item-level `item_level_paired_contrast`
  (`common/stats_engine.py:99-128`) is now the primary test in all three analyzers;
  seed-level kept as robustness. `DEFAULT_SEEDS` extended to 5 (`Stage2/config_stage2.py:12`).
- 1.2 Iso-loss → adaptive 500-step validation cadence near band crossings
  (`common/train_loop.py:518-526`), per-contrast loss-gap audit with tolerance 0.05
  (`Stage1/analyze_stage1.py:146-159`), `iso_checkpoints/index.csv` written
  (`common/train_loop.py:311-343`).
- 1.3 Contamination → short-phrase substring matching (`Dataset/contamination_filter.py:28-38,133-137`),
  GLUE sentences screened (`:76-92`).
- 1.4 Stream-ablation identity → `Stream_Count`/`Merge_At` in MLM summary schema and all
  lookups (`common/io_schemas.py:46-58`, `common/train_loop.py:449-464`,
  `Stage1/eval_bias_stage1.py:87-104`, `Stage3/eval_bias_stage3.py:94-159`).
- 1.5 Silent row drops → counted (`Scored_Row_Count`/`Failed_Row_Count`/`Tied_Pair_Count`),
  `Needs_Review` actually set, GLUE load-failures skip instead of writing 0.0,
  PLL ties counted separately.
- 1.6 Decision logic → all three analyzers now implement the pre-registered rules
  (item-level primary p, NO-GO legs, `Hyperloop <= Looped` tolerance, monotone
  n1/n2/n4 check, n=1≈Looped sanity, signed correlation corroborating-only,
  paper outline gated on GO, Indian direction reported).
- 1.7 Ablation budget confound → all arms (n=1,2,4) now train at the same 400M budget.
- 1.8 Environment → flash `cu_seqlens` int32 + truthful provenance demotion
  (`common/attention.py:118-122,409-422`); roberta-base anchor replaces ModernBERT under
  the transformers 4.46 pin (documented); WinoBias `trust_remote_code=True`; LR schedule
  sized via `SEQ_FILL_RATIO`; corpus rerun guard; tokenizer in manifest; integrity check
  wired into all three train scripts; gradient checkpointing enabled for base size.
- 2.3 Parameter-matched Vanilla baseline → `VanillaBERT6` registered and analyzed as an
  explicitly exploratory contrast (`common/architectures.py:1372-1381`,
  `Stage2/analyze_stage2.py:308-324`). **This arm is your single most important
  reviewer-defense — keep it prominent in the paper.**
- 2.4 Parameter claims → capacity gap disclosed in module docstring; n=1≈Looped caveat
  documented; `Total_Parameters` documented as disk footprint.
- 2.5 (partial) Canonical Nangia shared-token rate added for external calibration;
  WinoBias renamed to masked-pronoun stereotype-consistency with the correct construct note.
- 2.6 (mostly) CKA, token drift, eval-time early-merge at merge_at∈{1,2,3}, per-loop
  stream disagreement, GLUE<55 screen, Pareto fix, Stage-3 trajectory flag,
  `confirmatory_family.csv`/`exploratory_results.csv` naming, CLS/SEP ids, `cohens_d`
  signed-∞, `exploratory_contrast` alternative param — all done.

### A.2 Remaining open items (fix before final runs — all small)

1. **PLL still scored in bf16** (`common/bias_metrics.py:72,148,272`). Log-softmax is
   upcast after the forward, but the forward itself runs under bf16 autocast; rounding is
   the same order as small PLL gaps on borderline pairs. Run the PLL/SS-PLL/WinoBias
   forwards in fp32 (disable autocast inside the scorer). This is the last quantitative
   fix outstanding.
2. **Capability gate legs 1 & 3 missing** (`Stage1/analyze_stage1.py:187-210` implements
   only the baseline-bias leg). Add the above-chance GLUE screen (pull SST-2/RTE forward
   into Stage 1) and an above-random coreference control before trusting WinoBias in
   Stage 2. If the model is too undertrained to show these, your bias numbers are noise,
   and a reviewer who notices will reject on that alone.
3. **No corpus-level stereotype statistics.** FineWeb-Edu is heavily filtered; if the
   stereotypes your benchmarks test barely occur in the corpus, a null result means
   "never learned", not "architecturally mitigated". Write one cheap script counting
   benchmark-pair lexeme co-occurrence in the training corpus and report it as a sanity
   table. This is also your defense if baseline bias comes out near chance.
4. **Indian dataset deliverables incomplete** — spec wants 4 per-category CSVs +
   `provenance/provenance_report.json`; code still writes one merged CSV and a global
   markdown report (`Dataset/download_eval_datasets.py:113`,
   `Dataset/build_provenance_report.py`).
5. **Stage 3 seq=256/100M tail absent** (`Stage3/config_stage3.py:26` — seq=128 only).
   Either add it or amend the spec and say so in the paper.
6. Minor: dead `use_bf16` flag (`common/attention.py:203`); external-calibration failure
   logs ERROR instead of hard-failing; failed eval rows frozen forever by resume
   (`start_idx = max(Row_Index)+1` — now counted/flagged but never retried); stale seed
   comment in `Stage1/config_stage1.py:7`; Stage 2's calibration gate deliberately
   deviates from spec (above-chance canonical rate vs reproducing anchor ordering,
   `Stage2/analyze_stage2.py:92-119`) — **write this deviation into a pre-registration
   amendment**, reviewers respect amendments, they do not respect silent deviations.
7. Verify on the GPU box (code-verified only): flash-attn actually engages with the
   int32 fix (5-line probe), and the integrity check passes end-to-end once.

---

## PART B — 2025–2026 literature you must engage (this is what decides acceptance)

Your docs cite only Devlin 2019, Lan 2020, Saunshi 2025, Zeitoun 2026, and the benchmark
papers. The looped/recurrent-depth field moved fast in 2025–2026, and any reviewer at a
good journal will know this literature. Every item below was verified to exist.

### B.0 Citation errors in your current docs — fix first (embarrassing if a reviewer catches them)

- **"Zhu et al. (2025), arXiv:2603.08391" is a misattribution.** That arXiv ID is
  **Frey et al. (2026), "Adaptive Loops and Memory in Transformers: Think Harder or
  Know More?"** (Lamarr Institute/Fraunhofer IAIS; ICLR 2026 Latent & Implicit Thinking
  Workshop). The real Zhu et al. looped-LM paper is **Ouro: "Scaling Latent Reasoning
  via Looped Language Models", arXiv:2510.25741** (Zhu et al., Oct 2025). Your docs
  currently credit Ouro's idea to the wrong paper.
- **Both papers are arguably SUPPORT for SCH, not counter-evidence.** Your docs label
  the Zhu citation "counter-evidence". Read carefully: Ouro shows looped models' gains
  come from *knowledge manipulation*, not increased knowledge storage; Frey et al. show
  looping improves math (manipulation) but not commonsense (storage), with *similar
  per-parameter memorization* to standard transformers. That is exactly the
  manipulation-vs-storage dissociation your Stereotype Consolidation Hypothesis predicts
  should leave less room for shallow stereotype associations. Reframe: these are your
  strongest mechanistic allies — but they also force you to sharpen SCH, because
  per-parameter memorization is *preserved* under looping; your claim must be about
  *total* capacity at matched quality, not per-parameter memorization.
- **"Bae et al. (2025)" is untraceable as written.** The intended reference is almost
  certainly **Bae et al. (2025), "Mixture-of-Recursions: Learning Dynamic Recursive
  Depths for Adaptive Token-Level Computation", arXiv:2507.10524** (or Bae et al. 2024,
  "Relaxed Recursive Transformers", arXiv:2410.20672). Cite the exact one you mean.

### B.1 Missing must-cite recurrent-depth work (2025–2026)

Not citing Huginn and Ouro in a 2026 weight-sharing paper is a desk-reject-level omission
at architecture-aware venues.

- **Geiping et al. (2025), "Scaling up Test-Time Compute with Latent Reasoning: A
  Recurrent Depth Approach" (Huginn-3.5B), arXiv:2502.05171** — the canonical LM-scale
  recurrent-depth pretraining work (3.5B params, 800B tokens).
- **Zhu et al. (2025), Ouro, arXiv:2510.25741** — looped pretraining at 7.7T tokens;
  2.6B matches ~8–12B standard models. Use its knowledge-manipulation finding to frame SCH.
- **"How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language
  Models" (2026), arXiv:2604.21106** — iso-depth scaling laws; directly relevant to your
  iso-loss/iso-depth compute-matching design. Cite it to justify your matching protocol
  and, if possible, relate your matched-loss comparisons to its recurrence-value curves.
- **Kohli et al. (2026), "Loop, Think, & Generalize", arXiv:2604.07822** — implicit
  compositional generalization in recurrent-depth transformers.
- **Knupp et al. (2026), "Depth-Recurrent Attention Mixtures", arXiv:2601.21582** —
  attention-specific recurrence; adjacent to your encoder loop design.
- **Jeddi et al. (2026), "LoopFormer"** and **Prairie et al. (2026), "Parcae"** and
  **Goyal et al. (2026), "ELT"** — elastic/adaptive looped training and stable looped
  scaling; related-work completeness.
- **Bae et al. (2025), Mixture-of-Recursions, arXiv:2507.10524** — token-level adaptive
  recursion; position your fixed-loop design against it.

### B.2 The multi-stream/hyper-connection neighborhood (positions HyperloopBERT itself)

- **Chen et al. (2025), "ParScale: Parallel Scaling Law for Language Models",
  arXiv:2505.10475 (NeurIPS 2025)** — P parallel computation streams with learnable
  aggregation, gains ~O(log P). Your 4-stream Hyperloop with CWSA is conceptually its
  looped-encoder cousin. You MUST differentiate: ParScale scales *quality* via parallel
  streams; you study what parallel streams do to *bias*. Also cite
  **Wu et al. (2025), "Parallel Loop Transformer", arXiv:2510.24824**, which already
  combines looping + parallelism — your closest architectural neighbor.
- **Zhu et al. (2025), "Hyper-Connections" (ICLR 2025)** — the base technique behind
  your hyper-connection matrices; and **Xie et al. (2025), "MHC: Manifold-Constrained
  Hyper-Connections", arXiv:2512.24880**, which shows unconstrained hyper-connections
  can destabilize at scale and fixes it with manifold constraints. **Actionable:** check
  the learned hyper-connection matrices in your trained Hyperloop models for the
  instability MHC describes (report matrix statistics in the mechanistic section — cheap,
  no retraining) and cite MHC either as validation (your scale is stable) or as a fix.
- Optional: **ParaThinker, arXiv:2509.04475** (native parallel thinking) for the
  parallel-streams framing.

### B.3 The one methodological attack you must pre-empt — intermediate-loop readouts

**"Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models"
(2026), arXiv:2606.24898** shows that in looped models trained with loss only at the
final loop, intermediate-loop outputs are not meaningful under the shared readout head
— the readout is a blind spot. **This directly targets your loop-wise stereotype
trajectories (CONVERGENT/AMPLIFYING/OSCILLATING) and per-loop stream disagreement**:
your MLM head is trained on final-depth outputs, so PLL probed at loop k<12 may reflect
readout misalignment, not stereotype dynamics. Pre-empt it:

- Cite it explicitly and add a validity control: show trajectory conclusions are
  consistent with the eval-time early-merge intervention (merge_at∈{1,2,3} — which you
  already have) and/or train a tiny per-loop readout probe on frozen features and show
  the trajectory shape survives. Without this, a reviewer armed with 2606.24898 can
  dismiss your entire mechanistic section.

### B.4 The closest prior art reviewers WILL raise: compression–fairness

Your finding ("fewer unique parameters → less stereotype at matched quality") sits
inside a known literature on compression and fairness that your docs never mention:

- **Hooker et al. (2020), "Characterising Bias in Compressed Models"** — pruning
  disproportionately harms underrepresented subgroups (the pessimistic direction).
- **Xu et al. (2022), "Can Model Compression Improve NLP Fairness?", arXiv:2201.08542** —
  compression can *reduce* bias via regularization (your direction).
- **Gonçalves & Strubell (2023), "Understanding the Effect of Model Compression on
  Social Bias in LLMs"** — quantization regularizes bias in BERT-family models.
- **Ramesh, Sitaram & Choudhary (2023), "A Comparative Study on the Impact of Model
  Compression Techniques on Fairness in Language Models" (Microsoft, ACL 2023)** —
  distillation tends to amplify bias.
- **Mohammadshahi & Ioannou (2025)** — distillation temperature vs class-wise fairness.

Position precisely: compression removes parameters from a *trained* model; you vary
weight sharing *at pretraining* under iso-loss. Your `VanillaBERT6` parameter-matched
arm is the empirical separator — frame it as answering exactly this literature.

### B.5 Missing debiasing-method comparison (the most predictable reviewer question)

"Why architecture-level mitigation when post-hoc debiasing exists?" Your repo contains
zero references to CDA, INLP, Sent-Debias, Mabel, or FairFil. Minimum viable defense:

- Related-work paragraph covering CDA (counterfactual data augmentation), INLP
  (Ravfogel et al. 2020), Sent-Debias (Liang et al. 2020), Mabel (He et al. 2022),
  FairFil (2024), with the standard critique: post-hoc methods trade utility, are
  unstable across metrics, and can leave residual bias detectable by other probes.
- **Cheap optional experiment** (no new pretraining): apply projection-based debiasing
  (INLP-style) to your trained VanillaBERT embeddings and show (a) it degrades MLM/GLUE
  or leaves PLL bias on held-out probes, while (b) Hyperloop achieves the reduction
  *intrinsically*. One afternoon of compute, kills the objection.

### B.6 Benchmark/corpus updates (2024–2026)

- **IndiBias, arXiv:2403.20147 (NAACL 2024)** — Indian-context CrowS-Pairs-style pairs
  (Hindi + English). Your protocol already handles this format; adding the English
  IndiBias subset as a secondary confirmation instrument is cheap and strengthens the
  Indian-context claim alongside Indian-BhED.
- **IndiCASA, arXiv:2510.02742 (Oct 2025)** — Indian-context contrastive embedding
  similarity framework (Caste/Religion/Disability/Gender/Socioeconomic). Cite; optional
  as an embedding-level complement to your PLL metrics.
- **IndRegBias, arXiv:2601.06477** — Indian regional bias, English + code-mixed.
  Cite as scope boundary (your tokenizer is English-only).
- **IndoBias, arXiv:2606.01260 (2026)** — culturally-grounded Indonesian benchmark;
  cite in related work to show awareness of the post-CrowS-Pairs benchmark wave.
- Keep citing Blodgett et al. (2021) for PLL construct validity (you do); consider also
  acknowledging benchmark-reliability critiques generally — your SS-PLL agreement check
  is already the right defense, say so explicitly in the paper.

---

## PART C — Making the method "excellent": strategy for a good journal

1. **Reframe the contribution as the finding + the protocol, never the architecture.**
   Your own docs already say this; make the abstract say it. "First controlled iso-loss
   study of weight sharing vs stereotype consolidation, with a mechanistic multi-stream
   vehicle" survives a negative result; "we built HyperloopBERT" does not.
2. **Sharpen SCH against Ouro/Frey (B.0):** predict that looping reduces *total*
   stereotype storage at matched quality while preserving per-parameter memorization —
   and add one sentence predicting the failure mode (high-frequency corpus stereotypes
   will still leak in). A hypothesis with a stated failure mode reads as science, not
   advocacy.
3. **Lead the mechanistic section with the readout-blind-spot control (B.3)** — it
   converts your biggest weakness into evidence you read the 2026 literature.
4. **Artifact strategy for journal review:** release the trained checkpoints, the
   iso_checkpoints index, per-item PLL CSVs, and the analysis notebooks on Hugging Face
   / GitHub with the pre-registration (and its amendment, A.2.6) in the repo. Journals
   like TACL/JAIR weight reproducibility heavily; this is nearly free since your
   schemas already log everything.
5. **Venue fit:** TACL or JAIR for the full staged study; if the result is null, the
   iso-loss protocol + capability gate still makes a solid negative-result paper —
   pre-registered nulls are publishable precisely because your gates now work.
6. **Limitations section must already contain:** the ~18.9M capacity gap of Hyperloop,
   full-sentence PLL vs canonical Nangia (both reported), English-only tokenizer scope,
   FineWeb-Edu corpus filtering (with your A.2.3 statistics table), the seq=256 tail
   omission or amendment, and the roberta-base anchor substitution.

---

## Revised action order

1. Fix A.2.1 (fp32 PLL scoring) — last quantitative code fix; then freeze scoring code.
2. Add A.2.2 (GLUE + coreference capability legs) and A.2.3 (corpus stereotype stats) —
   these gate whether Stage 1 results mean anything.
3. Verify flash + integrity on the GPU box (A.2.7); fix A.2.4–A.2.6 deliverables and
   write the pre-registration amendment.
4. Run seed 42 end-to-end exactly as designed; then extend seeds.
5. While training runs: fix the citations (B.0), draft the related-work sections around
   B.1/B.2/B.4/B.5, and design the readout-blind-spot control (B.3) and optional INLP
   arm (B.5) so they can run on the trained checkpoints without new pretraining.
6. Before submission: run the B.2 hyper-connection stability check and the B.5 INLP
   probe on final checkpoints; write limitations per C.6.
