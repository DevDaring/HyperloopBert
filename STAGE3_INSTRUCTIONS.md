# STAGE 3 — MECHANISM + NOVEL ARCHITECTURE — Full Instructions for Copilot

> **Role of this file.** Three jobs, in order:
> 1. **BUILD** the Stage 3 code as specified.
> 2. **DECIDE** go/publish from the result CSVs using §4.
> 3. **IF GO**, recommend where to submit the full paper under the strict journal constraints
>    in §5, and write `results/stage3/journal_targets_stage3.md`.
>
> **Run Stage 3 only if Stage 2 returned GO.** No source code here; generate it. Detect and
> report unknowns; never guess silently.
>
> **Hard global rules:** global Python env only (no venv); no emoji anywhere; no hardcoded
> secrets (load from `.env`); cite every implemented paper/dataset in a comment block.

---

## 0. SCIENTIFIC CONTEXT (why Stage 3 exists)

Stages 1-2 established (GO) that stereotype preference decreases with parameter-sharing degree
at matched validation loss, robustly (PLL+SS-PLL), with external-calibration validity. The
**finding** is now in hand. Stage 3 explains the mechanism and introduces the novel
architecture as the mechanistic vehicle — and, critically, any Stage 3 failure is now
unambiguously a Hyperloop/stream problem, not "maybe SCH is false," because SCH was already
screened in.

**Stage 3 research question:**
> Does training with more parallel residual streams causally reduce stereotype encoding at
> matched loss (from-scratch dose-response), and what is the mechanism by which streams
> modulate bias across loop depth?

Iso-loss (matched validation loss) remains the primary, non-negotiable basis.

**Framing (lead with the finding, not the architecture):** the contribution is the finding
plus the controlled mechanistic evidence; HyperloopBERT + CWSA is the means to discover and
explain it. Even a null Hyperloop result (`Looped ~= Hyperloop`) is publishable, because the
finding is already established (§4.2).

---

## 1. SHARED INFRASTRUCTURE (condensed; reuse Stages 1-2)

Reuse `common/` and `Dataset/`: global-env install (no venv) with the exact pinned recipe and
FlashAttention-2 wheel (`flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`,
Python 3.12, CUDA 12.x, torch 2.5, L4 sm_89; varlen padding-aware path; SDPA/eager fallback;
BF16 only; log active path), case-insensitive `.env` loader (never log values), the
multi-provider JSON-first `llm_utils` (Gemini primary `gemini-3-flash-preview` 4 keys;
DeepSeek `deepseek-chat` 2 keys; Mistral `mistral-small-latest` 2 keys; OpenRouter 2 keys;
round-robin within a provider; **no cross-tier fallback**; deterministic JSON parsing; **no
LLM judge** — metrics are model-intrinsic), integrity (dedup + corruption + manifest) on every
rerun, the shared 30,522-WordPiece tokenizer, and the shared FineWeb-Edu pool.

Stage 3 code lives in:
```
Stage3/  train_stage3.py train_stream_ablation_stage3.py eval_bias_stage3.py
         mechanistic_stage3.py analyze_stage3.py config_stage3.py
Dry_Run/ dry_run_stage3.py
```
No emoji; no hardcoded keys (including `Dry_Run/`); cite all papers in comments.

---

## 2. STAGE 3 BUILD SPECIFICATION (additions over Stage 2)

### 2.1 HyperloopBERT + CWSA (the novel architecture)
From `common/architectures.py`, compute-matched to Looped:
- begin(2) -> middle(2, looped x4) -> end(2); effective depth 12; unique layers 6;
  `num_streams = 4`.
- **The shared middle block is applied exactly once per loop on a stream-mixed input** (NOT
  once per stream). Per loop: **depth-connection** (read: mix the n streams into one block
  input via a learnable matrix) -> middle block -> **width-connection** (write: scatter block
  output back into all n streams). Streams are residual carriers mixed by learnable
  hyper-connection matrices at loop boundaries only, so Hyperloop is compute-matched to Looped.
- **CWSA (CLS-Weighted Stream Aggregation):** final stream combination is a learned
  soft-attention over each stream's `[CLS]` representation (encoder-specific contribution; not
  simple averaging). Cite Zeitoun et al. 2026 (arXiv:2604.21254); ours = first controlled
  encoder-only adaptation + CWSA.
- Store per-loop stream tensors in a `stream_snapshots` attribute for mechanistic analysis.
- Also implement **EarlyMergeHyperloopBERT** (Hyperloop with streams merged early at
  `merge_at in {1,2,3}`), labeled an OOD intervention, not causal proof.

### 2.2 Primary training set
Architectures: Vanilla, ALBERT, Looped, **Hyperloop**, at **Base-ish**, **3-5 seeds**, **500M
tokens** (400M @ seq=128 then 100M @ seq=256 for sequence-length adaptation), iso-loss
snapshots throughout. (Vanilla/ALBERT/Looped may be reused from Stage 2 if trained on the same
tokenizer/pool/budget; otherwise retrain to keep the budget identical — record which.)

### 2.3 Stream-count ablation (the primary CAUSAL test)
`train_stream_ablation_stage3.py`: train **from scratch** HyperloopBERT with `n in {1, 2, 4}`:
- `n=1` degenerates to LoopedBERT (sanity check: bias should approximate Looped),
- `n=2` intermediate diversity,
- `n=4` full Hyperloop.
3 seeds each (42,43,44), 300M tokens @ seq=128. Because only stream count varies (same
architecture, data, procedure), a monotone bias-vs-n relationship at matched loss is a clean
dose-response — the strongest causal evidence in the program. Confirmatory contrast: n=4 vs
n=1 (permutation test).

### 2.4 Full mechanistic suite (`mechanistic_stage3.py`)
1. **Loop-wise stereotype trajectory** (Looped vs Hyperloop): preference at each loop boundary
   via the MLM head on captured hidden states; classify CONVERGENT / AMPLIFYING / OSCILLATING;
   does Hyperloop show a qualitatively different trajectory?
2. **Loop-wise representation similarity (CKA)**: linear-kernel CKA between loop-boundary
   representations; does multi-stream diversity prevent premature representational convergence?
3. **Stream-disagreement <-> bias correlation** (Hyperloop `stream_snapshots`): per loop,
   mean pairwise `1 - cosine` across streams; Pearson + Spearman vs Effect_Size; **explicitly
   labeled correlational, not causal.**
4. **Early-merge OOD intervention** (`merge_at in {1,2,3}` on trained Hyperloop; compare to
   full 4-stream and Looped): corroborating evidence, not proof.
5. **Demographic token drift**: curated terms per category (caste/gender/religion/race);
   cosine drift of the contextual embedding from loop 0 to loop L; stereotypical vs
   anti-stereotypical context; Looped vs Hyperloop.

### 2.5 Full evaluation
Full iso-loss bias evaluation on all snapshots (the most expensive eval stage). Full GLUE
(SST-2, MRPC, QNLI, RTE) for the Pareto. StereoSet appendix only (never in main tables).

---

## 3. RESULT REGISTRATION (the CSVs the decision reads)

Full column names (in `common/io_schemas.py`); `Timestamp` on every row; identity columns as
before. Add for Stage 3:
- `results/stage3/ablations/stream_count_ablation.csv`: identity + `Stream_Count, Band,
  Overall_Stereotype_Preference_Rate, Mean_Effect_Size, Std_Across_Seeds, Bootstrap_CI_Low,
  Bootstrap_CI_High`.
- `results/stage3/mechanistic/bias_trajectory.csv`: identity + `Dataset, Category, Loop_Depth,
  Mean_Preference_Rate, Std_Preference_Rate, Mean_Effect_Size, Trajectory_Shape`.
- `results/stage3/mechanistic/representation_similarity.csv`: identity + `Loop_Pair, CKA`.
- `results/stage3/mechanistic/stream_disagreement.csv`: identity + `Loop_Depth,
  Stream_Disagreement, Effect_Size, Pearson_R, Pearson_P, Spearman_R, Spearman_P`.
- `results/stage3/mechanistic/early_merge_intervention.csv`: identity + `Merge_At,
  Overall_Stereotype_Preference_Rate, Mean_Effect_Size`.
- `results/stage3/mechanistic/demographic_token_drift.csv`: identity + `Category,
  Demographic_Term, Context_Type, Loop_Depth, Cosine_Drift`.
- `results/stage3/stats/confirmatory_family.csv`: `Contrast, Metric, Dataset, Band,
  Raw_P_Value, Holm_Corrected_P_Value, Cohens_D, Significant_At_0.05`; plus
  `exploratory_results.csv`.

---

## 4. GO / PUBLISH DECISION PROTOCOL

`analyze_stage3.py` prints one of: **GO-PUBLISH-FULL**, **GO-PUBLISH-NULL**, **INVESTIGATE**.
Primary instrument = Multi-CrowS-Pairs English, PLL, iso-loss matched, Base-ish.

### 4.1 Confirmatory family (Holm-Bonferroni on this set only)
At the primary band on Multi-CrowS-Pairs PLL: (1) Vanilla vs Hyperloop (pre-registered
primary), (2) Vanilla vs Looped, (3) Looped vs Hyperloop, (4) n=4 vs n=1 (ablation primary).
Everything else is exploratory (effect sizes + CIs, uncorrected, labeled).

### 4.2 Decision rule (defaults; adjust with written justification)
- **GO-PUBLISH-FULL** (the strong outcome) if:
  1. `Vanilla > Hyperloop` at matched loss, `Holm_Corrected_P_Value < 0.05` (primary), AND
  2. `Hyperloop <= Looped` (Hyperloop at least matches Looped; "<" is a bonus), AND
  3. the stream-count ablation shows a monotone dose-response at matched loss
     (`bias(n=4) <= bias(n=2) <= bias(n=1)`, with `n=1 ~= Looped` sanity check), ideally with
     n=4-vs-n=1 surviving correction.
- **GO-PUBLISH-NULL** (still a full paper) if the sharing effect (Vanilla > {Looped, Hyperloop})
  holds but Hyperloop only ties Looped (`Looped ~= Hyperloop`) and/or the stream dose-response
  is flat. The finding is already established (Stages 1-2); frame Hyperloop as the mechanistic
  vehicle and report the null on stream diversity honestly. This is publishable.
- **INVESTIGATE** (do not publish yet) only if the core sharing effect *vanishes at full
  scale* (Vanilla ~= Looped at Base-ish 500M), contradicting Stages 1-2. Find the confound
  (budget, band selection, eval contamination) before submission.

### 4.3 What to print
Verdict; confirmatory-family table (raw + Holm-corrected p, Cohen d); the iso-loss ordering
Vanilla/Looped/Hyperloop; the stream-count dose-response (n=1/2/4) with the n=1≈Looped sanity
check; trajectory-shape comparison; CKA convergence finding; stream-disagreement correlation
(labeled correlational); early-merge corroboration; token-drift summary; Pareto point; and a
one-paragraph interpretation that states which outcome (FULL vs NULL) was obtained.

---

## 5. IF GO — WHERE TO SUBMIT THE FULL PAPER (Copilot decides; strict constraints)

Run on GO-PUBLISH-FULL or GO-PUBLISH-NULL. Produce `results/stage3/journal_targets_stage3.md`.
**What Stage 3 publishes:** a full paper — the controlled finding (parameter sharing reduces
stereotype encoding at matched loss), scaled across model sizes (Stage 1) and across the
sharing spectrum (Stage 2), with a from-scratch stream-count causal dose-response and a
mechanistic account of how streams modulate bias across loop depth, plus the best bias-quality
Pareto point and an India-centric evaluation contribution. Treat as **Tier S** if the primary
and ablation both hold with significance and the mechanism is coherent; **Tier M** if it is the
GO-PUBLISH-NULL outcome (finding solid, Hyperloop ties Looped).

### 5.1 Hard constraints (exclude any violator entirely)
1. **SCIE / SCI indexed** (Science Citation Index Expanded). **ESCI rejected.** "Indexed in
   Web of Science" is NOT sufficient (WoS = SCIE + ESCI). Scopus-only / DOAJ-only insufficient.
2. **APC-free** for the author. Acceptable: subscription-only; hybrid via the free
   subscription path (confirm it is the default, not vestigial); diamond OA; **ACM journals**
   (IIIT Kalyani has **ONOS** -> APC-free ACM OA). Rejected: any gold OA with APC (even
   "modest"), mandatory-OA hybrids, submission fees, "waiver on request".
3. **Scope match** to the journal's stated aims-and-scope.

### 5.2 IMPORTANT correction about the original target list
The original program named **TACL (primary)** and Neurocomputing (fallback). **TACL is
open-access (MIT Press) and its SCIE status is doubtful** — under the author's hard
constraints, do **not** assume TACL qualifies. **Verify TACL's current indexing on the
Clarivate Master Journal List; if it is ESCI / not SCIE, exclude it and target a
SCIE-confirmed venue instead.** Likewise **TMLR is diamond OA (no APC) but is not SCIE
indexed** — exclude it under the hard constraint and list it only under "explicitly excluded".

### 5.3 Mandatory verification (web access; do not skip)
For each candidate: verify SCIE via Clarivate MJL (`https://mjl.clarivate.com/`), requiring the
explicit phrase **"Science Citation Index Expanded"** (queries
`"<journal>" "Science Citation Index Expanded"`, `"<journal>" site:mjl.clarivate.com`); reject
if only ESCI. Verify APC on the official OA/author page; for ACM confirm IIIT Kalyani on the
current ONOS list. Ambiguous -> "Needs manual verification" section, excluded from the ranking.

### 5.4 Candidate pool for the full paper (subject to §5.3 verification)
Foreground "controlled mechanistic study of parameter sharing and stereotype encoding,
with a novel encoder architecture and an India-centric evaluation."
- **Neurocomputing (Elsevier)** — SCIE, hybrid (subscription free); strong fit for a method +
  benchmark + ablation + mechanism paper; the realistic primary target.
- **Knowledge-Based Systems (Elsevier)** — SCIE, hybrid, broad; accepts architecture/fairness
  work with strong empirics.
- **Neural Networks (Elsevier)** — SCIE, hybrid; excellent fit for the architecture and
  representation-analysis (CKA, trajectories) content.
- **ACM TALLIP** — ACM-ONOS APC-free; strongest if the India-centric instrument is a headline
  contribution; verify SCIE.
- **ACM TIST** — ACM-ONOS; broader intelligent-systems scope; verify SCIE.
- **Information Sciences (Elsevier)** — SCIE, hybrid; broad, accepts strong ML method papers
  (Tier S only).
- **Pattern Recognition (Elsevier)** — SCIE, hybrid; representation/method fit (Tier S only).
- **TACL** — only if §5.2 verification confirms current SCIE status; otherwise excluded.

### 5.5 User-specific exclusions (apply automatically)
- **Expert Systems with Applications** — excluded for bias papers (two prior rejections on a
  related bias paper).
- **Information Processing & Management** — author has an active submission; check status
  first; avoid double-submission.
- **Frontiers / MDPI / PLOS / TMLR** — excluded (ESCI/non-SCIE or APC); list only under
  "explicitly excluded (and why)".

### 5.6 Output format
Same structure as Stage 1 §5.5 (`results/stage3/journal_targets_stage3.md`): paper tier +
justification tied to the FULL vs NULL outcome; one-sentence recommendation; ranked primary
recommendations each with [verified] SCIE status + date, APC model, scope fit, honest
acceptance estimate (Very Low..Very High + percent), why-it-fits, risks, optional reframing
(only if it lifts the estimate >= 1 tier with no new experiments); a low-IF SCIE fallback
(always present); short-format alternatives only where the journal offers them;
needs-manual-verification; explicitly-excluded (including TACL/TMLR if they fail); one-sentence
bottom-line action. Be honest; do not pad; never label a journal SCIE without the explicit
Clarivate confirmation.

### 5.7 Recommended pre-submission order (state this in the output)
After picking the venue, the author should run a venue-specific pre-mortem (why would *this*
journal reject), revise, then a supervisor-readiness check, before submitting. Reference this
as the next action; do not perform it here.

---

## 6. STAGE 3 DELIVERABLES CHECKLIST
- [ ] HyperloopBERT + CWSA + EarlyMergeHyperloopBERT implemented, compute-matched to Looped;
      `stream_snapshots` stored.
- [ ] Primary set Vanilla/ALBERT/Looped/Hyperloop at Base-ish, 3-5 seeds, 500M tokens
      (400M@128 + 100M@256), iso-loss snapshots.
- [ ] From-scratch stream-count ablation n in {1,2,4}, 3 seeds, 300M tokens; n=1≈Looped sanity
      check; n=4-vs-n=1 confirmatory contrast.
- [ ] Full mechanistic suite: trajectory, CKA, stream-disagreement correlation (labeled
      correlational), early-merge OOD, token drift.
- [ ] Full iso-loss bias eval on all snapshots; full GLUE Pareto; StereoSet appendix only.
- [ ] Confirmatory stats with Holm on the small family; exploratory labeled.
- [ ] Figures: iso-loss ordering; stream-count dose-response bar chart; trajectory panels;
      CKA; stream-disagreement scatter; Pareto (PNG+PDF, colorblind-safe, no emoji).
- [ ] `analyze_stage3.py` prints GO-PUBLISH-FULL / GO-PUBLISH-NULL / INVESTIGATE per §4.
- [ ] IF GO: `results/stage3/journal_targets_stage3.md` per §5 with verified SCIE + APC-free
      venues (TACL/TMLR handled per §5.2) and honest acceptance ranking.
- [ ] `stage3_paper_outline.md` finding-first; Stages 1-2 fold in as scaling + spectrum
      sections. No hardcoded secrets; no emoji.

---

## 7. CITATIONS TO EMBED IN CODE COMMENTS (Stage 3 additions)
```
# CITATION: Zeitoun, A., Torroba-Hennigen, L., & Kim, Y. (2026). Hyperloop Transformers.
#           arXiv:2604.21254. MIT.   [Hyperloop base; ours = first encoder-only adaptation + CWSA]
# CITATION: Saunshi, N. et al. (2025). On the Power of Looped Transformers. arXiv. [SCH basis]
# SUPPORT:  Zhu, R.-J. et al. (2025). Ouro. arXiv:2510.25741; Frey, M. et al. (2026). arXiv:2603.08391.
#           [SCH support: manipulation not storage. arXiv:2603.08391 is Frey 2026, not Zhu.]
# CITATION: Zhu, D. et al. (2025). Hyper-Connections. ICLR 2025; Xie, Z. et al. (2025). MHC. arXiv:2512.24880.
#           [hyper-connection basis + stability check reported in stream_analysis_stage3.py]
# CITATION: Nadeem, M. et al. (2021). StereoSet. ACL.  [appendix only]
# CKA reference: Kornblith, S. et al. (2019). Similarity of Neural Network Representations
#               Revisited (CKA). ICML.
# (Stage 1-2 citations for Devlin, Bae, Lan, Nangia, Khandelwal, Zhao, Blodgett still apply.)
# BENCHMARK VALIDITY: PLL/SS-PLL construct-validity caveats (Blodgett et al. 2021) addressed by
#   PLL+SS-PLL convergence, continuous effect sizes, multi-dataset convergence, Holm correction
#   on the confirmatory family only, explicit confirmatory/exploratory separation.
# ETHICAL USE: datasets contain stereotypical content by design; fairness-audit/research only.
```
