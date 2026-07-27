# STAGE 2 — IS IT SHARING, AND IS IT ROBUST? — Full Instructions for Copilot

> **Role of this file.** Three jobs, in order:
> 1. **BUILD** the Stage 2 code as specified.
> 2. **DECIDE** go/no-go from the result CSVs using §4.
> 3. **IF GO**, recommend where to submit under the strict journal constraints in §5, and
>    write `results/stage2/journal_targets_stage2.md`.
>
> **Run Stage 2 only if Stage 1 returned GO.** No source code here; generate it. Detect and
> report unknowns; never guess silently.
>
> **Hard global rules:** global Python env only (no venv); no emoji anywhere; no hardcoded
> secrets (load from `.env`); cite every implemented paper/dataset in a comment block.

---

## 0. SCIENTIFIC CONTEXT (why Stage 2 exists)

Stage 1 established (GO) that Looped < Vanilla in stereotype preference at matched loss across
scale. Stage 2 retires the next risk: **is it parameter sharing specifically, and does the
effect survive a real validity check and real statistics?** A two-point comparison can be a
fluke; a single metric can be a construct-validity artifact.

**Stage 2 research question:**
> Is reduced stereotype preference monotone in the *degree* of parameter sharing at matched
> validation loss, and is it robust across a second validity metric, a structurally different
> bias instrument, external calibration, and multiplicity-corrected statistics?

The iso-loss (matched validation loss) comparison remains the primary, non-negotiable basis.

---

## 1. SHARED INFRASTRUCTURE (condensed; reuse Stage 1's build)

Reuse everything from `common/` and `Dataset/` already built for Stage 1: global-env install
(no venv), the exact pinned recipe with FlashAttention-2 wheel
(`flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`, Python 3.12, CUDA
12.x, torch 2.5, L4 sm_89), case-insensitive `.env` loader (never log key values), the
multi-provider JSON-first `llm_utils` (Gemini primary `gemini-3-flash-preview` 4 keys;
DeepSeek `deepseek-chat` 2 keys; Mistral `mistral-small-latest` 2 keys; OpenRouter 2 keys;
round-robin within a provider; **no cross-tier fallback**; deterministic JSON parsing; **no
LLM judge** — bias metrics are model-intrinsic), integrity (dedup + corruption + manifest)
on every rerun, the shared 30,522-WordPiece tokenizer, and the shared FineWeb-Edu pool.

Stage 2 code lives in:
```
Stage2/  train_stage2.py eval_bias_stage2.py eval_glue_stage2.py
         external_calibration_stage2.py loop_trajectory_stage2.py analyze_stage2.py config_stage2.py
Dry_Run/ dry_run_stage2.py
```
No emoji; no hardcoded keys (including in `Dry_Run/`); cite all papers in comments.

---

## 2. STAGE 2 BUILD SPECIFICATION (additions over Stage 1)

### 2.1 Add a third architecture — the sharing spectrum
- **ALBERTLoopedBERT** (from `common/architectures.py`): ONE shared layer applied 12 times
  (unique layers = 1), loop-index embeddings before each iteration, **no embedding
  factorization** (isolates weight-sharing from embedding compression). Cite Lan et al. 2020.

Now the sharing spectrum is: Vanilla (12 unique) -> Looped (6 unique) -> ALBERT (1 unique),
all compute-matched at effective depth 12. The Stage 2 question becomes whether bias is
**monotone in sharing degree** at matched loss.

### 2.2 Data budget, sizes, seeds
- Token budget: up to **300-500M** (a longer prefix of the shared pool). Single-phase seq=128
  is acceptable; optionally add a short seq=256 tail if convergence needs it (record which).
- Sizes: **Base-ish** as primary, plus **one smaller size** (Small) as a scale check that the
  spectrum holds across scale.
- Seeds: **3-5** (configurable `--seeds`). Iso-loss bands re-calibrated per §iso_loss for the
  new convergence range; only bands crossed by all three architectures are usable.

### 2.3 Add SS-PLL (second validity metric)
In `common/bias_metrics.py`, add **SS-PLL (Shared-Token PLL)**: score only the unmodified
shared tokens between the stereo/anti sentences (addresses Blodgett et al. 2021 surface-form
critique). Report a contrast as **robust** only when PLL and SS-PLL agree in direction. Add
`SS_PLL_Stereotypical, SS_PLL_AntiStereotypical, PLL_SS_PLL_Agreement` to the bias CSVs.

### 2.4 Add WinoBias (structurally different instrument)
Download type-1 pro/anti dev+test (from `wino_bias` or the uclanlp/corefBias repo) to
`datasets_eval/winobias/`. Score with the model: report `Pro_Stereotype_Accuracy,
Anti_Stereotype_Accuracy, Pro_Anti_Gap`. This is convergent validity across instrument *types*
(pairwise PLL vs coreference). Cite Zhao et al. 2018 (WinoBias).

### 2.5 Add external calibration (no training)
`external_calibration_stage2.py`: run the identical bias pipeline on `bert-base-uncased`,
`albert-base-v2`, `answerdotai/ModernBERT-base`. Flag `External_Calibration=true`. Use to
(a) show the pipeline reproduces known published bias scores (validity), and (b) show
scratch-ALBERT reproduces albert-base-v2's bias *pattern* across categories (anchor: the
scratch model is valid even at lower absolute quality). Save
`results/stage2/bias/external_calibration.csv`.

### 2.6 Add a GLUE quality screen + Pareto
`eval_glue_stage2.py`: fine-tune each primary model on **SST-2 + RTE** (sufficient), head =
`Linear(hidden, num_labels)` on pooler output, lr 2e-5, 3 epochs, batch 32 (16 on OOM). Use
for a bias-vs-utility Pareto and as a quality screen: exclude configs with GLUE average < 55%
(chance ~50% on binary tasks) from the primary iso-loss analysis, logged.

### 2.7 Add the loop-trajectory teaser (NO new training)
`loop_trajectory_stage2.py`: the only mechanistic analysis that needs no new training — it
reuses Stage 1/2 checkpoints via forward hooks. For Looped (and Vanilla layers 3/6/9/12 for
reference), track stereotype preference at each loop boundary by applying the MLM head to the
captured intermediate hidden state. Classify trajectory shape: CONVERGENT / AMPLIFYING /
OSCILLATING. Defer CKA, stream-disagreement, early-merge, and token-drift to Stage 3.

---

## 3. RESULT REGISTRATION (the CSVs the decision reads)

Full column names (in `common/io_schemas.py`); `Timestamp` on every row; identity columns as
in Stage 1. Add for Stage 2:
- `results/stage2/bias/<dataset>_summary.csv`: identity + `Band, Token_Marker,
  Overall_Stereotype_Preference_Rate, Macro_Average_Preference_Rate,
  <Category>_Preference_Rate, Mean_Effect_Size, Bootstrap_CI_Low, Bootstrap_CI_High,
  PLL_SS_PLL_Agreement`.
- `results/stage2/bias/winobias_summary.csv`: identity + `Pro_Stereotype_Accuracy,
  Anti_Stereotype_Accuracy, Pro_Anti_Gap`.
- `results/stage2/bias/external_calibration.csv`: `Model_Name` + dataset/category columns +
  `External_Calibration=true`.
- `results/stage2/glue/summary_table.csv`: identity + `Task, Accuracy, F1, GLUE_Average`.
- `results/stage2/mechanistic/loop_trajectory.csv`: identity + `Dataset, Category, Loop_Depth,
  Mean_Preference_Rate, Std_Preference_Rate, Mean_Effect_Size, Trajectory_Shape`.
- `results/stage2/stats/confirmatory_family.csv`: `Contrast, Metric, Dataset, Band,
  Raw_P_Value, Holm_Corrected_P_Value, Cohens_D, Significant_At_0.05`; plus
  `exploratory_results.csv` (uncorrected, labeled hypothesis-generating).

---

## 4. GO / NO-GO DECISION PROTOCOL

`analyze_stage2.py` prints one of: **GO**, **PAUSE-FIX-MEASUREMENT**, **NO-GO**.
Primary instrument = Multi-CrowS-Pairs English, PLL, iso-loss matched, Base-ish size.

### 4.1 Confirmatory family (Holm-Bonferroni applied to THIS small set only)
At the primary band on Multi-CrowS-Pairs PLL: (1) Vanilla vs Looped (primary), (2) Vanilla vs
ALBERT, (3) Looped vs ALBERT. Everything else (per-category, Indian, WinoBias, SS-PLL
convergence, Small-size, Pareto) is **exploratory** (effect sizes + bootstrap CIs,
uncorrected, labeled).

### 4.2 Decision rule (defaults; adjust with written justification)
- **GO** if **all** hold:
  1. Sharing spectrum direction holds: `Pref(Vanilla) > Pref(Looped)` AND
     `Pref(Vanilla) > Pref(ALBERT)` at the primary band (monotone
     `Vanilla > Looped > ALBERT` is the strongest form but not required).
  2. The primary contrast (Vanilla vs Looped) survives Holm correction:
     `Holm_Corrected_P_Value < 0.05`.
  3. PLL and SS-PLL **agree in direction** on the primary contrast
     (`PLL_SS_PLL_Agreement = true`).
  4. External calibration is sane: the pipeline reproduces the expected ordering on
     `bert-base-uncased` / `albert-base-v2` (no validity failure).
- **PAUSE-FIX-MEASUREMENT** if the effect appears in only one metric (PLL but not SS-PLL),
  or collapses under Holm correction, or external calibration fails to reproduce known
  patterns. The effect may be real but fragile; fix measurement before Stage 3.
- **NO-GO** if the sharing direction reverses at Base-ish under correction (contradicts Stage
  1). Investigate the contradiction before any further spend.

### 4.3 What to print
Verdict; the confirmatory-family table (raw and Holm-corrected p, Cohen d); the
monotonicity check across the spectrum; PLL/SS-PLL agreement; WinoBias direction; external-
calibration validity check; the loop-trajectory shape per architecture; and a one-paragraph
interpretation.

---

## 5. IF GO — WHERE TO SUBMIT (Copilot decides; strict constraints)

Run only if verdict is GO. Produce `results/stage2/journal_targets_stage2.md`. **What Stage 2
publishes:** a stronger short paper or journal letter — a sharing-degree dose-response of
stereotype preference at matched loss, with multi-metric validity (PLL+SS-PLL), a second
instrument (WinoBias), external calibration, a quality-utility Pareto, and a first mechanistic
signal (loop trajectory). Treat as **Tier M** (toward S if the spectrum is cleanly monotone
with strong significance and the external calibration is convincing).

### 5.1 Hard constraints (exclude any violator entirely)
1. **SCIE / SCI indexed** (Science Citation Index Expanded). **ESCI rejected.** "Web of
   Science" alone is insufficient (WoS = SCIE + ESCI). Scopus-only / DOAJ-only insufficient.
2. **APC-free** for the author. Acceptable: subscription-only; hybrid via free subscription
   path (confirm it is the default); diamond OA; **ACM journals** (IIIT Kalyani has **ONOS**,
   giving APC-free ACM OA). Rejected: any gold OA with APC, mandatory-OA hybrids, submission
   fees, "waiver on request".
3. **Scope match** to the journal's stated aims-and-scope.

### 5.2 Mandatory verification (web access; do not skip)
For each candidate: verify SCIE via Clarivate MJL (`https://mjl.clarivate.com/`), requiring
the explicit phrase **"Science Citation Index Expanded"** (queries
`"<journal>" "Science Citation Index Expanded"`, `"<journal>" site:mjl.clarivate.com`); reject
if only ESCI. Verify APC on the official OA/author page; for ACM confirm IIIT Kalyani on the
current ONOS list. Ambiguous -> "Needs manual verification" section, excluded from the ranking.

### 5.3 Candidate pool for a Stage-2 paper (subject to §5.2 verification)
A fuller contribution than Stage 1, so mid-tier full journals are now in range alongside
short formats. Foreground "controlled study of architectural parameter sharing and bias" plus
the India-centric instrument.
- **Knowledge-Based Systems (Elsevier)** — SCIE, hybrid (subscription free), broad; accepts
  fairness/bias and architecture-analysis work; good Tier-M fit.
- **Neurocomputing (Elsevier)** — SCIE, hybrid; strong for method + benchmark + ablation; has
  a Short Communication track if you keep it short.
- **Natural Language Engineering (Cambridge)** — SCIE, hybrid; NLP analysis fit.
- **ACM TALLIP** — ACM-ONOS APC-free; strongest if the India-centric instrument is central;
  verify SCIE.
- **ACM Transactions on Intelligent Systems and Technology (TIST)** — ACM-ONOS; broader scope;
  verify SCIE.
- **Neural Networks (Elsevier)** — SCIE, hybrid; fits architecture/representation analysis.

### 5.4 User-specific exclusions (apply automatically)
- **Expert Systems with Applications** — excluded for bias papers (two prior rejections on a
  related bias paper).
- **Information Processing & Management** — author has an active submission; check status
  before recommending; avoid double-submission.
- **Frontiers / MDPI / PLOS / TMLR** — excluded (ESCI/non-SCIE or APC); list only under
  "explicitly excluded (and why)".

### 5.5 Output format
Same structure as Stage 1 §5.5 (`results/stage2/journal_targets_stage2.md`): paper tier +
justification; one-sentence recommendation; ranked primary recommendations each with
[verified] SCIE status + date, APC model, scope fit, honest acceptance estimate
(Very Low..Very High + percent), why-it-fits, risks, optional reframing (only if it lifts the
estimate >= 1 tier with no new experiments); a low-IF SCIE fallback (always present);
short-format alternatives only where the journal truly offers them; needs-manual-verification;
explicitly-excluded; one-sentence bottom-line action. Be honest; do not pad; never label a
journal SCIE without the explicit Clarivate confirmation.

---

## 6. STAGE 2 DELIVERABLES CHECKLIST
- [ ] ALBERT added; sharing spectrum Vanilla/Looped/ALBERT at Base-ish (+ Small scale check),
      3-5 seeds, iso-loss snapshots.
- [ ] SS-PLL added; WinoBias added; external calibration on 3 public models; GLUE (SST-2+RTE)
      screen + Pareto; loop-trajectory teaser (no new training).
- [ ] Confirmatory stats with Holm-Bonferroni on the small family only; everything else
      labeled exploratory with CIs.
- [ ] Figures: sharing-spectrum iso-loss plot; bias-vs-GLUE Pareto; loop-trajectory; external-
      calibration table (PNG+PDF, colorblind-safe, no emoji).
- [ ] `analyze_stage2.py` prints GO / PAUSE-FIX-MEASUREMENT / NO-GO per §4.
- [ ] IF GO: `results/stage2/journal_targets_stage2.md` per §5 with verified SCIE + APC-free
      venues and honest acceptance ranking.
- [ ] `stage2_paper_outline.md` with finding-first abstract, validity section, stats split,
      limitations. No hardcoded secrets; no emoji.

---

## 7. CITATIONS TO EMBED IN CODE COMMENTS (Stage 2 additions)
```
# CITATION: Lan, Z. et al. (2020). ALBERT: A Lite BERT. ICLR.   [ALBERT sharing; no factored embedding here]
# CITATION: Zhao, J. et al. (2018). Gender Bias in Coreference Resolution (WinoBias). NAACL.
# CITATION: Blodgett, S.L. et al. (2021). Stereotyping Norwegian Salmon. ACL. [SS-PLL motivation]
# CITATION: Nadeem, M. et al. (2021). StereoSet. ACL.           [appendix only, if used]
# (Stage 1 citations for Devlin, Saunshi, Bae [arXiv:2507.10524], Geiping [arXiv:2502.05171],
#  Zhu/Ouro [arXiv:2510.25741], Frey [arXiv:2603.08391], Nangia, Khandelwal still apply.)
```
