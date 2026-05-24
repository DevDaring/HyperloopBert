# STAGE 1 — DOES THE EFFECT EXIST? — Full Instructions for Copilot

> **Role of this file.** You (Copilot) have three jobs, in order:
> 1. **BUILD** the Stage 1 code exactly as specified below.
> 2. **DECIDE** go/no-go from the produced result CSVs using the explicit rules in §4.
> 3. **IF GO**, recommend exactly where to submit this study, under the strict journal
>    constraints in §5 (SCIE-indexed, APC-free, acceptance-optimized), and write the
>    recommendation to `results/stage1/journal_targets_stage1.md`.
>
> No source code is in this file. Generate the code yourself. Ask no questions you can
> resolve from this file. Where a real value cannot be known in advance (converged loss,
> whether a model id is currently served, a journal's current indexing), **detect and
> report it — never guess silently.**
>
> **Hard global rules:** global Python environment only (no venv/conda); no emoji anywhere
> in code, comments, logs, figures, or docs; no hardcoded secrets anywhere (load every key
> from `.env`); cite every implemented paper/dataset in a comment block above the code.

---

## 0. SCIENTIFIC CONTEXT (why Stage 1 exists)

This is the first of three sequential, separately publishable stages of a de-risked research
program. The full program tests the **Stereotype Consolidation Hypothesis (SCH)**: that
weight-sharing in looped transformer encoders reduces the encoding of stereotypical
associations at **matched model quality**. SCH is an open question (counter-evidence: Zhu et
al., 2025, arXiv:2603.08391), not an assumption.

Stage 1 is the cheapest possible test of whether the effect exists at all, before any
expensive architecture work. If Stage 1 fails, the whole program stops for ~50 GPU-hours
instead of ~510.

**Stage 1 research question:**
> Does cross-layer parameter sharing reduce stereotype preference at matched MLM validation
> loss, and does that hold across model scale?

**The one non-negotiable:** the primary comparison is at **matched validation loss
(iso-perplexity)**, not at a fixed token budget. A fixed-token comparison is attackable —
a weaker model looks "less biased" only because it is less confident about everything.
Iso-loss removes quality as the alternative explanation. Do not drop this.

---

## 1. SHARED INFRASTRUCTURE (condensed; build if not already present)

Hardware target: one NVIDIA L4 (24 GB, Ada, sm_89), sequential jobs only.

### 1.1 Repository layout
The runnable Stage 1 code lives in `Stage1/`. Shared code lives in `common/` (imported, not
duplicated). Dataset assets are produced once by `Dataset/` and reused.

```
repo_root/
  .env.example        install.sh        requirements.txt    README.md
  common/   env_loader.py llm_utils.py attention.py architectures.py bias_metrics.py
            stats_engine.py integrity.py io_schemas.py plotting.py iso_loss.py logging_setup.py
  Dataset/  download_training_corpus.py download_eval_datasets.py train_tokenizer.py
            contamination_filter.py build_provenance_report.py validate_and_manifest.py
  Stage1/   train_stage1.py eval_bias_stage1.py analyze_stage1.py config_stage1.py
  Dry_Run/  dry_run_dataset.py dry_run_stage1.py
  data/ datasets_eval/ checkpoints/ models/ results/ figures/ logs/   (all gitignored)
```

### 1.2 Install (no venv, global environment)
`install.sh` runs this exact, verified recipe; pin these versions in `requirements.txt`:

```
python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
 && python3 -m pip install "numpy<2.0" transformers==4.46.0 accelerate==0.34.0 datasets==2.16.0 \
      bitsandbytes==0.46.1 pandas==2.2.2 tqdm==4.65.0 python-dotenv==1.0.0 requests==2.31.0 \
      sentencepiece==0.2.0 protobuf==4.25.0 \
 && wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl -O /tmp/flash_attn.whl \
 && python3 -m pip install --no-deps /tmp/flash_attn.whl
```
Also install (compatible versions): `tokenizers, huggingface_hub, scipy, scikit-learn,
matplotlib, seaborn`. The wheel tag implies **Python 3.12, CUDA 12.x, torch 2.5,
cxx11abiFALSE, linux x86-64**. L4 (sm_89) is supported by FlashAttention-2. Verify these at
runtime; if a `terraform/` directory exists, parse it to confirm GPU/OS/CUDA, else probe at
runtime and print one PASS/FAIL compatibility line. Never hard-fail solely on missing
terraform.

### 1.3 Flash-attention
In `common/attention.py`, custom self-attention must use FlashAttention-2 when available,
using the **padding-aware / varlen** path (MLM is bidirectional with padding masks). Fallback
order: FlashAttention -> PyTorch SDPA -> eager. BF16 only (never FP16). Log the active path
once per model build.

### 1.4 Secrets and `.env`
`common/env_loader.py` loads `.env`, resolves names **case-insensitively**, treats empty as
missing, and **never logs values**. Stage 1 only strictly needs `HF_KEY` (alias `HF_TOKEN`).
The LLM utility keys (GCP_KEY1..4, DEEPSEEK_KEY1..2, MISTRAL_KEY1..2, OPENROUTER_KEY1..2)
must still be present in `.env.example` and tested by the dry run, even though Stage 1's core
metrics need no LLM.

### 1.5 LLM utility (built and dry-run-tested even if unused by Stage 1 science)
`common/llm_utils.py`: provider tiers Gemini primary (model `gemini-3-flash-preview`, 4 keys),
DeepSeek secondary (`deepseek-chat`, 2 keys), Mistral tertiary (`mistral-small-latest`, 2
keys), OpenRouter alternate (2 keys). Round-robin **within** a provider's own keys. **No
automatic cross-tier fallback** — a failed provider surfaces the error; never silently switch
judges. Every generative call requests strict JSON (native structured-output + schema in the
prompt), parses deterministically (strip fences, first balanced object, validate schema). On
parse failure mark the row `Needs_Review=true`; **never** invoke a second LLM as a judge.
**Stage 1 bias metrics (PLL) are model-intrinsic and require no judge.**

### 1.6 Integrity on every rerun
`common/integrity.py` runs before consuming any data: detect/quarantine corrupt JSONL lines,
detect/drop exact duplicates (hash of normalized text), and verify file paths/row-counts/MD5
against `data/datasets_manifest.json`; re-download or repair on mismatch.

---

## 2. STAGE 1 BUILD SPECIFICATION

### 2.1 Dataset prerequisites (from `Dataset/`, produced once)
- **Training corpus:** FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample-10BT`, streaming).
  Collect a shared pool large enough for the whole program (~700M tokens); Stage 1 consumes a
  **200M-token prefix**. `data/fineweb-edu/{train,validation}.jsonl`.
- **Shared tokenizer:** one 30,522-token WordPiece (BertPreTokenizer, min_frequency 2,
  specials `[PAD][UNK][CLS][SEP][MASK]`) trained on the train split only. One tokenizer for
  the entire program eliminates the tokenization confound. Non-negotiable.
- **Eval datasets (Stage 1 uses two):**
  - Multi-CrowS-Pairs English (`{namespace}/Multi-CrowS-Pairs`, default namespace `Debk`,
    flag `--dataset-namespace`). Categories: Race_Color, Gender, Socioeconomic, Nationality,
    Religion, Age, Sexual_Orientation, Physical_Appearance, Disability.
  - Indian Multilingual Bias English (`{namespace}/Indian-Multilingual-Bias-Dataset`):
    `Caste.csv, Gender.csv, India_Religious.csv, Race.csv`. Categories: Caste, Gender,
    Religion, Race_Ethnicity. Write the provenance report
    `datasets_eval/indian_bias/provenance/provenance_report.json` (base = Indian-BhED,
    Khandelwal et al. 2023; extension = supplementary; if English-subset IAA kappa < 0.6 use
    original Indian-BhED English as primary).
- **Contamination removal:** remove any training document containing > 3 contiguous words
  matching any eval sentence; log the count.

### 2.2 Architectures (Stage 1 uses TWO only)
From `common/architectures.py`, encoder-only, bidirectional, GELU, absolute positions, MLM
(15% masking: 80% mask / 10% random / 10% original), **compute-matched at effective depth 12**:
- **VanillaBERT** — 12 independent layers (unique layers 12). Cite Devlin et al. 2019.
- **LoopedBERT** — begin(2) -> middle(2, looped x4) -> end(2) = effective depth 12, unique
  layers 6; loop-index embeddings before each middle iteration. Cite Saunshi et al. 2025;
  Bae et al. 2025.

Do **not** build ALBERT, Hyperloop, CWSA, or any mechanistic hooks in Stage 1.

### 2.3 Scale axis (three sizes; effective depth held at 12)
Vary hidden width only, so the depth-matched Vanilla-vs-Looped comparison holds at each scale:

| Size | Hidden | Heads | Intermediate | Approx Vanilla params |
|---|---|---|---|---|
| Tiny | 256 | 4 | 1024 | ~15-25M |
| Small | 512 | 8 | 2048 | ~40-60M |
| Base-ish | 768 | 12 | 3072 | ~90-110M |

### 2.4 Training
- One run per `(architecture, size, seed)` to the 200M-token budget, single-phase, seq=128.
- During the run, save snapshots at adaptive **iso-loss bands** (see §2.5) and at **token
  markers** 50M/100M/200M (the latter for the secondary endpoint plot only).
- Seeds: run **seed 42 only** first to decide go/no-go (`--seeds 1`). If the effect is
  visible, rerun with `--seeds 3` (42,43,44) before writing the paper.
- Uniform `LR = 5e-4` (identical for both architectures — per-architecture LR would be a
  confound), AdamW (0.9/0.98, eps 1e-6, wd 0.01), linear schedule, 10% warmup, grad clip 1.0,
  BF16, gradient checkpointing, flash-attention, effective batch 64 (micro-batch +
  accumulation; on OOM halve micro-batch, floor 4).
- Checkpoint every 2000 steps (full RNG/optimizer/scheduler/token state); validate every 2000
  steps on a fixed 5K-sample set (log loss, pseudo-perplexity, mask accuracy); check bands.
- Resume via `checkpoints/pipeline_state.json`; skip completed jobs on restart.

### 2.5 Iso-loss bands (adaptive; from `common/iso_loss.py`)
Small/shared models converge to higher loss than BERT-base, so do not hardcode 3.5/3.0/2.6.
Procedure: (1) monitor the largest, least-shared model first (Vanilla, Base-ish) to find the
loss floor; (2) choose 3-4 bands spanning a range that **every** architecture at **every**
size reaches (only bands crossed by all configs are usable for the primary comparison);
typical small-model bands ~ 4.0 / 3.7 / 3.4 / 3.1; (3) record the chosen bands in
`config_stage1.py` and the manifest. At each first crossing of a band by a `(model,size,seed)`,
save a snapshot and queue it for bias evaluation.

### 2.6 Bias evaluation
PLL only (SS-PLL is deferred to Stage 2), on Multi-CrowS-Pairs English (overall + 9 categories
+ macro-average) and Indian Bias English (overall + 4 categories), at every iso-loss snapshot
and at every token marker. For each pair: `Stereotype_Preferred = 1 if PLL(stereo) > PLL(anti)`,
`Effect_Size = PLL(stereo) - PLL(anti)`. Append every 50 rows with a resumable `Row_Index`.

### 2.7 Quality screen
A model snapshot with pseudo-perplexity > 60 is "not yet learned" and is excluded from the
primary comparison with a logged warning (a model that has learned nothing cannot have
meaningful bias).

---

## 3. RESULT REGISTRATION (the CSVs the decision in §4 reads)

Full column names only (defined in `common/io_schemas.py`); `Timestamp` on every row.
Identity columns: `Stage, Architecture, Model_Size, Hidden_Size, Seed, Unique_Parameters,
Total_Parameters, Effective_Depth, Shared_Ratio`.

- `results/stage1/mlm/summary_table.csv`: identity + `Validation_Loss, Pseudo_Perplexity,
  Mask_Accuracy, Tokens_Processed, Tokens_Per_Second, GPU_Hours, Token_Marker, Band`.
- `results/stage1/bias/<dataset>_<arch>_<size>_seed<seed>_<band>_progress.csv`: `Row_Index,
  Dataset, Category, Sentence_Stereotypical, Sentence_AntiStereotypical, PLL_Stereotypical,
  PLL_AntiStereotypical, Effect_Size, Stereotype_Preferred,` identity, `Validation_Loss, Band,
  Token_Marker, Needs_Review, Timestamp`.
- `results/stage1/bias/<dataset>_summary.csv`: identity + `Band, Token_Marker,
  Overall_Stereotype_Preference_Rate, Macro_Average_Preference_Rate,
  <Category>_Preference_Rate (per category), Mean_Effect_Size, Bootstrap_CI_Low,
  Bootstrap_CI_High`.
- `results/stage1/iso_checkpoints/index.csv`: identity + `Band, Snapshot_Path,
  Validation_Loss_At_Snapshot, Crossed_At_Step`.
- `results/stage1/stats/primary_contrast.csv` (when >=2 seeds): `Contrast, Metric, Dataset,
  Band, Model_Size, Mean_Delta_Preference, Bootstrap_CI_Low, Bootstrap_CI_High,
  Permutation_P_Value, Cohens_D`.

---

## 4. GO / NO-GO DECISION PROTOCOL (Copilot computes this and prints the verdict)

`analyze_stage1.py` must compute and print one of: **GO**, **EXTEND-SEEDS**, **NO-GO**, using
the **primary instrument** = Multi-CrowS-Pairs English, PLL, iso-loss matched. (Indian Bias is
a secondary confirmation, reported but not decisive.)

### 4.1 Per-size delta
For each size in {Tiny, Small, Base-ish}, at the **deepest iso-loss band that BOTH Vanilla and
Looped crossed** (and where both have pseudo-perplexity < 60):
```
delta(size) = Overall_Stereotype_Preference_Rate(Vanilla) - Overall_Stereotype_Preference_Rate(Looped)
```
Positive delta = Vanilla more biased than Looped at matched quality = the SCH direction.

### 4.2 Decision rule (defaults; Copilot may adjust with written justification)
- **GO** if **all** hold:
  1. `delta(size) > +0.02` for **>= 2 of 3** sizes, AND
  2. no size shows a reversal `delta(size) < -0.02`, AND
  3. with >= 2 seeds: the pooled primary contrast (Vanilla vs Looped at the primary band) has
     `Permutation_P_Value < 0.05` **or** a bootstrap CI of mean delta excluding 0.
- **EXTEND-SEEDS** if the direction is right (>= 2 of 3 sizes positive) but significance is
  not yet established at 1 seed. Action: rerun with `--seeds 3`, then re-decide.
- **NO-GO (STOP THE PROGRAM)** if either: mean `|delta|` across sizes `< 0.02`, or the sign is
  inconsistent (some sizes clearly reversed). This means `Vanilla ~= Looped` — SCH direction
  is not supported; do not spend Stage 2/3 compute.

### 4.3 What to print
A decision block: the verdict; the per-size delta table; the primary-band used per size; the
permutation p / CI when available; the Indian-Bias direction as secondary confirmation; and a
one-paragraph plain-language interpretation.

---

## 5. IF GO — WHERE TO SUBMIT (Copilot decides the venue; strict constraints)

Only run this section if the verdict is **GO** (with >= 2 seeds, i.e. publishable). Produce
`results/stage1/journal_targets_stage1.md`. **What Stage 1 publishes:** a short paper /
communication reporting a controlled finding (parameter sharing reduces stereotype preference
at matched validation loss, replicated across three model scales, on a general benchmark and
an India-centric instrument). Treat this as paper **Tier M** (solid, controlled, but two
architectures and small scratch models) unless the deltas are large and clean across all three
sizes with significance, in which case it may rise toward Tier M+/S.

### 5.1 Hard constraints (a journal violating ANY is excluded entirely — do not mention it)
1. **Indexing = SCIE / SCI** (Science Citation Index Expanded). **ESCI is rejected.**
   "Indexed in Web of Science" is NOT sufficient — WoS contains both SCIE and ESCI.
   Scopus-only and DOAJ-only are not sufficient.
2. **APC-free for the author.** Acceptable: subscription-only journals; hybrid journals via
   the free subscription path (confirm the subscription path is the default, not vestigial);
   diamond OA; **ACM journals** (the author's institution, IIIT Kalyani, has **ONOS** coverage
   giving APC-free OA in ACM venues). Rejected: gold OA with any APC (even "modest"),
   mandatory-OA hybrids, submission fees, "waiver on request".
3. **Scope match** — read the journal's aims-and-scope, do not infer from title.

### 5.2 Mandatory verification (do not skip; you have web access)
For every candidate, before it may appear in the ranked list:
- Verify SCIE via the **Clarivate Master Journal List** (`https://mjl.clarivate.com/`). Look
  for the explicit line **"Science Citation Index Expanded"** — not merely "Web of Science
  Core Collection". Queries: `"<journal>" "Science Citation Index Expanded"` and
  `"<journal>" site:mjl.clarivate.com`. If only "Emerging Sources Citation Index" appears,
  **reject**.
- Verify APC on the journal's official Open Access / Author Information page. For ACM
  candidates, confirm IIIT Kalyani is on the current ACM-ONOS participating list.
- If indexing or APC is ambiguous after both checks, mark **"Needs manual verification"** and
  exclude from the ranked recommendations (separate section).

### 5.3 Candidate pool for a Stage-1 short paper (all subject to §5.2 verification)
Foreground the **India-centric, controlled, scaling** angle — it is the distinctive
contribution and raises fit at regional/low-resource and analysis venues.
- **ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP)** —
  strongest scope match if the India-centric instrument is foregrounded; ACM-ONOS = APC-free;
  verify SCIE.
- **Neurocomputing (Elsevier)** — SCIE, hybrid (subscription path free); has a **Short
  Communication** track that fits a focused controlled finding.
- **Pattern Recognition Letters (Elsevier)** — SCIE, short-letter format; fits a tight
  empirical result.
- **Natural Language Engineering (Cambridge)** — SCIE, hybrid; good for NLP analysis pieces.
- **Knowledge-Based Systems (Elsevier)** — SCIE, hybrid, broad; accepts fairness/bias work
  (full-length, only if the result carries a full paper).

### 5.4 User-specific exclusions (apply automatically)
- **Expert Systems with Applications** — excluded for bias papers (author rejected there twice
  on a related bias paper).
- **Information Processing & Management** — the author has an active submission; check status
  before recommending; do not double-submit.
- **Frontiers / MDPI / PLOS / TMLR** — excluded on indexing (often ESCI / not SCIE) or APC
  grounds; mention only as "explicitly excluded (and why)".

### 5.5 Output file format (`results/stage1/journal_targets_stage1.md`)
```
# Journal Targets (Stage 1): <paper working title>
Paper tier: <M / M+ / S> - <one-sentence justification tied to the Stage 1 deltas>
Recommended format: short paper / communication / letter
Date: <YYYY-MM-DD>

## Recommendation in one sentence
<e.g. "Submit to ACM TALLIP as a focused India-centric controlled study for highest APC-free SCIE acceptance odds.">

## Primary recommendations (ranked by acceptance probability)
### 1. <Journal> - <Publisher> - IF <X.X>
- Indexing: SCIE [verified via Clarivate MJL on <date>] OR [Needs manual verification]
- APC: <Subscription / Hybrid subscription path free / ACM-ONOS / Diamond OA>
- Scope fit: <one sentence>
- Estimated acceptance: <Very Low/Low/Moderate/High/Very High + percent range, honest>
- Why this fits: <2-3 sentences>
- Risks: <one sentence on the likely reviewer objection>
- Optional reframing: <only if it raises the estimate by >= 1 tier and needs no new experiments>
### 2. ... ### 3. ...

## Low-IF SCIE fallback (always present)
<at least one safety-net SCIE + APC-free venue>

## Short-format alternatives (only if the journal actually offers them)
## Needs manual verification
## Explicitly excluded (and why)
## Bottom-line action
<one sentence: target first + whether to reframe/shorten before submission>
```
Be honest with acceptance estimates (a 30% is a 30%); do not pad with aspirational venues;
always include the low-IF SCIE fallback; never call a journal SCIE without the explicit
Clarivate "Science Citation Index Expanded" confirmation.

---

## 6. STAGE 1 DELIVERABLES CHECKLIST
- [ ] `Stage1/` builds and runs on the global env; flash-attention path logged; no venv.
- [ ] Vanilla + Looped, three sizes, 200M-token runs with iso-loss + token-marker snapshots.
- [ ] PLL bias eval on Multi-CrowS-Pairs + Indian Bias at every snapshot; appended every 50
      rows; resumable; quality screen applied.
- [ ] Headline figure: stereotype preference vs validation loss, Vanilla vs Looped, one panel
      per size, one figure per dataset (PNG + PDF, colorblind-safe, no emoji).
- [ ] Secondary figure: preference vs token budget (50M/100M/200M), labeled secondary.
- [ ] Result CSVs with full column names; bootstrap CIs; permutation test at >= 2 seeds.
- [ ] `analyze_stage1.py` prints GO / EXTEND-SEEDS / NO-GO per §4 with the delta table.
- [ ] IF GO: `results/stage1/journal_targets_stage1.md` produced per §5 with verified SCIE +
      APC-free venues and an honest acceptance ranking.
- [ ] `stage1_paper_outline.md`: finding-first abstract (matched-loss result filled post-run),
      method, India-centric instrument as novelty, limitations, the go/no-go statement.
- [ ] No hardcoded secrets anywhere (including `Dry_Run/`); no emoji anywhere.

---

## 7. CITATIONS TO EMBED IN CODE COMMENTS (Stage 1)
```
# CITATION: Devlin, J. et al. (2019). BERT. NAACL.                         [VanillaBERT]
# CITATION: Saunshi, N. et al. (2025). On the Power of Looped Transformers. arXiv. [SCH basis]
# CITATION: Bae, J. et al. (2025). Looped encoder adaptation.              [LoopedBERT]
# COUNTER:  Zhu, L. et al. (2025). arXiv:2603.08391.                       [SCH counter-evidence]
# CITATION: Nangia, N. et al. (2020). CrowS-Pairs. EMNLP.                  [Multi-CrowS-Pairs]
# CITATION: Khandelwal, K. et al. (2023). Indian-BhED. arXiv:2309.08573.   [Indian instrument]
# CITATION: Blodgett, S.L. et al. (2021). Stereotyping Norwegian Salmon. ACL. [PLL validity caveat]
# DATA WARNING: contains stereotypical content by design; research/fairness-audit use only.
```
