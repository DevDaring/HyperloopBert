# BUILD SPECIFICATION — Parameter Sharing & Stereotype Encoding (3-Stage, De-risked)

> **This document is a build prompt for a coding agent. It contains NO source code.**
> It is a complete, unambiguous specification. A coding tool should be able to read this
> document top to bottom and produce a working repository without asking further questions.
> Where a real value cannot be known ahead of time (e.g. exact converged loss, exact
> model id availability), the spec says so and tells the agent to *detect and report*,
> never to guess silently.

---

## 0. MISSION

Implement the experimental pipeline for the research program:

> **"Parameter Sharing Reduces Stereotype Memorization: A Controlled, Scaled, Mechanistic
> Study of Looped and Hyper-Connected Encoder Architectures."**
>
> Target venues: short paper / communication first (Stage 1), then a conference short
> paper or journal letter (Stage 2), then a full paper — TACL (primary) or Neurocomputing
> (Elsevier, SCIE Q1, fallback) — after Stage 3.

The single original monolithic plan (a ~510 GPU-hour, 26-model, all-at-once pipeline) is
**deliberately decomposed into three sequential stages**. Each stage is a real go/no-go
gate AND independently publishable. Compute escalates only after the effect is confirmed.
The big budget is spent last, not first.

**The one non-negotiable kept in every stage: iso-loss (matched validation-loss)
comparison.** Endpoint/token-budget comparison alone is attackable ("the smaller/shared
model only looks less biased because it is weaker"). Comparing bias at matched validation
loss removes model quality as an alternative explanation. Do not drop this.

### 0.1 Hardware target

- One NVIDIA **L4 (24 GB GDDR6, Ada Lovelace, compute capability sm_89)**.
- All training jobs run **sequentially** on the single GPU. No multiprocessing across jobs.
- Flash-Attention 2 is supported on sm_89 and **must** be used (see §3).

### 0.2 Scientific framing (lead with the finding, not the architecture)

The contribution is the **finding** that parameter sharing reduces stereotype encoding at
matched model quality — controlled evidence for the **Stereotype Consolidation Hypothesis
(SCH)**. The Hyperloop architecture (Stage 3) is the *mechanistic vehicle* used to explain
*how*, not the contribution itself.

**SCH (open question, not assumption):** Weight-sharing in looped transformers induces a
memorization-reasoning tradeoff (Saunshi et al., 2025) that reduces encoding of
stereotypical associations. The 2025-2026 looped-LM literature sharpens (not refutes) SCH:
Ouro (Zhu et al., 2025, arXiv:2510.25741) shows looped gains come from knowledge
*manipulation*, not increased storage; Frey et al. (2026, arXiv:2603.08391) find looping aids
manipulation while *per-parameter* memorization is preserved. SCH must therefore be stated as
a claim about **total** stereotype-storage capacity at matched quality (NOT per-parameter
memorization), with an explicit failure mode: high-frequency corpus stereotypes still leak in.
(Attribution note: arXiv:2603.08391 is Frey et al. 2026, not Zhu -- an earlier "Zhu
counter-evidence" citation was an error.)

### 0.3 The three-stage logic and why it de-risks the program

| | Stage 1 — *Does the effect exist?* | Stage 2 — *Is it sharing, and is it robust?* | Stage 3 — *Mechanism + novel architecture* |
|---|---|---|---|
| Question | Does sharing reduce stereotype preference at matched loss, across scale? | Is it the *degree* of sharing (dose-response), surviving validity + stats? | *Why* — stream diversity as causal mechanism (Hyperloop + CWSA) |
| Architectures | Vanilla vs Looped | Vanilla vs Looped vs ALBERT | + Hyperloop, + stream ablation n=1/2/4 |
| Sizes (hidden width) | Tiny / Small / Base-ish | Base-ish (+ one smaller scale check) | Base-ish |
| Tokens (max budget) | 200M (snapshots at 50M/100M/200M markers) | up to 300-500M | 500M (+ 300M ablations) |
| Seeds | 1 to decide, 2-3 to publish | 3-5 | 3-5 primary, 3 ablation |
| Bias eval | Multi-CrowS + Indian (PLL) | + SS-PLL + WinoBias + external calibration + GLUE screen + loop-trajectory teaser | full suite + iso-loss snapshots + full mechanistic suite |
| Rough L4 GPU-h | ~40-80 | ~150-200 | ~250-300 |
| Deliverable | communication / workshop short paper | conference short paper or journal letter | full paper (TACL / Neurocomputing) |

**Go/no-go gates (decision rules, must be printed by each stage at completion):**

- **Stage 1 GO** if `Vanilla bias > Looped bias` at matched loss, consistent across >= 2 of
  3 sizes. **NO-GO / STOP THE PROGRAM** if `Vanilla ~= Looped` — SCH direction is dead,
  learned for ~50 GPU-h instead of ~510.
- **Stage 2 GO** if the sharing->bias relationship is monotone (or at least
  `Vanilla > {Looped, ALBERT}`), survives Holm-Bonferroni on the confirmatory family, and
  PLL and SS-PLL agree. **PAUSE** if it only appears in one metric or collapses under
  correction — fix measurement before building Hyperloop.
- **Stage 3 GO/publish** even on a null Hyperloop result (`Looped ~= Hyperloop`), because
  the finding is already established and Hyperloop is framed as the mechanistic vehicle.
  The only true failure mode (SCH false) was screened out in Stage 1.

---

## 1. REPOSITORY LAYOUT

The runnable, stage-specific code lives in the **five required folders**. Shared modules
imported by all stages live in **`common/`** (a deliberate, documented addition — shared
architecture, metric, and utility code must not be duplicated across stages; duplication
would let the stages drift apart and break cross-stage comparability). Do not duplicate
shared logic into each stage folder.

```
repo_root/
  .env.example               # template with ALL key names, NO real values
  README.md                  # one-stop guide (see §14)
  requirements.txt           # pinned versions (see §3)
  install.sh                 # exact install recipe from §3 (no venv)
  terraform/                 # IF PRESENT: inspected for OS/GPU compatibility (see §3.3)

  common/                    # shared library, imported by every stage
    env_loader.py            # loads .env, resolves keys case-insensitively (§4)
    llm_utils.py             # multi-provider JSON-first client, round-robin (§5)
    attention.py             # flash-attention self-attention + fallbacks (§3, §6)
    architectures.py         # VanillaBERT, LoopedBERT, ALBERTLoopedBERT, HyperloopBERT,
                             #   EarlyMergeHyperloopBERT, size presets (§6)
    bias_metrics.py          # PLL, SS-PLL, effect size, WinoBias scoring (§6, §9)
    stats_engine.py          # bootstrap CI, permutation test, Holm-Bonferroni, Cohen d (§6)
    integrity.py             # dedup + corruption checks, manifest verify (§7.5)
    io_schemas.py            # canonical CSV column definitions (§12) - single source of truth
    plotting.py              # figure helpers, no emoji, colorblind-safe palette
    iso_loss.py              # iso-loss band selection, snapshot bookkeeping (§6.4)
    logging_setup.py         # rotating file + console logging

  Dataset/                   # download + validate + tokenizer + manifest (§7)
    download_training_corpus.py
    download_eval_datasets.py
    train_tokenizer.py
    contamination_filter.py
    build_provenance_report.py
    validate_and_manifest.py

  Stage1/                    # §8
    train_stage1.py
    eval_bias_stage1.py
    analyze_stage1.py        # iso-loss curve, go/no-go decision, figures, tables
    config_stage1.py         # stage-1-only constants (sizes, budgets, bands, seeds)

  Stage2/                    # §9
    train_stage2.py
    eval_bias_stage2.py
    eval_glue_stage2.py
    external_calibration_stage2.py
    loop_trajectory_stage2.py
    analyze_stage2.py
    config_stage2.py

  Stage3/                    # §10
    train_stage3.py
    train_stream_ablation_stage3.py
    eval_bias_stage3.py
    mechanistic_stage3.py    # CKA, stream disagreement, early-merge, token drift
    analyze_stage3.py
    config_stage3.py

  Dry_Run/                   # §11
    dry_run_dataset.py
    dry_run_stage1.py
    dry_run_stage2.py
    dry_run_stage3.py
    dry_run_report.json      # written at runtime; not committed

  data/                      # downloaded corpora + tokenizer (gitignored)
  datasets_eval/             # downloaded eval datasets (gitignored)
  checkpoints/               # pipeline_state.json + training checkpoints (gitignored)
  models/                    # final + iso-band snapshots (gitignored)
  results/                   # all CSVs (gitignored)
  figures/                   # all figures (gitignored)
  logs/                      # run logs (gitignored)
```

Every stage and the dataset stage must support **resume** via a `pipeline_state.json`
under `checkpoints/`. On restart, skip stages/jobs already marked complete. Every script
must support `--dry-run` delegating to the matching file in `Dry_Run/`.

---

## 2. GLOBAL EXECUTION CONTRACT

- **No venv, no conda.** Use the global Python environment only. All installs go to the
  global interpreter (see §3).
- **All secrets come from `.env`.** No key, token, or credential may be hardcoded anywhere,
  including in any `Dry_Run/` or test file. If a literal key string is found in any file
  during code review, it is a defect: remove it and load from environment instead (§4).
- **No emoji** anywhere in code, comments, logs, README, figures, or any generated text.
- **Resumability:** checkpoint frequently; never lose progress; scripts may run unattended
  for days. Retry transient I/O (HF downloads) on the same resource; never lose state.
- **Integrity on every rerun:** before consuming any dataset, run the integrity check (§7.5)
  to detect duplicate and corrupted records; re-download/repair as needed.
- **Determinism:** set and log all RNG seeds (`random`, `numpy`, `torch`, `torch.cuda`).
- **Citations in code:** wherever a paper, dataset, benchmark, or method is implemented or
  compared, place a citation comment block (§13) directly above the relevant code so it is
  impossible to miss when writing the paper.
- **Detailed result registration:** every result file uses full, unabbreviated column names
  defined once in `common/io_schemas.py` (§12). Never abbreviate (`Architecture` not `arch`,
  `Validation_Loss` not `val_loss`, `Stereotype_Preferred` not `pref`).

---

## 3. ENVIRONMENT, INSTALLATION, FLASH-ATTENTION

### 3.1 Canonical install recipe (verified working on the user's GCP L4)

`install.sh` must run exactly this (global environment, no venv). This recipe is
authoritative; do not substitute versions without recording the change in README.

```
python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
 && python3 -m pip install "numpy<2.0" transformers==4.46.0 accelerate==0.34.0 datasets==2.16.0 \
      bitsandbytes==0.46.1 pandas==2.2.2 tqdm==4.65.0 python-dotenv==1.0.0 requests==2.31.0 \
      sentencepiece==0.2.0 protobuf==4.25.0 \
 && wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl -O /tmp/flash_attn.whl \
 && python3 -m pip install --no-deps /tmp/flash_attn.whl
```

Additional packages this project needs (append to the recipe and to `requirements.txt`):
`tokenizers`, `huggingface_hub`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`.
Install these pinned to versions compatible with `transformers==4.46.0` and `numpy<2.0`.

After install, run the verification probe (already in the user's recipe) confirming
`torch.__version__`, `torch.version.cuda`, `bitsandbytes.__version__`, and
`flash_attn.__version__` import cleanly. If any line fails, abort and print a precise
remediation message (which package, which expected version).

**Critical runtime facts implied by the wheel tag** (the agent must verify, not assume):
- `cp312` -> Python **3.12** must be the runtime interpreter.
- `cu12torch2.5` -> CUDA 12.x runtime, torch 2.5.x.
- `cxx11abiFALSE` -> must match torch's C++ ABI; if a future torch build flips this, the
  wheel will fail to import and the agent must report it.
- `linux_x86_64` -> Linux x86-64 only.
- L4 is sm_89 (Ada). FlashAttention-2 supports sm_80/86/89/90, so L4 is supported.
  FlashAttention requires BF16 or FP16; this project uses **BF16** (L4 native).

### 3.2 Flash-attention usage requirement

Search/verify current FlashAttention-2 requirements against the actual runtime before
enabling it (the install recipe is the baseline; confirm GPU compute capability and Python
tag at runtime). Then:

- In `common/attention.py`, implement the self-attention used by all custom architectures
  so it calls FlashAttention-2 when available and the inputs are eligible.
- MLM uses **bidirectional** attention with **padding masks**. Use the variable-length /
  padding-aware FlashAttention path (unpadded varlen API) so padded tokens are excluded
  correctly. If varlen is not wired, fall back rather than silently masking incorrectly.
- Fallback order if FlashAttention is unavailable or ineligible: PyTorch scaled dot-product
  attention (SDPA) -> eager attention. Log which path is active once per model build.
- BF16 only with FlashAttention. Never FP16.
- Wrap FlashAttention import and `torch.compile` in try/except; both are non-fatal.

### 3.3 OS / terraform compatibility check

The user maintains terraform infrastructure. **If a `terraform/` directory (or any `*.tf`
files) exists in the repo**, the dataset dry run and `install.sh` preflight must parse it
to confirm:
- the GPU instance type resolves to an L4 (or another sm_80/86/89/90 GPU),
- the OS image is Linux x86-64 with Python 3.12 available,
- the CUDA driver is 12.x compatible.

If terraform files are absent, log a clear warning that compatibility could not be
auto-verified from infrastructure-as-code, and fall back to runtime probes (query the
device compute capability, the Python version, and `torch.version.cuda`). Print a single
PASS/FAIL compatibility line. Never hard-fail solely because terraform is missing.

---

## 4. SECRETS AND `.env` CONTRACT

All keys are read by `common/env_loader.py`. It must resolve names **case-insensitively**
(the user's notes mix `GCP_Key1` and `GCP_key4`), trim whitespace, and treat empty strings
as missing. Provide `.env.example` with every name below and **no values**.

```
# ---- HuggingFace (datasets + model hub) ----
HF_KEY=                 # alias also accepted: HF_TOKEN

# ---- Google AI Studio / Gemini : PRIMARY judge & extraction, round-robin over 4 keys ----
GCP_KEY1=
GCP_KEY2=
GCP_KEY3=
GCP_KEY4=

# ---- DeepSeek : SECONDARY, round-robin over 2 keys ----
DEEPSEEK_KEY1=
DEEPSEEK_KEY2=

# ---- Mistral : TERTIARY, round-robin over 2 keys ----
MISTRAL_KEY1=
MISTRAL_KEY2=

# ---- OpenRouter : alternate router, round-robin over 2 keys ----
OPENROUTER_KEY1=
OPENROUTER_KEY2=

# ---- Optional ----
OPENAI_API_KEY=
```

`env_loader.py` exposes a typed accessor returning, per provider, the ordered list of
present keys, and a boolean "available" flag. At startup every script logs which providers
are available (by count of keys found), and **never logs the key values**.

---

## 5. LLM UTILITY MODULE (`common/llm_utils.py`)

Used **only where a generative/extraction step is genuinely needed** (e.g. optional dataset
content validation, optional answer extraction, optional judge cross-checks). **The core
bias metrics (PLL, SS-PLL, WinoBias scoring) are model-intrinsic and require NO LLM judge.**
Build the utility regardless and exercise it fully in every dry run.

### 5.1 Provider tiers and the no-cross-fallback rule

- **Primary:** Gemini via Google AI Studio. Model id: `gemini-3-flash-preview`. Round-robin
  across `GCP_KEY1..4`.
- **Secondary:** DeepSeek (OpenAI-compatible endpoint). Model id default: `deepseek-chat`.
  Round-robin across `DEEPSEEK_KEY1..2`.
- **Tertiary:** Mistral. Model id: `mistral-small-latest` ("mistral small"). Round-robin
  across `MISTRAL_KEY1..2`.
- **Alternate:** OpenRouter (OpenAI-compatible router). Model id configurable. Round-robin
  across `OPENROUTER_KEY1..2`.

**No automatic cross-tier fallback.** The caller selects the provider explicitly (default
`gemini`). If the selected provider fails, the call surfaces the error; the system must
**not** silently retry on a different tier — switching judges mid-run would contaminate
results. Within a single provider, transient errors (timeouts, 5xx, rate limits) may be
retried a small bounded number of times **by advancing the round-robin to that provider's
next own key only**. Document this behavior in the module docstring and in README.

### 5.2 JSON-first contract (no separate judge needed)

Every generative call must:
1. Request structured JSON via the provider's native mechanism when available
   (Gemini `responseMimeType: application/json` + response schema; OpenAI-compatible
   `response_format={"type":"json_object"}` for DeepSeek/Mistral/OpenRouter).
2. Include the exact expected JSON schema in the prompt text as a backstop.
3. Set a low temperature for determinism (e.g. 0.0-0.2) for any judgement/extraction call.
4. Parse the response with a robust extractor: strip markdown code fences, locate the first
   balanced `{...}` object, `json.loads` it, then validate keys/types against the expected
   schema.
5. On parse/validation failure: log the raw response to `logs/`, mark the affected record
   `needs_review=true` in the output CSV, and continue. **Do not** invoke a second LLM as a
   judge to fix it. The deterministic JSON parser is the only judge.

### 5.3 Dry-run obligations for this module

Each dry run (§11) must, for **every configured provider and every key**, send one minimal
JSON-returning request, confirm a parseable JSON response, and confirm the **model id is
valid**. If `gemini-3-flash-preview` (or any id) is not currently served, the report must
say so explicitly with the provider's error, so the user can swap the id — never fail
silently and never substitute a different model.

---

## 6. SHARED LIBRARY DETAILS (`common/`)

### 6.1 Architectures (`architectures.py`)

All architectures are encoder-only, bidirectional, GELU, absolute position embeddings,
MLM objective (15% masking: 80% `[MASK]`, 10% random, 10% original). All are
**compute-matched at effective depth 12** (12 transformer-layer applications per forward
pass); they differ only in how many **unique** weight sets they hold.

Shared hyperparameters scale with a **size preset** (the "scale" axis is hidden width,
holding effective depth = 12 constant so the depth-matched Vanilla-vs-Looped comparison is
preserved at every scale):

| Size preset | Hidden | Heads | Intermediate | Approx Vanilla params |
|---|---|---|---|---|
| Tiny | 256 | 4 | 1024 | ~15-25M |
| Small | 512 | 8 | 2048 | ~40-60M |
| Base-ish | 768 | 12 | 3072 | ~90-110M |

Common: `MAX_POSITION_EMBEDDINGS=512`, `VOCAB_SIZE=30522`. Record per model:
`Unique_Parameters`, `Total_Parameters`, `Effective_Depth`, `Hidden_Size`,
`Shared_Ratio = 1 - unique_layers/12`. Log at model creation.

Classes to implement (with citation comment blocks from §13 directly above each):

- **VanillaBERT** — 12 independent layers. Effective depth = unique layers = 12.
  (Devlin et al., 2019.)
- **LoopedBERT** — begin(2) -> middle(2, looped x4) -> end(2). Effective depth
  2 + 2*4 + 2 = 12. Unique layers = 6. Add loop-index embeddings
  (`Embedding(num_loops, hidden)`) before each middle iteration to break symmetry.
  (Bae et al., 2025, arXiv:2507.10524; Saunshi et al., 2025; encoder-only adaptation.)
- **ALBERTLoopedBERT** — ONE shared layer applied 12 times. Unique layers = 1.
  **No embedding factorization** (isolates weight-sharing from embedding compression).
  Loop-index embeddings added before each of the 12 iterations. (Lan et al., 2020.)
- **HyperloopBERT** (Stage 3) — begin(2) -> middle(2, looped x4) -> end(2),
  `num_streams=4`. Effective depth 12, unique layers 6. The shared middle block is applied
  **exactly once per loop on a stream-mixed input** (NOT once per stream): per loop,
  depth-connection (read: mix n streams into one block input) -> middle block ->
  width-connection (write: scatter block output back into all n streams). Streams are
  residual carriers mixed by learnable hyper-connection matrices at loop boundaries only,
  so Hyperloop is compute-matched to Looped. **CWSA (CLS-Weighted Stream Aggregation):**
  final stream combination is a learned soft-attention over each stream's `[CLS]`
  representation (encoder-specific; not simple averaging). Store per-loop stream tensors in
  a `stream_snapshots` attribute for mechanistic analysis.
  (Zeitoun et al., 2026, arXiv:2604.21254 — first controlled encoder-only adaptation + CWSA.)
- **EarlyMergeHyperloopBERT** (Stage 3) — Hyperloop with streams merged early at
  `merge_at in {1,2,3}`. Labeled OOD intervention, **not** causal proof.

Every architecture returns `last_hidden_state`, `pooler_output`, `mlm_logits`, and (where
applicable) intermediate snapshots needed for mechanistic hooks.

### 6.2 Bias metrics (`bias_metrics.py`)

- **PLL (Pseudo-Log-Likelihood, primary):** for each token position, mask it, predict it
  from all other tokens, accumulate log-prob, normalize by subword count.
  For a pair: `Stereotype_Preferred = 1 if PLL(stereo) > PLL(anti) else 0`;
  `Effect_Size = PLL(stereo) - PLL(anti)` (continuous).
- **SS-PLL (Shared-Token PLL, validity check):** score only the unmodified shared tokens
  between the stereo/anti sentences (addresses Blodgett et al., 2021 surface-form critique).
  A result is "robust" only when PLL and SS-PLL agree in direction.
- **WinoBias scoring:** pro/anti stereotype coreference; report
  `Pro_Stereotype_Accuracy`, `Anti_Stereotype_Accuracy`, `Pro_Anti_Gap`.
  (Nangia et al., 2020; Khandelwal et al., 2023; Zhao et al. WinoBias; Blodgett et al., 2021.)

### 6.3 Statistics (`stats_engine.py`)

`bootstrap_ci` (n=2000, 95%), `paired_permutation_test` (n=10000, one-sided),
`holm_bonferroni_correction` (applied to the SMALL confirmatory family only),
`cohens_d`, `seed_variance_report` (mean/std/min/max/coefficient_of_variation),
`pearson_and_spearman` (both r, both p, labeled exploratory/uncorrected).

### 6.4 Iso-loss bookkeeping (`iso_loss.py`)

Iso-loss bands are **adaptive** because small/shared models converge to higher loss than
BERT-base. Procedure:
1. Calibration: monitor the **largest, least-shared** model first (Vanilla, Base-ish). Its
   converged validation loss defines the floor.
2. Choose 3-4 bands spanning a range that **every** architecture at **every** size in the
   stage actually reaches (only bands crossed by all configs are usable for the primary
   iso-loss comparison). Typical small-model bands: `4.0, 3.7, 3.4, 3.1`. Do not hardcode
   `3.5/3.0/2.6` blindly; record the chosen bands in the stage config and in the manifest.
3. During training, after each validation, when `val_loss <= band` is first crossed for a
   `(model, seed)`, save a snapshot to `models/iso_band_models/<model>_<size>_seed<seed>_loss<band>/`
   and queue it for bias evaluation. Also save snapshots at the token markers (50M/100M/200M
   for Stage 1) for the secondary endpoint plot.

### 6.5 IO schemas (`io_schemas.py`)

Single source of truth for every CSV's columns (§12). All writers import from here. Append
incrementally (every 50 rows for long bias evaluations) with a resumable `Row_Index` so an
interrupted evaluation continues, not restarts.

---

## 7. DATASET STAGE (`Dataset/`)

Runs once; outputs are shared by all three stages. One shared tokenizer and one shared
training pool guarantee cross-stage comparability.

### 7.1 Training corpus — FineWeb-Edu

`datasets.load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train",
streaming=True)`. Collect a single pool large enough for the **largest** stage (Stage 3:
500M training + 80M validation + headroom; collect ~700M tokens to be safe). Estimate
tokens from raw bytes (~5.5 bytes per token) while streaming. Write
`data/fineweb-edu/train.jsonl` and `data/fineweb-edu/validation.jsonl`, one
`{"text": "..."}` JSON object per line. Each stage's data loader consumes a **prefix** of
this pool up to that stage's token budget — never re-download per stage.

Citation comment: this is a CONTROLLED study, not a SOTA model; the budget is enough signal
to differentiate architectures' bias encoding, not to reach BERT-base perplexity.

### 7.2 Evaluation datasets

A CLI flag `--dataset-namespace` (default `ANONYMOUS`) controls the HF org prefix (for anonymous
submission, mirror to an anonymous org and pass it here).

- **Multi-CrowS-Pairs** — `{namespace}/Multi-CrowS-Pairs`, English CSV ->
  `datasets_eval/multicrows/crows_pair_english.csv`. (Nangia et al., 2020 + extension.)
  Categories: Race_Color, Gender, Socioeconomic, Nationality, Religion, Age,
  Sexual_Orientation, Physical_Appearance, Disability.
- **Indian Multilingual Bias** — `{namespace}/Indian-Multilingual-Bias-Dataset`, English
  files -> `datasets_eval/indian_bias/english/{Caste.csv, Gender.csv, India_Religious.csv,
  Race.csv}`. Categories: Caste, Gender, Religion, Race_Ethnicity.
  (Khandelwal et al., 2023, Indian-BhED, arXiv:2309.08573 + multilingual extension.)
- **WinoBias** — try `wino_bias`; else fetch the type-1 pro/anti dev+test files from the
  uclanlp/corefBias repo -> `datasets_eval/winobias/`. (Stage 2+.)
- **StereoSet** — appendix only, never in main tables. (Stage 3 appendix.)
- **GLUE** subsets `sst2`, `mrpc`, `qnli`, `rte` -> `datasets_eval/glue/`. (Stage 2+.)

### 7.3 Provenance report (Indian dataset)

Write `datasets_eval/indian_bias/provenance/provenance_report.json` documenting base dataset
(Indian-BhED), extension source, extension relationship, validation status (SUPPLEMENTARY),
inter-annotator-agreement note, and the recommendation: if English-subset IAA kappa < 0.6,
use original Indian-BhED English as primary and demote the extension to supplementary. Print
at startup; embed verbatim in the paper data section. The India-centric instrument must be
defensible; reviewers will scrutinize the extension.

### 7.4 Contamination removal

Collect all sentence text from CrowS-Pairs, Multi-CrowS-Pairs English, Indian Bias English,
and WinoBias into `data/fineweb-edu/contamination_phrases.txt`. Remove any training document
containing > 3 contiguous words matching any contamination phrase (substring check). Log the
removed-document count.

### 7.5 Integrity: duplicate + corruption checks (run EVERY rerun, all stages)

`common/integrity.py`, invoked at the start of the dataset stage AND at the start of each
training stage:
- **Corruption:** every JSONL line parses; `text` field present and non-empty; minimum
  length threshold; valid UTF-8; no truncated final line. Quarantine bad lines to
  `data/fineweb-edu/quarantine.jsonl` and log counts.
- **Duplicates:** hash each document (e.g. SHA-1 of normalized text); detect and drop exact
  duplicates; report duplicate rate. For eval datasets, detect duplicate pairs.
- **Manifest verification:** compare current file paths, row counts, and MD5 hashes against
  `data/datasets_manifest.json`. On mismatch, re-download or repair the affected file, then
  rewrite the manifest. The manifest records every produced file with row count + MD5.

### 7.6 Tokenizer (shared, trained once)

Train ONE 30,522-token WordPiece tokenizer (BertPreTokenizer, `min_frequency=2`, special
tokens `[PAD] [UNK] [CLS] [SEP] [MASK]`) on the FineWeb-Edu **train** split only. Save
`data/tokenizer/tokenizer.json` + `tokenizer_config.json` loadable by
`PreTrainedTokenizerFast`. One shared tokenizer eliminates the tokenization confound: any
bias difference between architectures is attributable only to architecture. Non-negotiable.

---

## 8. STAGE 1 — DOES THE EFFECT EXIST? (publishable as a communication / short paper)

### 8.1 Research question

> Does cross-layer parameter sharing reduce stereotype preference at matched MLM validation
> loss, and does that hold across model scale?

Architectures: **Vanilla-BERT vs Looped-BERT only.** No ALBERT, no Hyperloop, no CWSA, no
GLUE, no StereoSet, no external calibration, no mechanistic suite. Sizes: Tiny, Small,
Base-ish. Same tokenizer, same data prefix, same LR, same effective depth.

### 8.2 Training

- One training run per `(architecture, size, seed)` to the Stage-1 max budget **200M
  tokens** (single-phase, seq=128 is sufficient for Stage 1). Save iso-loss-band snapshots
  (§6.4) AND token-marker snapshots at 50M/100M/200M during the single run.
- Seeds: run **seed 42 only** first to decide go/no-go; if the effect is visible, run seeds
  43 (and 44) before writing the paper. Configurable via `--seeds`.
- Uniform `LR = 5e-4`, AdamW (betas 0.9/0.98, eps 1e-6, weight_decay 0.01), linear schedule
  with 10% warmup, grad clip 1.0, BF16, gradient checkpointing, flash-attention, effective
  batch 64 via micro-batch + accumulation, OOM auto-halving micro-batch (floor 4).
- Checkpoint every 2000 steps (full RNG + optimizer + scheduler + token count); validate
  every 2000 steps on a fixed 5K-sample validation set; log loss/PPL/mask-accuracy and check
  iso-loss bands.

### 8.3 Bias evaluation

PLL only (SS-PLL deferred to Stage 2), on **Multi-CrowS-Pairs English** (overall + 9
categories + macro-average) and **Indian Bias English** (overall + 4 categories). Evaluate
at every iso-loss-band snapshot and at the token markers. Append every 50 rows, resumable.

### 8.4 Deliverables and decision

- **Headline figure:** stereotype preference (y) vs validation loss (x), with Vanilla and
  Looped as two curves, one panel per size (Tiny/Small/Base-ish), one figure per eval
  dataset. The effect = Looped curve below Vanilla curve at matched loss, at every size.
- **Secondary figure:** preference vs token budget (50M/100M/200M) endpoints — reported for
  completeness, explicitly NOT the primary claim.
- **Statistics:** bootstrap CIs always; paired permutation test (Vanilla vs Looped at the
  primary iso-loss band on Multi-CrowS-Pairs) once >= 2 seeds exist.
- **Go/no-go (printed):** GO if `Vanilla > Looped` at matched loss across >= 2 of 3 sizes;
  NO-GO/STOP if `Vanilla ~= Looped`.

### 8.5 Paper artifact (communication / short paper)

`analyze_stage1.py` must emit, into `results/stage1/` and `figures/`:
- the headline + secondary figures (PNG + PDF, no emoji, colorblind-safe),
- result tables as CSV **and** LaTeX,
- a `stage1_paper_outline.md`: title, finding-first abstract (with the matched-loss result
  filled in post-run), method (2 architectures, 3 sizes, iso-loss protocol), the India-centric
  instrument as the geographic novelty, limitations (scratch pretraining at 200M tokens; PLL
  construct-validity caveat to be strengthened in Stage 2), and the go/no-go statement.

Result CSV schemas: §12.

---

## 9. STAGE 2 — IS IT SHARING, AND IS IT ROBUST? (conference short paper / journal letter)

Run only if Stage 1 = GO. Retires the risk: "is it sharing specifically, and does it survive
a real validity check and real statistics?"

### 9.1 Additions over Stage 1

- **ALBERTLoopedBERT** as a third point -> sharing spectrum Vanilla (12 unique) -> Looped
  (6) -> ALBERT (1). Test whether bias is **monotone in sharing degree**.
- **SS-PLL** alongside PLL; report as robust only where both agree.
- **WinoBias** as a structurally different third instrument (convergent validity).
- **External calibration** (no training): run the identical bias pipeline on
  `bert-base-uncased`, `albert-base-v2`, `answerdotai/ModernBERT-base`. Flag
  `External_Calibration=true`. Use to (a) show the pipeline reproduces known published bias
  scores and (b) show scratch-ALBERT reproduces albert-base-v2's bias *pattern* across
  categories (validity anchor even at lower absolute quality).
- **GLUE** (SST-2 + RTE is sufficient): bias-vs-utility Pareto + quality screen (exclude
  configs with GLUE avg < 55%, logged).
- **Loop-wise bias trajectory** (the one mechanistic teaser that needs **no new training** —
  reuses checkpoints via forward hooks): track stereotype preference at each loop boundary
  for Looped (and Vanilla layers 3/6/9/12 for reference); classify trajectory shape
  (CONVERGENT / AMPLIFYING / OSCILLATING). Defer CKA, stream-disagreement, early-merge, token
  drift to Stage 3.
- Seeds: **3-5**. Token budget: up to 300-500M (single pool prefix). Add **one smaller scale
  check** alongside Base-ish to confirm the spectrum holds across scale.

### 9.2 Statistics (confirmatory)

Confirmatory family (Holm-Bonferroni applied here only): Vanilla vs Looped, Vanilla vs
ALBERT, Looped vs ALBERT, all at the primary metric (Multi-CrowS-Pairs PLL, iso-loss
matched). Everything else (per-category, Indian, WinoBias, SS-PLL convergence) is exploratory
(effect sizes + bootstrap CIs, uncorrected, labeled hypothesis-generating).

### 9.3 Deliverables and decision

- Monotone dose-response of bias in sharing degree at iso-loss, with significance, PLL/SS-PLL
  agreement, external-calibration validity, and the loop-trajectory teaser.
- **Go/no-go (printed):** GO if monotone (or at least `Vanilla > {Looped, ALBERT}`) and
  survives Holm with SS-PLL agreement; PAUSE if single-metric-only or collapses under
  correction.
- `analyze_stage2.py` emits figures (sharing-spectrum iso-loss plot; Pareto; loop trajectory;
  external-calibration table), CSV + LaTeX tables, and `stage2_paper_outline.md`.

---

## 10. STAGE 3 — MECHANISM + NOVEL ARCHITECTURE (full paper: TACL / Neurocomputing)

Run only if Stage 2 = GO. By now sharing-reduces-bias is established, so any Stage 3 failure
is unambiguously a Hyperloop/stream problem, not "maybe SCH is false."

### 10.1 Additions over Stage 2

- **HyperloopBERT + CWSA** (the novelty), compute-matched to Looped (§6.1). Full primary
  set: Vanilla, ALBERT, Looped, Hyperloop, Base-ish, 3-5 seeds, 500M tokens
  (400M @ seq=128 then 100M @ seq=256 for sequence-length adaptation), iso-loss snapshots.
- **Stream-count ablation trained from scratch, n in {1,2,4}** (the *causal* dose-response
  and strongest evidence): `hyperloop_n1` (must collapse to Looped — sanity check),
  `hyperloop_n2`, full `n=4`. 3 seeds each, 300M tokens @ seq=128. Permutation test n=4 vs
  n=1 is a confirmatory contrast.
- **Full mechanistic suite** (§ analyses):
  1. Loop-wise stereotype trajectory (Looped vs Hyperloop; does Hyperloop show a different
     trajectory type?).
  2. Loop-wise representation similarity via **CKA** (does multi-stream diversity prevent
     premature representational convergence?).
  3. Stream-disagreement <-> bias **correlation** (Hyperloop `stream_snapshots`; Pearson +
     Spearman; explicitly labeled correlational, not causal).
  4. **Early-merge** OOD intervention (`merge_at in {1,2,3}`; corroborating, not proof).
  5. **Demographic token drift** (curated terms per category; cosine drift across loops in
     stereotypical vs anti-stereotypical context; Looped vs Hyperloop).
- Full iso-loss bias evaluation on all snapshots (most expensive eval stage).
- StereoSet appendix; full GLUE (SST-2, MRPC, QNLI, RTE).

### 10.2 Confirmatory family (Holm-Bonferroni)

Vanilla vs Hyperloop (pre-registered primary), Vanilla vs Looped, Looped vs Hyperloop,
n=4 vs n=1 — all at Multi-CrowS-Pairs PLL, iso-loss matched. Everything else exploratory.

### 10.3 Deliverables

`Vanilla > Looped > Hyperloop` (or `Looped ~= Hyperloop`) at iso-loss, with a from-scratch
stream-count causal dose-response and the mechanistic story for *how* streams modulate bias,
plus the best bias-quality Pareto point. Full paper; Stages 1-2 fold in as the scaling and
sharing-spectrum sections. Even a null Hyperloop result is publishable (§0.3).
`analyze_stage3.py` emits all figures + tables (CSV + LaTeX) + `stage3_paper_outline.md`.

---

## 11. DRY_RUN FOLDER (`Dry_Run/`)

One dry run per stage folder (Stage1/2/3) plus one for the dataset stage. Each dry run must
**execute every model, every API/provider, and every pipeline step it covers on at least one
tiny instance**, and check code integrity. None of them train to convergence; all exit fast
and write a machine-readable report.

Each dry run performs:
1. **Env check:** `.env` present; per-provider key counts; `HF_KEY` works (download a 10-row
   slice of each dataset that stage uses). Never print key values.
2. **Code integrity:** import every module the stage uses; byte-compile all stage files
   (syntax check); verify expected functions/classes exist with expected signatures; assert
   no hardcoded key-like literals exist in any file under the stage and under `Dry_Run/`.
3. **Model smoke test:** instantiate **every** architecture used by the stage at Tiny size;
   run a forward pass, a backward pass, and a 5-step training loop; confirm CUDA + BF16 +
   flash-attention path is active (log which attention path was used).
4. **API/key/model test:** for **every** provider and **every** key, send one minimal
   JSON-returning request; confirm parseable JSON and a **valid model id**; report any
   invalid id (e.g. if `gemini-3-flash-preview` is not served) with the provider's error.
   No cross-tier fallback during the test.
5. **Eval smoke test:** run each bias metric and each analysis the stage uses on a handful of
   rows; confirm output CSVs have the exact schema from `io_schemas.py`.
6. **Iso-loss + integrity smoke test:** confirm band-crossing snapshot logic fires; run the
   dedup/corruption check on a tiny slice.
7. **GPU-hour estimate:** print a per-step throughput probe and an estimated GPU-hour budget
   for the full stage.
8. **Report:** write `Dry_Run/dry_run_report.json` with PASS/FAIL per junction; exit without
   full training. Top-level PASS only if every junction passes.

---

## 12. RESULT REGISTRATION — CANONICAL CSV SCHEMAS (full column names, defined in `io_schemas.py`)

Every result is registered in detail; never abbreviate. Add `Timestamp` to every row.
Common identity columns reused across files: `Stage`, `Architecture`, `Model_Size`,
`Hidden_Size`, `Seed`, `Unique_Parameters`, `Total_Parameters`, `Effective_Depth`,
`Shared_Ratio`.

- **MLM quality** `results/<stage>/mlm/summary_table.csv`:
  identity + `Validation_Loss`, `Pseudo_Perplexity`, `Mask_Accuracy`, `Tokens_Processed`,
  `Tokens_Per_Second`, `GPU_Hours`, `Token_Marker`, `Band`.
- **Bias per-example** `results/<stage>/bias/<dataset>_<arch>_<size>_seed<seed>_<band>_progress.csv`:
  `Row_Index`, `Dataset`, `Category`, `Sentence_Stereotypical`,
  `Sentence_AntiStereotypical`, `PLL_Stereotypical`, `PLL_AntiStereotypical`,
  `SS_PLL_Stereotypical`, `SS_PLL_AntiStereotypical`, `Effect_Size`, `Stereotype_Preferred`,
  identity columns, `Validation_Loss`, `Band`, `Token_Marker`, `External_Calibration`,
  `Needs_Review`, `Timestamp`.
- **Bias summary** `results/<stage>/bias/<dataset>_summary.csv`:
  identity + `Band`, `Token_Marker`, `Overall_Stereotype_Preference_Rate`,
  `Macro_Average_Preference_Rate`, per-category `<Category>_Preference_Rate`,
  `Mean_Effect_Size`, `Bootstrap_CI_Low`, `Bootstrap_CI_High`, `PLL_SS_PLL_Agreement`.
- **WinoBias** `results/<stage>/bias/winobias_summary.csv`:
  identity + `Pro_Stereotype_Accuracy`, `Anti_Stereotype_Accuracy`, `Pro_Anti_Gap`.
- **External calibration** `results/<stage>/bias/external_calibration.csv`:
  `Model_Name`, dataset/category columns as above, `External_Calibration=true`.
- **GLUE** `results/<stage>/glue/summary_table.csv`:
  identity + `Task`, `Accuracy`, `F1`, `GLUE_Average`.
- **Iso-loss index** `results/<stage>/iso_checkpoints/index.csv`:
  identity + `Band`, `Snapshot_Path`, `Validation_Loss_At_Snapshot`, `Crossed_At_Step`.
- **Mechanistic (Stage 3)**: `bias_trajectory.csv` (identity + `Dataset`, `Category`,
  `Loop_Depth`, `Mean_Preference_Rate`, `Std_Preference_Rate`, `Mean_Effect_Size`,
  `Trajectory_Shape`), `representation_similarity.csv` (CKA pairs),
  `stream_disagreement.csv` (`Loop_Depth`, `Stream_Disagreement`, `Effect_Size`,
  `Pearson_R`, `Pearson_P`, `Spearman_R`, `Spearman_P`),
  `early_merge_intervention.csv` (`Merge_At`, bias columns),
  `demographic_token_drift.csv` (`Category`, `Demographic_Term`, `Context_Type`,
  `Loop_Depth`, `Cosine_Drift`).
- **Stream ablation (Stage 3)** `results/stage3/ablations/stream_count_ablation.csv`:
  identity + `Stream_Count`, `Band`, bias columns, std across seeds.
- **Stats** `results/<stage>/stats/confirmatory_family.csv`:
  `Contrast`, `Metric`, `Dataset`, `Band`, `Raw_P_Value`, `Holm_Corrected_P_Value`,
  `Cohens_D`, `Significant_At_0.05`; plus `exploratory_results.csv` (uncorrected, labeled).

---

## 13. CITATIONS TO EMBED IN CODE COMMENTS (place above the relevant implementation)

Use a clearly delimited comment block so nothing is missed at writing time. Verify arXiv ids
at build time where shown as approximate.

```
# CITATION: Devlin, J. et al. (2019). BERT. NAACL.                      [VanillaBERT baseline]
# CITATION: Lan, Z. et al. (2020). ALBERT: A Lite BERT. ICLR.           [ALBERT sharing; no factored embedding here]
# CITATION: Saunshi, N. et al. (2025). Reasoning with Latent Thoughts:
#           On the Power of Looped Transformers. arXiv.                 [memorization-reasoning tradeoff = SCH basis]
# CITATION: Bae, S. et al. (2025). Mixture-of-Recursions. arXiv:2507.10524.  [LoopedBERT positioning]
# CITATION: Geiping, J. et al. (2025). Huginn recurrent-depth. arXiv:2502.05171. [recurrent-depth pretraining]
# CITATION: Zeitoun, A., Torroba-Hennigen, L., & Kim, Y. (2026).
#           Hyperloop Transformers. arXiv:2604.21254. MIT.             [HyperloopBERT base; ours = encoder-only + CWSA]
# SUPPORT:  Zhu, R.-J. et al. (2025). Ouro. arXiv:2510.25741; Frey, M. et al. (2026). arXiv:2603.08391.
#           [SCH support: manipulation not storage; per-parameter memorization preserved.
#            arXiv:2603.08391 is Frey et al. 2026, NOT Zhu -- prior attribution was an error.]
# CITATION: Voria, G. et al. (2026). Tracing Stereotypes: From Biased Neurons to Fairer Models.
#           arXiv:2601.05663.                                          [mechanistic ally: stereotypes localize to neurons]
# CITATION: Nangia, N. et al. (2020). CrowS-Pairs. EMNLP.              [Multi-CrowS-Pairs base]
# CITATION: Khandelwal, K. et al. (2023). Indian-BhED. arXiv:2309.08573.[Indian bias instrument base]
# CITATION: Zhao, J. et al. (2018). Gender Bias in Coreference (WinoBias). NAACL.
# CITATION: Blodgett, S.L. et al. (2021). Stereotyping Norwegian Salmon. ACL. [PLL validity; SS-PLL motivation]
# CITATION: Nadeem, M. et al. (2021). StereoSet. ACL.                   [appendix only]
# DATASET WARNING: contains stereotypical content by design; research/fairness-audit use only.
```

Also embed verbatim, where indicated: the SCH support/positioning block (above the architecture
definitions), the pre-registered primary endpoint block (above the bias evaluation), the
benchmark-validity disclaimer, and the ethical-use notice.

---

## 14. README.md REQUIREMENTS (one-stop solution)

README must let anyone reproduce everything from scratch. Sections:
1. Title, finding-first abstract (placeholders filled per stage), and the 3-stage de-risking
   rationale with the go/no-go gates.
2. Hardware (L4 24GB, sm_89), OS, Python 3.12; the terraform-compatibility note (§3.3).
3. Exact install instructions = the `install.sh` recipe (§3.1); no venv, global environment.
4. The full `.env` contract (§4) with key names and provider tiers; reminder that no key is
   ever hardcoded and tests load from environment only.
5. The LLM utility contract (§5): provider tiers, round-robin, **no cross-tier fallback**,
   JSON-first (no separate judge), and how to swap a model id if a dry run reports it invalid.
6. Repository layout (§1) including why `common/` exists.
7. How to run, in order: dataset stage -> Stage 1 dry run -> Stage 1 -> read go/no-go ->
   Stage 2 -> ... -> Stage 3. Include each script's flags (`--dry-run`, `--seeds`,
   `--dataset-namespace`, resume behavior).
8. Dataset citations in BibTeX; the Indian-dataset provenance report; contamination-removal
   note; integrity/dedup behavior on rerun.
9. Result file tree with the canonical CSV schemas (§12).
10. Per-stage GPU-hour estimates and what each stage publishes.
11. Limitations: English-only primary analysis; scratch pretraining (not SOTA quality);
    PLL/SS-PLL construct validity addressed but not fully resolved; iso-loss bands valid only
    within all models' shared convergence range.
12. The full citation block (§13).

No emoji anywhere in the README.

---

## 15. CODING STANDARDS (apply throughout)

- Production-grade: handle exceptions, log errors with context, retry transient I/O on the
  same resource, checkpoint frequently, never lose progress over multi-day unattended runs.
- No emoji in any code, comment, log line, figure label, or generated text.
- No hardcoded secrets anywhere; `Dry_Run/` and any test file must load keys from environment
  only. A literal key string in any file is a defect to remove.
- Detailed logging to `logs/` (rotating) and console; log throughput, GPU memory, band
  crossings, integrity counts, and which attention path is active.
- All RNG seeds set and logged.
- Every CSV uses the canonical full column names from `io_schemas.py`.
- Resume everywhere via `checkpoints/pipeline_state.json`; skip completed work on restart.
- Integrity check (dedup + corruption + manifest) at the start of every stage and rerun.
- Citations as comment blocks above implementations (§13).

---

## 16. RUNBOOK (the order the user will actually run)

```
# 0. Install (global env, no venv)
bash install.sh

# 1. Dataset stage (once): download, integrity, contamination, provenance, tokenizer, manifest
python3 Dry_Run/dry_run_dataset.py          # verify HF access, slices, integrity, env
python3 Dataset/download_training_corpus.py
python3 Dataset/download_eval_datasets.py
python3 Dataset/contamination_filter.py
python3 Dataset/build_provenance_report.py
python3 Dataset/train_tokenizer.py
python3 Dataset/validate_and_manifest.py

# 2. Stage 1 (the gate)
python3 Dry_Run/dry_run_stage1.py           # models + APIs + pipeline + integrity, all PASS
python3 Stage1/train_stage1.py --seeds 1    # seed 42 only, to decide
python3 Stage1/eval_bias_stage1.py
python3 Stage1/analyze_stage1.py            # prints GO / NO-GO; writes figures, tables, outline
# if GO: rerun with --seeds 3 before writing the communication / short paper

# 3. Stage 2 (only if Stage 1 GO)
python3 Dry_Run/dry_run_stage2.py
python3 Stage2/train_stage2.py --seeds 3
python3 Stage2/eval_bias_stage2.py
python3 Stage2/eval_glue_stage2.py
python3 Stage2/external_calibration_stage2.py
python3 Stage2/loop_trajectory_stage2.py
python3 Stage2/analyze_stage2.py            # prints GO / PAUSE

# 4. Stage 3 (only if Stage 2 GO)
python3 Dry_Run/dry_run_stage3.py
python3 Stage3/train_stage3.py --seeds 3
python3 Stage3/train_stream_ablation_stage3.py --ablation-seeds 3
python3 Stage3/eval_bias_stage3.py
python3 Stage3/mechanistic_stage3.py
python3 Stage3/analyze_stage3.py            # full-paper figures, tables, outline
```

---

## 17. FINAL CHECKLIST FOR THE CODING AGENT

- [ ] Five required folders present (Dry_Run, Dataset, Stage1, Stage2, Stage3) plus a
      documented `common/` shared package.
- [ ] No venv anywhere; global environment install via the exact §3.1 recipe.
- [ ] Flash-Attention 2 wired into custom attention with SDPA/eager fallback; BF16 only;
      attention path logged; terraform/runtime compatibility checked (§3.3).
- [ ] `.env` contract (§4) implemented; case-insensitive key resolution; no key values
      logged; **no hardcoded keys anywhere, including dry-run/test files**.
- [ ] LLM utility (§5): Gemini primary (4 keys), DeepSeek secondary (2), Mistral tertiary
      (2), OpenRouter alternate (2); round-robin within a provider; **no cross-tier
      fallback**; JSON-first with deterministic parsing; **no separate LLM judge required**.
- [ ] Every dry run executes all models + all providers/keys + the full pipeline once and
      checks code integrity and model-id validity; writes `dry_run_report.json`.
- [ ] Dataset stage: shared tokenizer (once), shared training pool (consumed by prefix per
      stage), contamination removal, Indian provenance report, manifest with MD5 + row counts.
- [ ] Integrity (dedup + corruption + manifest) runs on every rerun before consuming data.
- [ ] Iso-loss snapshotting implemented with adaptive bands; primary comparison is at matched
      validation loss, endpoint comparison is secondary.
- [ ] Stage 1 = Vanilla vs Looped, 3 sizes, PLL, iso-loss curve, prints GO/NO-GO, emits a
      communication/short-paper outline + figures + CSV/LaTeX tables.
- [ ] Stage 2 = + ALBERT (sharing spectrum), SS-PLL, WinoBias, external calibration, GLUE
      screen, loop-trajectory teaser, confirmatory stats; prints GO/PAUSE.
- [ ] Stage 3 = + Hyperloop+CWSA, from-scratch stream ablation n in {1,2,4}, full mechanistic
      suite, full iso-loss eval; full-paper outputs.
- [ ] All results registered with full column names from `io_schemas.py`; `Timestamp` on
      every row; bias evaluations append every 50 rows and resume.
- [ ] Citations as comment blocks above every implemented paper/dataset/benchmark (§13).
- [ ] README.md is a complete one-stop guide (§14). No emoji anywhere.
```
```
