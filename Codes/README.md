# Parameter Sharing Reduces Stereotype Memorization

A controlled, scaled, mechanistic study of looped and hyper-connected encoder
architectures. The contribution is the FINDING that cross-layer parameter
sharing reduces stereotype encoding at matched model quality (evidence for the
Stereotype Consolidation Hypothesis, SCH); the Hyperloop architecture is the
mechanistic vehicle used to explain how, not the contribution itself.

Target venues: communication / workshop short paper (Stage 1), conference
short paper or journal letter (Stage 2), full paper for TACL (primary) or
Neurocomputing (Elsevier, fallback) after Stage 3.

## The three-stage de-risking design

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Question | Does the effect exist? | Is it sharing, and is it robust? | Mechanism + novel architecture |
| Architectures | Vanilla vs Looped | + ALBERT (sharing spectrum) | + Hyperloop + CWSA, stream ablation n in {1,2,4} |
| Gate | GO if Vanilla > Looped at matched loss across >= 2 of 3 sizes | GO if monotone in sharing degree, survives Holm, PLL/SS-PLL agree | Publishable even on a null Hyperloop result |

The one non-negotiable in every stage: iso-loss (matched validation-loss)
comparison. Endpoint comparison alone is attackable ("the shared model only
looks less biased because it is weaker"); comparing at matched validation loss
removes model quality as the alternative explanation. Validation masking is
FIXED (single dedicated RNG seed, common/train_loop.py) so every architecture,
size, and seed is validated on the identical masked token set. Validation runs
on a fine cadence (every 500 steps) once a model is within 0.15 nat of the
next uncrossed band, each crossing is registered with its ACTUAL loss in
results/<stage>/iso_checkpoints/index.csv, and every contrast audits the
per-seed loss gap against ISO_LOSS_TOLERANCE.

Statistical protocol: the PRIMARY test everywhere is an item-level paired
permutation (sentence pairs are the resampling unit, per-item Effect_Size
deltas averaged across seeds) -- real power at 3 seeds. Seed-level contrasts
are robustness checks only: with 3 seeds their minimum achievable p is
1/2^3 = 0.125 and they must never gate a decision. A capability gate requires
VanillaBERT (base) to show statistically detectable baseline bias before any
contrast is interpreted (a model that never learned the stereotypes cannot be
shown to "reduce" them).

## Hardware and environment

- One NVIDIA L4 (24 GB, sm_89). All jobs run sequentially on the single GPU.
- Linux x86-64, Python 3.12, CUDA 12.x driver, BF16 (never FP16).
- FlashAttention-2 (varlen, padding-aware) with SDPA -> eager fallback; the
  active attention path is logged once per model build.
- No venv, no conda: the global Python environment only.

Install: `bash install.sh` (exact pinned recipe + verification probe).
If no terraform files are present, compatibility is verified by runtime probes
(GPU compute capability, Python version, torch CUDA version).

## Secrets (.env contract)

All keys load from `.env` via `common/env_loader.py` (case-insensitive names,
whitespace trimmed, empty = missing). No key is ever hardcoded or logged.
See `.env.example` for every key name: `HF_KEY` (alias `HF_TOKEN`),
`GCP_KEY1..4` (Gemini, primary), `DEEPSEEK_KEY1..2` (secondary),
`MISTRAL_KEY1..2` (tertiary), `OPENROUTER_KEY1..2` (alternate router).

LLM utility (`common/llm_utils.py`): round-robin WITHIN a provider only,
NO automatic cross-tier fallback (switching judges mid-run would contaminate
results), JSON-first responses parsed by a deterministic extractor -- no LLM
judge. The core bias metrics (PLL, SS-PLL, WinoBias) are model-intrinsic and
use no LLM at all.

## Repository layout

```
common/     shared library (architectures, attention, metrics, stats, iso-loss,
            io_schemas, train_loop, integrity, env/llm utils, plotting)
Dataset/    download + contamination filter + tokenizer + provenance + manifest
Stage1/     Vanilla vs Looped, 3 sizes, PLL, iso-loss curve, GO/NO-GO
Stage2/     + ALBERT, SS-PLL, WinoBias, GLUE screen, external calibration,
            loop-trajectory teaser, confirmatory stats (Holm)
Stage3/     + Hyperloop + CWSA, stream ablation, mechanistic suite, full paper
Dry_Run/    per-stage smoke tests, all exit fast, write dry_run_report.json
data/ datasets_eval/ checkpoints/ models/ results/ figures/ logs/   (gitignored)
```

`common/` exists so shared architecture/metric/utility code is never
duplicated across stages -- duplication would let the stages drift apart and
break cross-stage comparability. One shared WordPiece tokenizer (30,522 tokens,
trained once on the FineWeb-Edu train split) eliminates the tokenization
confound entirely.

## How to run

```
bash install.sh

# 1. Dataset stage (once)
python3 Dry_Run/dry_run_dataset.py
python3 Dataset/download_training_corpus.py
python3 Dataset/download_eval_datasets.py     # also builds WinoBias pronoun CSVs
python3 Dataset/contamination_filter.py
python3 Dataset/build_provenance_report.py
python3 Dataset/train_tokenizer.py
python3 Dataset/validate_and_manifest.py

# 2. Stage 1 (the gate)
python3 Dry_Run/dry_run_stage1.py
python3 Stage1/train_stage1.py --seeds 1      # seed 42 only, to decide
python3 Stage1/eval_bias_stage1.py
python3 Stage1/analyze_stage1.py              # prints GO / NO-GO
# if GO: rerun with --seeds 3 before writing the paper

# 3. Stage 2 (only if Stage 1 GO)
python3 Dry_Run/dry_run_stage2.py
python3 Stage2/train_stage2.py --seeds 3
python3 Stage2/eval_bias_stage2.py
python3 Stage2/eval_glue_stage2.py
python3 Stage2/external_calibration_stage2.py
python3 Stage2/loop_trajectory_stage2.py
python3 Stage2/analyze_stage2.py              # prints GO / PAUSE

# 4. Stage 3 (only if Stage 2 GO)
python3 Dry_Run/dry_run_stage3.py
python3 Stage3/train_stage3.py --seeds 3      # primary set + stream-count ablations (equal budgets)
python3 Stage3/eval_bias_stage3.py
python3 Stage2/loop_trajectory_stage2.py --stage 3
python3 Stage3/stream_analysis_stage3.py      # disagreement, CKA, EVAL-TIME early merge, token drift
python3 Stage3/analyze_stage3.py              # confirmatory family + full-paper outputs
```

Design decisions vs the original spec (documented deviations):
- All stream-count arms (n = 1/2/4) train at the SAME 400M-token budget, so
  the dose-response varies only stream count -- never budget.
- EarlyMerge is applied at EVAL TIME to the trained 4-stream model
  (merge_at in {1,2,3}); it is never trained, and never labeled causal.
- The seq=256 adaptation tail is omitted (single-L4 budget); all training is
  seq=128, stated as a limitation.
- roberta-base replaces ModernBERT as the third calibration anchor
  (ModernBERT needs transformers>=4.48; the project pins 4.46).

Common flags: `--dry-run`, `--seeds N`, `--sizes ...`, `--dataset-namespace ORG`,
`--resume`.

Resume behaviour: training checkpoints (model + optimizer + scheduler + RNG +
iso-band tracker + data cursor) are written every 2000 optimizer steps to
`checkpoints/<stage>/<run_id>/latest.pt`; `--resume` continues mid-run from
the last checkpoint and skips completed runs via
`results/<stage>/pipeline_state.json`. Bias evaluations append every 50 rows
with a resumable `Row_Index`; summary rows are deduplicated unconditionally so
re-runs can never corrupt seed pairing.

## Datasets

- Training: FineWeb-Edu `sample-10BT` (streamed prefix; one shared pool for
  all stages, each stage consumes a prefix of it).
- Multi-CrowS-Pairs English (Nangia et al. 2020 + extension) -- primary.
- Indian Multilingual Bias (Khandelwal et al. 2023, Indian-BhED base) -- the
  India-centric instrument; provenance report embedded in the paper.
- WinoBias (Zhao et al. 2018), scored with the masked-pronoun protocol
  (Kurita et al. 2019) -- convergent validity (Stage 2+).
- GLUE SST-2/RTE (+ MRPC/QNLI in Stage 3) -- bias-vs-utility Pareto + screen.
- StereoSet -- appendix only.

Contamination removal: training documents containing an 8-gram (word-level,
lowercased) overlap with any evaluation sentence are removed before training.
(The spec's minimum was 4 contiguous words; 8-grams are used because 4-gram
matching removes large volumes of benign boilerplate -- deviation documented
here deliberately.) Integrity (corruption + dedup + MD5 manifest) runs on
every rerun before data is consumed.

## Result files

All CSVs use full, unabbreviated column names defined once in
`common/io_schemas.py` (never abbreviate; `Timestamp` on every row).
Key files per stage under `results/<stage>/`: `mlm/summary_table.csv`,
`bias/<dataset>_summary.csv` (+ per-example `*_progress.csv`),
`bias/winobias_summary.csv`, `glue/summary_table.csv`,
`mechanistic/loop_trajectory.csv`, `mechanistic/stream_disagreement.csv`,
`stats/confirmatory_family.csv`, `stats/exploratory_stats.csv`.

## GPU-hour estimates

Stage 1 ~40-80 h; Stage 2 ~150-200 h; Stage 3 ~250-300 h (L4, sequential).
Each dry run prints a per-step throughput probe and a stage estimate.

## Limitations

- English-only primary analysis.
- Scratch pretraining at 200-500M tokens: models differentiate architecture
  effects but are far from SOTA quality; claims are architecture contrasts at
  matched loss, never absolute quality claims.
- PLL/SS-PLL construct validity (Blodgett et al. 2021) is addressed via
  SS-PLL agreement and WinoBias convergence, not fully resolved.
- Iso-loss bands are valid only within the convergence range shared by all
  models in a stage.
- EarlyMerge is an out-of-distribution intervention -- corroborating evidence,
  not causal proof. The causal claim rests on the from-scratch stream-count
  dose-response.

## Citations

```
Devlin, J. et al. (2019). BERT. NAACL.                       [VanillaBERT]
Lan, Z. et al. (2020). ALBERT. ICLR.                         [ALBERT sharing]
Saunshi, N. et al. (2025). Reasoning with Latent Thoughts:
  On the Power of Looped Transformers. arXiv.                [SCH basis]
Bae, S. et al. (2025). Mixture-of-Recursions: Learning Dynamic
  Recursive Depths for Adaptive Token-Level Computation.
  arXiv:2507.10524.                          [adaptive recursion; LoopedBERT positioning]
Geiping, J. et al. (2025). Scaling up Test-Time Compute with
  Latent Reasoning: A Recurrent Depth Approach (Huginn-3.5B).
  arXiv:2502.05171.                          [canonical LM-scale recurrent-depth pretraining]
Zeitoun, A., Torroba-Hennigen, L., & Kim, Y. (2026).
  Hyperloop Transformers. arXiv:2604.21254.                  [HyperloopBERT base]
Zhu, R.-J. et al. (2025). Ouro: Scaling Latent Reasoning via
  Looped Language Models. arXiv:2510.25741.  [SCH support: looped gains from knowledge
                                              MANIPULATION, not increased storage]
Frey, M. et al. (2026). Adaptive Loops and Memory in Transformers:
  Think Harder or Know More? arXiv:2603.08391.  [SCH support: looping aids manipulation,
                                              not storage; per-parameter memorization preserved]
Voria, G. et al. (2026). Tracing Stereotypes in Pre-trained
  Transformers: From Biased Neurons to Fairer Models.
  arXiv:2601.05663.                          [mechanistic ally: stereotypes localize to small
                                              neuron subsets in BERT -- replaces older intrinsic
                                              bias-localization references]
Nangia, N. et al. (2020). CrowS-Pairs. EMNLP.                [Multi-CrowS-Pairs]
Khandelwal, K. et al. (2023). Indian-BhED. arXiv:2309.08573. [Indian instrument]
Zhao, J. et al. (2018). WinoBias. NAACL.
Kurita, K. et al. (2019). Measuring Bias in Contextualized
  Word Representations. GeBNLP @ ACL.                        [masked-pronoun protocol]
Blodgett, S.L. et al. (2021). Stereotyping Norwegian Salmon.
  ACL.                                                       [SS-PLL motivation]
Wang, A., Phan, M., Ho, D.E., & Koyejo, S. (2025). Fairness
  through Difference Awareness: Measuring Desired Group
  Discrimination in LLMs. ACL 2025 (Best Paper).
  arXiv:2502.01926.                        [CONSTRUCT SCOPE: our PLL preference rate is a
                                            CORRELATION benchmark in their descriptive /
                                            normative / correlation taxonomy. We do NOT
                                            claim difference-aware fairness; see the
                                            limitations + future-work notes.]
Nadeem, M. et al. (2021). StereoSet. ACL.                    [appendix only]
Phipson, B. & Smyth, G.K. (2010). Permutation p-values
  should never be zero. SAGMB.                               [+1 correction]
```

DATASET WARNING: the evaluation datasets contain stereotypical content by
design. Research and fairness-audit use only.
