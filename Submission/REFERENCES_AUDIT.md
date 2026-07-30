# REFERENCES_AUDIT

Every entry in `reference.bib` checked for existence, venue and citation use.
Where a published version exists it is cited in place of the preprint.
**No entry was written from memory without verification; nothing was invented.**

| Key | Type | Title | Status | Cited | Note |
|---|---|---|---|---|---|
| `devlin2019bert` | inproceedings | BERT}: Pre-training of Deep Bidirectional Transforme | standard | yes | canonical venue, widely indexed |
| `lan2020albert` | inproceedings | ALBERT}: A Lite {BERT} for Self-supervised Learning  | standard | yes | canonical venue, widely indexed |
| `turc2019wellread` | article | Well-Read Students Learn Better: On the Importance o | arXiv only | NOT CITED | no peer-reviewed version found; re-check at camera-ready |
| `nangia2020crows` | inproceedings | CrowS-Pairs}: A Challenge Dataset for Measuring Soci | standard | yes | canonical venue, widely indexed |
| `zhao2018winobias` | inproceedings | Gender Bias in Coreference Resolution: Evaluation an | standard | yes | canonical venue, widely indexed |
| `blodgett2021salmon` | inproceedings | Stereotyping {N}orwegian Salmon: An Inventory of Pit | standard | yes | canonical venue, widely indexed |
| `nadeem2021stereoset` | inproceedings | StereoSet}: Measuring Stereotypical Bias in Pretrain | standard | yes | canonical venue, widely indexed |
| `kurita2019measuring` | inproceedings | Measuring Bias in Contextualized Word Representation | standard | yes | canonical venue, widely indexed |
| `wang2025difference` | inproceedings | Fairness through Difference Awareness: Measuring Des | **verified** | yes | ACL 2025 Long Papers, pp. 6867-6893 — verified (Best Paper) |
| `khandelwal2023indianbhed` | inproceedings | Indian-BhED}: A Dataset for Measuring India-Centric  | **verified** | yes | ACM GoodIT 2024, doi 10.1145/3677525.3678666 — verified |
| `saunshi2025looped` | inproceedings | Reasoning with Latent Thoughts: On the Power of Loop | **verified** | yes | ICLR 2025 — verified via proceedings + DBLP |
| `geiping2025huginn` | article | Scaling up Test-Time Compute with Latent Reasoning:  | arXiv only | yes | no peer-reviewed version found; re-check at camera-ready |
| `zhu2025ouro` | article | Ouro: Scaling Latent Reasoning via Looped Language M | arXiv only | yes | no peer-reviewed version found; re-check at camera-ready |
| `frey2026adaptive` | article | Adaptive Loops and Memory in Transformers: Think Har | arXiv only | yes | no peer-reviewed version found; re-check at camera-ready |
| `bae2025mor` | inproceedings | Mixture-of-Recursions: Learning Dynamic Recursive De | **verified** | yes | NeurIPS 2025 poster — verified |
| `zeitoun2026hyperloop` | article | Hyperloop Transformers | arXiv only | yes | no peer-reviewed version found; re-check at camera-ready |
| `zhu2025hyperconnections` | inproceedings | Hyper-Connections | standard | yes | canonical venue, widely indexed |
| `xie2025mhc` | article | MHC}: Manifold-Constrained Hyper-Connections | arXiv only | yes | no peer-reviewed version found; re-check at camera-ready |
| `hooker2020characterising` | article | Characterising Bias in Compressed Models | arXiv only | yes | no peer-reviewed version found; re-check at camera-ready |
| `xu2022compression` | article | Can Model Compression Improve {NLP} Fairness? | arXiv only | yes | no peer-reviewed version found; re-check at camera-ready |
| `ramesh2023comparative` | inproceedings | A Comparative Study on the Impact of Model Compressi | standard | yes | canonical venue, widely indexed |
| `voria2026tracing` | article | Tracing Stereotypes in Pre-trained Transformers: Fro | arXiv only | yes | no peer-reviewed version found; re-check at camera-ready |
| `dao2022flashattention` | inproceedings | FlashAttention}: Fast and Memory-Efficient Exact Att | standard | yes | canonical venue, widely indexed |
| `kornblith2019similarity` | inproceedings | Similarity of Neural Network Representations Revisit | standard | NOT CITED | canonical venue, widely indexed |
| `geiping2023cramming` | inproceedings | Cramming: Training a Language Model on a Single {GPU | standard | NOT CITED | canonical venue, widely indexed |
| `phipson2010permutation` | article | Permutation {P}-values Should Never Be Zero: Calcula | standard | yes | canonical venue, widely indexed |
| `wolf2020transformers` | inproceedings | Transformers: State-of-the-Art Natural Language Proc | standard | NOT CITED | canonical venue, widely indexed |
| `penedo2024fineweb` | inproceedings | The {FineWeb} Datasets: Decanting the Web for the Fi | standard | yes | canonical venue, widely indexed |
| `wang2018glue` | inproceedings | GLUE}: A Multi-Task Benchmark and Analysis Platform  | standard | yes | canonical venue, widely indexed |
| `ekstrand2022fairness` | article | Fairness in Information Access Systems | **verified** | yes | FnTIR 16(1-2):1-177, doi 10.1561/1500000079 — verified |
| `singh2018fairness` | inproceedings | Fairness of Exposure in Rankings | **verified** | yes | KDD 2018, pp. 2219-2228, doi 10.1145/3219819.3220088 — verified |

## Upgrades applied in this revision

Four entries were moved from preprint to published venue after verification:
Saunshi et al. (arXiv:2502.17416 → ICLR 2025), Bae et al. (arXiv:2507.10524 →
NeurIPS 2025), Khandelwal et al. (arXiv:2309.08573 → ACM GoodIT 2024 with DOI),
and Wang et al. (arXiv:2502.01926 → ACL 2025). Two IR-fairness references were
added for Rule 24 and both were verified against the publisher record before
being written into the file.

## A defect found and fixed

BibTeX does not permit `%` comments **inside** an entry. Fourteen such comment
lines were silently causing BibTeX to skip their entries, so several references
were being dropped from the bibliography without any error surfacing. All were
moved above their entries. The bibliography now resolves 27 of 27 citations with
zero undefined references.

## Remaining risk

Nine entries have no peer-reviewed version at the time of writing and are kept as
arXiv records, flagged in the file. Four of them are 2025/2026 preprints in a
fast-moving area and should be re-checked at camera-ready. None is load-bearing
for a claim in the paper; all are cited as related work.
