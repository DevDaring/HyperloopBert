# REFERENCES_FINAL_AUDIT

Every record flagged in the final review was checked against the official arXiv
listing. All four flags were correct; all four are now fixed.

| Ref | Field | Previous record | Verified source | Correction applied |
|---|---|---|---|---|
| `xie2025mhc` | first author | Xie, **Zhijian** | arXiv:2512.24880 lists **Zhenda Xie** (20 authors) | corrected, and the first seven authors are now named instead of `and others` |
| `xie2025mhc` | title casing | `{MHC}` | listing shows **mHC** | corrected to `{mHC}` |
| `zeitoun2026hyperloop` | first author | Zeitoun, **Abdelrahman** | arXiv:2604.21254 lists **Abbas Zeitoun** | corrected |
| `zhu2025ouro` | title | *Ouro: Scaling Latent Reasoning via Looped Language Models* | arXiv:2510.25741 title is **Scaling Latent Reasoning via Looped Language Models**; "Ouro" is the model family, named only in the abstract | title corrected; a comment records the model name |
| `frey2026adaptive` | venue | arXiv only | presented at the **LIT Workshop, ICLR 2026** | `note` field added; kept as arXiv record since the workshop version is not separately indexed |

## Method

Each record was checked by fetching the arXiv abstract page directly and reading
the author list and title as displayed. No venue metadata was inferred, and no
record was upgraded without a source that states the venue.

## Status of the remaining entries

The four venue upgrades made in the previous revision (Saunshi → ICLR 2025,
Bae → NeurIPS 2025, Khandelwal → ACM GoodIT 2024, Wang → ACL 2025) and the two
IR-fairness additions (Ekstrand → FnTIR 16(1–2):1–177, Singh & Joachims →
KDD 2018 pp. 2219–2228) were each verified against the publisher record at the
time they were added and are unchanged.

Entries still carried as arXiv-only, with no peer-reviewed version located:
`turc2019wellread`, `geiping2025huginn`, `zhu2025ouro`, `frey2026adaptive`,
`zeitoun2026hyperloop`, `xie2025mhc`, `hooker2020characterising`,
`xu2022compression`, `voria2026tracing`. None is load-bearing for a claim; all
are cited as related work. These should be re-checked at camera-ready.

## Lesson recorded

Three of the four errors were in author names or titles of recent preprints that
had been entered from memory rather than copied from the listing. Every future
entry is to be transcribed from the source page.
