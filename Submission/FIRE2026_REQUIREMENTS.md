# FIRE 2026 — Submission Requirements (fetched 2026-07-28)

Source: https://fire.irsi.org.in/fire/2026/call_for_papers

## Hard constraints

| Item | Requirement |
|---|---|
| **Deadline** | **2026-08-15** (notification 2026-10-15, camera-ready 2026-10-31) |
| **Template** | ACM ICPS, 2-column |
| **Documentclass** | `\documentclass[sigconf,natbib=true,anonymous=true]{acmart}` |
| **Page limit** | **max 9 pages of content**, references excluded |
| **Review** | Regular Papers: **DOUBLE-BLIND** (Perspective also double-blind; Resource/Demo single-blind) |
| **Submission system** | Microsoft CMT3 — https://cmt3.research.microsoft.com/FIRE2026 |
| **Format** | PDF only |
| **Required metadata** | ACM CCS concepts + keywords |

## DOUBLE-BLIND — anonymisation checklist (CRITICAL)

Because Regular Papers are double-blind, the submitted PDF must NOT contain:

- [ ] Author names / affiliations (`anonymous=true` handles the header, but check the body)
- [ ] **The Hugging Face repo `Debk/HyperloopBERT`** — the username identifies the author
- [ ] **The GitHub repo `DevDaring/HyperloopBert`** — same problem
- [ ] The `Debk/...` dataset namespace used by `--dataset-namespace`
- [ ] Any acknowledgements, grant numbers, or institution names
- [ ] Self-citations phrased as "our previous work [12]" — use third person

**Mitigation:** the pre-registration already anticipates this (spec 7.2 supports an
anonymised mirror via `--dataset-namespace`). For submission, host an anonymised copy
(e.g. an anonymous.4open.science link or a fresh anonymous HF account) and cite that.
Restore the real links in the camera-ready.

## Topical fit

The CFP explicitly lists **"explainability and fairness"** among topics of interest, and
FIRE has a strong multilingual / Indian-language focus. Our work is a fairness-evaluation
methodology paper, which fits the fairness track directly.

Note: the India-centric instrument was dropped from the empirical run (the available mirror
is Bengali, unusable by an English-only tokenizer — see BERT_Findings.md §2.2 bug 11).
Given FIRE's regional focus this should be stated explicitly as a limitation and as
future work, not quietly omitted.

## Files

- `FIRE_HyperloopBERT.tex` — the paper
- `reference.bib` — bibliography (conference/journal versions preferred over arXiv)
- `images/` — figures
