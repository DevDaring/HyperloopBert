# ANONYMITY_AUDIT — FIRE 2026 double-blind

Checked on the **rendered PDF text**, not only the source, since that is what a
reviewer receives.

| Check | Method | Result |
|---|---|---|
| `\documentclass` carries `anonymous=true` | source grep | present |
| acmart anonymous mode actually active | `Anon.` running head in rendered PDF | **confirmed present** |
| Author names / affiliations | pdftotext grep | none; author block is `Anonymous Author(s)` |
| `Debk` (HF namespace) | pdftotext grep | 0 hits |
| `DevDaring` (GitHub owner) | pdftotext grep | 0 hits |
| `huggingface` / `github` | pdftotext grep | 0 hits |
| Any `http` URL | pdftotext grep | 0 hits |
| Acknowledgements / grant / funded | pdftotext grep | 0 hits |
| Personal names | pdftotext grep | 0 hits |
| Self-citation in first person | source review | none; no self-citations present |
| CCS concepts + keywords render | rendered PDF | both present |
| Content pages ≤ 9 | pdfinfo + references location | **5 content pages**, references begin p.6 |

## Deliberate wording choices for anonymity

- The word "released" was removed where it implied a named public artifact.
- The Reproducibility section refers to "an anonymised archive for review" and
  carries a `% CAMERA-READY: restore the concrete repository and checkpoint URLs
  here.` comment. LaTeX comments do not render.
- "Fixed in advance" was softened to "specified before analysis of the comparison
  point" wherever the timestamped provenance cannot be cited without identifying
  the authors.

## Residual risk

The architecture name *HyperloopBERT* is used throughout. The corresponding
public model repository returns HTTP 401 (private) and so is not discoverable by
search, but if that repository is made public before the notification date it
would become a deanonymisation vector. **Recommendation: keep it private until
after 2026-10-15.**

## Generative AI disclosure and anonymity

The required disclosure is written as a non-identifying section before the
references and does not use an acknowledgements environment, which the anonymous
acmart template suppresses.
