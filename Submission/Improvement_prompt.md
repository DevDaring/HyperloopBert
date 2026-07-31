# Claude Code Instructions — Revise `FIRE_HyperloopBERT.tex` (Sanyal-readiness pass)

## 0. Context and working method (read first)

You are editing an anonymous double-blind submission to **FIRE 2026** titled
"Same Models, Opposite Conclusions: Testing the Robustness of Architecture Bias
Comparisons". The goal is narrow: make the paper survive a strict structural/prose
review by the author's supervisor before it is sent to him. This is a polish pass,
not a rewrite.

Before editing:

1. Read the full `FIRE_HyperloopBERT.tex`.
2. Identify the figure-generating scripts in the repo (the Figure 3 plotting code in
   particular — one fix below lives there, not in the `.tex`).
3. Keep the LaTeX compilable at every step (balanced environments, all `\ref`/`\label`
   resolve, no broken table syntax).
4. Keep a `CHANGELOG.md` listing each edit and the section it touched.
5. **Do not invent numbers, citations, or results.** Every number already in the paper
   is correct; your job is placement and prose, not values.

**Severity tags** used below: `[MAJOR]` = the supervisor will call this out, fix before
sending; `[MINOR]` = polish; `[STYLE]` = optional; `[NOTE]` = judgment call for the
author, do not auto-change.

---

## 1. Verdict (honest — read before touching anything)

This paper is **already above the usual bar**. The prose varies sentence length, avoids
promotional verbs, cites densely and thematically in Related Work, binds every equation
to its symbols, and is unusually honest about its own limits (single seed, capability
gate not met, negative result stated plainly). None of that should be changed — see
Section 6, "What NOT to touch."

The remaining issues are structural leftovers and a few consistency bugs, not a framing
problem. The framing passes cleanly: problem (are architecture-vs-architecture bias
comparisons robust to evaluation choices?), novelty (varies weight-sharing *before*
training, matches on validation loss, then stress-tests the comparison), and result (a
significant single-point effect reverses across matched points and scoring rules) are
all legible. Fix the six `[MAJOR]` items and the supervisor has little left to grab.

---

## 2. Global style guardrails (hold these on every edited passage)

The paper mostly satisfies these already; apply them only where an edit is made, and do
not "correct" passages that already comply.

- No em-dashes beyond ~4 per page (paper is roughly within budget; do not add more).
- Banned promotional verbs (`leverage`, `harness`, `employ`, `utilise`, `showcase`,
  `delve`, `foster`) — the paper is currently clean; keep it that way.
- Prefer active voice; do not introduce new passive constructions.
- British/Indian spelling is used consistently (`modelling`, `behaviour`, `optimise`,
  `artefact`, `colour`) — keep it consistent (one exception in Section 4).
- Numbers live in tables/figures; prose carries the qualitative finding plus at most the
  single headline delta.
- Contributions list is verb-first (fixed in 3.4).
- Model and dataset names italicised on first use (fixed in 4.1).

---

## 3. Priority fixes (`[MAJOR]`)

### 3.1 Introduction roadmap does not match the section structure `[MAJOR]`

**Where:** Introduction, final paragraph.
**Original:**

> "Section 2 reviews the relevant measurement and weight-sharing literature. Section 4
> defines the encoders, the matching protocol and the three scoring rules. Section 6
> reports the robustness analyses, and Section 8 states what the evidence does not
> support."

**Issue:** The paper has ten sections (1 Introduction, 2 Related Work, 3 Data, 4 Method,
5 Experimental Setup, 6 Results, 7 Discussion, 8 Limitations, 9 Reproducibility, 10
Conclusion). The roadmap names only 2, 4, 6, 8 — it silently skips Data (§3),
Experimental Setup (§5), Discussion (§7), Reproducibility (§9), and Conclusion (§10),
and it labels §8 (Limitations) as "states what the evidence does not support", which
actually happens in §6.9 and §7. A structural reviewer reads this first and sees a stale
roadmap. This is the single most likely thing to be flagged.

**Fix:** Rewrite the roadmap so it names the main sections in order with correct numbers.
Either walk all sections (§2 Related Work → §3 Data → §4 Method → §5 Experimental Setup
→ §6 Results → §7 Discussion → §8 Limitations → §9 Reproducibility → §10 Conclusion), or
name the load-bearing ones with numbers that match their real content. Do not leave a
version that references §8 for "what the evidence does not support" while §7 Discussion
and §6.9 carry that argument.

### 3.2 Experimental Setup (§5) and Reproducibility (§9) duplicate the training config `[MAJOR]`

**Where:** Section 5, first paragraph; Section 9.
**Original (§5):**

> "Training uses the AdamW optimiser at a peak learning rate of 3 × 10⁻⁴, batch size
> 512, a warmup over the first tenth of the steps, and gradient clipping at 1.0, in
> mixed precision on one H100 GPU."

**Original (§9):**

> "trained with AdamW, peak learning rate 3 × 10⁻⁴ (HyperloopBERT 1.5×10⁻⁴), batch size
> 512, 10% warmup, gradient clipping 1.0, bfloat16 autocast, one seed (42), on one H100
> GPU."

**Issue:** The same optimiser, learning rate, batch size, warmup, clipping, precision,
and GPU are stated twice. Repeating a configuration verbatim in two sections is the prose
analogue of duplicating a table and a figure; the supervisor will ask why §5 exists if
§9 already has it. It also explains why the §5 number set was omitted from the roadmap.

**Fix (choose one, prefer the first):**

- Keep the full configuration in §9 Reproducibility (its natural home) and trim §5 to the
  parts that carry the narrative and are *not* pure config: the deepest-matched-point
  token range (1.53–2.07 B), the HyperloopBERT learning-rate exception and why it was
  tuned separately, and the single-seed / benchmark-item-bootstrap caveat. Have §5 defer
  the rest with a pointer such as "full training configuration in Section 9."
- Or merge §5 entirely into §4 (Method) and §9, removing the standalone section. If §5 is
  removed, update the roadmap in 3.1 and any `\ref` to it.

### 3.3 Figure 3(a) legend contradicts the paper's scoring-rule name `[MAJOR]`

**Where:** Figure 3(a), legend/label (in the plotting code, not the `.tex`).
**Original:** legend reads "full-sentence PLL / changed-token".
**Paper text (§4.3):**

> "The shared-token rule takes I to be only the tokens the two sentences have in common."

**Issue:** The paper defines and uses the name **shared-token** everywhere (Table 3,
Table 6, §4.3, §6.5, Discussion). Figure 3(a) calls the same rule **changed-token**,
which is not just inconsistent naming but the semantic opposite: the rule scores the
tokens the sentences *share*, and explicitly does not score the demographic tokens that
*change*. A reviewer who notices the figure says the reverse of the method loses trust in
the results.

**Fix:** In the Figure 3 plotting script, relabel the series from "changed-token" to
"shared-token" so the figure matches Table 6 and the text. Regenerate the figure. Verify
no other figure or caption uses "changed-token".

### 3.4 Contributions list is noun-first; the supervisor's pattern is verb-first `[MAJOR]`

**Where:** Introduction, "This paper makes three contributions."
**Original:**

> "(1) A comparison protocol that holds training progress fixed ... (2) Evidence that the
> apparent architecture effect is sensitive to both choices ... (3) Evidence that an
> aggregate stereotype score conceals sign changes ..."

**Issue:** The supervisor's own papers open each contribution with a verb (propose,
present, show, introduce, demonstrate). Noun-first items ("A protocol that...",
"Evidence that...") read as a list of nouns rather than claims of work done. The
"avoid we" preference applies to body prose, not the contributions list; verb-first with
"We" is the expected form here.

**Fix:** Convert to verb-first, keeping one sentence each:

- "We introduce a comparison protocol that holds training progress fixed by matching
  models on validation loss, then checks whether the conclusion survives the choice of
  matched point and scoring rule."
- "We show that the apparent architecture effect is sensitive to both choices: it
  reverses sign across matched training levels, does not survive correction under either
  of two other scoring rules, and replicates in direction but not significance on a
  second benchmark."
- "We show that an aggregate stereotype score conceals sign changes among individual bias
  categories, and that a capability check rules out reading these measurements as
  evidence about downstream behaviour."

### 3.5 Abstract number density `[MAJOR]`

**Where:** Abstract.
**Original (excerpt):**

> "... the same 7 billion tokens, 28 billion in total, using one random seed. Training
> produced 21 distinct snapshots. These were scored on two stereotype benchmarks, of
> 1508 and 2106 items, under three scoring rules."

**Issue:** The supervisor's standing instruction is to keep the abstract to a small number
of figures; this one front-loads setup counts (7 B, 28 B, 21, 1508, 2106) that read as
mechanical. The *result* fractions later in the abstract ("two of the three", "one point
out of five", "one difference of three") are the argument of a negative-result paper and
should stay — they are qualitative and load-bearing.

**Fix:** Cut the *setup* numbers, keep the *result* fractions. Convert "7 billion tokens,
28 billion in total" to "identical data", "two stereotype benchmarks, of 1508 and 2106
items" to "two stereotype benchmarks", and either drop "21 distinct snapshots" or reduce
to "a sequence of matched snapshots". Retain at most one setup number if one feels
necessary. Do not touch the result fractions. Target roughly 2–3 numerals total in the
abstract.

### 3.6 Result-table captions do not define their columns `[MAJOR]`

**Where:** Tables 4, 5, 6, 7 captions.
**Issue:** Column headers use quantities and acronyms the captions never define. Table 4
and Table 7 have an `Effect`, a `Δ`, and a `p_Holm` column with no caption gloss; Table 7
additionally uses `SS` and `LMS` as bare headers (these are the stereotype score and the
language-modelling score, defined only implicitly in §6.6 prose). A reader cannot read
these tables standalone, and undefined acronyms in headers are a reliable flag.

**Fix:** Add a one-line definition of each non-obvious column to the caption (or a table
footnote):

- `Effect` = mean paired stereotype effect (mean of e over items).
- `95% interval` = item-bootstrap confidence interval on the effect.
- `Δ` = contrast against the unshared VanillaBERT baseline.
- `p_Holm` = Holm-corrected sign-flip permutation p-value within the contrast family.
- Table 7: `SS` = stereotype score; `LMS` = language-modelling score.
  Keep column precision consistent down each column (already the case — do not change the
  values).

---

## 4. Minor fixes (`[MINOR]`)

### 4.1 Italicise model and dataset names on first use `[MINOR]`

Italicise `VanillaBERT`, `LoopedBERT`, `HyperloopBERT`, `ALBERTLoopedBERT`, and the
datasets `CrowS-Pairs`, `StereoSet`, `WinoBias`, `FineWeb-Edu` at their first occurrence
(the supervisor's papers do this). Plain thereafter. Keep spelling identical across
Abstract, Intro, Method, Results, Conclusion.

### 4.2 Citation `[3]` placement in Section 5 `[MINOR]`

The sentence "in mixed precision on one H100 GPU [3]" attaches the FlashAttention
reference to the GPU/precision claim. FlashAttention is an attention implementation, not
a GPU or a precision mode. If FlashAttention is actually used, cite it where attention or
throughput is described; if it is not used, remove `[3]` from that sentence. Verify
against the training code.

### 4.3 Dataset category spelling `[MINOR]`

The text renders the CrowS-Pairs category as "race-colour" (British), but the dataset's
official label is "race-color". For a category name taken from the source, match the
dataset's spelling, or note the anglicisation once. This is the one place the otherwise
consistent British spelling collides with a proper noun.

### 4.4 Abstract voice `[STYLE]`

The abstract is passive throughout ("were built", "was trained", "were scored"). This is
acceptable in an abstract; only tighten if 3.5 is already being edited and an active
recast reads cleanly. Low priority — do not force it.

---

## 5. Notes for the author — do NOT auto-change (`[NOTE]`)

### 5.1 Generative AI Use Disclosure section

Keep it. ACM/FIRE policy expects disclosure of writing-assistance tools, so removing it
would be wrong. The disclosure is not what a strict reviewer objects to; the prose
quality is, and this pass addresses that. No edit needed beyond leaving it in place.

### 5.2 Citation verification was not performed here

This pass is writing-and-structure only. Before submission, verify that the 2025–2026
preprint references resolve to real records and correct arXiv IDs (Frey et al. 2026;
Voria et al. 2026; Zeitoun et al. 2026; Xie et al. 2025; Saunshi, Bae, Geiping, Zhu
2025), and update any that have since been published to their venue of record. The
well-known references (BERT, ALBERT, CrowS-Pairs, StereoSet, GLUE, FineWeb, Hyper-
Connections) are not in question.

### 5.3 Venue-fit framing

The FIRE/IR relevance rests on the argument that these encoders underlie retrieval and
ranking (made in the Introduction and Discussion) and on the India-centric instrument
noted in Limitations. That connection is present and adequate; keep it visible when
editing the Introduction, but no new material is required.

---

## 6. What NOT to touch (preserve the paper's strengths)

Do not "improve" any of the following — they are why the paper is strong, and edits will
degrade it:

- The honest hedging and the single-seed caveat ("No claim in this paper is supported by
  replication across seeds"; "this is an absence of evidence ... rather than evidence of
  absence"). Keep the calibrated language exactly.
- Section 6.9 ("The capability gate is not met") and its conclusion that the measurements
  do not speak to downstream behaviour. Do not soften it.
- The thematic, densely cited Related Work — do not convert to chronological narrative or
  drop citations.
- The equation-to-symbol bindings after Eq. (1), (2), (3) — every symbol is defined; do
  not trim these.
- The Discussion's synthesis and the negative-result conclusion. Do not add a triumphant
  or promotional closing.
- Any result value in any table. Values are correct; only captions/placement change.

---

## 7. Verification checklist (run before reporting done)

- [ ] LaTeX compiles; all `\ref`/`\label` resolve; no broken tables.
- [ ] Roadmap names sections in order with numbers matching their real content (3.1).
- [ ] Training configuration stated once, not duplicated across §5 and §9 (3.2).
- [ ] Figure 3(a) legend reads "shared-token"; figure regenerated; no "changed-token"
  remains anywhere (3.3).
- [ ] Contributions are verb-first, one sentence each (3.4).
- [ ] Abstract carries ~2–3 numerals; result fractions retained, setup counts trimmed
  (3.5).
- [ ] Tables 4–7 captions define Effect, Δ, p_Holm, SS, LMS (3.6).
- [ ] Model/dataset names italicised on first use (4.1).
- [ ] `[3]` citation placement resolved (4.2); "race-color" spelling resolved (4.3).
- [ ] No banned verbs, no new em-dashes, no new passive voice introduced.
- [ ] Section 6 items left untouched.
- [ ] `CHANGELOG.md` lists every change; §5.2 citation check flagged for the author.

```

```
