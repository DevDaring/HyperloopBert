# Claude Code Instructions — Convert `FIRE_HyperloopBERT.tex` to formal technical register

## 0. What this pass is, and what it is NOT

The structural revision from the previous pass is **already applied** in this draft — the
introduction roadmap names every section in order, the contributions are verb-first, the
Figure 3(a) legend reads "shared-token", Tables 4/8 define their columns, the training
configuration is stated once (§9) not duplicated, model and dataset names are italicised.
**Do not redo, revisit, or undo any of that.**

This pass fixes exactly one thing: **writing register**. The prose is clear but written in
a conversational, narrative style — the register of a well-written blog post, not a
technical paper. The supervisor read only the abstract and marked it red for this reason.
Your job is to convert the whole paper into a plain, formal, technical academic register,
sentence by sentence, without changing any content, number, citation, result, or the
(already-fixed) structure.

**Working method.** Edit `FIRE_HyperloopBERT.tex` in place. Keep it compilable. Keep a
`CHANGELOG.md`. Change wording only — never a numeric value, a citation key, a table cell,
a symbol definition, or a section order. When a rewrite risks changing meaning, keep the
meaning and change only the register.

---

## 1. The target register — calibrate to this before editing

The target has three properties at once. All three must hold in every sentence you touch.

**(a) Formal, not conversational.** Remove idioms, rhetorical one-liners, reader-address,
and storytelling transitions. State the technical fact plainly.

**(b) Plain, not ornate.** Do NOT fix casual prose by reaching for elevated vocabulary.
No buzzwords, no rare words. The banned lists in Section 5 are hard constraints.

**(c) Short and even, not dramatic and not long.** Prefer declarative sentences of about
15–25 words. Do not use sentence fragments for effect. Do not build long clause-stacked
sentences either. Split anything with three or more commas or a nested clause.

### The single most important rule (read twice)

There are TWO ways to fail academic register, and this paper must avoid BOTH:

- **Too casual** (the current problem): "harder than it looks", "does not hold up", "the
  model is asked", "flips its sign", "Audits followed."
- **Too ornate** (the overcorrection to avoid): "we leverage", "furthermore", "a robust
  and comprehensive framework", 40-word sentences with three subordinate clauses.

The target is the middle band, which is the register of the supervisor's own papers:
plain words, precise technical terms, short-to-medium declarative sentences, formal but
readable. When in doubt, choose the plainer word and the shorter sentence — but make the
phrasing formal, not chatty.

A short sentence is not the problem; a *dramatic idiomatic* sentence is. "The effect does
not replicate." is fine — it is short, plain, and technical. "The result does not hold
up." is not — it is an idiom. Keep the first kind, remove the second.

---

## 2. The abstract — rewrite it to match this exactly

The abstract is what was red-marked, so it is the anchor. Rewrite it to the register
below. This version preserves every claim and number in the current abstract and changes
only wording and cadence. You may use it directly, adjusting only if a fact differs from
the current draft.

> Pre-trained language models are reused across many downstream systems, so a stereotypical
> association encoded during pre-training can propagate to every system built on the model.
> A natural question is whether one architecture encodes less of that association than
> another. This comparison is confounded with model capability. A model further along in
> training assigns sharper probabilities on nearly every probe, including the minimal pairs
> that stereotype benchmarks score. A difference measured after fixed training therefore
> reflects both the architecture and the amount of training. Matching the models at equal
> validation loss is the standard control, and this study evaluates whether it is
> sufficient. Four encoders are trained from scratch on identical data with a single random
> seed, differing only in how far they reuse weights across depth, and a snapshot is
> recorded each time an encoder reaches a target validation loss. The snapshots are scored
> on two stereotype benchmarks under three scoring rules. At the deepest matched point, two
> of the three weight-reusing encoders record a lower stereotype effect than the unshared
> baseline, and both contrasts survive multiplicity correction. The effect does not
> replicate. It holds at only one of five matched points for the looped encoder and reverses
> at the others, becomes indistinguishable from zero once realised loss is adjusted for
> continuously, and fails correction under both alternative scoring rules, one of which
> reverses its sign. On the second benchmark the direction agrees, but only one of three
> contrasts remains significant, and for a different encoder. All four encoders perform at
> chance on a pronoun-resolution test, so these measurements reflect lexical association
> rather than task behaviour. Architecture-level bias comparisons should therefore be
> reported across several matched training levels and scoring rules. Code, checkpoints, and
> per-item scores are available at https://anonymous.4open.science/r/HyperloopBert/.

**Why each original phrase was replaced** — apply the same reasoning across the paper:

- "a stereotype it encodes can spread to all of them" → "propagate to every system built
  on the model" ("spread to all of them" is casual and vague).
- "The question is harder than it looks." → folded into "This comparison is confounded with
  model capability." (the idiom is removed; the technical reason is stated).
- "A better-trained model answers more sharply" → "assigns sharper probabilities on nearly
  every probe" (a model does not "answer sharply"; it assigns probabilities).
- "mixes the design with how far each model has trained" → "reflects both the architecture
  and the amount of training" ("mixes … with how far" is loose; "confounds/reflects" is the
  technical verb).
- "This study asks whether that fix is enough." → "this study evaluates whether it is
  sufficient" ("fix" and "enough" are casual).
- "Four encoders were built" → "Four encoders are trained" ("built" is casual for a model).
- "look less biased than the unshared one" → "record a lower stereotype effect than the
  unshared baseline" ("look less biased" is conversational).
- "The result does not hold up." → "The effect does not replicate." (idiom → technical
  term; both are short, which is fine).
- "flips its sign" → "reverses its sign".

---

## 3. Catalogue of register tells to fix throughout (before → after)

These patterns recur across the paper. Apply the same conversions wherever they appear, not
only at the lines quoted.

### Group A — idioms and loose phrasing → technical phrasing

- "harder than it looks" / "harder than it appears" → "is confounded with model capability"
  / "is more difficult to measure than it first appears" (state the reason, drop the idiom).
- "does not hold up" → "does not replicate" / "does not survive the robustness checks".
- "is enough" / "that fix" → "is sufficient" / "that control".
- "worth inspecting" / "it is worth …" → "is examined" / delete and state the action.
- "matters beyond …" → "has implications beyond …" / "is relevant beyond …".
- "answers more sharply" → "assigns sharper probabilities".
- "look less biased" / "less biased" → "record(s) a lower stereotype effect".
- "carries less of that stereotype" → "encodes less of that association".
- "were built" (of a model) → "were trained" / "were constructed".

### Group B — dramatic fragments and narrative transitions → plain topic sentences

The Related Work section in particular uses a storytelling cadence. Replace it.

- "Measurement came first." → "Measurement of stereotype association developed first."
- "Audits followed." → "Audits of these benchmarks followed."
- "They stop short of the question that follows." → "These audits do not address the
  question that follows."
- "Asking it needs two models differing in one designed way, and that axis comes from a
  separate literature." → "Answering it requires two models that differ along a single
  design axis, which a separate literature provides."
- "One line does connect capacity to bias, and it is the closest precedent." → "A third
  line relates model capacity to measured bias and is the closest precedent."
- "Two things are left open." → "This line leaves two questions open."
- "That test matters beyond encoder design." → "This test is relevant beyond encoder
  design."
  Keep the thematic grouping and every citation — only the transition wording changes.

### Group C — personification of the model → operational description

- "the model is asked how likely the original word was in that slot" → "the probability of
  the original token at that position is evaluated".
- "higher means the model finds the sentence more plausible" → "a higher value indicates
  that the model assigns greater probability to the sentence".
- "a model holding the stereotype should find the same surrounding words more likely" →
  "under this rule, an encoder that encodes the stereotype assigns higher probability to
  the shared context when the stereotypical term is present".

### Group D — reader-engaging or informal framing → neutral statement

- "it is natural to ask which of the two designs carries less …" → "a natural question is
  whether one design encodes less …".
- "so it is worth inspecting the underlying model output" → "the underlying model output is
  therefore examined directly".
- "The single-point result was convincing for defensible reasons." → "The single-point
  result has the features of a sound finding." (then the existing list of those features).
- "which is the difficulty" → "which is the central difficulty".
- "This section describes what was built and how it was measured." → delete or replace with
  the content sentence that already follows it ("This section defines the four encoders, the
  matching protocol, and the three scoring rules.").

---

## 4. Section-by-section targets

Work through in order and apply Sections 1–3. The density of register issues is highest in
the **Abstract** (Section 2 above), **Related Work** (Group B), and **§4.3 scoring rules**
(Group C). The Method equations, the Results tables, the Reproducibility section, and the
Limitations section are already close to the target register and need only light touch —
fix the specific phrases flagged above, do not rewrite passages that are already plain and
formal. In particular, do not touch:

- the equation-to-symbol definitions after Eqs. (1)–(3),
- the statistical description (permutation test, Holm correction, bootstrap, Cohen's d),
- the honesty and hedging in §6.10, §6.11, §7, and §8.

After each section, re-read your edits against Section 1(c): if any sentence you wrote runs
past ~30 words or stacks clauses, split it.

---

## 5. Hard constraints (the two rules stated explicitly)

**5.1 Banned vocabulary — do not introduce any of these while "formalising".**

- Promotional / buzzwords: leverage, harness, employ, utilise, showcase, pave, pioneer,
  foster, underscore, unlock, supercharge, revolutionise, seamless, cutting-edge,
  state-of-the-art, holistic, paradigm, novel (outside the contributions list), robust and
  comprehensive as vague praise, end-to-end (overuse).
- Rare / C2 words: delve, intricate, pivotal, paramount, realm, landscape, tapestry,
  multifaceted, unprecedented, quintessential, panoply, myriad, plethora, nuanced (as
  filler), interplay (as filler).
- AI-tell connectives: Furthermore, Moreover, Additionally, Notably, Importantly, It is
  worth noting, It should be emphasised, To this end.
  Allowed connectives, used sparingly: Therefore, However, In particular, We observe that,
  In contrast, Specifically.

**5.2 Sentence format — no long complicated sentences.**

- Target 15–25 words per sentence; hard ceiling around 30.
- At most one subordinate clause per sentence.
- Split any sentence with three or more commas or a nested "…, which …, …" structure.
- Do not use sentence fragments for emphasis. Every sentence is a complete clause.
- Keep some short declaratives for rhythm — short is good, dramatic-idiomatic is not.

---

## 6. What to preserve (do not touch)

- Every numerical value, every table cell, every citation key, every symbol and its
  definition. This is a wording pass only.
- The negative-result framing and all hedging and honesty (single-seed caveat, capability
  gate not met, "absence of evidence rather than evidence of absence", the direction-
  convention attribution to the 218 antistereo pairs). Keep the calibrated qualifiers.
- The already-fixed structure: roadmap, verb-first contributions, table columns, figure
  labels, §5/§9 split, italicised names.
- British/Indian spelling (modelling, behaviour, artefact, realise, colour). Keep it
  consistent; do not switch to US spelling while editing.
- The Generative AI Use Disclosure section — leave it in place unchanged.

---

## 7. Verification checklist (run before reporting done)

- [ ] LaTeX compiles; no numbers, citations, symbols, or table cells changed.
- [ ] Abstract rewritten to the Section 2 register; no idioms; ≤ a handful of numerals.
- [ ] No instance remains of: "harder than it looks/appears", "does not hold up", "look
  less biased", "were built" (of a model), "flips its sign", "is enough", "worth
  inspecting", "the model is asked", "the model finds … plausible".
- [ ] Related Work narrative transitions ("Measurement came first", "Audits followed", "Two
  things are left open", "That test matters") converted to plain topic sentences.
- [ ] No banned buzzword or rare word introduced (Section 5.1).
- [ ] No sentence exceeds ~30 words or stacks clauses (Section 5.2); no fragments for
  effect.
- [ ] Equation definitions, statistics, hedging, and structure left untouched.
- [ ] British spelling consistent; GenAI disclosure intact.
- [ ] `CHANGELOG.md` lists every reworded passage by section.

```

```
