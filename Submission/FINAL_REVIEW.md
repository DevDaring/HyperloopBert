# Claude Code Instructions — Optional final polish, `FIRE_HyperloopBERT.tex`

## 0. Status — read first

The paper is already at the bar for formal technical writing. The register conversion is
done and the structure, citations, equations, and argument are sound. **This pass is
optional polish, not a required fix.** Every item below is minor. Do not undertake any
larger rewrite, and do not touch anything not listed here. Keep the LaTeX compilable and
log each change in `CHANGELOG.md`. Change no numeric value, citation, symbol, or section
order.

If the author decides to submit as-is, that is a reasonable decision; these are
refinements, not corrections.

---

## 1. `[optional]` §6.2 — trim the numbers that merely restate Table 4

**Where:** Section 6.2, the paragraph beginning "Read in isolation, this table supports…".

**Issue (the one place the paper diverges from Sanyal's style).** Sanyal's preference is
that a table holds the exact numbers and the prose carries the qualitative finding plus at
most the headline delta. This paragraph restates most of Table 4 in prose: it gives
Δ = 0.0237, Δ = 0.0241, p_Holm = 0.0003, Δ = 0.0116, the interval [−0.0008, 0.0240], and
p_Holm = 0.065 — all already visible in the Table 4 row for each encoder.

**Important scope limit.** This applies to §6.2 ONLY. In §6.4 (the reversal, Δ = −0.0222),
§6.6 (the sign flip under the direction convention, +0.0076 vs −0.0035, +0.0919 vs
−0.0519), and §6.7 (StereoSet Δ and p), the specific numbers ARE the argument and are
often not fully tabulated. Leave those numbers in place — removing them would weaken the
paper. Do not "tidy" numbers anywhere except §6.2.

**Fix for §6.2.** Rewrite so the prose states the finding and keeps only the numbers that
are NOT in Table 4 (the Cohen's d values and the aggregate range), pointing to the table
for the rest. For example:

> Read in isolation, Table 4 supports the conclusion that weight reuse accompanies a lower
> stereotype effect: LoopedBERT and HyperloopBERT fall below the unshared baseline and both
> contrasts survive Holm correction, whereas ALBERTLoopedBERT does not, with a bootstrap
> interval covering zero. All three effects are small in standardised terms, at Cohen's
> d = 0.115, 0.098 and 0.047 respectively. The benchmark's own aggregate, the percentage of
> the 1508 pairs preferring the stereotypical member, separates the four far less, placing
> all between 55.4 and 55.8 with overlapping intervals.

This keeps every finding and every non-table number, and defers the exact Δ and p values to
Table 4 where they already appear. Confirm the Cohen's d values and the 55.4–55.8 range
against the current draft before writing.

---

## 2. `[minor]` Figure 3(b) axis label — "race-colour" → "race-color"

**Where:** the Figure 3(b) plotting script (not the `.tex`).

**Issue.** The body text (§3.2 and §6.8) uses "race-color", the label the dataset itself
uses. The Figure 3(b) y-axis still reads "race-colour". This is a text/figure
inconsistency on a proper-noun category label.

**Fix.** In the plotting script, change the category tick label from "race-colour" to
"race-color" so it matches the text and the dataset. Regenerate the figure. Leave the rest
of the British/Indian spelling in the prose unchanged — this one is a dataset label, not a
spelling choice.

---

## 3. `[minor]` §6.6 — remove the stray first-person "we"

**Where:** Section 6.6.
**Original:**

> "Holding the token set fixed at the official shared tokens, we cross aggregation with
> direction convention over all 1508 pairs."

**Issue.** The paper is otherwise written in a consistent impersonal voice (outside the
contributions list), so this single "we cross" stands out.

**Fix.** Recast impersonally, for example:

> "With the token set fixed at the official shared tokens, aggregation is crossed with the
> direction convention over all 1508 pairs."

Do not hunt for other "we" instances — the verb-first "We specify / We show" in the
contributions list is correct and should stay.

---

## 4. `[optional]` §4.3 — split the longest methods sentence

**Where:** Section 4.3, the sentence defining the permutation test.
**Original:**

> "The null hypothesis E[d_i] = 0 is tested by a paired sign-flip permutation test: the
> signs of the N differences are independently randomised over m = 10^4 draws, and the
> p-value is the Phipson–Smyth estimate (b + 1)/(m + 1), where b counts the draws at least
> as extreme as the observed |Δ| [15]."

**Issue.** About 45 words with a colon and three clauses. Methods sentences may be denser,
but this one splits cleanly.

**Fix (optional).** Break after the test is named:

> "The null hypothesis E[d_i] = 0 is tested by a paired sign-flip permutation test. The
> signs of the N differences are independently randomised over m = 10^4 draws, and the
> p-value is the Phipson–Smyth estimate (b + 1)/(m + 1), where b counts the draws at least
> as extreme as the observed |Δ| [15]."

---

## 5. Do not touch

- Any number, citation, symbol, or table cell outside the §6.2 trim in Section 1.
- The structure, roadmap, contributions, table columns, figure legends (other than the
  §2 label), and §5/§9 split — all already correct.
- The honesty and hedging throughout (§6.10, §6.11, §7, §8).
- British/Indian spelling in the prose; the Generative AI Use Disclosure section.

## 6. Verification checklist

- [ ] LaTeX compiles; no numbers/citations/symbols changed except the §6.2 prose trim.
- [ ] §6.2 keeps the finding and the non-table numbers (Cohen's d, 55.4–55.8) and defers
  the exact Δ/p to Table 4; §6.4/§6.6/§6.7 numbers left untouched.
- [ ] Figure 3(b) axis reads "race-color"; figure regenerated.
- [ ] §6.6 recast impersonally; contributions "We" left intact.
- [ ] §4.3 permutation sentence split (if done).
- [ ] Nothing outside the listed items changed; `CHANGELOG.md` updated.
