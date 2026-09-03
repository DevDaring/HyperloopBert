# Unresolved findings

Findings still open when the loop stopped (round limit (4) reached with findings open).

### linter-4-5 - R25 (MAJOR)

- Location: Introduction
- Quote: "conceptual framework"
- Problem: The Introduction never reaches the 'conceptual framework' stage; R25 requires motivation, background, research gap, objectives, conceptual framework, methodology, contributions in that order. | The Introduction follows the required order: motivation, background, research gap, objectives, methodology, contributions, but the conceptual framework stage is missing.
- Raised by: linter, linter-ds

### opus-4-2 - R14 (MAJOR)

- Location: Introduction, CrowS-Pairs example paragraph
- Quote: "The model probability of the original token is recorded. The log probabilities are averaged. The anti-stereotypical score is subtracted from the stereotypical score."
- Problem: Five successive subject-verb-object sentences of near-identical length give a machine-like staccato rhythm that recurs throughout the paper.
- Raised by: opus
- Hint: Join the scoring steps into two sentences that read as one procedure.

### opus-4-6 - R25 (MAJOR)

- Location: Introduction, final contribution paragraph
- Quote: "This paper makes these contributions. The protocol evaluates \emph{CrowS-Pairs}, \emph{StereoSet}, and \emph{WinoBias}."
- Problem: The paragraph is a block of result statements announced by a list stem, so it reads as a generated summary rather than the close of an argument.
- Raised by: opus
- Hint: State the contribution as what the study establishes, then hand the reader to Related Work.

### opus-4-3 - R15 (MAJOR)

- Location: Introduction, first paragraph
- Quote: "Encoder choice therefore matters before any downstream task is known."
- Problem: The sentence states no measurable claim: it does not say what encoder choice changes or by what evidence.
- Raised by: opus
- Hint: Say what quantity differs between encoders before fine-tuning.

### opus-4-4 - R13 (MAJOR)

- Location: Introduction, first paragraph
- Quote: "Fine-tuning can change the association, but the initial encoder can still affect model selection."
- Problem: Neither clause names a model, dataset, or metric, so "affect model selection" cannot be checked.
- Raised by: opus
- Hint: State that practitioners pick encoders using measured stereotype scores, and cite the source for that practice.

### opus-4-1 - R25 (MAJOR)

- Location: Introduction, paragraph beginning "The comparison varies encoder design"
- Quote: "A tokeniser is the fixed procedure that splits text into model tokens. A lexical association is a model preference for words or sentences that co-occur"
- Problem: Five consecutive stand-alone definitions form a glossary block rather than a continuous argument, and the paragraph could be moved anywhere without loss.
- Raised by: opus
- Hint: Introduce each term at the point where the study first needs it, inside the sentence that uses it.

### opus-4-5 - R15 (MAJOR)

- Location: Introduction, paragraph beginning "The practical question"
- Quote: "A model later in training can assign more concentrated probabilities on benchmark pairs."
- Problem: "More concentrated probabilities" is not a measured quantity anywhere in the paper and gives the reader nothing checkable.
- Raised by: opus
- Hint: Say that lower validation loss changes the size of the paired score difference.

### linter-4-3 - R11 (MAJOR)

- Location: Matching, scoring, and inference (line 473)
- Quote: "MASK"
- Problem: Term 'MASK' is used 2 times and is never explained in plain words at first use.
- Raised by: linter

### opus-4-9 - R13 (MAJOR)

- Location: Method, Architectures, opening sentence
- Quote: "Architecture is the design choice tested in the study, so all non-architectural inputs are held fixed where the runs allow."
- Problem: "Where the runs allow" hides which input was not held fixed, and the reader must reach the setup section to learn it. | The subsection opens with context.
- Raised by: linter-ds, opus
- Hint: Name the learning rate as the single exception in this sentence.

### kimi-4-5 - R13 (MAJOR)

- Location: Results, continuous-adjustment paragraph
- Quote: "no architecture coefficient in the \emph{CrowS-Pairs} item-effect regression is distinguishable from zero at the significance rule used for bold entries"
- Problem: The decision rule for bold entries is never stated anywhere; I could not tell whether bold means p < 0.05 on one p-value column, on both, or after Holm correction.
- Raised by: kimi
- Hint: State the threshold once, e.g., "bold marks entries with both p-values below 0.05."

### opus-4-10 - R13 (MAJOR)

- Location: Results, continuous-adjustment paragraph (sec:cont)
- Quote: "Remaining capability differences are therefore large enough to affect the single-point reading."
- Problem: "The single-point reading" is not a named quantity and "large enough" states no comparison or threshold.
- Raised by: opus
- Hint: Compare the loss coefficient in Table 5 with the contrasts in Table 3 explicitly.

### opus-4-7 - R15 (MAJOR)

- Location: Results, subsection heading (line 536)
- Quote: "Single-point comparison and concrete predictions"
- Problem: The subsection reports masked-pronoun probe probabilities, not predictions, so the heading does not describe its content.
- Raised by: opus
- Hint: Name the subsection after the matched-point comparison and the probe outputs it reports.

### linter-4-4 - R14 (MAJOR)

- Location: Single-point comparison and concrete predictions (line 594)
- Quote: "The first probe is ``The scientist said that [MASK] was ready for the lab.'' The other probes replace ``scientist'' with ``teacher'' and ``secretary.'' On the ..."
- Problem: Sentence is 41 words; R14 sets a hard maximum near 28.
- Raised by: linter

### linter-4-1 - R6 (MAJOR)

- Location: line 469
- Quote: "WinoBias"
- Problem: Run-in paragraph heading; R6 allows sections and subsections only.
- Raised by: linter

### linter-4-2 - R6 (MAJOR)

- Location: line 754
- Quote: "CrowS-Pairs"
- Problem: Run-in paragraph heading; R6 allows sections and subsections only.
- Raised by: linter

### checker-4-1 - R23 (MAJOR)

- Location: tab:datastats
- Problem: Float 'tab:datastats': float added. A float may be dropped only when R10 shows it duplicates another; otherwise restore it.
- Raised by: checker

### checker-4-3 - R23 (MAJOR)

- Location: tab:gate
- Problem: Float 'tab:gate': float added. A float may be dropped only when R10 shows it duplicates another; otherwise restore it.
- Raised by: checker

### checker-4-4 - R23 (MAJOR)

- Location: tab:reprosettings
- Problem: Float 'tab:reprosettings': float added. A float may be dropped only when R10 shows it duplicates another; otherwise restore it.
- Raised by: checker

### checker-4-2 - R23 (MAJOR)

- Location: tab:runfacts
- Problem: Float 'tab:runfacts': float added. A float may be dropped only when R10 shows it duplicates another; otherwise restore it.
- Raised by: checker

