# Unresolved findings

Findings still open when the loop stopped (round limit (4) reached with findings open).

### opus-4-1 - R14 (MAJOR)

- Location: Abstract
- Quote: "At the lowest-loss matched point on CrowS-Pairs, where models are compared after reaching the same held-out-text loss target, the full-sentence rule gives"
- Problem: This abstract sentence runs to about 38 words with a nested relative clause, well past the 28-word hard maximum. | The abstract's central result sentence runs about 35 words with an embedded definitional clause; I had to read it twice to parse what was compared. | Sentence is 36 words; R14 sets a hard maximum near 28.
- Raised by: kimi, linter, opus
- Hint: Split into one sentence defining the matched point and one stating the result.

### opus-4-11 - R25 (MAJOR)

- Location: Experimental Setup
- Quote: "At the common rate its loss became non-finite."
- Problem: The same fact is stated twice within two paragraphs of the same short section, and again in Method, Discussion and Limitations. | Each analysis paragraph opens with a formulaic one-line bridge of the same shape, revealing a stack of former subsections rather than a continuous argument.
- Raised by: opus
- Hint: State the non-finite loss once in Experimental Setup and cross-reference it later.

### linter-4-2 - R12 (MAJOR)

- Location: Introduction (line 100)
- Quote: "It must also keep the same sign under both direction conventions."
- Problem: 'It' has no named referent in this sentence or the one before it. | In the abstract I could not tell what quantity is differenced: scores of the two sentences in a pair, or scores of two models; this is only clarified in the Introduction.
- Raised by: kimi, linter

### linter-4-4 - R14 (MAJOR)

- Location: Introduction (line 43)
- Quote: "Before task-specific fine-tuning, encoders can assign different scores to the same stereotype benchmark items. These pre-training scores do not measure how a la"
- Problem: Sentence is 29 words; R14 sets a hard maximum near 28. | Sentence is 41 words; R14 sets a hard maximum near 28.
- Raised by: linter

### opus-4-3 - R25 (MAJOR)

- Location: Introduction, paragraph 4
- Quote: "A checkpoint is a saved model state at a training step. Lower validation loss can change the paired score difference on benchmark items."
- Problem: The paragraph is a chain of stand-alone glossary definitions in uniform short declaratives, so it reads as an inserted term list rather than an argument.
- Raised by: opus
- Hint: Attach each definition to the sentence that first needs the term instead of listing them consecutively.

### opus-4-10 - R13 (MAJOR)

- Location: Method, Architectures (hyper-connection equation)
- Quote: "The paper does not tabulate those configuration fields, so exact reruns require the released model configuration."
- Problem: The stream count n and width d are used in the equation but never given, so the HyperloopBERT description is incomplete in the paper itself. | Subsection opens with context, but the sentence contains an undefined term 'HyperloopBERT' at the opening, violating R4/R5.
- Raised by: linter-ds, opus
- Hint: Give the stream count and width values in Table 3 or in this sentence.

