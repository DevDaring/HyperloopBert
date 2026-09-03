# Changelog

Per-round fix log from the Writer (ChatGPT). Each entry maps a finding to the change made.

## Round 1

Findings given: 0. Fixes reported: 0.

_No fix entries returned for this round._

## Round 2

Findings given: 32. Fixes reported: 32.

| Finding | Rule | Location | Change | Status |
| --- | --- | --- | --- | --- |
| linter-1-5 | R11 | Abstract and Introduction | Defined weight reuse, matched training point, lexical association, Holm correction, ALBERT, contrast, capability check, and the named benchmarks before using those terms. | fixed |
| checker-1-5 | R3 | front matter | Removed the CCSXML block and optional numerical significance arguments from the CCS descriptors so classification identifiers and weights no longer appear as prose numbers. | fixed |
| grok-1-2 | R17 | Abstract | Replaced the claim about a different encoder with a precise statement that only the HyperloopBERT contrast remains corrected on StereoSet. | fixed |
| grok-1-3 | R13 | Abstract | Named the official scoring rule and specified that the LoopedBERT and ALBERTLoopedBERT contrasts reverse sign. | fixed |
| kimi-1-3 | R14 | Abstract | Split the long abstract sentence about matched points, loss adjustment, and scoring rules into short separate sentences. | fixed |
| linter-1-19 | R14 | Method / Architectures | Split the hyper-connection equation explanation into several short sentences that define the matrices, streams, and residual special case separately. | fixed |
| linter-1-11 | R12 | Results / Capability gate | Rewrote the capability-gate interpretation to name the failed WinoBias check and the four encoders instead of using ambiguous pronouns. | fixed |
| linter-1-29 | R26 | Data | Folded the short Pre-training corpus subsection into the Data section as a normal paragraph. | fixed |
| opus-1-1 | R15 | Discussion and Results openings | Replaced vague phrasing with a concrete statement that the single-point analysis used paired contrasts, bootstrap intervals, permutation tests, and Holm-corrected p-values. | fixed |
| linter-1-23 | R25 | Introduction | Reordered the Introduction to move from motivation, to background, research gap, objectives, conceptual framework, methodology, and contributions. | fixed |
| grok-1-1 | R17 | Introduction / contributions | Removed the unsupported four-dependence-model claim and stated only the continuous realised-loss adjustment reported in Table model. | fixed |
| opus-1-15 | R15 | Introduction | Replaced figurative claims with literal statements that the study rechecks the CrowS-Pairs comparison under matched points and scoring rules. | fixed |
| linter-1-30 | R26 | Method and Results | Folded the Method capability gate into the Method text and folded the Results capability gate into the Results section as paragraphs with preserved labels. | fixed |
| opus-1-14 | R15 | Related Work opening | Replaced the matched rhetorical triplet with a single plain sentence describing the prior literatures used by the study. | fixed |
| grok-1-4 | R13 | Results / Capability gate | Dropped the unsupported interval-covering-chance claim and reported only the minimum and maximum accuracy values shown in Table gate. | fixed |
| opus-1-2 | R15 | Results / Capability gate | Replaced the metaphor about bounding measurements with a literal statement that the failed WinoBias check limits effects to lexical association claims. | fixed |
| opus-1-5 | R25 | Results / Sensitivity analyses | Rewrote the numbered check sequence as connected paragraphs that each link back to the single-point result. | fixed |
| opus-1-10 | R10 | Results / scorer check | Kept exact scorer values in Table rules2 and described Figure robust panel a only as an interval view of the scorer pattern rather than a source of exact values. | fixed |
| opus-1-9 | R13 | Results / Single-point comparison and Method | Deleted the unspecified post-hoc margin sentence and stated in the concrete CrowS-Pairs example that sent_more is anti-stereotypical for the antistereo item. | fixed |
| opus-1-18 | R15 | Results / scorer validation | Removed the drift-and-cancels assertion and stated that all encoders were scored with the same code and environment. | fixed |
| checker-1-1 | R23 | Floats | Kept the dataset, run-facts, capability-gate, and reproducibility-settings floats in the complete source without changing their numeric values. | fixed |
| linter-ds-1-1 | R1 | Abstract | Reopened the abstract with field context about encoders in retrieval and classification, then explained weight reuse before the problem statement. | fixed |
| linter-ds-1-13 | R4 | Section and subsection openings | Added context sentences to Data, Method, Architectures, Results, Limitations, Reproducibility, Conclusion, and the declaration section. | fixed |
| kimi-1-16 | R11 | Data | Defined an unaligned shared-token pair before Table datastats uses the term. | fixed |
| grok-1-7 | R17 | Discussion | Reframed the reporting recommendation as what this study needed rather than as a requirement imposed on prior work. | fixed |
| kimi-1-12 | R11 | Introduction, Related Work, Method, and Reproducibility | Defined effective depth, head, streams, Holm correction, residual connection, pruning, boundary tokens, Phipson-Smyth estimate, and architecture bundle before or at first use. | fixed |
| opus-1-20 | R15 | Method / Architectures | Replaced the abstract phrase about every realisation with a concrete statement that the result applies to the encoder-scale masked-language-model setting trained here. | fixed |
| grok-1-9 | R19 | Related Work | Removed the repeated direction-depends claim and kept the compression literature summary to the specific role it plays in this study. | fixed |
| opus-1-21 | R17 | Reproducibility | Qualified the rerun claim by stating that released scripts reproduce stored scores within the CPU rerun tolerance in Table reprosettings. | fixed |
| grok-1-5 | R17 | Results / Single-point comparison | Limited the single-point conclusion to LoopedBERT and HyperloopBERT, the two encoders whose CrowS-Pairs contrasts survive Holm correction. | fixed |
| checker-ms-1-18 | R24 | Results / Capability gate | Preserved and expanded the failed capability-gate result while adding context before the result sentence. | fixed |
| checker-ms-1-1 | R10 | Float captions | Adjusted the example-table caption to state that the table contains concrete examples and kept other captions concise and specific to their float contents. | fixed |

## Round 3

Findings given: 36. Fixes reported: 36.

| Finding | Rule | Location | Change | Status |
| --- | --- | --- | --- | --- |
| grok-2-3 | R13 | Abstract | Replaced the vague null statement with a sentence naming the CrowS-Pairs item-effect regression, VanillaBERT reference, and realised validation-loss covariate. | fixed |
| grok-2-4 | R13 | Abstract | Rewrote the StereoSet sentence to name the intrasentence items, the paired effect, HyperloopBERT versus VanillaBERT, and the lowest-loss matched point. | fixed |
| linter-2-9 | R14 | Conclusion | Split and replaced the abstract methodological sentence with two shorter checkable sentences about reporting matched capability levels, scoring rules, and direction conventions. | fixed |
| linter-2-3 | R12 | Data | Changed 'This label' to 'The direction label' so the referent is named in the sentence. | fixed |
| linter-2-7 | R14 | Discussion | Split the long capability sentence and cited Table model and Table reprosettings for the measured association and seed limitation. | fixed |
| linter-2-8 | R14 | Discussion | Split the reporting recommendation into two short sentences stating what to report and why to report it. | fixed |
| opus-2-13 | R15 | Discussion first paragraph | Replaced the abstract opening with a concrete statement that the corrected CrowS-Pairs contrast does not repeat across matched points or alternative scorers. | fixed |
| grok-2-2 | R17 | Experimental Setup | Reframed HyperloopBERT as a stability-adjusted architecture bundle because it required a lower peak learning rate. | fixed |
| linter-2-10 | R25 | Introduction | Reordered and rewrote the introduction to begin with system motivation before definitions, gap, objectives, framework, method, and contributions. | fixed |
| opus-2-1 | R25 | Introduction first paragraph | Removed the dictionary-style opening and folded definitions into a motivated paragraph about retrieval systems, classifiers, and inherited associations. | fixed |
| opus-2-4 | R12 | Related Work | Changed 'This work' to 'The looped and hyper-connection studies cited above' to name the referent. | fixed |
| opus-2-16 | R25 | Reproducibility | Combined the training configuration into one procedural sentence that points to Table reprosettings, while retaining short definitions of AdamW, bfloat16, and H100. | fixed |
| grok-2-5 | R17 | Results continuous adjustment | Led the interpretation with limited power and stated that the loss-adjusted null is not evidence that architecture has no effect. | fixed |
| kimi-2-2 | R18 | Results concrete predictions | Added full masked-pronoun probe text with the [MASK] position before interpreting Table qual. | fixed |
| opus-2-10 | R10 | Results scorer analysis | Removed the sentence saying Figure robust repeats Table rules2 and clarified that the table gives exact values while the figure shows uncertainty and category patterns. | fixed |
| opus-2-7 | R15 | Results sensitivity analyses | Replaced figurative wording about a warning with a literal statement about continuous adjustment for realised capability and loss-adjusted coefficients. | fixed |
| opus-2-8 | R25 | Results sensitivity analyses | Replaced repeated verdict openers with transitions linking the matched-point, continuous-adjustment, scorer, benchmark, category, and gate checks. | fixed |
| linter-2-2 | R11 | Sensitivity analyses | Replaced the split token string with 'bert-base-uncased' and defined it as the public BERT checkpoint used to validate the CrowS-Pairs scorer implementation. | fixed |
| linter-2-1 | R6 | Method capability gate | Removed run-in-heading style by keeping the WinoBias capability-gate material as ordinary paragraph text under the existing subsection. | fixed |
| checker-2-1 | R23 | tab:datastats | Preserved the complete tab:datastats float and all numeric values. | fixed |
| checker-2-3 | R23 | tab:gate | Preserved the complete tab:gate float and all numeric values. | fixed |
| checker-2-4 | R23 | tab:reprosettings | Preserved the complete tab:reprosettings float and all numeric values. | fixed |
| checker-2-2 | R23 | tab:runfacts | Preserved the complete tab:runfacts float and all numeric values. | fixed |
| kimi-2-3 | R11 | Abstract | Added plain-language explanations for Holm correction and the alternative scorers, and removed result-specific counts from the abstract wording. | fixed |
| linter-ds-2-1 | R1 | Abstract | Kept field context at the abstract opening and shortened the opening context to avoid an overlong first sentence. | fixed |
| linter-ds-2-6 | R4 | Architectures | Kept the subsection opening contextual and retained the transition from method purpose to the architecture table. | fixed |
| opus-2-14 | R15 | Discussion first paragraph | Removed the defensive execution-error sentence and replaced it with the specific checks applied identically across encoders. | fixed |
| kimi-2-5 | R11 | Experimental Setup and Results | Replaced ambiguous 'deepest' wording in prose with 'lowest-loss matched point' where it referred to validation-loss matching. | fixed |
| checker-ms-2-1 | R10 | Floats | Preserved all non-duplicative floats and clarified the distinct roles of tables and figures in the results text. | fixed |
| opus-2-22 | R25 | Introduction final paragraph | Removed the bare whole-paper roadmap and replaced it with a transition to Related Work. | fixed |
| opus-2-2 | R15 | Introduction second paragraph | Started directly with the CrowS-Pairs pair and the task instead of describing the paper layout. | fixed |
| kimi-2-7 | R13 | Limitations | Deleted the unidentified post-hoc sensitivity check and named the analyses that were performed after examining the single-point result. | fixed |
| grok-2-6 | R13 | Method architectures | Scoped the HyperloopBERT result to the encoder-scale masked-language-model comparison on CrowS-Pairs and StereoSet at the analysed matched checkpoints. | fixed |
| kimi-2-8 | R15 | Results capability gate | Replaced 'perform at chance' with a statement that WinoBias accuracies straddle the chance level and cited Table gate. | fixed |
| kimi-2-4 | R11 | Results continuous adjustment | Defined wild cluster bootstrap as resampling snapshot-level residual patterns while preserving dependence among item rows from the same snapshot. | fixed |
| checker-ms-2-19 | R24 | Limitations | Retained the limitations section, the failed capability gate, the single-seed caveat, and calibrated qualifiers. | fixed |

## Round 4

Findings given: 64. Fixes reported: 64.

| Finding | Rule | Location | Change | Status |
| --- | --- | --- | --- | --- |
| opus-3-3 | R11 | Abstract | Rewrote the abstract sentence to define the measured effect as a per-item stereotype score difference and to state that HyperloopBERT required a lower learning rate. | fixed |
| linter-3-6 | R14 | Experimental Setup | Split the long HyperloopBERT bundle sentence into two shorter sentences. | fixed |
| grok-3-2 | R17 | Introduction | Changed the replication-like wording from rechecks the comparison to runs the comparison. | fixed |
| linter-3-8 | R25 | Introduction | Reordered the Introduction to start with the practical motivation, then background, gap, objective, framework, methodology, and contributions. | fixed |
| linter-3-4 | R14 | Introduction | Split the contrast and Holm-correction definitions into short sentences. | fixed |
| opus-3-8 | R13 | Introduction | Replaced the imprecise architecture-inference sentence with a sentence about attributing an architecture difference from one benchmark score. | fixed |
| linter-ds-3-19 | R25 | Introduction | Added an explicit practical motivation at the opening of the Introduction. | fixed |
| opus-3-6 | R25 | Introduction | Removed the glossary-style paragraph and introduced definitions at the points where the argument needs them. | fixed |
| linter-3-5 | R14 | Matching, scoring, and inference | Split the paired-difference and contrast sentence into shorter sentences. | fixed |
| kimi-3-7 | R13 | Method, capability gate | Added sentences explaining that WinoBias replaces the pronoun with [MASK], scores candidate pronouns, and counts a prediction as correct when the higher-scoring pronoun matches the gold referent. | fixed |
| opus-3-7 | R15 | Reproducibility | Replaced the generic opening with a sentence naming the exact materials needed to rerun the study. | fixed |
| opus-3-1 | R15 | Reproducibility | Deleted the three tautological definition sentences for AdamW, bfloat16, and H100 and named them once with a table citation. | fixed |
| opus-3-5 | R25 | Results / Sensitivity analyses | Rewrote the sensitivity subsection so each check follows from the previous finding instead of using ordinal template openings. | fixed |
| grok-3-3 | R17 | Results / Single-point comparison | Removed the unsupported no-reduction claim and stated only that the two contrasts against VanillaBERT are similar and that no direct test is reported. | fixed |
| opus-3-4 | R10 | Results, scorer check | Removed the text use of the scorer panel and cropped Figure fig:robust to show only category contrasts, leaving scorer exact values in Table tab:rules2. | fixed |
| opus-3-2 | R15 | Results, scorer check | Replaced the circular validation sentence with a plain statement that bert-base-uncased was rescored with this implementation. | fixed |
| checker-ms-3-3 | R24 | Limitations | Expanded the limitations section to restore the single-seed, non-convergence, HyperloopBERT bundle, exploratory-analysis, benchmark-quality, category, and English-only caveats. | fixed |
| linter-3-7 | R14 | Single-point comparison and concrete predictions | Split the masked-pronoun probe description into short sentences. | fixed |
| linter-3-2 | R14 | Abstract | Split the long abstract sentence about inherited lexical association into shorter sentences. | fixed |
| linter-3-1 | R6 | Method capability gate | Kept WinoBias discussion as ordinary paragraph text and did not use a run-in heading. | fixed |
| checker-3-1 | R23 | tab:datastats | Retained Table tab:datastats with all numeric values unchanged. | fixed |
| checker-3-3 | R23 | tab:gate | Retained Table tab:gate with all numeric values unchanged. | fixed |
| checker-3-4 | R23 | tab:reprosettings | Retained Table tab:reprosettings with all numeric values unchanged. | fixed |
| checker-3-2 | R23 | tab:runfacts | Retained Table tab:runfacts with all numeric values unchanged and only renamed plateau row labels. | fixed |
| grok-3-1 | R17 | Title | Narrowed the title to this weight-reusing encoder comparison and its matching and scoring checks. | fixed |
| kimi-3-2 | R11 | Abstract | Explained in the abstract that the released scorer assigns the stereotypical role from the dataset direction label. | fixed |
| opus-3-18 | R12 | Abstract | Replaced those choices with the named matched training point and scoring rule. | fixed |
| linter-ds-3-6 | R4 | Architectures | Verified that the subsection opens with context before technical details and retained the opening. | fixed |
| linter-ds-3-4 | R4 | Data | Verified that the Data section opens with context and retained the opening. | fixed |
| kimi-3-4 | R11 | Data, CrowS-Pairs paragraph | Defined shared tokens before defining an unaligned shared-token pair. | fixed |
| linter-ds-3-16 | R4 | Declaration on Generative AI | Verified that the declaration opens with context and retained the required wording. | fixed |
| grok-3-5 | R17 | Discussion | Changed advice to other reports into a statement about what this study's comparison supports. | fixed |
| linter-ds-3-12 | R4 | Discussion | Verified that the Discussion opens with context and retained the opening. | fixed |
| kimi-3-14 | R12 | Discussion | Named both conventions in the sentence: treating sent_more as stereotypical and using the dataset direction label. | fixed |
| linter-ds-3-8 | R4 | Experimental Setup | Verified that the section opens with context and retained the opening. | fixed |
| opus-3-12 | R13 | Experimental Setup | Stated in prose that each encoder was trained once and cited Table tab:reprosettings. | fixed |
| kimi-3-10 | R11 | Table tab:runfacts | Renamed early plateau exit rows to initial flat-loss region exit. | fixed |
| linter-ds-3-2 | R4 | Introduction | Verified that the Introduction opens with field context and practical motivation. | fixed |
| opus-3-14 | R15 | Introduction | Replaced the abstract conceptual-framework sentence with plain sentences stating what is varied, held fixed, and measured. | fixed |
| linter-ds-3-13 | R4 | Limitations | Verified that the Limitations section opens with context and retained the opening. | fixed |
| opus-3-16 | R15 | Limitations | Deleted the sentence commenting on FIRE's regional focus and kept the factual statement about the unused caste-and-religion instrument. | fixed |
| linter-ds-3-7 | R4 | Matching, scoring, and inference | Verified that the subsection opens with context and retained the opening. | fixed |
| linter-ds-3-17 | R5 | Method | Verified that text remains between the section heading and the first subsection heading. | fixed |
| kimi-3-5 | R11 | Method, Architectures | Added a sentence defining shared ratio as the fraction of effective-depth block applications that reuse block parameters. | fixed |
| kimi-3-6 | R11 | Method, Matching, scoring, and inference | Defined a band as a pre-set target-loss level whose crossing selects a snapshot. | fixed |
| grok-3-6 | R19 | Related Work | Compressed the CrowS-Pairs and StereoSet mechanics to one cited sentence. | fixed |
| linter-ds-3-3 | R4 | Related Work | Verified that Related Work opens with context and retained the opening. | fixed |
| opus-3-17 | R15 | Related Work | Replaced over-reading a single aggregate with a concrete statement about item-quality problems weakening conclusions from one benchmark average. | fixed |
| linter-ds-3-14 | R4 | Reproducibility | Rewrote the Reproducibility opening to state what a reader needs to rerun the study. | fixed |
| linter-ds-3-18 | R5 | Results | Verified that text remains between the Results heading and the first subsection heading. | fixed |
| grok-3-7 | R13 | Results / Sensitivity analyses | Named the exploratory comparisons as CrowS-Pairs matched-point contrasts against VanillaBERT. | fixed |
| opus-3-15 | R15 | Results, StereoSet check | Replaced instrument-dependent wording with a concrete statement that CrowS-Pairs and StereoSet flag different encoders after correction. | fixed |
| kimi-3-13 | R13 | Results, StereoSet check | Added a sentence defining the StereoSet paired effect as a pseudo-log-likelihood difference between stereotypical and anti-stereotypical fillers in the blank position. | fixed |
| opus-3-10 | R15 | Results, capability gate | Replaced straddle wording with a plain statement that accuracies for every encoder and split lie just above and just below chance. | fixed |
| opus-3-11 | R15 | Results, continuous adjustment | Removed the unused OLS abbreviation and stated that the model is fitted by least squares. | fixed |
| kimi-3-8 | R13 | Results, continuous adjustment | Replaced reported threshold with the named decision rule used for bold entries in Table tab:model. | fixed |
| kimi-3-9 | R15 | Results, single-point comparison | Replaced track each other closely with a literal statement that the validation-loss curves nearly overlap in Figure fig:training. | fixed |
| checker-ms-3-4 | R24 | Capability gate result | Preserved the failed capability-gate result in Results, Limitations, and the gate table. | fixed |
| linter-ds-3-11 | R4 | Sensitivity analyses | Verified that the subsection opens with context and retained the opening in revised form. | fixed |
| linter-ds-3-10 | R4 | Single-point comparison and concrete predictions | Verified that the subsection opens with context and retained the opening. | fixed |
| kimi-3-11 | R11 | Table tab:rules2 caption | Changed deepest matched point to lowest-loss matched point in the caption. | fixed |
| kimi-3-12 | R13 | Tables tab:rules2, tab:band, tab:qual, tab:stereoset | Explained Sig. and bolding in Table tab:rules2 and expanded ALBERT headers to ALBERTLoopedBERT where the ambiguity occurred. | fixed |
| checker-ms-3-2 | R10 | duplicate_pairs | Recorded the confirmed duplicate between the scorer panel of Figure fig:robust and Table tab:rules2 and removed the panel from the displayed figure. | fixed |
| checker-ms-3-1 | R10 | float_summaries | Added one-line float summaries in this fix log. | fixed |

