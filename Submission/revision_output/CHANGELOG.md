# Changelog

Per-round fix log from the Writer (ChatGPT). Each entry maps a finding to the change made.

## Round 1

Findings given: 19. Fixes reported: 19.

| Finding | Rule | Location | Change | Status |
| --- | --- | --- | --- | --- |
| seed-0-1 | R25 | Introduction | Added a conceptual-framework paragraph between the objective paragraph and the methodology paragraph, defining the architecture comparison in terms of architecture, realised validation loss, scorer convention, and capability. | fixed |
| seed-0-2 | R14 | Introduction, CrowS-Pairs example paragraph | Combined the repeated scoring-step sentences into two procedural sentences that describe masking, probability recording, averaging, and subtraction without the staccato sequence. | fixed |
| seed-0-3 | R25 | Introduction, final contribution paragraph | Rewrote the contribution paragraph as an argument about what the study establishes and closed by handing the reader to Related Work. | fixed |
| seed-0-4 | R15 | Introduction, first paragraph | Replaced the vague sentence about encoder choice with a measurable claim that different encoders can assign different stereotype-association scores before fine-tuning. | fixed |
| seed-0-5 | R13 | Introduction, first paragraph | Named CrowS-Pairs and StereoSet as stereotype benchmarks used for pre-selection evidence and cited their benchmark papers in the model-selection sentence. | fixed |
| seed-0-6 | R25 | Introduction, paragraph beginning 'The comparison varies enc | Removed the glossary-style block and introduced tokeniser, matched point, contrast, and Holm correction inside the argument where each term is needed. | fixed |
| seed-0-7 | R15 | Introduction, paragraph beginning 'The practical question' | Replaced 'more concentrated probabilities' with the checkable statement that lower validation loss can change the size of paired score differences on benchmark items. | fixed |
| seed-0-8 | R11 | Matching, scoring, and inference | Defined the token \texttt{[MASK]} as the placeholder inserted at a hidden position before later uses in scoring and WinoBias. | fixed |
| seed-0-9 | R13 | Method, Architectures, opening sentence | Named the HyperloopBERT peak learning rate as the single non-architectural exception and cited Table~\ref{tab:reprosettings}. | fixed |
| seed-0-10 | R13 | Results, continuous-adjustment paragraph | Stated that bold marks terms for which both the snapshot and bootstrap p-values pass the stated table decision rule. | fixed |
| seed-0-11 | R13 | Results, continuous-adjustment paragraph | Replaced the vague 'single-point reading' claim with an explicit comparison between the realised-loss coefficient in Table~\ref{tab:model} and the single-point contrasts in Table~\ref{tab:band}. | fixed |
| seed-0-12 | R15 | Results, subsection heading | Renamed the subsection to 'Matched-point comparison and probe outputs' so the heading matches the reported matched-point comparison and masked-pronoun probabilities. | fixed |
| seed-0-13 | R14 | Matched-point comparison and probe outputs | Split the long probe sentence into shorter sentences about the scientist, teacher, and secretary probes. | fixed |
| seed-0-14 | R6 | Method, capability-gate paragraph | Kept \label{sec:gate} inside the capability-gate paragraph and used no run-in heading before the WinoBias capability description. | fixed |
| seed-0-15 | R6 | Results, scorer paragraph | Kept \label{sec:scorer} inside a normal paragraph and used no run-in heading before the CrowS-Pairs scoring-rule analysis. | fixed |
| seed-0-16 | R23 | tab:datastats | Retained Table~\ref{tab:datastats} with all numeric values unchanged. | fixed |
| seed-0-17 | R23 | tab:gate | Retained Table~\ref{tab:gate} with all numeric values unchanged. | fixed |
| seed-0-18 | R23 | tab:reprosettings | Retained Table~\ref{tab:reprosettings} with all numeric values unchanged. | fixed |
| seed-0-19 | R23 | tab:runfacts | Retained Table~\ref{tab:runfacts} with all numeric values unchanged. | fixed |

## Round 2

Findings given: 62. Fixes reported: 62.

R23 guard: 1 float(s) had changed values; restored.

| Finding | Rule | Location | Change | Status |
| --- | --- | --- | --- | --- |
| checker-1-1 | R3 | front matter (title/abstract) | Removed the CCSXML block with classification identifiers and significance digits, and kept digit-free \ccsdesc commands. | fixed |
| grok-1-4 | R17 | Abstract | Replaced the lexical-association-only claim with a statement that architecture attribution and downstream model-selection claims are not supported in this study. | fixed |
| grok-1-8 | R13 | Abstract | Named the CrowS-Pairs contrasts and stated that the loss-adjusted regression and alternative scorer checks do not preserve them. | fixed |
| opus-1-1 | R15 | Abstract | Removed the detached definitions of weight reuse and matched training point from the abstract and described the actual comparison of saved model states and scoring rules. | fixed |
| opus-1-2 | R11 | Abstract | Replaced the undefined word bundle with a plain statement that the HyperloopBERT comparison mixes stream design with a training-stability change. | fixed |
| opus-1-3 | R15 | Abstract; Results; Conclusion | Replaced all uses of remains corrected with statements that the HyperloopBERT contrast stays significant after Holm adjustment. | fixed |
| opus-1-11 | R13 | Data and Results labels sec:scorer, sec:cont, sec:category | Changed references to paragraph labels as Sections into references to the named Results analyses or the Results section. | fixed |
| opus-1-7 | R15 | Data, CrowS-Pairs paragraph | Folded the field and direction-label definitions into one prose sentence tied to the CrowS-Pairs item structure. | fixed |
| grok-1-5 | R17 | Discussion | Replaced the conditional model-selection claim with the statement that the comparison does not support model-selection claims in this study. | fixed |
| opus-1-5 | R25 | End of Introduction, Related Work, Data, Method, Discussion, | Removed repeated template hand-off sentences and kept only transitions that carry substantive content. | fixed |
| grok-1-2 | R17 | Introduction | Changed inheritance by later systems into an untested motivation about model-selection evidence before fine-tuning. | fixed |
| grok-1-3 | R17 | Introduction | Replaced establishes with reports for this run and named the analysis choices being checked. | fixed |
| linter-1-11 | R25 | Introduction | Reordered the introduction into motivation, background, gap, objective, conceptual framework, methodology, and contribution paragraphs. | fixed |
| linter-1-9 | R14 | Introduction | Split the long contrast and Holm-correction sentence into shorter definition sentences. | fixed |
| linter-1-8 | R14 | Introduction | Split the long motivation sentence about pre-trained encoders and later systems into shorter sentences. | fixed |
| opus-1-13 | R13 | Introduction, conceptual framework paragraph | Specified the three choices as matched checkpoint, scoring rule, and direction convention, and defined the contrast sign convention. | fixed |
| opus-1-9 | R15 | Limitations | Removed the repeated sentence and kept one sentence stating that a single seed means no run is repeated. | fixed |
| linter-1-10 | R14 | Matched-point comparison and probe outputs | Split the long probe-output sentence into separate sentences about the scientist and teacher probes. | fixed |
| grok-1-6 | R19 | Related Work | Condensed benchmark-audit findings into one cited sentence and moved to the architecture-comparison gap. | fixed |
| opus-1-8 | R15 | Reproducibility | Removed stacked glossary sentences and defined AdamW, bfloat16, and H100 in the training-setup sentence. | fixed |
| kimi-1-9 | R13 | Results, Sensitivity analyses, Table model text | Stated that bold requires both p-values to be below the chosen five-percent alpha and clarified the Sig. rule for Table rules2. | fixed |
| linter-1-7 | R14 | front matter (title/abstract) | Removed the long CCSXML text block from the front matter. | fixed |
| linter-1-5 | R6 | line 630 | Kept he and she only as inline pronoun labels in prose and table entries, not as a run-in paragraph heading. | fixed |
| grok-1-1 | R17 | title | Changed the title to Architecture Bias Comparisons Under Matched Loss and Scoring Choices. | fixed |
| linter-ds-1-1 | R1 | Abstract | Retained a contextual opening and revised the abstract as a single paragraph within the venue limit. | fixed |
| opus-1-16 | R15 | Conclusion | Replaced the negated construction with a direct sentence stating that no contrast stays significant under either alternative rule. | fixed |
| opus-1-6 | R25 | Data; Method; Experimental Setup | Rewrote decorative openings so each section begins with the substantive fact needed for the argument. | fixed |
| opus-1-15 | R15 | Discussion | Changed pipelines choose to developers of retrieval and classification systems select among pre-trained encoders. | fixed |
| opus-1-17 | R11 | Discussion | Removed the undefined term pipeline checks and stated directly that all encoders used the same data order, scorer code, and snapshot-selection rule. | fixed |
| checker-ms-1-1 | R10 | Floats section | Reviewed the cited table and figure and retained both because the finding states they do not duplicate information. | fixed |
| kimi-1-1 | R11 | Introduction, CrowS-Pairs example paragraph | Defined boundary tokens at first use as the special start and end markers. | fixed |
| kimi-1-3 | R11 | Introduction, comparison paragraph | Added a first-use definition of representation width as the size of each token vector inside the encoder. | fixed |
| opus-1-14 | R15 | Introduction | Replaced attributed from with a sentence saying one benchmark score cannot separate an architecture effect from a capability effect. | fixed |
| kimi-1-2 | R11 | Introduction, gap paragraph | Replaced sign with direction of the paired score difference before formally defining contrast sign later. | fixed |
| kimi-1-13 | R11 | Limitations | Replaced mirror with available hosted copy of the dataset. | fixed |
| linter-ds-1-14 | R26 | Method section | Kept the Method subsections because they are substantial and retained a short contextual paragraph before the first subsection. | fixed |
| kimi-1-4 | R13 | Method, antistereo example | Named sent_more as the anti-stereotypical sentence and removed the unsupported citation to the counts table for the item label. | fixed |
| grok-1-7 | R19 | Related Work | Reduced the compression-and-fairness recap to one cited sentence about pruning and compression changing fairness measurements. | fixed |
| kimi-1-14 | R11 | Reproducibility | Defined the manifest as a file listing each model's block count and file checksums. | fixed |
| grok-1-9 | R13 | Results / StereoSet | Named the StereoSet HyperloopBERT contrast that survives Holm correction and the CrowS-Pairs LoopedBERT and HyperloopBERT contrasts. | fixed |
| grok-1-10 | R13 | Results / matched-point comparison | Replaced similar with a statement about CrowS-Pairs Delta values against VanillaBERT and cited Table band in the same sentence. | fixed |
| linter-ds-1-15 | R26 | Results section | Kept the Results subsections because they cover substantial parts and retained a contextual paragraph before the first subsection. | fixed |
| kimi-1-5 | R11 | Results, Matched-point comparison and probe outputs | Defined the flat-loss region as the early training phase where validation loss stays nearly constant. | fixed |
| kimi-1-11 | R11 | Results, Sensitivity analyses, capability gate | Replaced coreference component with pronoun-resolution component. | fixed |
| kimi-1-8 | R12 | Results, Sensitivity analyses, continuous adjustment | Replaced This with The absence of a detectable architecture coefficient. | fixed |
| kimi-1-6 | R11 | Results, scorer paragraphs | Glossed bert-base-uncased as the public base BERT model trained on lowercased text. | fixed |
| kimi-1-7 | R11 | Results, scorer paragraphs | Defined prefix-and-suffix alignment as matching the common prefix and suffix of the two sentences. | fixed |
| opus-1-10 | R15 | Results, StereoSet paragraph | Replaced flag with a literal statement that the two benchmarks do not support the same set of corrected contrasts. | fixed |
| kimi-1-10 | R9 | Results, Table qual | Aligned the probability-ratio header across all probe columns and stated in the body what the bold entries mark. | fixed |
| linter-ds-1-11 | R4 | Section 'Conclusion' | Retained the contextual opening and revised the surrounding conclusion for clearer flow. | fixed |
| linter-ds-1-12 | R4 | Section 'Declaration on Generative AI' | Retained the required declaration wording with a contextual opening sentence. | fixed |
| linter-ds-1-8 | R4 | Section 'Discussion' | Retained the contextual opening and revised the section to progress from matched-point evidence to scorer dependence and limitations for model selection. | fixed |
| linter-ds-1-9 | R4 | Section 'Limitations' | Retained the contextual opening and removed only the repeated sentence. | fixed |
| linter-ds-1-2 | R4 | Section 'Method' | Retained a context paragraph before the first Method subsection. | fixed |
| linter-ds-1-10 | R4 | Section 'Reproducibility' | Retained the contextual opening and integrated definitions into the setup prose. | fixed |
| linter-ds-1-5 | R4 | Section 'Results' | Retained a contextual Results opening before the first subsection. | fixed |
| checker-ms-1-3 | R24 | Limitations | Preserved the single-seed caveat and its interpretation that intervals do not quantify variation between training runs. | fixed |
| checker-ms-1-4 | R24 | Limitations and capability-gate outcomes | Preserved the failed capability-gate result and the statement that downstream behaviour is not supported. | fixed |
| linter-ds-1-3 | R4 | Subsection 'Architectures' | Retained the contextual subsection opening that links architecture to the fixed non-architectural inputs. | fixed |
| linter-ds-1-6 | R4 | Subsection 'Matched-point comparison and probe outputs' | Retained the contextual subsection opening identifying the lowest-loss matched point. | fixed |
| linter-ds-1-4 | R4 | Subsection 'Matching, scoring, and inference' | Retained the contextual subsection opening about pre-set validation-loss targets. | fixed |
| linter-ds-1-7 | R4 | Subsection 'Sensitivity analyses' | Retained the contextual subsection opening that motivates checks from the single-point result. | fixed |

## Round 3

Findings given: 43. Fixes reported: 43.

| Finding | Rule | Location | Change | Status |
| --- | --- | --- | --- | --- |
| opus-2-1 | R15 | Introduction, prior-work paragraph | Replaced the ambiguous sentence with two plain sentences stating that audits found unclear items and that averaging one association score over such items weakens conclusions. | fixed |
| checker-2-1 | R3 | front matter (title/abstract) | Removed the CCSXML and ccsdesc metadata block so classification identifiers and significance weights no longer appear as uncited prose-like numbers in the source. | fixed |
| grok-2-1 | R13 | Abstract | Named the full-sentence scorer and stated that the measured quantity is the mean paired item difference for LoopedBERT and HyperloopBERT versus VanillaBERT. | fixed |
| linter-2-13 | R25 | Introduction | Reordered the Introduction into motivation, background, research gap, objectives, conceptual framework, methodology, and contributions, with explicit transition sentences for each stage. | fixed |
| linter-2-9 | R14 | Introduction (line 119) | Split the long definition paragraph into shorter sentences and removed the ambiguous prior-work sentence. | fixed |
| opus-2-6 | R15 | Introduction, conceptual framework paragraph | Rewrote the conceptual-framework claim to state that a comparison is accepted only if the contrast keeps the same sign at all matched points, under all scoring rules, and under both direction conventions. | fixed |
| opus-2-4 | R25 | Introduction, final paragraph | Replaced the results-dump closing paragraph with a contribution paragraph that states the protocol, the main checks, and the resulting limits. | fixed |
| opus-2-2 | R25 | Introduction, paragraph beginning This paper asks whether | Distributed definitions across the motivation, background, gap, objective, framework, and methodology paragraphs so the paragraph now advances the argument instead of forming a glossary block. | fixed |
| kimi-2-9 | R13 | Method, Architectures, hyper-connection equation | Added a sentence after the equation stating that the implementation fixes n as the stream count and d as the shared representation width, with exact values in the released configuration. | deferred |
| opus-2-5 | R6 | Method and Results paragraph labels | Moved the orphan labels sec:gate, sec:qualitative, sec:cont, sec:scorer, sec:stereoset, sec:category, and sec:capability onto separate lines before their paragraphs and rewrote the first sentences as transitions. | fixed |
| grok-2-6 | R13 | Abstract | Separated the WinoBias gate failure from the scorer and loss sensitivity failures, so the abstract no longer attributes all architecture limits to the gate failure. | fixed |
| kimi-2-1 | R11 | Abstract | Replaced unexplained technical phrases with plain descriptions: a regression that adjusts for validation loss and checks with different scoring rules. | fixed |
| linter-ds-2-1 | R1 | Abstract (line 26) | Retained the contextual opening sentence and kept the abstract as one paragraph within the stated venue limit. | fixed |
| opus-2-14 | R15 | Conclusion, final paragraph | Deleted the filler sentence The evidence supports a measurement recommendation and let the next sentence state the recommendation directly. | fixed |
| opus-2-13 | R15 | Data, evaluation paragraph | Merged the duplicate benchmark-training sentences into one sentence saying the encoders are scored on the benchmarks without fine-tuning. | fixed |
| opus-2-9 | R15 | Discussion, second paragraph | Replaced Still matters with a measurable statement that remaining differences in realised validation loss are associated with the CrowS-Pairs item effect. | fixed |
| opus-2-16 | R15 | Introduction, first paragraph | Replaced the rhetorical phrase motivate, but do not test with a literal statement that pre-fine-tuning scores do not measure later ranker or classifier behaviour. | fixed |
| linter-ds-2-34 | R26 | Method section (line 319) | Retained the Method opening and its two substantial subsections because the subsections organize architecture details separately from scoring and inference. | fixed |
| kimi-2-12 | R15 | Method, scoring rules | Rewrote the official-rule description to state that the released alignment selects token spans matched in both sentence members, then sums their log probabilities. | fixed |
| opus-2-18 | R15 | Method, scoring rules paragraph | Combined the two demographic-term sentences into one sentence stating that demographic terms are not scored but remain visible as conditioning context. | fixed |
| grok-2-4 | R17 | Related Work | Replaced the claim that the study contributes evidence behind an architecture-selection claim with a statement that the study tests the claim and does not support it. | fixed |
| grok-2-5 | R13 | Results / Matched-point comparison | Removed the untested closeness claim and replaced it with a statement naming LoopedBERT, HyperloopBERT, the looped layout, and the learning-rate exception. | fixed |
| linter-ds-2-35 | R26 | Results section (line 551) | Retained the Results opening and its two substantial subsections because they separate the single-point comparison from sensitivity analyses. | fixed |
| opus-2-12 | R14 | Results, loss-adjusted regression paragraph | Split the wild-cluster-bootstrap sentence into shorter sentences that first identify the check and then state what is resampled and preserved. | fixed |
| kimi-2-15 | R13 | Results, scorer analysis | Specified that bert-base-uncased was rescored with the official CrowS-Pairs rule for the aggregate stereotype score and clarified that its authors refers to the original BERT authors. | fixed |
| linter-ds-2-11 | R4 | Section Conclusion | Retained the contextual opening of the Conclusion. | fixed |
| linter-ds-2-15 | R4 | Section Data | Retained the contextual opening of the Data section. | fixed |
| linter-ds-2-12 | R4 | Section Declaration on Generative AI | Retained the required declaration text with a contextual opening sentence. | fixed |
| linter-ds-2-8 | R4 | Section Discussion | Retained the contextual opening of the Discussion section. | fixed |
| linter-ds-2-16 | R4 | Section Experimental Setup | Retained the contextual opening of the Experimental Setup section. | fixed |
| linter-ds-2-13 | R4 | Section Introduction | Retained and revised the contextual opening of the Introduction so it starts with the practical setting for retrieval and classification systems. | fixed |
| linter-ds-2-9 | R4 | Section Limitations | Retained the contextual opening of the Limitations section. | fixed |
| linter-ds-2-2 | R4 | Section Method | Retained the contextual opening of the Method section. | fixed |
| linter-ds-2-14 | R4 | Section Related Work | Retained the contextual opening of the Related Work section and kept the section without subheadings. | fixed |
| linter-ds-2-10 | R4 | Section Reproducibility | Retained the contextual opening of the Reproducibility section. | fixed |
| linter-ds-2-3 | R4 | Section Results | Retained the contextual opening of the Results section. | fixed |
| checker-ms-2-17 | R24 | Section 6 (Limitations) | Kept the single-seed limitation and its consequence for training-run variation. | fixed |
| checker-ms-2-18 | R24 | Section 6 (Limitations) | Kept the capability-gate failure and its citation to Table 7. | fixed |
| linter-ds-2-4 | R4 | Subsection Architectures | Retained the contextual opening of the Architectures subsection. | fixed |
| linter-ds-2-6 | R4 | Subsection Matched-point comparison and probe outputs | Retained the contextual opening of the matched-point subsection. | fixed |
| linter-ds-2-5 | R4 | Subsection Matching, scoring, and inference | Retained the contextual opening of the matching and scoring subsection. | fixed |
| linter-ds-2-7 | R4 | Subsection Sensitivity analyses | Retained the contextual opening of the sensitivity analyses subsection. | fixed |
| checker-ms-2-1 | R10 | float captions | Reviewed the captions and retained them because each caption names the distinct content of its own float without duplicating another table or figure. | fixed |

## Round 4

Findings given: 63. Fixes reported: 63.

| Finding | Rule | Location | Change | Status |
| --- | --- | --- | --- | --- |
| opus-3-1 | R25 | Introduction | Removed template stage openers and rewrote the Introduction to flow from motivation, to background, to research gap, to objectives, to conceptual framework, to methodology, and then to contributions. | fixed |
| grok-3-5 | R13 | Abstract | Named the StereoSet intrasentence item set and the paired filler-effect contrast, and defined the matched point and full-sentence rule in plain words. | fixed |
| kimi-3-1 | R11 | Abstract | Added that the encoders differ by how much they reuse block parameters across depth and defined HyperloopBERT streams as separate copies of token vectors. | fixed |
| grok-3-3 | R17 | Abstract and Results capability-gate paragraph | Rephrased the WinoBias outcome as a limit of the unfine-tuned masked-head probe rather than evidence about fine-tuned rankers or classifiers. | fixed |
| linter-3-5 | R14 | Data | Split the long sentence defining race-color, sent_more, and the direction label into three shorter sentences. | fixed |
| grok-3-1 | R17 | Introduction | Removed the claim that the study fully separates architecture and stated that HyperloopBERT does not isolate stream design from its lower learning rate. | fixed |
| kimi-3-6 | R11 | Introduction | Defined both direction conventions and defined sent_more before using those criteria; the depth marker is defined in the architecture subsection before use. | fixed |
| linter-3-3 | R14 | Introduction | Split the opening explanation of encoder scores and stereotype benchmarks into shorter sentences. | fixed |
| linter-3-4 | R14 | Introduction | Split the Holm correction definition into two shorter sentences. | fixed |
| opus-3-3 | R15 | Introduction | Replaced the abstract statement about a framework accepting comparisons with the concrete sign-stability criteria. | fixed |
| opus-3-2 | R15 | Introduction | Replaced the tautological research-gap sentence with the specific statement that no cited prior study compares weight-reusing encoders at matched validation loss. | fixed |
| linter-3-7 | R14 | Results, matched-point comparison and probe outputs | Split the probe-description and probe-result sentence into separate shorter sentences. | fixed |
| linter-3-6 | R14 | Method, matching, scoring, and inference | Split the CrowS-Pairs antistereo example into shorter sentences that name the paired strings, sent_more role, and label separately. | fixed |
| opus-3-5 | R13 | Method, Architectures | Replaced the vacuous configuration sentence with a statement that n is the configured stream count and d is the shared representation width, and explicitly noted that the paper does not tabulate those fields. | fixed |
| opus-3-7 | R15 | Method, capability gate | Rewrote the capability-gate paragraph to state the three pre-set conditions and that all had to hold before downstream model-selection claims. | fixed |
| grok-3-4 | R13 | Results, Sensitivity analyses | Named the full-sentence CrowS-Pairs VanillaBERT-minus-LoopedBERT contrast and stated that it changes sign across matched points. | fixed |
| opus-3-8 | R25 | Results, Sensitivity analyses | Varied the paragraph transitions so each check follows from the specific uncertainty left by the preceding result. | fixed |
| linter-3-1 | R6 | Method | Kept WinoBias as body text under the capability-gate paragraph and did not use it as a run-in paragraph heading. | fixed |
| kimi-3-3 | R11 | Abstract | Defined Holm correction as a multiple-testing correction across contrasts against the baseline and avoided undefined alpha in the abstract. | fixed |
| kimi-3-5 | R11 | Abstract | Glossed the WinoBias capability gate as a pre-set language-understanding check. | fixed |
| linter-ds-3-1 | R1 | Abstract | Retained the contextual opening and kept the abstract as a single paragraph under the venue limit. | fixed |
| opus-3-15 | R15 | Abstract | Replaced the figurative final sentence with literal statements about non-preserved CrowS-Pairs contrasts and near-chance WinoBias performance. | fixed |
| linter-ds-3-6 | R4 | Architectures | Retained the contextual opening sentences for the Architectures subsection and clarified the table description. | fixed |
| linter-ds-3-15 | R4 | Conclusion | Retained the contextual opening of the Conclusion. | fixed |
| linter-ds-3-4 | R4 | Data | Retained the contextual opening of the Data section. | fixed |
| linter-ds-3-16 | R4 | Declaration on Generative AI | Retained the required declaration section and its contextual opening sentence. | fixed |
| opus-3-14 | R15 | Discussion | Replaced the figurative opening with a literal statement that the corrected contrast holds at one point and not under the other checks. | fixed |
| linter-ds-3-8 | R4 | Experimental Setup | Retained the contextual opening of Experimental Setup and made the HyperloopBERT exception explicit. | fixed |
| kimi-3-11 | R11 | Experimental Setup | Defined flat-loss region before Table runfacts uses the term. | fixed |
| opus-3-13 | R15 | Experimental Setup | Replaced the vague statement about usable runs with the specific non-finite-loss reason for the HyperloopBERT lower learning rate. | fixed |
| linter-ds-3-2 | R4 | Introduction | Retained a contextual opening that introduces retrieval systems and classifiers before the problem. | fixed |
| linter-ds-3-13 | R4 | Limitations | Retained the contextual opening of the Limitations section. | fixed |
| linter-ds-3-10 | R4 | Matched-point comparison and probe outputs | Retained the contextual opening of the subsection and added a direct figure description. | fixed |
| linter-ds-3-7 | R4 | Matching, scoring, and inference | Retained the contextual opening of the subsection. | fixed |
| linter-ds-3-5 | R4 | Method | Retained the contextual opening of the Method section. | fixed |
| kimi-3-10 | R12 | Method, Architectures | Replaced 'this implementation' with 'HyperloopBERT' and rewrote the category-analysis referent as 'The category analyses'. | fixed |
| kimi-3-12 | R11 | Method, Matching, scoring, and inference | Explained that confidence intervals come from resampling benchmark items with replacement and taking percentile cutoffs. | fixed |
| kimi-3-8 | R11 | Related Work | Defined masked-position probing as scoring hidden positions to measure an association. | fixed |
| grok-3-8 | R19 | Related Work | Kept the compression-fairness point to one cited sentence and removed repeated elaboration. | fixed |
| linter-ds-3-3 | R4 | Related Work | Retained the contextual opening of Related Work and kept the section without subheadings. | fixed |
| linter-ds-3-14 | R4 | Reproducibility | Retained the contextual opening of Reproducibility. | fixed |
| linter-ds-3-19 | R26 | Results | Retained the consolidated Results structure with two substantive subsections and a contextual opening paragraph. | fixed |
| opus-3-11 | R15 | Results, Matched-point comparison and probe outputs | Replaced the vague aggregate-interpretation sentence with a concrete statement that the probes show direct model outputs behind the aggregate pattern. | fixed |
| linter-ds-3-11 | R4 | Sensitivity analyses | Retained and revised the subsection opening so it states the specific purpose of varying analysis choices. | fixed |
| checker-ms-3-17 | R10 | duplicate_pairs | Checked candidate float pairs and kept them distinct: tables report exact values, while figures show trends or category variation. | fixed |
| checker-ms-3-10 | R10 | fig:delta | Added body text stating that Figure delta plots CrowS-Pairs contrasts against realised validation loss. | fixed |
| checker-ms-3-14 | R10 | fig:robust | Added body text stating that Figure robust shows CrowS-Pairs contrasts by bias category. | fixed |
| checker-ms-3-7 | R10 | fig:training | Added body text stating that Figure training shows validation-loss curves over training tokens. | fixed |
| checker-ms-3-1 | R10 | float_summaries | Ensured each float has a one-line caption and is described in nearby body text as a table of exact values or a figure of trends or variation. | fixed |
| checker-ms-3-19 | R24 | sec:capability | Retained the same capability-gate result that WinoBias accuracies lie around chance and that the gate fails. | fixed |
| checker-ms-3-18 | R24 | sec:limits | Retained the single-seed caveat and clarified that no run is repeated. | fixed |
| checker-ms-3-4 | R10 | tab:arch | Described Table arch in body text as comparing encoders by weight reuse and parameter count. | fixed |
| checker-ms-3-8 | R10 | tab:band | Described Table band in body text as reporting realised loss, effect, interval, contrast, and Holm-corrected p-value. | fixed |
| checker-ms-3-2 | R10 | tab:data | Described Table data in body text as showing concrete examples from the evaluation tasks. | fixed |
| checker-ms-3-3 | R10 | tab:datastats | Described Table datastats in body text as listing exact data choices and dataset sizes. | fixed |
| checker-ms-3-15 | R10 | tab:gate | Described Table gate in body text as reporting WinoBias, GLUE, and overall gate outcomes. | fixed |
| checker-ms-3-11 | R10 | tab:model | Described Table model in body text as reporting the snapshot-level regression and bootstrap results. | fixed |
| checker-ms-3-9 | R10 | tab:qual | Described Table qual in body text as reporting masked-pronoun probabilities for hand-built probes. | fixed |
| checker-ms-3-16 | R10 | tab:reprosettings | Described Table reprosettings in body text as listing implementation settings for reruns. | fixed |
| checker-ms-3-5 | R10 | tab:rules | Described Table rules in body text as summarising the three scoring rules. | fixed |
| checker-ms-3-12 | R10 | tab:rules2 | Described Table rules2 in body text as giving contrasts and Holm-survival counts under scoring rules. | fixed |
| checker-ms-3-6 | R10 | tab:runfacts | Described Table runfacts in body text as giving run-level facts that affect interpretation. | fixed |
| checker-ms-3-13 | R10 | tab:stereoset | Described Table stereoset in body text as reporting paired effect, SS, LMS, contrast, and Holm-corrected p-value. | fixed |

