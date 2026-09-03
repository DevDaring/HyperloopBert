# Glossary

Every term the paper uses that a reader new to the domain would not know.

| Term | Plain definition | Defined at |
| --- | --- | --- |
| [CLS] | [CLS] is a special boundary token that marks the start of the sequence and is excluded from full-sentence scoring. | Method |
| [MASK] | [MASK] is the special token inserted where the masked language model must predict a missing token. | Method and Results |
| [SEP] | [SEP] is a special boundary token that marks the end or separation boundary and is excluded from full-sentence scoring. | Method |
| AdamW | AdamW is the optimiser named in the reproducibility settings for pre-training. | Reproducibility |
| ALBERT | ALBERT is a BERT variant that shares encoder-block parameters across layers. | Introduction |
| ALBERTLoopedBERT | ALBERTLoopedBERT is an encoder that reuses one block across the whole stack, following ALBERT. | Introduction |
| antistereo | The direction label antistereo means that sent_more is the anti-stereotypical member. | Data |
| architecture bundle | An architecture bundle is a design that changes more than one mechanism at the same time. | Method |
| band | A band is a pre-set target-loss level whose crossing selects a snapshot. | Method |
| BERT | BERT is the original bidirectional transformer encoder design cited as the unshared reference architecture. | Introduction |
| bert-base-uncased | bert-base-uncased is the released BERT checkpoint rescored to check the CrowS-Pairs scorer implementation. | Results |
| bfloat16 | bfloat16 is the reduced-precision floating-point format named in the reproducibility settings for autocast. | Reproducibility |
| capability gate | The capability gate is the pre-specified set of checks required before interpreting stereotype measurements as more than lexical association. | Method |
| checkpoint | A checkpoint is a saved model state at a training step. | Introduction |
| CLS | The full-sentence rule scores every position except the special boundary tokens [CLS] and [SEP] . | Matching, scoring, and inference (line 411) |
| contrast | A contrast is the mean paired difference between the unshared baseline and another encoder over the same benchmark items. | Introduction |
| correlation-type benchmark | A correlation-type benchmark measures association rather than unequal treatment in downstream decisions. | Related Work |
| CrowS-Pairs | CrowS-Pairs is a stereotype benchmark of paired sentences that differ in a demographic term. | Introduction |
| effective depth | Effective depth means the number of transformer-block applications made to each token. | Introduction |
| FineWeb-Edu | FineWeb-Edu is a filtered subset of web text selected for educational content. | Data |
| FIRE | FIRE is the Forum for Information Retrieval Evaluation venue named in the paper. | front matter |
| full-sentence rule | The full-sentence rule scores every position except special boundary tokens. | Introduction and Method |
| GLUE | GLUE is a general language understanding benchmark used in the planned capability gate. | Method |
| H100 | H100 is the GPU type named in the reproducibility settings for pre-training. | Reproducibility |
| Holm correction | Holm correction is a step-down procedure that controls family-wise error when several contrasts are tested against the same baseline. | Introduction |
| HyperloopBERT | HyperloopBERT is a looped encoder that adds parallel representation streams mixed by learned weights. | Introduction |
| information retrieval system | An information retrieval system ranks documents for a user query. | Introduction |
| language encoder | A language encoder maps a text sequence into contextual token representations. | Abstract |
| lexical association | A lexical association is a model preference for words or sentences that co-occur with a social group in benchmark text. | Abstract and Introduction |
| LMS | LMS is the StereoSet language-modelling score, the benchmark percentage favouring a meaningful completion over an unrelated one. | Results |
| LoopedBERT | LoopedBERT is an encoder that reuses a small core of blocks across depth. | Introduction |
| lowest-loss matched point | The lowest-loss matched point is the common matched checkpoint band with the lowest validation-loss target reached by all compared encoders. | Results |
| MASK | **UNDEFINED - the paper uses this term without explaining it** | Matching, scoring, and inference (line 473) |
| masked language model | A masked language model is an encoder trained to predict tokens hidden from their surrounding words. | Introduction |
| masked language modelling head | The masked language modelling head is the output layer that assigns probabilities to tokens. | Introduction |
| matched point | A matched point is a checkpoint selected by crossing the same validation-loss target. | Introduction |
| matched training point | A matched training point compares saved model states with similar held-out text loss. | Abstract |
| model capability | Model capability here means language-modelling quality, measured by validation loss on held-out text. | Introduction |
| official rule | The official rule is the CrowS-Pairs released scorer that uses shared-token alignment, sums log probabilities, and follows the direction label. | Method |
| Phipson-Smyth estimate | The Phipson-Smyth estimate avoids a zero permutation p-value when a finite number of random sign flips is sampled. | Method |
| pruning | Pruning means removing model weights after training. | Related Work |
| pseudo-log-likelihood | A pseudo-log-likelihood is the average log probability assigned to original tokens when each token is masked in turn. | Related Work |
| race-color | race-color is the CrowS-Pairs label for its race and colour category. | Data |
| residual connection | A residual connection adds a block input to the block output before the next layer. | Related Work |
| sent\_more | The field sent\_more stores one member of the sentence pair. | Data (line 208) |
| sent_more | sent_more is the CrowS-Pairs field that stores one member of the sentence pair. | Data |
| SEP | The full-sentence rule scores every position except the special boundary tokens [CLS] and [SEP] . | Matching, scoring, and inference (line 411) |
| shared ratio | The shared ratio is the fraction of effective-depth block applications that reuse block parameters. | Method |
| shared tokens | Shared tokens are tokens that occur in both members of a sentence pair. | Data |
| shared-token rule | The shared-token rule scores only tokens that the two paired sentences have in common. | Method |
| social stereotype association | A social stereotype association is a higher model score for text that links a social group with a common stereotype. | Introduction |
| SS | SS is the StereoSet stereotype score, the benchmark percentage favouring the stereotypical completion. | Results |
| stereo | The direction label stereo means that sent_more is the stereotypical member. | Data |
| StereoSet | StereoSet is a stereotype benchmark that presents a context with stereotypical, anti-stereotypical, and unrelated completions. | Abstract and Data |
| stereotype benchmark | A stereotype benchmark estimates social associations by comparing model scores for paired texts. | Introduction |
| stream | A stream is one parallel copy of token representations inside HyperloopBERT. | Introduction |
| text classifier | A text classifier assigns labels to text documents. | Introduction |
| tokeniser | A tokeniser is the fixed procedure that splits text into model tokens. | Introduction |
| transformer block | A transformer block is the repeated attention-and-feed-forward unit used in BERT. | Introduction |
| unaligned shared-token pair | An unaligned shared-token pair is a pair for which the shared-token alignment cannot recover the common token sequence. | Data |
| validation loss | Validation loss is the average negative log probability on held-out text, so lower values indicate better language modelling. | Introduction |
| VanillaBERT | VanillaBERT is the unshared baseline with independent transformer blocks. | Introduction |
| weight reuse | Weight reuse means that the same block parameters are applied at more than one depth in the network. | Abstract and Introduction |
| wild cluster bootstrap | A wild cluster bootstrap resamples snapshot-level residual patterns while preserving dependence among item rows from the same snapshot. | Results |
| WinoBias | WinoBias is a pronoun-resolution dataset used here as a capability check rather than a bias measure. | Abstract and Data |
