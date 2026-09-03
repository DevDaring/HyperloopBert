# Glossary

Every term the paper uses that a reader new to the domain would not know.

| Term | Plain definition | Defined at |
| --- | --- | --- |
| [CLS] | The token [CLS] marks the start of the sequence. | Method |
| [MASK] | The token [MASK] is the placeholder inserted at a hidden position. | Method |
| [SEP] | The token [SEP] marks the end or separation boundary. | Method |
| AdamW | AdamW is an optimizer with decoupled weight decay. | Reproducibility |
| ALBERT | ALBERT is a BERT variant that shares encoder-block parameters across layers. | Introduction |
| ALBERTLoopedBERT | ALBERTLoopedBERT reuses one block across all block positions, following ALBERT. | Abstract and Introduction |
| antistereo | The direction label antistereo means that sent_more is the anti-stereotypical member. | Method |
| architecture bundle | An architecture bundle is a design that changes more than one mechanism at the same time. | Method |
| band | A band is a pre-set target-loss level whose crossing selects a snapshot. | Method |
| BERT | BERT is the original bidirectional transformer encoder design used as the basis for these encoders. | Introduction |
| bert-base-uncased | bert-base-uncased is the public base BERT model trained on lowercased text by the original BERT authors. | Results |
| bfloat16 | bfloat16 is a low-precision floating-point format used during training. | Reproducibility |
| boundary tokens | Boundary tokens are the special start and end markers in the token sequence. | Method |
| capability gate | The capability gate is a pre-set set of checks for basic language-understanding ability. | Abstract and Method |
| checkpoint | A checkpoint is a saved model state at a training step. | Introduction |
| CLS | The full-sentence rule scores every position except the special boundary tokens [CLS] and [SEP] . | Matching, scoring, and inference (line 414) |
| contrast | A contrast is the mean paired difference between VanillaBERT and another encoder on the same benchmark items. | Introduction |
| CrowS-Pairs | CrowS-Pairs is a paired-sentence stereotype benchmark. | Abstract and Introduction |
| direction convention | A direction convention decides whether sent_more is treated as stereotypical or whether the dataset direction label assigns the stereotypical role. | Introduction |
| effective depth | Effective depth is the number of transformer-block applications made to each token. | Introduction |
| entry blocks | Entry blocks are the non-reused blocks before the looped core. | Method |
| exit blocks | Exit blocks are the non-reused blocks after the looped core. | Method |
| FineWeb-Edu | FineWeb-Edu is a filtered subset of web text selected for educational content. | Data |
| flat-loss region | A flat-loss region is an early training phase where validation loss stays nearly constant. | Experimental Setup |
| full-sentence rule | The full-sentence rule scores every ordinary token in a sentence while excluding special boundary tokens. | Abstract and Introduction |
| GLUE | GLUE is a general language understanding benchmark used in the planned gate. | Method |
| H100 | H100 names the GPU type used for pre-training. | Reproducibility |
| Holm correction | Holm correction is a multiple-testing correction for several contrasts against one baseline. | Abstract and Introduction |
| hyper-connections | Hyper-connections generalise a residual connection to several parallel streams with learned mixing. | Related Work |
| HyperloopBERT | HyperloopBERT is a looped encoder that adds parallel representation streams mixed by learned weights. | Abstract and Introduction |
| intrasentence split | In the StereoSet intrasentence split, each item supplies a context sentence with a blank and candidate fillers. | Data |
| language encoder | A language encoder maps a text sequence into token vectors whose values depend on the surrounding text. | Abstract and Introduction |
| learned depth marker | A learned depth marker is a learned vector added to token states before each pass through a reused core. | Method |
| LMS | LMS is the StereoSet benchmark percentage favouring a meaningful completion over an unrelated one. | Results |
| LoopedBERT | LoopedBERT is an encoder that reuses a small core of transformer blocks across depth. | Abstract and Introduction |
| manifest | The manifest lists each model's block count and file checksums. | Reproducibility |
| MASK | The token [MASK] is the placeholder inserted at a hidden position. | Matching, scoring, and inference (line 412) |
| masked language model | A masked language model is an encoder trained to predict tokens hidden from their surrounding words. | Introduction |
| masked language modelling head | The masked language modelling head is the output layer that assigns probabilities to tokens. | Introduction |
| masked-position probing | Masked-position probing means scoring hidden positions to measure an association. | Related Work |
| matched point | A matched point is a checkpoint selected by crossing the same validation-loss target. | Abstract and Introduction |
| model capability | Model capability here means language-modelling quality, measured by validation loss on held-out text. | Introduction |
| official rule | The official rule is the CrowS-Pairs released implementation that selects aligned spans, sums log probabilities, and uses the direction label for roles. | Method |
| percentile item bootstrap | A percentile item bootstrap resamples benchmark items with replacement and takes percentile cutoffs as interval limits. | Method |
| Phipson-Smyth estimate | The Phipson-Smyth estimate avoids a zero permutation p-value when a finite number of random sign flips is sampled. | Method |
| pseudo-log-likelihood | A pseudo-log-likelihood is the average log probability assigned to original tokens when each token is masked in turn. | Related Work |
| race-color | The dataset label race-color names the CrowS-Pairs race and colour category. | Data |
| representation width | Representation width is the size of each token vector inside the encoder. | Introduction |
| residual connection | A residual connection adds a block input to the block output before the next layer. | Related Work |
| sent\_more | The two direction conventions either treat sent\_more , the dataset field holding one sentence member, as stereotypical or use the dataset direction label. | Introduction (line 101) |
| sent_more | The field sent_more stores one member of a CrowS-Pairs sentence pair. | Introduction and Data |
| SEP | The full-sentence rule scores every position except the special boundary tokens [CLS] and [SEP] . | Matching, scoring, and inference (line 414) |
| shared ratio | The shared ratio is the fraction of effective-depth block applications that reuse block parameters. | Method |
| shared tokens | Shared tokens are tokens that occur in both members of a sentence pair. | Data |
| shared-token alignment | Shared-token alignment matches the common prefix and common suffix of two token sequences. | Method |
| shared-token rule | The shared-token rule scores only tokens that the two sentences have in common. | Method |
| social stereotype association | A social stereotype association is a higher model score for text that links a social group with a common stereotype. | Introduction |
| SS | SS is the StereoSet benchmark percentage favouring the stereotypical completion. | Results |
| StereoSet | StereoSet is a sentence-completion stereotype benchmark. | Abstract and Introduction |
| stereotype benchmark | A stereotype benchmark measures whether a model gives higher scores to text that links a social group with a stereotype. | Abstract and Introduction |
| stream | A stream is one parallel copy of token representations inside HyperloopBERT. | Abstract and Introduction |
| token vectors | Token vectors are numerical representations of tokens whose values are used by the encoder. | Abstract |
| tokeniser | A tokeniser is the fixed procedure that splits text into model tokens. | Introduction |
| transformer block | A transformer block is the repeated attention-and-feed-forward unit used in BERT. | Introduction |
| unaligned shared-token pair | An unaligned shared-token pair is a pair for which shared-token alignment cannot recover the common token sequence. | Data |
| uncased | Uncased means trained on lowercased text. | Results |
| validation loss | Validation loss is the average negative log probability on held-out text. | Introduction |
| VanillaBERT | VanillaBERT is the unshared baseline with independent transformer blocks. | Abstract and Introduction |
| weight reuse | Weight reuse means that the same block parameters are applied at more than one depth in the network. | Introduction |
| wild cluster bootstrap | A wild cluster bootstrap is an inference check over snapshots that resamples snapshot-level residual patterns. | Results |
| WinoBias | WinoBias is a pronoun-resolution dataset used here as a capability check rather than a bias measure. | Abstract and Data |
