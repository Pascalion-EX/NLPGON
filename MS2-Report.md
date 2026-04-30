1. Abstract

This notebook implements an end-to-end Arabic extractive question answering system. The system reads SQuAD-style Arabic QA data, applies Arabic-specific normalization and cleaning, constructs span-aligned training examples, builds a vocabulary, creates a reusable inference pipeline, and evaluates two neural architectures: a Bidirectional LSTM with additive attention and a custom Transformer encoder.

The notebook demonstrates a complete QA engineering workflow rather than only a single model. Its main strengths are the explicit Arabic preprocessing pipeline, careful answer-span alignment logic, and modular inference design. Its main limitation is the very small number of usable span-aligned examples after filtering, which strongly affects model reliability and metric stability.

2. Objective

The notebook aims to build a complete extractive QA system for Arabic text where the model predicts the start and end positions of the answer span inside the context. The implementation covers:

dataset discovery and parsing
exploratory data analysis
Arabic text preprocessing
tokenization experiments
answer span extraction and alignment
vocabulary construction
training and evaluation of sequence models
deployment-style inference through a unified pipeline
3. Dataset Description

The dataset is a cleaned Arabic QA subset derived from ElDa7ee7 Season 8 and stored in SQuAD-style JSON files. The notebook loads:

7 training files
7 test files

After flattening the nested JSON format into (question, context, answer) triplets, the raw dataset size becomes:

168 training triplets
72 test triplets
240 total triplets

The exploratory analysis shows that the dataset is relatively short-form:

Mean context length: 13.3 words
Mean question length: 34.4 characters
Mean answer length: 27.3 characters

The language mix analysis shows the corpus is strongly Arabic-dominant:

Average Arabic character ratio: 94.3%
Average Latin character ratio: 1.59%
Average digit ratio: 0.81%

This confirms that the task is primarily Arabic QA, with limited code-switching.

4. Preprocessing Pipeline

A major part of the notebook is the Arabic-specific normalization pipeline. It applies 7 steps:

Unicode normalization
Alef normalization
Ending normalization (ى → ي, ة → ه)
Diacritic removal
Tatweel removal
Punctuation normalization
Whitespace collapsing

This preprocessing is appropriate for Arabic QA because many equivalent forms can otherwise fragment the vocabulary and break exact span matching.

The impact on average context length is modest:

Average context chars before preprocessing: 77.60
Average context chars after preprocessing: 77.10
Average reduction: 0.54%

This indicates that preprocessing mainly standardizes the text rather than aggressively compressing it.

5. Tokenization and Vocabulary

The notebook implements three tokenization strategies:

Word-level tokenization
Character-level tokenization
Simple subword tokenization using frequency-based character n-grams

This is a good experimental design choice because Arabic can benefit from subword modeling due to morphology and spelling variation. However, the actual training pipeline uses the shared word-level vocabulary.

Vocabulary statistics:

Total vocabulary size: 1381
Special tokens: 5
Unique content tokens: 1376
Singleton tokens: 1041 (75.4%)
Most frequent token: "ما" with frequency 95
Average token frequency: 1.74

The high singleton rate indicates a sparse vocabulary, which is expected with a small dataset and explains some generalization difficulty.

The test-set out-of-vocabulary rate is also relatively high:

Total test tokens: 1183
OOV tokens: 321
OOV rate: 27.13%

This is an important finding. It means more than one quarter of test tokens are unseen in the training vocabulary, which limits a pure word-level model.

6. Answer Span Alignment

Because extractive QA depends on exact span positions, the notebook includes a careful alignment module that converts raw character offsets into cleaned token offsets. The procedure:

preprocesses raw tokens while preserving raw character spans
maps raw answer offsets into token-level spans
tries overlap-based alignment first
falls back to local search and then full exact search if needed

This is one of the strongest engineering parts of the notebook.

However, the alignment stage also reveals the biggest bottleneck:

Train spans found: 22 / 168 (13.1%)
Test spans found: 7 / 72 (9.7%)

After filtering for valid spans and then splitting the training data:

Train: 19 samples
Validation: 3 samples
Test: 7 samples

This is the main reason the final metrics should be interpreted cautiously. The notebook is technically correct in filtering invalid spans, but the usable dataset becomes extremely small.

7. Inference Pipeline Design

The notebook defines a reusable QAPipeline class that performs:

preprocessing
tokenization
concatenation as question + [SEP] + context
encoding and padding
model inference
constrained best-span search
answer decoding

This design is sound and modular. It allows the same inference interface to be used with different models through a model_fn adapter.

The span search adds practical constraints:

answer must lie in the context region only
maximum answer length is limited
a mild length penalty discourages overlong spans
a “left-boundary rescue” tries to recover slightly earlier starts when semantically helpful

This is a thoughtful heuristic design for extractive QA.

The profiling cell also shows the pipeline is lightweight in this prototype setting. The main latency contributors are preprocessing and the mock forward pass.

8. Model 1: Bidirectional LSTM with Additive Attention
Architecture

The first neural model is a BiLSTM QA model with:

embedding dimension: 128
two bidirectional LSTM layers
hidden size: 128
layer normalization
dropout: 0.30
additive attention over sequence states
separate linear start and end heads
Parameter count
Trainable parameters: 903,810
Training setup
optimizer: AdamW
learning rate: 3e-4
weight decay: 1e-2
gradient clipping: 1.0
epochs: 20
checkpoint selection based on validation F1
Best validation result
Best epoch: 3
Best validation F1: 42.06%
Test result
Cross-entropy loss: 2.8479
Exact Match: 14.29%
Token-level F1: 42.38%
Interpretation

The BiLSTM shows relatively better validation behavior than the Transformer. It also achieves the only nonzero test Exact Match score. This suggests that on tiny data, the recurrent model may be slightly more stable in learning local lexical patterns.

9. Model 2: Custom Transformer Encoder
Architecture

The second model is a custom Transformer QA model with:

token embedding dimension: 128
token-type embeddings
learned positional embeddings
2 Transformer encoder blocks
4 attention heads
feed-forward dimension: 512
dropout: 0.20
separate MLP start and end heads

The model also builds a context-only mask so that answer logits outside the context are suppressed. This is a correct design decision for extractive QA.

Parameter count
Trainable parameters: 639,362
Training setup
optimizer: AdamW
learning rate: 2e-4
weight decay: 1e-2
scheduler: ReduceLROnPlateau on validation F1
epochs: 25
checkpoint selection based on validation F1
Best validation result
Best epoch: 2
Best validation F1: 11.11%
Best validation EM: 0.00%
Test result
Cross-entropy loss: 2.2167
Exact Match: 0.00%
Token F1: 48.16%
Interpretation

The Transformer achieves the best test F1 in the notebook, but its validation F1 is much lower than the BiLSTM and its Exact Match is zero. This suggests the model often predicts semantically overlapping spans that do not exactly match the gold answer boundaries.

A qualitative example confirms this behavior:

predicted: "في هروبها من الرادارات"
gold: "هروبها من الرادارات"

This is a near-correct answer semantically, but it fails exact matching because of the extra token at the beginning.

10. Comparative Results
Model	Best Val F1	Test Loss	Test EM	Test F1
BiLSTM + Attention	42.06%	2.8479	14.29%	42.38%
Custom Transformer	11.11%	2.2167	0.00%	48.16%
11. Discussion

The notebook succeeds as a technical prototype and teaching implementation. It covers the full QA pipeline correctly and cleanly. The code organization is strong, especially in preprocessing, span alignment, and modular inference.

However, the experimental reliability is limited by four issues:

11.1 Severe data shrinkage

Only 22 of 168 training items survive span alignment, and only 19 remain in the final training split. This is the dominant limitation.

11.2 Tiny validation and test sets

Validation size is 3 and test size is 7. Under these conditions, EM and F1 are highly unstable.

11.3 High OOV rate

The 27.13% OOV rate shows that word-level vocabulary is not robust enough for this Arabic setting.

11.4 Metric mismatch

The Transformer produces partially correct spans more often than exactly correct spans, which explains higher F1 but zero EM.

12. Technical Issues Noticed in the Notebook

There is one implementation issue in the Transformer training loop:

bad_epochs += 0 is used instead of incrementing the counter on non-improving epochs.

Because of that, the printed “no improvement” status remains misleading and the early stopping condition never actually progresses. Checkpoint saving still works, but early stopping logic is effectively broken.

13. Conclusion

This notebook is a good end-to-end Arabic extractive QA prototype. Its strongest contributions are:

Arabic-aware preprocessing
span-safe alignment logic
reusable inference pipeline
comparison between recurrent and Transformer architectures

The final results show that:

the BiLSTM is more stable on the tiny validation split
the Transformer gives better token overlap on the test set
neither result should be overinterpreted because the usable dataset is extremely small