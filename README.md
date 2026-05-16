# Milestone 3: Arabic RAG

## Features

- Arabic transcript retrieval
- Multilingual embeddings
- FAISS vector store
- Strict grounded prompting
- Multi-turn memory
- Out-of-domain detection
- Retry and fallback model strategy
- Streamlit interface
- Evaluation logs
- Two-LLM comparison

## 2. Main Features

### Arabic Transcript Retrieval

The system retrieves relevant Arabic transcript chunks based on the user question.

Implemented in:

```text
rag_pipeline.py
```

Main function:

```python
retrieve_context()
```

The retriever searches the FAISS vector store and returns the top relevant transcript chunks with their metadata.

---

### Multilingual Embeddings

The project uses a multilingual embedding model to represent Arabic, English, and mixed Arabic-English text.

Embedding model:

```text
intfloat/multilingual-e5-small
```

Implemented in:

```text
build_vectorstore.py
rag_pipeline.py
```

The embedding model converts transcript chunks and user queries into numerical vectors.

---

### FAISS Vector Store

FAISS is used as the local vector database.

It stores transcript embeddings and retrieves the closest chunks to the user query.

Implemented in:

```text
build_vectorstore.py
rag_pipeline.py
```

Build FAISS index:

```python
FAISS.from_documents()
```

Load FAISS index:

```python
FAISS.load_local()
```

---

### Strict Grounded Prompting

The chatbot is instructed to answer only from the retrieved transcript context.

Implemented in:

```text
rag_pipeline.py
```

Main prompt variable:

```python
STRICT_SYSTEM_PROMPT
```

The prompt prevents the model from guessing or using outside knowledge.

If the answer is not found in the context, the chatbot returns:

```text
لا توجد معلومات كافية في السياق للإجابة على هذا السؤال.
```

---

### Multi-Turn Memory

The chatbot supports follow-up questions.

Example:

```text
User: من هو جون كينيدي؟
User: وماذا حدث له؟
```

The system tracks:

```text
active_episode
active_entity
chat_memory
```

Implemented in:

```text
rag_pipeline.py
```

Main functions:

```python
add_to_memory()
get_sliding_window_memory()
rewrite_followup_query()
```

The system uses sliding-window memory, meaning it keeps only the most recent conversation turns.

---

### Out-of-Domain Detection

The system detects questions that are not supported by the selected transcripts.

Example:

```text
ما هو سعر الدولار اليوم؟
```

The system rejects this because it requires external/current knowledge.

Implemented in:

```text
rag_pipeline.py
```

Main functions:

```python
is_obvious_external_question()
is_out_of_domain()
```

The rejection message is:

```text
لا توجد معلومات كافية في الحلقات المختارة للإجابة على هذا السؤال.
```

---

### Retry and Fallback Model Strategy

The system includes retry logic and a fallback model to avoid crashing if the main model fails.

Implemented in:

```text
rag_pipeline.py
```

Primary model:

```text
llama-3.1-8b-instant
```

Fallback model:

```text
llama-3.3-70b-versatile
```

Main functions:

```python
call_llm_with_retry()
load_llm()
```

---

### Streamlit Interface

The project includes a Streamlit web interface for chatting with the RAG system.

Implemented in:

```text
app.py
```

Run with:

```bash
streamlit run app.py
```

The interface shows:

- Chat history
- Generated answer
- Retrieved sources
- Chunk content
- Retrieval query
- Out-of-domain flag
- Model used
- Logs

---

### Evaluation Logs

The system evaluates generated answers against QA pairs.

Implemented in:

```text
evaluate.py
```

Evaluation output:

```text
logs/evaluation_logs.csv
```

Metrics used:

- Semantic similarity
- ROUGE-L
- Groundedness

---

### Two-LLM Comparison

The project compares two LLMs on the same QA questions.

Implemented in:

```text
compare_llms.py
```

Compared models:

```text
llama-3.1-8b-instant
llama-3.3-70b-versatile
```

Output file:

```text
logs/llm_comparison_logs.csv
```

---

## 3. Project Structure

```text
Milestone3/
│
├── app.py
├── build_vectorstore.py
├── rag_pipeline.py
├── evaluate.py
├── compare_llms.py
├── summarize_results.py
├── requirements.txt
├── technical_report.md
├── README.md
├── .gitignore
│
├── data/
│   ├── transcripts/
│   │   ├── episode_1.txt
│   │   ├── episode_2.txt
│   │   └── ...
│   │
│   └── qa/
│       ├── episode_1.csv
│       ├── episode_2.csv
│       └── ...
│
├── vectorstore/
│   └── faiss_index/
│       ├── index.faiss
│       └── index.pkl
│
└── logs/
    ├── evaluation_logs.csv
    ├── llm_comparison_logs.csv
    ├── evaluation_summary.csv
    ├── llm_comparison_summary.csv
    ├── best_cases.csv
    └── worst_cases.csv
```

---

## 4. Dataset Setup

Place selected transcript files inside:

```text
data/transcripts/
```

Place matching QA files inside:

```text
data/qa/
```

Only 3 to 5 episodes should be used, according to the milestone requirements.

Example:

```text
data/transcripts/جون كينيدي  الدحيح.txt
data/transcripts/الأخطبوط  الدحيح.txt
data/transcripts/الساموراي  الدحيح.txt
```

Example QA files:

```text
data/qa/جون كينيدي  الدحيح.csv
data/qa/الأخطبوط  الدحيح.csv
data/qa/الساموراي  الدحيح.csv
```

---

## 5. Environment Setup

Create a clean Conda environment:

```bash
conda create -n ms3rag python=3.11 -y
conda activate ms3rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 6. Requirements

Example `requirements.txt`:

```text
numpy==1.26.4
pandas
scikit-learn
requests
python-dotenv
streamlit
rouge-score
langchain-core
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-groq
sentence-transformers
faiss-cpu
```

---

## 7. API Key Setup

Create a `.env` file in the project root:

```bash
nano .env
```

Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Do not upload `.env` to GitHub.

Make sure `.env` is included in `.gitignore`.

---

## 8. Build the Vector Store

Run:

```bash
python build_vectorstore.py
```

This script:

1. Loads transcript files
2. Cleans text lightly
3. Splits text into chunks
4. Creates multilingual embeddings
5. Stores embeddings in FAISS
6. Saves the FAISS index locally

Expected output:

```text
Loading transcripts...
Loaded X transcript files.
Cleaning and chunking transcripts...
Created X chunks.
Building FAISS vectorstore...
Vectorstore saved to: vectorstore/faiss_index
```

---

## 9. Run Terminal Chatbot

Run:

```bash
python rag_pipeline.py
```

Example questions:

```text
من هو جون كينيدي؟
```

```text
وماذا حدث له؟
```

```text
ما هو سعر الدولار اليوم؟
```

To stop the chatbot:

```text
exit
```

---

## 10. Run Streamlit App

Run:

```bash
streamlit run app.py
```

The app provides a web interface for the chatbot.

It displays:

- User question
- Generated answer
- Retrieved transcript chunks
- Episode source
- Chunk ID
- Similarity score
- Out-of-domain result
- Model used
- Conversation logs

---

## 11. Run Evaluation

Run:

```bash
python evaluate.py
```

This evaluates the system using QA files from:

```text
data/qa/
```

The output is saved to:

```text
logs/evaluation_logs.csv
```

The evaluation uses:

### Semantic Similarity

Measures meaning similarity between the generated answer and the reference answer.

### ROUGE-L

Measures text overlap between generated answer and gold answer.

### Groundedness

Measures whether the generated answer is supported by the retrieved transcript context.

---

## 12. Compare Two LLMs

Run:

```bash
python compare_llms.py
```

This compares:

```text
llama-3.1-8b-instant
llama-3.3-70b-versatile
```

The output is saved to:

```text
logs/llm_comparison_logs.csv
```

The comparison uses:

- Semantic similarity
- ROUGE-L
- Groundedness

---

## 13. Summarize Results

Run:

```bash
python summarize_results.py
```

This generates:

```text
logs/evaluation_summary.csv
logs/llm_comparison_summary.csv
logs/best_cases.csv
logs/worst_cases.csv
```

These files are used in the technical report.

---

## 14. Evaluation Metrics

### Semantic Similarity

This metric compares the meaning of the generated answer with the gold answer.

In this project, it is implemented using character-level TF-IDF similarity.

---

### ROUGE-L

ROUGE-L measures the longest common subsequence between the generated answer and reference answer.

It is useful for checking text overlap.

---

### Groundedness

Groundedness checks whether the generated answer is supported by the retrieved transcript chunks.

A higher groundedness score means the generated answer is more connected to the retrieved context.

---

## 15. How the RAG Pipeline Works

The complete pipeline is:

```text
User question
↓
Rewrite follow-up question if needed
↓
Embed question
↓
Search FAISS vector store
↓
Retrieve top-k chunks
↓
Detect out-of-domain questions
↓
Build strict grounded prompt
↓
Call Groq LLM
↓
Return answer
↓
Display retrieved sources and logs
```

---

## 16. Out-of-Domain Example

Question:

```text
ما هو سعر الدولار اليوم؟
```

Expected output:

```text
لا توجد معلومات كافية في الحلقات المختارة للإجابة على هذا السؤال.
```

The system marks:

```text
Out of Domain: True
```

---

## 17. Multi-Turn Example

Question 1:

```text
من هو جون كينيدي؟
```

Question 2:

```text
وماذا حدث له؟
```

The second question is rewritten internally to include the active entity:

```text
جون كينيدي
```

This helps the retriever search the correct transcript chunks.

---
## 21. Main Commands

Build vector store:

```bash
python build_vectorstore.py
```

Run chatbot:

```bash
python rag_pipeline.py
```

Run Streamlit:

```bash
streamlit run app.py
```

Run evaluation:

```bash
python evaluate.py
```

Compare LLMs:

```bash
python compare_llms.py
```

Summarize results:

```bash
python summarize_results.py
```
