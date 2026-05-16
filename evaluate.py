import os
import time
import pandas as pd

from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_pipeline import answer_question


QA_DIR = "Data/QA"
OUTPUT_LOG = "logs/evaluation_logs.csv"

MAX_QUESTIONS_PER_FILE = 10


def semantic_similarity(reference, generated):
    """
    Lightweight semantic similarity using character-level TF-IDF.
    Good enough for a baseline evaluation without heavy BERTScore.
    """

    if not reference or not generated:
        return 0.0

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5)
    )

    matrix = vectorizer.fit_transform([reference, generated])
    score = cosine_similarity(matrix[0], matrix[1])[0][0]

    return float(score)


def rouge_l_score(reference, generated):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, generated)
    return float(scores["rougeL"].fmeasure)


def grounding_score(generated_answer, retrieved_docs):
    """
    Simple grounding score:
    Measures whether generated answer overlaps semantically with retrieved context.
    """

    retrieved_text = " ".join([doc["content"] for doc in retrieved_docs])

    return semantic_similarity(retrieved_text, generated_answer)


def evaluate_file(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = ["question", "answer"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}")

    df = df.head(MAX_QUESTIONS_PER_FILE)

    rows = []

    for _, row in df.iterrows():
        question = str(row["question"])
        gold_answer = str(row["answer"])

        print(f"Evaluating: {question}")

        try:
            result = answer_question(question)

            generated_answer = result["answer"]
            retrieved_docs = result.get("retrieved_docs", [])

            semantic = semantic_similarity(gold_answer, generated_answer)
            rouge_l = rouge_l_score(gold_answer, generated_answer)
            groundedness = grounding_score(generated_answer, retrieved_docs)

            retrieved_sources = [
                {
                    "episode": doc["metadata"].get("episode_title"),
                    "chunk_id": doc["metadata"].get("chunk_id"),
                    "score": doc.get("score")
                }
                for doc in retrieved_docs
            ]

            rows.append({
                "qa_file": os.path.basename(csv_path),
                "question": question,
                "gold_answer": gold_answer,
                "generated_answer": generated_answer,
                "model_used": result.get("model_used"),
                "out_of_domain": result.get("out_of_domain"),
                "retrieval_query": result.get("retrieval_query"),
                "semantic_similarity": semantic,
                "rouge_l": rouge_l,
                "groundedness": groundedness,
                "retrieved_sources": str(retrieved_sources),
                "status": "success"
            })

        except Exception as error:
            rows.append({
                "qa_file": os.path.basename(csv_path),
                "question": question,
                "gold_answer": gold_answer,
                "generated_answer": "",
                "model_used": "",
                "out_of_domain": "",
                "retrieval_query": "",
                "semantic_similarity": 0,
                "rouge_l": 0,
                "groundedness": 0,
                "retrieved_sources": "",
                "status": f"error: {error}"
            })

        time.sleep(1)

    return rows


def main():
    os.makedirs("logs", exist_ok=True)

    qa_files = [
        os.path.join(QA_DIR, file)
        for file in os.listdir(QA_DIR)
        if file.endswith(".csv")
    ]

    if not qa_files:
        raise FileNotFoundError(f"No QA CSV files found in {QA_DIR}")

    all_rows = []

    for csv_path in qa_files:
        print("=" * 80)
        print("Evaluating file:", csv_path)
        print("=" * 80)

        rows = evaluate_file(csv_path)
        all_rows.extend(rows)

    output_df = pd.DataFrame(all_rows)
    output_df.to_csv(OUTPUT_LOG, index=False, encoding="utf-8-sig")

    print("\nSaved evaluation log to:", OUTPUT_LOG)

    print("\nAverage scores:")
    print(output_df[["semantic_similarity", "rouge_l", "groundedness"]].mean())


if __name__ == "__main__":
    main()