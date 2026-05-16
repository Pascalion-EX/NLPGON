import os
import time
import pandas as pd

from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_pipeline import answer_question


QA_DIR = "Data/QA"
OUTPUT_LOG = "logs/llm_comparison_logs.csv"

MAX_QUESTIONS_PER_FILE = 5

MODELS_TO_COMPARE = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile"
]


def semantic_similarity(reference, generated):
    if not reference or not generated:
        return 0.0

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5)
    )

    matrix = vectorizer.fit_transform([reference, generated])
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def rouge_l_score(reference, generated):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, generated)
    return float(scores["rougeL"].fmeasure)


def grounding_score(generated_answer, retrieved_docs):
    retrieved_text = " ".join([doc["content"] for doc in retrieved_docs])
    return semantic_similarity(retrieved_text, generated_answer)


def load_questions():
    qa_files = [
        os.path.join(QA_DIR, file)
        for file in os.listdir(QA_DIR)
        if file.endswith(".csv")
    ]

    if not qa_files:
        raise FileNotFoundError(f"No QA CSV files found in {QA_DIR}")

    rows = []

    for csv_path in qa_files:
        df = pd.read_csv(csv_path).head(MAX_QUESTIONS_PER_FILE)

        for _, row in df.iterrows():
            rows.append({
                "qa_file": os.path.basename(csv_path),
                "question": str(row["question"]),
                "gold_answer": str(row["answer"])
            })

    return rows


def main():
    os.makedirs("logs", exist_ok=True)

    questions = load_questions()
    output_rows = []

    for model in MODELS_TO_COMPARE:
        print("=" * 80)
        print("Testing model:", model)
        print("=" * 80)

        for item in questions:
            question = item["question"]
            gold_answer = item["gold_answer"]

            print(f"[{model}] {question}")

            try:
                result = answer_question(
                    question=question,
                    forced_model=model
                )

                generated_answer = result["answer"]
                retrieved_docs = result.get("retrieved_docs", [])

                output_rows.append({
                    "model": model,
                    "qa_file": item["qa_file"],
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": generated_answer,
                    "semantic_similarity": semantic_similarity(gold_answer, generated_answer),
                    "rouge_l": rouge_l_score(gold_answer, generated_answer),
                    "groundedness": grounding_score(generated_answer, retrieved_docs),
                    "out_of_domain": result.get("out_of_domain"),
                    "status": "success"
                })

            except Exception as error:
                output_rows.append({
                    "model": model,
                    "qa_file": item["qa_file"],
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": "",
                    "semantic_similarity": 0,
                    "rouge_l": 0,
                    "groundedness": 0,
                    "out_of_domain": "",
                    "status": f"error: {error}"
                })

            time.sleep(1)

    df = pd.DataFrame(output_rows)
    df.to_csv(OUTPUT_LOG, index=False, encoding="utf-8-sig")

    print("\nSaved comparison log to:", OUTPUT_LOG)

    print("\nModel comparison averages:")
    print(
        df.groupby("model")[["semantic_similarity", "rouge_l", "groundedness"]]
        .mean()
        .sort_values("semantic_similarity", ascending=False)
    )


if __name__ == "__main__":
    main()