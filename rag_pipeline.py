import os
import time
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

VECTORSTORE_DIR = "vectorstore/faiss_index"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

TOP_K = 4

# FAISS distance: lower score = better match
# Tune this after testing.
OOD_DISTANCE_THRESHOLD = 0.322

PRIMARY_MODEL = "llama-3.1-8b-instant"
FALLBACK_MODEL = "llama-3.3-70b-versatile"

MAX_RETRIES = 3

OOD_MESSAGE = "لا توجد معلومات كافية في الحلقات المختارة للإجابة على هذا السؤال."


# ============================================================
# STEP 10: MULTI-TURN MEMORY
# ============================================================

chat_memory = []
active_episode = None
active_entity = None


def add_to_memory(user_question, assistant_answer, retrieved_docs=None):
    """
    Stores recent conversation turns and keeps track of the active episode/entity.
    This helps follow-up questions like:
    - وماذا حدث له؟
    - وماذا عنه؟
    """

    global active_episode, active_entity

    chat_memory.append({
        "user": user_question,
        "assistant": assistant_answer
    })

    # Store the dominant retrieved episode
    if retrieved_docs:
        episode_counts = {}

        for doc in retrieved_docs:
            episode = doc["metadata"].get("episode_title")

            if episode:
                episode_counts[episode] = episode_counts.get(episode, 0) + 1

        if episode_counts:
            active_episode = max(episode_counts, key=episode_counts.get)

    # Simple entity tracking for common selected episodes
    # You can add more entities later.
    q = user_question.lower()

    if "جون" in q and "كينيدي" in q:
        active_entity = "جون كينيدي"
    elif "الأخطبوط" in q or "اخطبوط" in q:
        active_entity = "الأخطبوط"
    elif "الساموراي" in q or "ساموراي" in q:
        active_entity = "الساموراي"
    elif "تاج محل" in q:
        active_entity = "تاج محل"


def get_sliding_window_memory(max_turns=4):
    """
    Keeps only the latest turns to avoid overloading the prompt.
    """

    recent_turns = chat_memory[-max_turns:]

    history_parts = []

    for turn in recent_turns:
        history_parts.append(f"User: {turn['user']}")
        history_parts.append(f"Assistant: {turn['assistant']}")

    if active_episode:
        history_parts.append(f"Active episode/topic: {active_episode}")

    if active_entity:
        history_parts.append(f"Active entity: {active_entity}")

    return "\n".join(history_parts)

def rewrite_followup_query(question):
    """
    Makes vague follow-up questions searchable.

    Example:
    وماذا حدث له؟
    becomes:
    ماذا حدث لجون كينيدي؟ اغتيال جون كينيدي نهاية جون كينيدي
    """

    q = question.strip()

    # Strong Kennedy-specific rewrite
    if active_entity == "جون كينيدي":
        if any(word in q for word in ["حدث له", "حصل له", "ماذا حدث", "وبعدين", "بعد كده", "مصيره", "نهايته"]):
            return "ماذا حدث لجون كينيدي؟ اغتيال جون كينيدي نهاية جون كينيدي موت جون كينيدي"

    vague_words = [
        "له", "لها", "عنه", "عنها",
        "هو", "هي", "ده", "دي",
        "ذلك", "هذه", "وبعدين", "بعد كده"
    ]

    is_followup = any(word in q for word in vague_words)

    if is_followup and active_entity:
        return f"{q} المقصود هو {active_entity}"

    if is_followup and active_episode:
        return f"{q} الموضوع هو {active_episode}"

    return q

# ============================================================
# VECTORSTORE + RETRIEVAL
# ============================================================

def load_vectorstore():
    """
    Loads the FAISS vectorstore created by build_vectorstore.py.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def retrieve_context(query, vectorstore, top_k=TOP_K):
    """
    Retrieves top-k similar transcript chunks.
    """

    docs_with_scores = vectorstore.similarity_search_with_score(
        query,
        k=top_k
    )

    retrieved_docs = []

    for doc, score in docs_with_scores:
        retrieved_docs.append({
            "content": doc.page_content,
            "score": float(score),
            "metadata": doc.metadata
        })

    return retrieved_docs


def build_context(retrieved_docs):
    """
    Builds the context block passed to the LLM.
    """

    context_parts = []

    for i, item in enumerate(retrieved_docs, start=1):
        metadata = item["metadata"]

        episode = metadata.get("episode_title", "Unknown episode")
        source = metadata.get("source_file", "Unknown source")
        chunk_id = metadata.get("chunk_id", "Unknown chunk")

        context_parts.append(
            f"""
[Chunk {i}]
Episode: {episode}
Source File: {source}
Chunk ID: {chunk_id}
Distance Score: {item["score"]}

Content:
{item["content"]}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# STEP 11: OUT-OF-DOMAIN DETECTION
# ============================================================

def is_obvious_external_question(question):
    """
    Detects questions that clearly require outside/current knowledge.
    """

    external_keywords = [
        "اليوم", "حاليا", "حالياً", "الآن", "دلوقتي",
        "سعر", "الدولار", "اليورو", "الجنيه", "الذهب",
        "الطقس", "الحرارة", "الأخبار", "خبر",
        "موعد", "نتيجة المباراة", "البورصة",
        "stock", "price", "weather", "today", "now",
        "current", "exchange rate"
    ]

    return any(keyword in question.lower() for keyword in external_keywords)


def is_out_of_domain(question, retrieved_docs):
    """
    OOD logic:
    1. Reject obvious external/current questions.
    2. Reject if retrieval distance is weak.
    """

    if is_obvious_external_question(question):
        return True

    if not retrieved_docs:
        return True

    best_distance = retrieved_docs[0]["score"]

    if best_distance > OOD_DISTANCE_THRESHOLD:
        return True

    return False


# ============================================================
# STRICT GROUNDED PROMPT
# ============================================================

STRICT_SYSTEM_PROMPT = """
You are a Retrieval-Augmented Generation chatbot.

Your job:
Answer the user's question using ONLY the retrieved context.

Strict rules:
1. Do not use outside knowledge.
2. Do not guess.
3. Do not invent names, events, dates, or facts.
4. If the user has a typo, do not change the main entity unless the retrieved context clearly supports the corrected entity.
5. If the retrieved context is about a very close matching entity, answer using that entity name exactly as it appears in the retrieved context.
6. If the answer is not clearly found in the retrieved context, say exactly:
   "لا توجد معلومات كافية في السياق للإجابة على هذا السؤال."
7. If the user asks in Arabic, answer in Arabic.
8. If the user asks in English, answer in English.
9. If the user mixes Arabic and English, you may answer using Arabic-English mixed language.
10. Keep the answer short, clear, and directly connected to the retrieved transcript context.
11. Mention the source episode name when useful.
12. For follow-up questions, use the conversation history only to resolve references, not to add outside facts.
"""


def build_user_prompt(question, context, chat_history=""):
    """
    Builds the final user prompt for the LLM.
    """

    return f"""
Retrieved Context:
{context}

Conversation History:
{chat_history}

User Question:
{question}

Important:
- Answer only from the retrieved context.
- Use the entity name exactly as it appears in the retrieved context.
- Do not introduce a different person, episode, or topic.
- If the context is insufficient, use the rejection sentence.
- If this is a follow-up question, use the active entity/topic from conversation history only to understand what the user refers to.

Answer:
"""


# ============================================================
# STEP 12: RETRY + FALLBACK MODEL
# ============================================================

def load_llm(model_name):
    """
    Loads Groq chat model.
    Make sure GROQ_API_KEY exists in .env.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing GROQ_API_KEY. Add it to your .env file like:\n"
            "GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxx"
        )

    return ChatGroq(
        model=model_name,
        temperature=0,
        api_key=api_key
    )


def call_llm_with_retry(messages):
    """
    Tries the primary model first.
    If it fails, retries.
    If it still fails, uses fallback model.
    """

    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]
    last_error = None

    for model_name in models_to_try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                llm = load_llm(model_name)
                response = llm.invoke(messages)

                return {
                    "content": response.content,
                    "model_used": model_name,
                    "attempts": attempt,
                    "error": None
                }

            except Exception as error:
                last_error = error
                print(
                    f"[Warning] Model {model_name} failed "
                    f"on attempt {attempt}: {error}"
                )

                time.sleep(2 * attempt)

    return {
        "content": "حدث خطأ أثناء الاتصال بالنموذج. حاول مرة أخرى لاحقًا.",
        "model_used": "none",
        "attempts": MAX_RETRIES,
        "error": str(last_error)
    }


# ============================================================
# FULL RAG ANSWER FUNCTION
# ============================================================

def answer_question(question):
    vectorstore = load_vectorstore()

    if is_obvious_external_question(question):
        retrieval_query = question
    else:
        retrieval_query = rewrite_followup_query(question)

    retrieved_docs = retrieve_context(
        query=retrieval_query,
        vectorstore=vectorstore,
        top_k=TOP_K
    )

    if is_out_of_domain(question, retrieved_docs):
        answer = OOD_MESSAGE

        add_to_memory(
            user_question=question,
            assistant_answer=answer,
            retrieved_docs=retrieved_docs
        )

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "out_of_domain": True,
            "model_used": None,
            "attempts": None,
            "retrieval_query": retrieval_query
        }

    context = build_context(retrieved_docs)
    chat_history = get_sliding_window_memory(max_turns=4)

    messages = [
        SystemMessage(content=STRICT_SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(question, context, chat_history))
    ]

    response = call_llm_with_retry(messages)

    answer = response["content"]

    add_to_memory(
        user_question=question,
        assistant_answer=answer,
        retrieved_docs=retrieved_docs
    )

    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "out_of_domain": False,
        "model_used": response.get("model_used"),
        "attempts": response.get("attempts"),
        "error": response.get("error"),
        "retrieval_query": retrieval_query
    }


# ============================================================
# TERMINAL CHAT TEST
# ============================================================

def print_result(result):
    print("\nAnswer:")
    print(result["answer"])

    print("\nModel Used:")
    print(result["model_used"])

    print("\nOut of Domain:")
    print(result["out_of_domain"])

    print("\nRetrieval Query:")
    print(result["retrieval_query"])

    print("\nRetrieved Sources:")
    for doc in result["retrieved_docs"]:
        print("-" * 50)
        print("Episode:", doc["metadata"].get("episode_title"))
        print("Chunk ID:", doc["metadata"].get("chunk_id"))
        print("Score:", doc["score"])

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print("RAG chatbot is running.")
    print("Type 'exit' to stop.\n")

    while True:
        try:
            question = input("Ask a question: ").strip()

            if question.lower() in ["exit", "quit", "خروج"]:
                print("Chat stopped.")
                break

            if not question:
                print("Please enter a question.")
                continue

            result = answer_question(question)
            print_result(result)

        except KeyboardInterrupt:
            print("\nChat stopped.")
            break

        except Exception as error:
            print("\nUnexpected error:")
            print(error)
            print("=" * 70)