import streamlit as st
from rag_pipeline import answer_question


st.set_page_config(
    page_title="Arabic RAG Chatbot - MS3",
    page_icon="🤖",
    layout="wide"
)

st.title("Milestone 3: Arabic RAG Chatbot")
st.caption("Answers are grounded only in retrieved transcript chunks.")

# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs" not in st.session_state:
    st.session_state.logs = []


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("Settings")

    st.info(
        "Current pipeline uses:\n"
        "- FAISS vectorstore\n"
        "- Multilingual embeddings\n"
        "- Groq API LLM\n"
        "- Sliding-window memory\n"
        "- Out-of-domain detection"
    )

    show_sources = st.checkbox("Show retrieved sources", value=True)
    show_logs = st.checkbox("Show logs", value=True)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.logs = []
        st.rerun()


# =========================
# DISPLAY CHAT HISTORY
# =========================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================
# CHAT INPUT
# =========================

user_question = st.chat_input("Ask a question about the selected episodes...")

if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = answer_question(user_question)

                answer = result["answer"]
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                st.session_state.logs.append({
                    "question": user_question,
                    "answer": answer,
                    "out_of_domain": result.get("out_of_domain"),
                    "model_used": result.get("model_used"),
                    "retrieval_query": result.get("retrieval_query"),
                    "retrieved_docs": result.get("retrieved_docs", [])
                })

                if show_sources:
                    with st.expander("Retrieved Sources"):
                        for i, doc in enumerate(result.get("retrieved_docs", []), start=1):
                            meta = doc["metadata"]

                            st.markdown(f"### Source {i}")
                            st.write("Episode:", meta.get("episode_title"))
                            st.write("Source file:", meta.get("source_file"))
                            st.write("Chunk ID:", meta.get("chunk_id"))
                            st.write("Score:", doc.get("score"))
                            st.text_area(
                                label=f"Chunk {i} content",
                                value=doc.get("content", ""),
                                height=180
                            )

                if show_logs:
                    with st.expander("Run Log"):
                        st.write("Out of domain:", result.get("out_of_domain"))
                        st.write("Model used:", result.get("model_used"))
                        st.write("Attempts:", result.get("attempts"))
                        st.write("Retrieval query:", result.get("retrieval_query"))

            except Exception as error:
                error_message = f"Unexpected error: {error}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })


# =========================
# LOG TABLE
# =========================

if show_logs and st.session_state.logs:
    st.divider()
    st.subheader("Conversation Logs")

    for i, log in enumerate(st.session_state.logs, start=1):
        with st.expander(f"Turn {i}: {log['question']}"):
            st.write("Answer:", log["answer"])
            st.write("Out of domain:", log["out_of_domain"])
            st.write("Model used:", log["model_used"])
            st.write("Retrieval query:", log["retrieval_query"])