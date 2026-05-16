import os
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# =========================
# CONFIG
# =========================

TRANSCRIPTS_DIR = "Data/Transcripts"
VECTORSTORE_DIR = "vectorstore/faiss_index"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120


# =========================
# STEP 4: LIGHT CLEANING
# =========================

def clean_text(text: str) -> str:
    """
    Light cleaning only.
    Do NOT remove punctuation.
    Do NOT remove English words.
    Do NOT stem or lemmatize Arabic.
    """

    # Normalize multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize multiple empty lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Strip beginning/end spaces
    text = text.strip()

    return text


def load_transcripts(transcripts_dir: str):
    documents = []

    transcript_files = list(Path(transcripts_dir).glob("*.txt"))

    if not transcript_files:
        raise FileNotFoundError(f"No transcript files found in {transcripts_dir}")

    for file_path in transcript_files:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned_text = clean_text(raw_text)

        doc = Document(
            page_content=cleaned_text,
            metadata={
                "episode_title": file_path.stem,
                "source_file": file_path.name
            }
        )

        documents.append(doc)

    return documents


# =========================
# STEP 5: CHUNKING
# =========================

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "؟", "!", "،", " "]
    )

    chunks = []

    for doc in documents:
        split_chunks = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(split_chunks):
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "episode_title": doc.metadata["episode_title"],
                        "source_file": doc.metadata["source_file"],
                        "chunk_id": i
                    }
                )
            )

    return chunks


# =========================
# STEP 6: EMBEDDINGS + FAISS
# =========================

def build_vectorstore(chunks):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    vectorstore.save_local(VECTORSTORE_DIR)

    return vectorstore


# =========================
# MAIN
# =========================

def main():
    print("Loading transcripts...")
    documents = load_transcripts(TRANSCRIPTS_DIR)
    print(f"Loaded {len(documents)} transcript files.")

    print("Cleaning and chunking transcripts...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Building FAISS vectorstore...")
    build_vectorstore(chunks)

    print(f"Vectorstore saved to: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()