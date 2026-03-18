"""
RAG Chatbot - Full Pipeline
Knowledge Base: PDF + Audio Lecture on "Databases for GenAI"
Stack: PyMuPDF | Whisper | ChromaDB | Azure OpenAI Embeddings + GPT-4
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Optional

# ── PDF extraction ──────────────────────────────────────────────────────────
import fitz  # PyMuPDF

# ── Audio transcription ──────────────────────────────────────────────────────
import whisper

# ── Text chunking ────────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Embeddings + LLM (Azure OpenAI) ─────────────────────────────────────────
from openai import AzureOpenAI

# ── Vector database ──────────────────────────────────────────────────────────
import chromadb

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — fill in your Azure OpenAI details here
# You can also set these as environment variables instead.
# ────────────────────────────────────────────────────────────────────────────

# Azure OpenAI credentials — get these from your Azure Portal
AZURE_OPENAI_API_KEY      = os.getenv("AZURE_OPENAI_API_KEY",  "YOUR_AZURE_API_KEY_HERE")
AZURE_OPENAI_ENDPOINT     = os.getenv("AZURE_OPENAI_ENDPOINT", "https://YOUR_RESOURCE_NAME.openai.azure.com/")
AZURE_OPENAI_API_VERSION  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# These must match the deployment names you created in Azure OpenAI Studio
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
LLM_DEPLOYMENT       = os.getenv("AZURE_LLM_DEPLOYMENT",       "gpt-4o-mini")

# Put your actual files here (relative paths are fine)
PDF_PATH        = "data/RAG Intro.pdf"
AUDIO_DIR       = "data/audio"   # folder with .mp3 / .wav / .m4a files
CHROMA_DB_DIR   = "chroma_db"    # persisted vector store
COLLECTION_NAME = "genai_lecture"

WHISPER_MODEL = "base"   # tiny / base / small / medium / large
CHUNK_SIZE    = 600      # characters per chunk
CHUNK_OVERLAP = 100
TOP_K         = 5        # retrieved chunks per query

# ────────────────────────────────────────────────────────────────────────────
# STEP 1 – LOAD & PROCESS DATA
# ────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    print(f"[PDF] Extracting text from: {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")          # plain text extraction
        if text.strip():
            pages.append(f"--- Page {i+1} ---\n{text.strip()}")
    doc.close()
    full_text = "\n\n".join(pages)
    print(f"[PDF] Extracted {len(full_text):,} characters from {len(pages)} pages.")
    return full_text


def transcribe_audio(audio_dir: str, model_name: str = WHISPER_MODEL) -> str:
    """Transcribe all audio files in a directory using OpenAI Whisper."""
    audio_dir_path = Path(audio_dir)
    supported = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm",
                 ".mp4", ".mkv", ".avi", ".mov", ".wmv"}  # video formats work too
    audio_files = [f for f in audio_dir_path.iterdir() if f.suffix.lower() in supported]

    if not audio_files:
        print(f"[Audio] No audio files found in '{audio_dir}'. Skipping transcription.")
        return ""

    print(f"[Audio] Loading Whisper model '{model_name}' (first run downloads it)…")
    model = whisper.load_model(model_name)
    transcripts = []

    for audio_file in sorted(audio_files):
        print(f"[Audio] Transcribing: {audio_file.name}")
        result = model.transcribe(str(audio_file), fp16=False)
        text = result["text"].strip()
        transcripts.append(f"--- Transcript: {audio_file.name} ---\n{text}")
        print(f"[Audio] Done ({len(text):,} chars)")

    full_transcript = "\n\n".join(transcripts)
    print(f"[Audio] Total transcript: {len(full_transcript):,} characters.")
    return full_transcript


# ────────────────────────────────────────────────────────────────────────────
# STEP 2 – CHUNK THE TEXT
# ────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, source_label: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    Returns a list of dicts with 'text', 'source', and 'chunk_id'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)
    chunks = []
    for i, chunk in enumerate(raw_chunks):
        chunk_id = hashlib.md5(f"{source_label}_{i}_{chunk[:50]}".encode()).hexdigest()
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk,
            "source": source_label,
            "chunk_index": i,
        })
    print(f"[Chunk] '{source_label}' → {len(chunks)} chunks")
    return chunks


# ────────────────────────────────────────────────────────────────────────────
# STEP 3 – EMBED & STORE IN CHROMADB
# ────────────────────────────────────────────────────────────────────────────

def make_azure_client() -> AzureOpenAI:
    """Create an AzureOpenAI client from environment / config constants."""
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def get_embeddings(texts: list[str], client: AzureOpenAI) -> list[list[float]]:
    """Batch-embed a list of strings using Azure OpenAI embeddings."""
    BATCH = 100
    all_embeddings = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        # In Azure, 'model' param is your deployment name
        response = client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=batch)
        all_embeddings.extend([r.embedding for r in response.data])
        time.sleep(0.5)   # respect rate limits
    return all_embeddings


def build_vector_store(chunks: list[dict], openai_client: AzureOpenAI,
                       db_dir: str = CHROMA_DB_DIR,
                       collection_name: str = COLLECTION_NAME) -> chromadb.Collection:
    """Embed all chunks and upsert them into a persistent ChromaDB collection."""
    print(f"\n[VectorDB] Setting up ChromaDB at '{db_dir}'…")
    chroma_client = chromadb.PersistentClient(path=db_dir)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Skip chunks already in the DB (useful on reruns)
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("[VectorDB] All chunks already indexed. Skipping embedding step.")
        return collection

    print(f"[VectorDB] Embedding {len(new_chunks)} new chunks…")
    texts = [c["text"] for c in new_chunks]
    embeddings = get_embeddings(texts, openai_client)

    collection.upsert(
        ids=[c["chunk_id"] for c in new_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]}
                   for c in new_chunks],
    )
    print(f"[VectorDB] ✅ Stored {len(new_chunks)} chunks. "
          f"Total in collection: {collection.count()}")
    return collection


def load_existing_vector_store(db_dir: str = CHROMA_DB_DIR,
                               collection_name: str = COLLECTION_NAME) -> chromadb.Collection:
    """Load an already-built ChromaDB collection (no re-embedding needed)."""
    chroma_client = chromadb.PersistentClient(path=db_dir)
    collection = chroma_client.get_collection(name=collection_name)
    print(f"[VectorDB] Loaded collection '{collection_name}' "
          f"({collection.count()} chunks).")
    return collection


# ────────────────────────────────────────────────────────────────────────────
# STEP 4 – RETRIEVE & GENERATE
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful teaching assistant for an AI Academy course on
"Databases for Generative AI". You have access to the lecture slides (PDF) and
audio transcripts as your knowledge base.

Answer questions clearly and accurately based ONLY on the provided context chunks.
If the context does not contain enough information, say so honestly.
Always cite which source (PDF or audio transcript) you are drawing from."""


def retrieve(query: str, collection: chromadb.Collection,
             openai_client: AzureOpenAI, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and retrieve the top-k most similar chunks."""
    response = openai_client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=[query])
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "distance": round(dist, 4),
        })
    return chunks


def generate_answer(query: str, context_chunks: list[dict],
                    openai_client: AzureOpenAI) -> str:
    """Build a prompt with retrieved context and call the Azure LLM."""
    context_str = "\n\n".join(
        f"[Source: {c['source']} | similarity distance: {c['distance']}]\n{c['text']}"
        for c in context_chunks
    )

    user_message = f"""Use the following context excerpts to answer the question.

CONTEXT:
{context_str}

QUESTION: {query}

ANSWER:"""

    response = openai_client.chat.completions.create(
        model=LLM_DEPLOYMENT,   # Azure deployment name, not model name
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_completion_tokens=800,
    )
    return response.choices[0].message.content.strip()


def ask(query: str, collection: chromadb.Collection,
        openai_client: AzureOpenAI) -> dict:
    """End-to-end RAG: retrieve → generate → return structured result."""
    print(f"\n{'='*60}")
    print(f"Q: {query}")
    print("="*60)

    chunks = retrieve(query, collection, openai_client)
    answer = generate_answer(query, chunks, openai_client)

    print(f"A: {answer}\n")
    print("📎 Sources used:")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {c['source']} (dist={c['distance']}): "
              f"{c['text'][:100].replace(chr(10), ' ')}…")

    return {"question": query, "answer": answer, "context_chunks": chunks}


# ────────────────────────────────────────────────────────────────────────────
# MAIN – orchestrate the full pipeline
# ────────────────────────────────────────────────────────────────────────────

def ingest(rebuild: bool = False):
    """Run the ingestion pipeline: load → chunk → embed → store."""
    openai_client = make_azure_client()
    all_chunks: list[dict] = []

    # ── PDF ──────────────────────────────────────────────────────────────────
    if Path(PDF_PATH).exists():
        pdf_text = extract_text_from_pdf(PDF_PATH)
        all_chunks += chunk_text(pdf_text, source_label="PDF")
    else:
        print(f"[Warning] PDF not found at '{PDF_PATH}'. "
              "Place your PDF there and rerun.")

    # ── Audio ─────────────────────────────────────────────────────────────────
    if Path(AUDIO_DIR).exists():
        audio_text = transcribe_audio(AUDIO_DIR)
        if audio_text:
            all_chunks += chunk_text(audio_text, source_label="AudioTranscript")
    else:
        print(f"[Warning] Audio directory not found at '{AUDIO_DIR}'. "
              "Place your audio files there and rerun.")

    if not all_chunks:
        print("\n[Error] No data to index. Add your PDF/audio files and rerun.")
        return None

    collection = build_vector_store(all_chunks, openai_client)
    return collection


def run_test_queries(collection: chromadb.Collection, openai_client: AzureOpenAI):
    """Run the three required test questions and save a log."""
    test_questions = [
        "What are the production Do's for RAG?",
        "What is the difference between standard retrieval and the ColPali approach?",
        "Why is hybrid search better than vector-only search?",
    ]

    log = []
    for q in test_questions:
        result = ask(q, collection, openai_client)
        log.append({
            "question": result["question"],
            "answer": result["answer"],
            "top_sources": [
                {"source": c["source"], "distance": c["distance"],
                 "snippet": c["text"][:200]}
                for c in result["context_chunks"]
            ],
        })

    # Save log to JSON
    log_path = "chatbot_answers_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Answer log saved to '{log_path}'")


def interactive_chat(collection: chromadb.Collection, openai_client: AzureOpenAI):
    """Simple REPL for chatting with the RAG system."""
    print("\n🤖 RAG Chatbot ready! Type 'quit' to exit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not query:
            continue
        ask(query, collection, openai_client)


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Chatbot for GenAI Databases lecture")
    parser.add_argument("--ingest",   action="store_true", help="Run ingestion pipeline")
    parser.add_argument("--test",     action="store_true", help="Run 3 test questions")
    parser.add_argument("--chat",     action="store_true", help="Interactive chat mode")
    parser.add_argument("--rebuild",  action="store_true", help="Force re-embed all chunks")
    args = parser.parse_args()

    openai_client = make_azure_client()

    if args.ingest or args.rebuild:
        collection = ingest(rebuild=args.rebuild)
    else:
        try:
            collection = load_existing_vector_store()
        except Exception:
            print("No existing DB found. Running ingestion first…")
            collection = ingest()

    if collection is None:
        exit(1)

    if args.test:
        run_test_queries(collection, openai_client)
    elif args.chat:
        interactive_chat(collection, openai_client)
    else:
        # Default: run tests then open chat
        run_test_queries(collection, openai_client)
        interactive_chat(collection, openai_client)