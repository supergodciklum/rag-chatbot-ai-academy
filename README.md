# RAG Chatbot — Databases for GenAI Lecture

**GitHub Repository:** https://github.com/supergodciklum/rag-chatbot-ai-academy.git

A complete Retrieval-Augmented Generation (RAG) pipeline that ingests a PDF
and audio recordings from the "Databases for GenAI" lecture and lets you chat
with the content.

## Architecture

```
PDF ──► PyMuPDF ──┐
                   ├──► Chunking (LangChain) ──► OpenAI Embeddings ──► ChromaDB
Audio ──► Whisper ─┘                                                       │
                                                                           ▼
User Question ──► Embed Question ──► Similarity Search ──► Top-K Chunks ──► GPT-4 ──► Answer
```

## Setup

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd rag_chatbot
pip install -r requirements.txt
```

> **ffmpeg is required** for Whisper audio processing:
> - macOS: `brew install ffmpeg`
> - Ubuntu/Debian: `sudo apt install ffmpeg`
> - Windows: Download from https://ffmpeg.org/download.html

### 2. Set your Azure OpenAI credentials

You need three values from your **Azure Portal → Azure OpenAI resource**:

```bash
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://YOUR-RESOURCE-NAME.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-01"

# These must match the deployment names you created in Azure OpenAI Studio:
export AZURE_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
export AZURE_LLM_DEPLOYMENT="gpt-4o-mini"
```

Or create a `.env` file in the project root:
```
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE-NAME.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_LLM_DEPLOYMENT=gpt-4o-mini
```

> **Where to find these values:**
> Azure Portal → your OpenAI resource → **Keys and Endpoint** (for key + endpoint)
> Azure OpenAI Studio → **Deployments** (for your deployment names)

### 3. Add your source files

```
rag_chatbot/
├── data/
│   ├── RAG Intro.pdf               ← Place the lecture PDF here
│   └── RAG Intro.mp4               ← Place the video/audio file here
```

> **Note:** The `data/` folder is intentionally **not included** in this repository because video files are too large to push to GitHub.
> You must add the source files manually before running the pipeline.

## Usage

### Ingest (build the vector index)
```bash
python rag_pipeline.py --ingest
```

This will:
1. Extract text from the PDF
2. Transcribe all audio files (downloads Whisper model on first run ~150MB)
3. Chunk all text
4. Embed chunks via OpenAI API
5. Store in ChromaDB (persisted to `./chroma_db/`)

### Run test questions
```bash
python rag_pipeline.py --test
```

Runs the three required test questions and saves results to `chatbot_answers_log.json`.

### Interactive chat
```bash
python rag_pipeline.py --chat
```

### Default (test + chat)
```bash
python rag_pipeline.py
```

### Force rebuild (re-embed everything)
```bash
python rag_pipeline.py --ingest --rebuild
```

## Configuration

Edit the constants at the top of `rag_pipeline.py`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | LLM for generation |
| `WHISPER_MODEL` | `base` | Whisper size (tiny/base/small/medium/large) |
| `CHUNK_SIZE` | `600` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `5` | Retrieved chunks per query |

## Design Decisions

### Why PyMuPDF over pdfplumber/pdfminer?
PyMuPDF is faster, handles more PDF variants, and extracts text in reading order more reliably.

### Why RecursiveCharacterTextSplitter?
It respects natural boundaries (paragraphs → sentences → words) before splitting by character count, producing more semantically coherent chunks than fixed-size splits.

### Why ChromaDB?
Easy local setup with persistence, no external service required. Production deployments should use Qdrant or Weaviate for better scalability and hybrid search support.

### Why `text-embedding-3-small`?
Good quality-to-cost ratio (5x cheaper than `text-embedding-3-large`) while performing well on retrieval benchmarks.

### Chunking strategy
600-char chunks with 100-char overlap. Overlap ensures context at boundaries is not lost — a sentence split across two chunks is still partially present in both.
