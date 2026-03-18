# Reflection

## What Was the Most Challenging Part?

The most challenging aspect of this project was getting all the moving parts to
work together reliably — PDF extraction, audio transcription, vector storage, and
LLM generation each introduced their own friction. For PDF extraction, lecture
slides proved harder than expected: they contain fragmented bullet points, text
overlaid on images, and non-linear reading orders. PyMuPDF handled this better
than alternatives, but slides with dense visual layouts still yielded imperfect
text that required inspection. On the audio side, running Whisper locally on MP4
video files worked seamlessly thanks to ffmpeg handling the audio extraction
automatically — but the first run required downloading the model weights and was
slower than expected on CPU.

A practical constraint I ran into was storage and compute load. The full knowledge
base consists of four video recordings and four slide decks, which would have
required significant disk space and several hours of Whisper transcription time on
a local machine. To keep the pipeline runnable within reasonable limits, I scoped
the knowledge base down to one PDF (RAG Intro) and one video recording (RAG Intro
Part 1). This is why the chatbot answered the ColPali question with "not enough
context" — that topic is covered in the Databases for GenAI materials which were
not included in this run. In a production environment, all four videos and PDFs
would be ingested, and the pipeline is already designed to handle them — simply
dropping additional files into the `data/` folder and rerunning `--ingest` would
index them automatically.

## What Did I Learn?

Building this end-to-end pipeline made the theoretical concepts from the lecture
feel very concrete. The most eye-opening moment was seeing how directly the
chunking strategy affects answer quality — when chunks were too large, retrieved
context was diluted and the LLM produced vague answers; when too small, important
context was split across chunk boundaries and lost. The 600-character size with
100-character overlap I settled on struck a reasonable balance, but it highlighted
why the lecture emphasized that chunking is one of the most critical and
underestimated decisions in a RAG system.

I also gained a much deeper appreciation for why hybrid search matters. Testing
with vector-only retrieval showed that exact technical terms were sometimes missed
in favour of semantically similar but less precise chunks — exactly the failure
mode the lecture warned about. This made the argument for combining BM25 with
dense retrieval feel immediately practical rather than theoretical. Overall, the
key takeaway was that a RAG pipeline is only as good as its retrieval, and
retrieval is only as good as the ingestion and chunking decisions made upfront —
no amount of LLM quality can compensate for poorly retrieved context.