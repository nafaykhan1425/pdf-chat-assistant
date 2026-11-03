# 📄 PDF Chat Assistant (RAG)

Upload any text-based PDF, such as a book, lecture notes or a research paper, and ask
questions about it in a chat interface. Every answer cites the page it came from, and
you can expand **Sources** to see the exact passages the AI used.

## How it works (RAG pipeline)

```
PDF ──► extract text per page ──► split into overlapping chunks ──► embed chunks (Gemini)
                                                                          │
Question ──► embed question ──► find the most similar chunks (cosine) ◄───┘
                                          │
                        top chunks + question ──► Gemini ──► answer with (Page N) citations
```

1. **Extract**: `pypdf` reads the text of each page.
2. **Chunk**: text is split into ~1000-character pieces with overlap, each remembering its page number.
3. **Embed**: chunks are converted to vectors with `gemini-embedding-001`.
4. **Retrieve**: your question is embedded and compared with every chunk using cosine similarity.
5. **Generate**: the top 5 chunks are sent to a Gemini chat model, which must answer only from them and cite pages.

## Features

- Chat with any text-based PDF
- Page citations in every answer
- Expandable source passages with relevance scores
- Follow-up questions understand the conversation
- Automatic retries and model fallback when the API is busy or a model is retired
- Handles encrypted, scanned and very large PDFs gracefully

## Project Structure

```
pdf-chat-assistant/
├── app.py            # Streamlit interface
├── rag.py            # RAG logic (extract, chunk, embed, retrieve, answer)
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

1. Clone the repository

   ```bash
   git clone https://github.com/<your-username>/pdf-chat-assistant.git
   cd pdf-chat-assistant
   ```

2. Create a virtual environment and install dependencies

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

4. Create `.streamlit/secrets.toml`:

   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

5. Run the app

   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub (never commit your API key).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, select the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:

   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

5. Click **Deploy**.

## Configuration

Settings you can change:

| Setting | Where | Default |
| --- | --- | --- |
| Embedding model | `EMBED_MODEL` in `rag.py` | `gemini-embedding-001` |
| Chat models (tried in order) | `CHAT_MODELS` in `rag.py` | `gemini-3.8-flash`, `gemini-3.5-flash`, `gemini-3.1-flash-lite` |
| Chunk size and overlap | `chunk_pages()` in `rag.py` | 1000 / 150 characters |
| Sections retrieved per question | `TOP_K` in `app.py` | 5 |
| Max file size | `MAX_FILE_MB` in `app.py` | 25 MB |
| Max indexed sections | `MAX_CHUNKS` in `app.py` | 1500 |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "No text found" | The PDF is scanned (images). Use a PDF with selectable text, or run OCR on it first. |
| `503` busy error | The app retries automatically. If it persists, wait a minute and try again. |
| `429` rate limit | Free-tier limit reached, especially when indexing large PDFs. Wait and retry. |
| `404` model not found | Google retired a model. Update the model names in `rag.py` using the [official list](https://ai.google.dev/gemini-api/docs/models). |
| Slow indexing | Large PDFs need many embedding calls. Try a smaller file or reduce `MAX_CHUNKS`. |

## Limitations

- Scanned PDFs without a text layer are not supported (no OCR).
- Tables and multi-column layouts may be extracted in a jumbled order.
- Vectors are kept in memory, so the document must be re-indexed after each app restart.
- Answers depend on retrieval quality; very vague questions may pull the wrong passages.

## Ideas for Improvement

- Support multiple PDFs at once
- Add OCR for scanned documents
- Hybrid search (keywords + embeddings)
- Persist the index with a vector database such as ChromaDB or FAISS

## License

MIT
