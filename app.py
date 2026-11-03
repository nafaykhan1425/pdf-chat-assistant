import hashlib
import io
import os

import streamlit as st
from google import genai
from google.genai import errors, types

import rag

st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄", layout="wide")

MAX_FILE_MB = 25
MAX_CHUNKS = 1500  # keeps embedding time and API usage reasonable
TOP_K = 5


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")


api_key = get_api_key()
if not api_key:
    st.error(
        "Gemini API key not found. Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` "
        "locally, or to Secrets on Streamlit Community Cloud."
    )
    st.stop()


@st.cache_resource
def get_client(key: str):
    return genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=rag.REQUEST_TIMEOUT_MS),
    )


client = get_client(api_key)

if "doc" not in st.session_state:
    st.session_state.doc = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar: upload and index the PDF
# ---------------------------------------------------------------------------
def process_pdf(file_bytes: bytes, name: str):
    status = st.sidebar.empty()
    status.info("Reading PDF...")
    pages, total_pages = rag.extract_pages(io.BytesIO(file_bytes))

    if not pages:
        status.empty()
        st.sidebar.error(
            "No text found. This looks like a scanned PDF (images only). "
            "Please upload a PDF with selectable text."
        )
        return None

    chunks = rag.chunk_pages(pages)
    if len(chunks) > MAX_CHUNKS:
        st.sidebar.warning(
            f"Large document: only the first {MAX_CHUNKS} sections were indexed."
        )
        chunks = chunks[:MAX_CHUNKS]

    status.info(f"Indexing {len(chunks)} sections...")
    bar = st.sidebar.progress(0.0)
    vectors = rag.embed_texts(
        client,
        [c["text"] for c in chunks],
        task_type="RETRIEVAL_DOCUMENT",
        progress_cb=bar.progress,
    )
    bar.empty()
    status.empty()
    return {
        "name": name,
        "pages": total_pages,
        "chunks": chunks,
        "vectors": vectors,
    }


st.sidebar.title("📄 PDF Chat Assistant")
uploaded = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is not None:
    data = uploaded.getvalue()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        st.sidebar.error(f"File is larger than {MAX_FILE_MB} MB.")
    else:
        file_id = hashlib.md5(data).hexdigest()
        current = st.session_state.doc
        if current is None or current["id"] != file_id:
            try:
                doc = process_pdf(data, uploaded.name)
                if doc:
                    doc["id"] = file_id
                    st.session_state.doc = doc
                    st.session_state.messages = []
            except ValueError as err:
                st.sidebar.error(str(err))
            except errors.APIError as err:
                if err.code in rag.RETRYABLE_CODES:
                    st.sidebar.error("The AI service is busy. Please try again in a minute.")
                else:
                    st.sidebar.error(f"API error ({err.code}): {err.message}")
            except Exception as err:
                st.sidebar.error(f"Could not read this PDF: {err}")

doc = st.session_state.doc
if doc:
    st.sidebar.success(f"**{doc['name']}**\n\n{doc['pages']} pages, {len(doc['chunks'])} sections indexed")
    if st.sidebar.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.sidebar.caption("Answers are generated from your PDF only and cite page numbers.")


# ---------------------------------------------------------------------------
# Main: chat
# ---------------------------------------------------------------------------
st.title("Chat with your PDF")

if not doc:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()


def show_sources(sources):
    with st.expander("📚 Sources"):
        for s in sources:
            st.markdown(f"**Page {s['page']}** (relevance {s['score']:.2f})")
            st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

question = st.chat_input("Ask something about your document...")

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        hits, log = [], []
        answer = None
        with st.status("Working on your question...", expanded=True) as status:
            try:
                # Short follow-ups ("explain more") need the previous question for context
                search_text = question
                if len(question.split()) < 6:
                    prev = [m["content"] for m in history if m["role"] == "user"]
                    if prev:
                        search_text = prev[-1] + " " + question

                status.write("1/3 Embedding your question...")
                q_vec = rag.embed_texts(client, [search_text], task_type="RETRIEVAL_QUERY")[0]

                status.write("2/3 Finding the most relevant pages...")
                hits = rag.retrieve(q_vec, doc["vectors"], doc["chunks"], k=TOP_K)

                status.write("3/3 Generating the answer...")
                answer = rag.answer_question(
                    client, question, hits, history,
                    on_status=lambda m: status.write(m), log=log,
                )
                status.update(label="Done", state="complete", expanded=False)
            except errors.APIError as err:
                log.append(f"API error {err.code} - {str(err.message)[:200]}")
                status.update(label="Something went wrong", state="error")
            except Exception as err:
                log.append(f"{type(err).__name__} - {str(err)[:200]}")
                status.update(label="Something went wrong", state="error")

        if answer is None:
            answer = (
                "Sorry, I couldn't get an answer. This is usually a temporary overload, "
                "a rate limit, an invalid API key or an outdated model name. "
                "See the diagnostic details below, then try again."
            )
            hits = []
            st.markdown(answer)
            with st.expander("🔍 Diagnostic details"):
                for line in log:
                    st.code(line)
        else:
            st.markdown(answer)
            show_sources(hits)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": hits}
    )
