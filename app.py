
import streamlit as st
import pymupdf
import chromadb
import ollama
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Folio | PDF Chat",
    page_icon="F",
    layout="wide"
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --paper: #f5f2e9;
        --surface: #fffdf8;
        --ink: #20352e;
        --muted: #69776f;
        --pine: #205b4b;
        --pine-dark: #174538;
        --copper: #98452f;
        --line: #dedfd4;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 242, 233, 0.92);
    }

    .block-container {
        max-width: 1080px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    .folio-hero {
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 4px solid var(--pine);
        border-radius: 14px;
        padding: 2rem 2.25rem;
        margin: 0 0 2rem;
        box-shadow: 0 8px 24px rgba(32, 53, 46, 0.045);
    }

    .folio-eyebrow,
    .section-kicker {
        color: var(--copper);
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.13em;
        line-height: 1.4;
        text-transform: uppercase;
    }

    .folio-title {
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2.25rem, 5vw, 3.5rem);
        font-weight: 600;
        letter-spacing: -0.045em;
        line-height: 1.05;
        margin: 0.55rem 0 0.65rem;
    }

    .folio-description {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.65;
        margin: 0;
        max-width: 620px;
    }

    .section-heading {
        border-bottom: 1px solid var(--line);
        margin: 1.75rem 0 1.15rem;
        padding-bottom: 0.85rem;
    }

    .section-title {
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.65rem;
        font-weight: 600;
        letter-spacing: -0.025em;
        line-height: 1.25;
        margin: 0.25rem 0 0;
    }

    .stMarkdown, [data-testid="stCaptionContainer"] {
        color: var(--ink);
    }

    .stButton > button,
    .stDownloadButton > button {
        align-items: center;
        background: var(--pine);
        border: 1px solid var(--pine);
        border-radius: 9px;
        color: #fffdf8;
        display: flex;
        font-weight: 650;
        justify-content: center;
        min-height: 2.8rem;
        padding: 0.55rem 1rem;
        transition: background 140ms ease, border-color 140ms ease,
                    transform 140ms ease;
        width: 100%;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: var(--pine-dark);
        border-color: var(--pine-dark);
        color: #fffdf8;
        transform: translateY(-1px);
    }

    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible {
        box-shadow: 0 0 0 3px var(--paper), 0 0 0 5px var(--copper);
        outline: none;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface);
        border: 1px dashed #9eac9f;
        border-radius: 12px;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--pine);
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--pine);
    }

    [data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
    }

    [data-testid="stChatMessage"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }

    [data-testid="stChatInput"] textarea {
        background: var(--surface);
        border-color: #aeb8ad;
        border-radius: 10px;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--pine);
        box-shadow: 0 0 0 1px var(--pine);
    }

    hr {
        border-color: var(--line);
    }

    @media (max-width: 640px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.25rem;
        }

        .folio-hero {
            border-radius: 11px;
            padding: 1.4rem 1.25rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <header class="folio-hero">
        <div class="folio-eyebrow">A quieter way to work with documents</div>
        <h1 class="folio-title">Folio</h1>
        <p class="folio-description">
            Bring a PDF. Find the details, get the gist, and ask better questions.
        </p>
    </header>
    """,
    unsafe_allow_html=True
)


def section_heading(step, title):
    st.markdown(
        f"""
        <div class="section-heading">
            <div class="section-kicker">{step}</div>
            <h2 class="section-title">{title}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_chroma_collection(collection_name):
    client = chromadb.PersistentClient(path="./chroma_db")

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


section_heading("01 · Add a document", "Start with your PDF")
pdf_file = st.file_uploader(
    "Choose a PDF to upload",
    type=["pdf"]
)

if pdf_file is not None:
    pdf_bytes = pdf_file.getvalue()

    st.download_button(
        label="Download Uploaded PDF",
        data=pdf_bytes,
        file_name=pdf_file.name,
        mime="application/pdf"
    )

    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    collection_name = f"pdf_{pdf_hash}"

    previous_hash = st.session_state.get("current_pdf_hash")

    if previous_hash != pdf_hash:
        st.session_state["pdf_ready"] = False
        st.session_state["chat_history"] = []
        st.session_state["pdf_summary"] = ""
        st.session_state["current_pdf_hash"] = pdf_hash
        st.session_state["pdf_name"] = pdf_file.name
        st.session_state["collection_name"] = collection_name
        st.session_state["document_ids"] = []
        st.session_state["selected_question"] = ""

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        page_texts = []

        for page_number, page in enumerate(pdf):
            page_text = page.get_text()

            if page_text.strip():
                page_texts.append({
                    "page": page_number + 1,
                    "text": page_text
                })

    if not page_texts:
        st.warning(
            "No selectable text was found. This PDF may need OCR."
        )

    else:
        st.success("Your PDF is ready to explore.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = []

        for page in page_texts:
            page_chunks = splitter.split_text(page["text"])

            for chunk in page_chunks:
                chunks.append({
                    "text": chunk,
                    "page": page["page"]
                })

        metric_pages, metric_chunks = st.columns(2)
        with metric_pages:
            st.metric("Pages with text", len(page_texts))
        with metric_chunks:
            st.metric("Passages prepared", len(chunks))

        with st.expander("Preview text chunks"):
            for index, chunk in enumerate(chunks[:10]):
                st.markdown(
                    f"**Chunk {index + 1} — Page {chunk['page']}**"
                )
                st.write(chunk["text"])
                st.divider()

        section_heading("02 · Get the overview", "Summarize this document")

        if st.button("Summarize PDF"):
            try:
                with st.spinner("Generating your PDF summary..."):
                    max_chars = 12000
                    text_parts = []
                    current_part = ""
                    summaries = []

                    for page in page_texts:
                        page_content = (
                            f"Page {page['page']}:\n"
                            f"{page['text']}\n\n"
                        )

                        if (
                            len(current_part) + len(page_content) > max_chars
                            and current_part
                        ):
                            text_parts.append(current_part)
                            current_part = ""

                        current_part += page_content

                    if current_part:
                        text_parts.append(current_part)

                    for part in text_parts:
                        response = ollama.chat(
                            model="llama3.2",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "Summarize the supplied PDF text "
                                        "in simple English. Include the "
                                        "main ideas, important facts, "
                                        "and key points. Do not invent "
                                        "information."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": part
                                }
                            ]
                        )

                        summaries.append(
                            response["message"]["content"]
                        )

                    if len(summaries) > 1:
                        final_response = ollama.chat(
                            model="llama3.2",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "Combine these partial summaries "
                                        "into one clear, well-organized "
                                        "summary in simple English. "
                                        "Avoid repetition and do not add "
                                        "unsupported facts."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": "\n\n".join(summaries)
                                }
                            ]
                        )

                        summary = final_response["message"]["content"]
                    else:
                        summary = summaries[0]

                    st.session_state["pdf_summary"] = summary

            except Exception as error:
                st.error(f"Could not summarize the PDF: {error}")

        if st.session_state.get("pdf_summary"):
            st.subheader("Your PDF Summary")
            st.markdown(st.session_state["pdf_summary"])

            st.download_button(
                label="Download Summary",
                data=st.session_state["pdf_summary"],
                file_name="pdf_summary.txt",
                mime="text/plain"
            )

        section_heading("03 · Prepare for questions", "Make it searchable")
        if st.button("Create Embeddings"):
            try:
                with st.spinner(
                    "Creating embeddings and storing in ChromaDB..."
                ):
                    model = load_embedding_model()
                    texts = [chunk["text"] for chunk in chunks]

                    embeddings = model.encode(
                        texts,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )

                    collection = get_chroma_collection(collection_name)

                    document_ids = [
                        f"chunk_{i}" for i in range(len(chunks))
                    ]

                    collection.upsert(
                        ids=document_ids,
                        documents=texts,
                        embeddings=embeddings.tolist(),
                        metadatas=[
                            {"page": chunk["page"]}
                            for chunk in chunks
                        ]
                    )

                    st.session_state["chunks"] = chunks
                    st.session_state["pdf_ready"] = True
                    st.session_state["pdf_name"] = pdf_file.name
                    st.session_state["document_ids"] = document_ids
                    st.session_state["collection_name"] = collection_name

                st.success(
                    f"Created and stored {len(embeddings)} "
                    "embeddings successfully!"
                )

                st.write("Embedding dimensions:", embeddings.shape[1])
                st.write(
                    "Documents stored:",
                    collection.count()
                )

            except Exception as error:
                st.error(f"An error occurred: {error}")


chat_step = (
    "04 · Explore the details"
    if st.session_state.get("pdf_ready", False)
    else "When you're ready"
)
section_heading(chat_step, "Ask your PDF")

if st.session_state.get("pdf_ready", False):
    st.caption(f"Reading: {st.session_state['pdf_name']}")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.subheader("A few ways to get started")

    suggested_questions = [
        "What is this PDF about?",
        "Summarize the main points.",
        "What are the important facts?",
        "What is the conclusion?",
        "List the key topics discussed."
    ]

    for i in range(0, len(suggested_questions), 2):
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                suggested_questions[i],
                key=f"suggested_question_{i}"
            ):
                st.session_state["selected_question"] = (
                    suggested_questions[i]
                )
                st.rerun()

        if i + 1 < len(suggested_questions):
            with col2:
                if st.button(
                    suggested_questions[i + 1],
                    key=f"suggested_question_{i + 1}"
                ):
                    st.session_state["selected_question"] = (
                        suggested_questions[i + 1]
                    )
                    st.rerun()

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDF...")

    if st.session_state.get("selected_question"):
        question = st.session_state.pop("selected_question")

    if question:
        st.session_state["chat_history"].append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                with st.spinner(
                    "Searching your PDF and generating an answer..."
                ):
                    model = load_embedding_model()

                    collection = get_chroma_collection(
                        st.session_state["collection_name"]
                    )

                    question_embedding = model.encode(
                        [question],
                        normalize_embeddings=True
                    )

                    document_ids = st.session_state.get(
                        "document_ids", []
                    )

                    if not document_ids:
                        st.error(
                            "Please create embeddings for your PDF first."
                        )
                        st.stop()

                    results = collection.query(
                        query_embeddings=question_embedding.tolist(),
                        n_results=min(3, len(document_ids)),
                        include=["documents", "metadatas"]
                    )

                    documents = results.get("documents", [[]])[0]
                    metadatas = results.get("metadatas", [[]])[0]

                    if not documents:
                        answer = (
                            "I couldn't find relevant information "
                            "in the PDF."
                        )
                        source_pages = []
                    else:
                        context_parts = []

                        for i in range(len(documents)):
                            context_parts.append(
                                f"Page {metadatas[i]['page']}:\n"
                                f"{documents[i]}"
                            )

                        context = "\n\n".join(context_parts)

                        response = ollama.chat(
                            model="llama3.2",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a helpful PDF assistant. "
                                        "Answer using only the supplied "
                                        "PDF context. If the answer is "
                                        "not present in the context, say "
                                        "the PDF does not provide enough "
                                        "information. Do not invent facts. "
                                        "Explain in simple English."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        f"PDF context:\n{context}\n\n"
                                        f"Question: {question}"
                                    )
                                }
                            ]
                        )

                        answer = response["message"]["content"]

                        source_pages = sorted({
                            metadata["page"]
                            for metadata in metadatas
                        })

                    st.markdown(answer)

                    if source_pages:
                        with st.expander("📖 View source passages"):
                            for i in range(len(documents)):
                                st.markdown(
                                    f"**Page {metadatas[i]['page']}**"
                                )
                                st.write(documents[i])
                                st.divider()

                        st.caption(
                            "Source pages: "
                            + ", ".join(
                                str(page) for page in source_pages
                            )
                        )

            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": answer
            })

        except Exception as error:
            st.error(f"Could not generate an answer: {error}")

else:
    st.info(
        "Upload a PDF and click Create Embeddings before chatting."
    )


if st.session_state.get("pdf_ready", False):
    chat_history = st.session_state.get("chat_history", [])

    if chat_history:
        chat_text = ""

        for message in chat_history:
            role = message["role"].capitalize()
            content = message["content"]
            chat_text += f"{role}:\n{content}\n\n"

        st.download_button(
            label="Export Chat History",
            data=chat_text,
            file_name="pdf_chat_history.txt",
            mime="text/plain"
        )

    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.rerun()
