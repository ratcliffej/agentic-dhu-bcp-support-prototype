import os
from pathlib import Path
import base64, streamlit as st
from llama_index.core import Settings, VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from urllib.parse import quote




def main():
    # V2 instance
    st.set_page_config(page_title="DHU BCP Support Assistant", page_icon=":books:")



    # Derive a per-user/session key (pick whatever makes sense for you)
    headers = st.context.headers
    session_id = headers.get("X-Forwarded-For") or st.session_state.get("session_id")

    if not session_id:
        session_id = str(st.session_state.setdefault("session_id", os.urandom(8).hex()))

    # Build memory (token_limit = how much history to keep)
    if "li_memory" not in st.session_state:
        st.session_state.li_memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

    # --- Sidebar notice (non-production banner) ---
    st.sidebar.markdown(
        """
        <style>
        /* Make a sticky banner at the top of the sidebar */
        [data-testid="stSidebar"] .sidebar-banner {
            position: sticky;
            top: 0;
            background: #fffae6;           /* light yellow */
            color: #333;
            padding: 12px 12px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            font-weight: 600;
            line-height: 1.3;
            z-index: 1;                     /* sit above other sidebar content */
            margin-bottom: 12px;
        }
        </style>
        <div class="sidebar-banner">
            ⚠️ Please note:<br><br> 
            This is a non-production demonstration environment using a limited, anonymised data set from the provided manual.
            <br><br>
            No not rely on it for decision making.
            <br><br>
            The links to the source document and page are disabled due to limitations of the prototyping environment.
        </div>
        """,
        unsafe_allow_html=True,
    )



    os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]

    Settings.llm = OpenAI(
        model="gpt-4.1-mini-2025-04-14",
        temperature=0.5,
        system_prompt=(
            "You are an expert on advising staff on Business Continuity Plan (BCP) queries. "
            "Assume that all questions are related to answering what to do in emergency or for unusual situations that need clear advice. "
            "Use bullet points where appropriate. "
            "Keep answers clear, concise and factual; avoid hallucinations; include citations."
        )
    )

    logo_path = Path(__file__).resolve().parents[1] / "img" / "logo.jpg"
    data = base64.b64encode(logo_path.read_bytes()).decode()
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/jpeg;base64,{data}" width="100">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("""
            # DHU BCP Support Assistant
            """)

    headers = st.context.headers
    ip = headers.get("X-Forwarded-For") or headers.get("Host")
    st.write("**User IP:**", ip)

    INPUT_DIR = "./docs"
    sig = _dir_signature(Path(INPUT_DIR).resolve())
    index, relative_files = get_index(INPUT_DIR, sig)

    st.markdown("**Sources:**\n" + "\n".join(f"- {rf}" for rf in relative_files))

    # Build the chat engine each run (cheap), or cache it too if you like
    chat_engine = index.as_chat_engine(chat_mode="condense_question", memory = st.session_state.li_memory, verbose=True) # type: ignore

    # --- chat UI (unchanged) ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about DHU policies and procedures."}
        ]

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt) # type: ignore
                st.write(response.response)
                # --- Cite sources (filename + page) with stub URLs (non-clickable) ---
                from urllib.parse import quote

                def _build_stub_url(md: dict, page: str | int | None):
                    """Return a text URL for later enabling as a real hyperlink.
                    Prefers a configured `docs_base_url`; otherwise shows a local file:// URI.
                    """
                    base = st.secrets.get("docs_base_url")
                    page_str = str(page) if page is not None else None

                    # Prefer a public base if provided (works in prod later)
                    if base:
                        name = md.get("file_name") or Path(md.get("file_path", "")).name
                        if not name:
                            return None
                        href = f"{str(base).rstrip('/')}/{quote(name)}"
                        if page_str:
                            href += f"#page={quote(page_str)}"
                        return href

                    # Fallback: local file URI (useful for debugging; not clickable in prototype)
                    fp = md.get("file_path")
                    if fp:
                        try:
                            uri = Path(fp).resolve().as_uri()
                            if page_str:
                                uri += f"#page={page_str}"
                            return uri
                        except Exception:
                            return None
                    return None

                def _fmt_citation(sn):
                    md = getattr(sn, "node", sn).metadata or {}

                    name = (
                        md.get("file_name")
                        or Path(md.get("file_path", "")).name
                        or md.get("filename")
                        or "document"
                    )
                    page_val = (
                        md.get("page_label")
                        or md.get("page_number")
                        or md.get("page_index")
                        or md.get("page")
                    )
                    page = str(page_val) if page_val is not None else "?"

                    score = getattr(sn, "score", None)
                    label = f"{name} p.{page}" + (f" (score {score:.2f})" if score is not None else "")

                    # Build a stub URL string for visibility (not clickable)
                    href = _build_stub_url(md, page if page != "?" else None)
                    return label, href

                if getattr(response, "source_nodes", None):
                    st.markdown("**Sources:**")
                    for sn in response.source_nodes[:5]:  # top 5
                        label, href = _fmt_citation(sn)
                        if href:
                            st.markdown(f"- {label} — URL: `{href}`")
                        else:
                            st.markdown(f"- {label}")

                st.session_state.messages.append({"role": "assistant", "content": response.response})

    # Optional: maintenance actions — reset chat + rebuild index
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset conversation"):
            # Clear LlamaIndex chat memory and UI transcript
            try:
                if "li_memory" in st.session_state:
                    st.session_state.li_memory.reset()
            except Exception:
                pass
            st.session_state.messages = [
                {"role": "assistant", "content": "Memory cleared. Ask me a question about DHU policies and procedures."}
            ]
            st.rerun()
    with col2:
        if st.button("Rebuild library"):
            get_index.clear()   # clears the cache so next run rebuilds
            st.rerun()

def _dir_signature(base: Path) -> tuple:
    """Return a signature that changes when any file under `base` changes."""
    return tuple(sorted(
        (str(p), p.stat().st_mtime_ns)
        for p in base.rglob("*") if p.is_file()
    ))

@st.cache_resource(show_spinner=False)
def get_index(input_dir: str, signature: tuple):
    """Build once; re-build only if `signature` changes."""
    base = Path(input_dir).resolve()

    # Collect files
    pdfs = list(base.rglob("*.pdf"))
    others = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() not in {".pdf"}]

    # Nice relative file list for display
    relative_files = []
    for p in list(base.rglob("*")):
        if p.is_file():
            try:
                rel = p.resolve().relative_to(base)
            except ValueError:
                rel = Path(os.path.relpath(p.resolve(), start=base))
            relative_files.append(str(rel))

    docs = []

    # 1) Load PDFs page-by-page so we keep page numbers in metadata
    if pdfs:
        reader = PyMuPDFReader()  # preserves page labels / numbers
        for pdf in pdfs:
            # Version-safe call: newer readers use `file=`, older use `file_path=`
            try:
                page_docs = reader.load_data(file_path=pdf)
            except TypeError:
                page_docs = reader.load_data(file_path=str(pdf))

            # Ensure filename metadata and guarantee a page number exists
            for i, d in enumerate(page_docs, start=1):
                d.metadata["file_path"] = str(pdf)
                d.metadata["file_name"] = pdf.name
                # Normalise page metadata across loaders
                if "page_label" not in d.metadata and "page_number" not in d.metadata and "page" not in d.metadata:
                    d.metadata["page_number"] = i
            docs.extend(page_docs)

    # 2) Load any other files (DOCX/TXT etc.) using a simple reader fallback
    if others:
        # This fallback may not add page numbers; but we still set filename metadata
        fallback_reader = SimpleDirectoryReader(input_files=[str(p) for p in others])
        other_docs = fallback_reader.load_data()
        for d in other_docs:
            p = Path(d.metadata.get("file_path", d.metadata.get("filename", "")) or "")
            if p:
                # d.metadata.setdefault("file_path", str(p))
                # d.metadata.setdefault("source", p.name)
                d.metadata["file_path"] = str(p)
                d.metadata["file_name"] = p.name
        docs.extend(other_docs)

    # Consistent chunking; metadata from Documents is carried onto Nodes
    parser = SentenceSplitter(chunk_size=900, chunk_overlap=120)

    index = VectorStoreIndex.from_documents(docs, transformations=[parser])
    return index, relative_files

main()
