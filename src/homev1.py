import os
from pathlib import Path
import base64, streamlit as st
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer





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
            This is a non-production demonstration environment using a limited, anonymised data set from the provided manual.<br><br>
            No not rely on it for decision making.
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
            "Assume that all questions are related to answering what to do in an emergency. "
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

    INPUT_DIR = "./data"
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
                st.session_state.messages.append({"role": "assistant", "content": response.response})

    # Optional: a manual “Rebuild library” button
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
    reader = SimpleDirectoryReader(input_dir=str(base), recursive=True)

    # Nice relative file list for display
    relative_files = []
    for p in reader.input_files:
        p = Path(p).resolve()
        try:
            rel = p.relative_to(base)
        except ValueError:
            rel = Path(os.path.relpath(p, start=base))
        relative_files.append(str(rel))

    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index, relative_files

main()
