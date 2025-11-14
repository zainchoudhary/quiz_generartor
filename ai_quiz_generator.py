import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
from langgraph.graph import StateGraph, END
from rag_system import RAG
import hashlib
from typing import TypedDict, Optional, List, Dict

api_key = os.environ.get("GOOGLE_API_KEY")
# ------------------ Caching RAG instance ------------------
@st.cache_resource
def get_rag():
    return RAG()


rag = get_rag()

# ------------------ Page Config ------------------
st.set_page_config(page_title="🧠 AI Quiz Generator", layout="centered")

# ------------------ Custom CSS (your original styles kept intact) ------------------
st.markdown(""" 
<style>
@import url('https://fonts.fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

:root {
    --primary-accent: #33CCFF;
    --secondary-dark: #0B1A2B;
    --background-main: #001f3f;
    --card-bg-light: #162231;
    --text-color: #F0F5FA;
    --muted-text: #9FB8C9;
}

body, .stApp {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text-color);
    background: var(--background-main);
    overflow-x: hidden;
}

header { visibility: hidden; height: 0; }

p { 
    text-align: center; 
    color: var(--muted-text);
    font-size: 1.1em;
}

/* --- Upgraded Neon Title with Navy Blue --- */
.neon-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4em;
    font-weight: 900;
    text-align: center;
    letter-spacing: 2px;
    background: linear-gradient(120deg, #0B1A2B, #00FFE0, #33CCFF); /* Navy + cyan gradient */
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 
        0 0 5px #0B1A2B,
        0 0 10px #00FFE0,
        0 0 20px #33CCFF,
        0 0 30px #33CCFF,
        0 0 40px #00FFE0,
        0 0 55px #33CCFF,
        0 0 75px #33CCFF;
}

@keyframes flicker {
    0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% { opacity: 1; }
    20%, 22%, 24%, 55% { opacity: 0.4; }
}

/* Floating Bubbles Animation */
.bubble {
    position: absolute;
    border-radius: 50%;
    opacity: 0.2;
    animation: float 20s linear infinite;
}

@keyframes float {
    0% { transform: translateY(100vh) scale(0); opacity:0.2;}
    50% { opacity:0.5; }
    100% { transform: translateY(-10vh) scale(1);}
}

/* Quiz Card Styles */
.quiz-card {
    background-color: var(--card-bg-light);
    padding: 25px 30px;
    border-radius: 20px;
    margin: 25px auto;
    border-left: 6px solid var(--primary-accent);
    box-shadow: 0 15px 40px rgba(0,0,0,0.7);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.quiz-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.8);
}

.quiz-question {
    font-size: 1.35em;
    font-weight: 600;
    margin-bottom: 15px;
    color: var(--primary-accent);
}

.quiz-option {
    padding: 12px 20px;
    background-color: var(--secondary-dark);
    border-radius: 12px;
    margin-bottom: 10px;
    cursor: pointer;
    transition: background 0.3s ease, transform 0.2s ease, color 0.3s ease;
    color: var(--text-color);
}
.quiz-option:hover {
    background-color: #1F3B5A;
    color: #33CCFF;
    transform: translateX(5px);
}

/* Buttons Styling */
.stDownloadButton button, 
.stButton button { 
    background: linear-gradient(135deg, #0B1A2B, #162231) !important;
    color: #33CCFF !important;
    font-weight: 700;
    font-size: 1.05em;
    padding: 12px 28px !important;
    border-radius: 20px !important;
    border: 2px solid #33CCFF !important;
    box-shadow: 0 5px 20px rgba(0,0,0,0.5);
    transition: all 0.3s ease;
}
.stDownloadButton button:hover,
.stButton button:hover {
    box-shadow: 0 10px 30px rgba(0,0,0,0.7);
    background: linear-gradient(135deg, #162231, #0B1A2B);
    color: #00FFE0 !important;
    transform: scale(1.05);
}
</style>

<div id="bubbles"></div>

<script>
const colors = ['#00FFE0','#33CCFF','#FF33CC','#FFDD33','#33FFAA'];
for(let i=0;i<30;i++){
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    let size = Math.random()*50+10;
    bubble.style.width = size+'px';
    bubble.style.height = size+'px';
    bubble.style.left = Math.random()*100+'vw';
    bubble.style.background = colors[Math.floor(Math.random()*colors.length)];
    bubble.style.animationDuration = (Math.random()*15+15)+'s';
    bubble.style.animationDelay = (Math.random()*5)+'s';
    document.getElementById('bubbles').appendChild(bubble);
}
</script>
""", unsafe_allow_html=True)

# --- Neon Title (No Blink) ---
st.markdown("<h1 class='neon-title'>🧠 AI Quiz Generator</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a PDF/DOCX or type a topic to generate MCQs</p>", unsafe_allow_html=True)


# ------------------ Graph State ------------------
class QuizState(TypedDict, total=False):
    file: Optional[bytes]  # stored file bytes (or None)
    file_name: Optional[str]  # filename of uploaded file
    text: str  # Extracted text content
    quiz_data: List[Dict]  # Generated MCQs
    num_mcqs: int  # Number of MCQs requested
    topic: str  # Topic for quiz generation
    doc_hash: Optional[str]  # Hash of the document content


graph = StateGraph(QuizState)


# ---------- Helper: compute file hash ----------
def compute_hash(b: bytes) -> str:
    """Computes SHA256 hash of file bytes for caching/deduplication."""
    return hashlib.sha256(b).hexdigest()


# ---------- Node Functions (with session_state persistence) ----------
def upload_node(state: QuizState) -> QuizState:
    """Handles UI for file upload and configuration."""
    # Initialize persistent keys in session_state
    if "file_bytes" not in st.session_state: st.session_state.file_bytes = None
    if "file_name" not in st.session_state: st.session_state.file_name = ""
    if "text" not in st.session_state: st.session_state.text = ""
    if "topic" not in st.session_state: st.session_state.topic = ""
    if "num_mcqs" not in st.session_state: st.session_state.num_mcqs = 5
    if "use_text" not in st.session_state: st.session_state.use_text = False
    if "last_doc_hash" not in st.session_state: st.session_state.last_doc_hash = None

    # UI controls
    use_text_new = st.checkbox("Choose Topic Manually", value=st.session_state.use_text)

    # ------------------ CRITICAL FIX ------------------
    # Check if the state changed from File Upload (False) to Manual Text (True)
    if st.session_state.use_text == False and use_text_new == True:
        # If switching to manual text, clear the existing RAG index and hash
        rag.clear_documents()
        st.session_state.last_doc_hash = None
        st.info("RAG index cleared. Generating quiz from manual topic.")

    st.session_state.use_text = use_text_new  # Update session state after check
    # --------------------------------------------------

    if not st.session_state.use_text:
        uploaded_file = st.file_uploader("📂 Upload PDF or DOCX", type=["pdf", "docx"])
        if uploaded_file is not None:
            # Store raw bytes & filename to persist across reruns
            st.session_state.file_bytes = uploaded_file.getvalue()
            st.session_state.file_name = uploaded_file.name

        # Clear manual topic/text if switching to file upload
        if st.session_state.text.strip():
            st.session_state.text = ""
            st.session_state.topic = ""
    else:
        st.session_state.text = st.text_area("Enter topic or text", value=st.session_state.text, height=150)
        # Use the entered text as the topic
        st.session_state.topic = st.session_state.text.strip()

        # Clear file info if switching to manual text
        if st.session_state.file_bytes:
            st.session_state.file_bytes = None
            st.session_state.file_name = ""

    st.session_state.num_mcqs = st.number_input(
        "Number of MCQs", 1, 50, st.session_state.num_mcqs, 1
    )

    # Calculate hash if file is present
    current_doc_hash = compute_hash(st.session_state.file_bytes) if st.session_state.file_bytes else None

    return {
        "file": st.session_state.file_bytes,
        "file_name": st.session_state.file_name,
        "text": st.session_state.text,
        "num_mcqs": int(st.session_state.num_mcqs),
        "topic": st.session_state.topic,
        "doc_hash": current_doc_hash
    }


def extract_text_node(state: QuizState) -> QuizState:
    """
    Step 1: Extract text content from the uploaded file bytes.
    Does NOT modify the RAG index.
    """
    file_bytes = state.get("file")
    file_name = state.get("file_name", "")
    extracted_text = state.get("text", "")  # Start with manual text if available

    # If there's a file, read it
    if file_bytes and file_name:
        bio = BytesIO(file_bytes)
        if file_name.lower().endswith(".pdf"):
            try:
                reader = PdfReader(bio)
                extracted_text = "\n".join([p.extract_text() or "" for p in reader.pages])
            except Exception as e:
                st.error(f"PDF read error: {e}")
                extracted_text = ""
        elif file_name.lower().endswith(".docx"):
            try:
                doc = Document(bio)
                extracted_text = "\n".join([p.text for p in doc.paragraphs])
            except Exception as e:
                st.error(f"DOCX read error: {e}")
                extracted_text = ""

        # If text extraction failed, but we had bytes, something went wrong.
        if not extracted_text.strip():
            st.warning(f"Could not extract meaningful text from {file_name}.")

    # Update session_state for persistence and return for graph flow
    st.session_state.text = extracted_text

    # Auto-derive topic if not manually set and we have extracted text
    topic = state.get("topic", "").strip()
    if not topic and extracted_text.strip():
        topic = " ".join(extracted_text.split()[:7])  # Small fallback topic
        st.session_state.topic = topic

    return {
        "file": file_bytes,
        "file_name": file_name,
        "text": extracted_text,
        "num_mcqs": state.get("num_mcqs", 5),
        "topic": topic,
        "doc_hash": state.get("doc_hash")
    }


def process_document_node(state: QuizState) -> QuizState:
    """
    Step 2: Handles the RAG pipeline steps (Chunking, Embedding, Indexing).
    Only re-processes if a new document is detected via its hash.
    """
    text = state.get("text", "").strip()
    current_doc_hash = state.get("doc_hash")
    last_doc_hash = st.session_state.last_doc_hash

    # Check for content from an uploaded file
    if current_doc_hash and text:
        if last_doc_hash != current_doc_hash:
            st.info("🔄 Document changed. Processing (Chunking, Embedding, Indexing)...")
            try:
                # This call performs Chunking, Embedding, and Indexing
                rag.add_document(text)
                st.session_state.last_doc_hash = current_doc_hash
                st.success("✅ Document processed and RAG index updated.")
            except Exception as e:
                st.error(f"RAG processing error: {e}")

    # Check for content from manual text input (which doesn't set a hash)
    elif not current_doc_hash and text and st.session_state.use_text:
        # When using manual topic, the index is assumed cleared by upload_node
        # We do nothing here, as the quiz generation step relies on general knowledge
        # when the RAG index is empty.
        pass

    elif not text:
        st.warning("No text or file content available for processing.")

    # Return the state unchanged for the next step
    return state


def generate_quiz_node(state: QuizState) -> QuizState:
    """
    Step 3: Handles the RAG pipeline steps (Retrieval and Generation).
    """
    topic = state.get("topic", "").strip()
    num_mcqs = int(state.get("num_mcqs", 5))
    quiz_data = []

    if not topic:
        st.warning("Please provide a topic (choose manually) or upload a file with content.")
        return {"quiz_data": [], "topic": topic, "num_mcqs": num_mcqs}

    try:
        # This call performs Retrieval (if index exists) and LLM Generation
        quiz_data = rag.generate_mcqs(topic, num_mcqs)
    except Exception as e:
        st.error(f"Error generating MCQs: {e}")
        quiz_data = []

    if not quiz_data:
        st.warning("⚠️ No quiz generated. Try changing the topic or upload a different document.")

    # Preserve original text content and other metadata
    return {
        "text": state.get("text", ""),
        "num_mcqs": num_mcqs,
        "quiz_data": quiz_data,
        "topic": topic
    }


def display_quiz_node(state: QuizState) -> QuizState:
    """
    Step 4: Displays the generated quiz and provides a download button.
    """
    quiz_data = state.get("quiz_data", [])
    if not quiz_data:
        st.info("Nothing to display.")
        return {"quiz_data": []}

    st.subheader("✅ Quiz Generated!")
    # Trick to force CSS to render sometimes
    st.markdown("<span style='display:none'>.</span>", unsafe_allow_html=True)

    for i, q in enumerate(quiz_data):
        question_text = q['question'].strip()
        st.markdown(f"""
        <div class='quiz-card' style='animation-delay:{i * 0.2}s'>
            <div class='quiz-question'>Q{i + 1}. {question_text}</div>
            <div class='quiz-option'>A) {q['options']['A']}</div>
            <div class='quiz-option'>B) {q['options']['B']}</div>
            <div class='quiz-option'>C) {q['options']['C']}</div>
            <div class='quiz-option'>D) {q['options']['D']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    # Download as Word
    buf = BytesIO()
    doc = Document()
    doc.add_heading("AI Generated Quiz", 0)
    for i, q in enumerate(quiz_data):
        doc.add_paragraph(f"Q{i + 1}. {q['question']}")
        for k, v in q["options"].items():
            doc.add_paragraph(f"{k}) {v}")
        doc.add_paragraph(f"Answer: {q['answer']}\n")
    doc.save(buf)
    buf.seek(0)
    st.download_button("⬇️ Download as Word", buf, "AI_Quiz.docx")
    return {"quiz_data": quiz_data}


# ---------- Build Graph ----------
graph.add_node("upload_node", upload_node)
graph.add_node("extract_text_node", extract_text_node)
graph.add_node("process_document_node", process_document_node)
graph.add_node("generate_quiz_node", generate_quiz_node)
graph.add_node("display_quiz_node", display_quiz_node)

graph.set_entry_point("upload_node")
graph.add_edge("upload_node", "extract_text_node")
graph.add_edge("extract_text_node", "process_document_node")
graph.add_edge("process_document_node", "generate_quiz_node")
graph.add_edge("generate_quiz_node", "display_quiz_node")
graph.add_edge("display_quiz_node", END)

# ---------- Streamlit trigger ----------
# Initial state with default values
initial_state = {"file": None, "file_name": "", "text": "", "quiz_data": [], "num_mcqs": 5, "topic": "",
                 "doc_hash": None}

# Run the first node for UI rendering and state collection
current_state = upload_node(initial_state)

# Only run the full pipeline when user explicitly clicks the button
if st.button("🚀 Generate Quiz", use_container_width=True):
    with st.spinner("Executing RAG Pipeline: Extracting Text..."):
        # 1. Extract Text
        current_state = extract_text_node(current_state)

    with st.spinner("Executing RAG Pipeline: Chunking, Embedding, & Indexing..."):
        # 2. Process Document (Chunking, Embedding, Indexing)
        current_state = process_document_node(current_state)

    with st.spinner("Executing RAG Pipeline: Retrieval & Generation..."):
        # 3. Generate Quiz (Retrieval and LLM Call)
        current_state = generate_quiz_node(current_state)

    # 4. Display Results
    display_quiz_node(current_state)