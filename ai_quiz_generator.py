import streamlit as st
import os
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
import re
import hashlib
from typing import TypedDict, Optional, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from rag_pipeline import run_rag_pipeline
from database import init_db, login_user, register_user, save_quiz, get_quiz_history

# ------------------- Session State Initialization -------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
if "use_text" not in st.session_state:
    st.session_state.use_text = False
if "current_step" not in st.session_state:
    st.session_state.current_step = "upload_node"  # default starting step

# ------------------- LLM Setup -------------------
google_api_key = os.environ.get("GOOGLE_API_KEY")
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key
    )
except Exception:
    st.error("⚠️ GOOGLE_API_KEY not found. LLM features disabled.")
    llm = None

# ------------------- Streamlit Page -------------------
st.set_page_config(page_title="🧠 AI Quiz Generator", layout="wide")
init_db()

# ------------------- Sidebar: Authentication -------------------
st.sidebar.header("Authentication")
if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user['username']}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.user = None
        st.session_state.quiz_data = []
        st.session_state.current_step = "upload_node"
        st.rerun()
else:
    auth_mode = st.sidebar.radio("Choose action", ["Login", "Register"])
    if auth_mode == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.user = user
                st.session_state.quiz_data = get_quiz_history(user["id"]) or []
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password.")
    else:  # Register
        new_username = st.sidebar.text_input("Choose username")
        new_password = st.sidebar.text_input("Choose password", type="password")
        if st.sidebar.button("Register"):
            success = register_user(new_username, new_password)
            if success:
                st.sidebar.success("Account created! You can now log in.")
            else:
                st.sidebar.error("Username already exists.")

# ------------------- Sidebar: Quiz History -------------------
if st.session_state.user:
    with st.sidebar.expander("📜 Quiz History", expanded=False):
        history = get_quiz_history(st.session_state.user["id"])
        if not history:
            st.info("No quizzes yet.")
        else:
            for idx, entry in enumerate(history, 1):
                st.markdown(f"**Quiz {idx}** - {entry['created_at']}")
                for q_num, q in enumerate(entry["quiz_data"], 1):
                    st.markdown(f"Q{q_num}. {q['question']}")
                    st.markdown(f"A) {q['options']['A']}")
                    st.markdown(f"B) {q['options']['B']}")
                    st.markdown(f"C) {q['options']['C']}")
                    st.markdown(f"D) {q['options']['D']}")
                    st.markdown(f"**Answer:** {q['answer']}")
                st.markdown("---")

# ------------------- Main App -------------------
if st.session_state.user:
    st.markdown("<h1>🧠 AI Quiz Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p>Upload a PDF/DOCX or type a topic to generate MCQs (Documents are saved per user)</p>", unsafe_allow_html=True)

    class QuizState(TypedDict, total=False):
        file: Optional[any]
        raw_text: str
        context_text: str
        manual_topic: str
        file_hash: str
        quiz_data: List[Dict]
        num_mcqs: int

    graph = StateGraph(QuizState)

    # ------------------- Upload Node -------------------
    def upload_node(state: QuizState) -> QuizState:
        def toggle_use_text():
            st.session_state.use_text = not st.session_state.use_text

        use_text = st.checkbox("Or type a topic manually", value=st.session_state.use_text, on_change=toggle_use_text)
        file = None
        text_input = ""

        if not st.session_state.use_text:
            file = st.file_uploader("📂 Upload PDF or DOCX", type=["pdf", "docx"])
        else:
            text_input = st.text_area("Enter topic or text", height=150, key="manual_text_input")

        num_mcqs = st.number_input("Number of MCQs", 1, 50, state.get("num_mcqs", 5), 1)
        return {"file": file, "manual_topic": text_input, "num_mcqs": int(num_mcqs)}

    # ------------------- Extract Text Node -------------------
    def extract_text_node(state: QuizState) -> QuizState:
        file = state.get("file")
        manual_topic = state.get("manual_topic", "")
        raw_text = ""
        file_hash = ""

        if file:
            file_bytes = file.read()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            file.seek(0)
            if file.name.lower().endswith(".pdf"):
                reader = PdfReader(file)
                raw_text = "\n".join([p.extract_text() or "" for p in reader.pages])
            elif file.name.lower().endswith(".docx"):
                doc = Document(BytesIO(file_bytes))
                raw_text = "\n".join([p.text for p in doc.paragraphs])
        elif manual_topic:
            raw_text = manual_topic
            file_hash = hashlib.sha256(manual_topic.encode('utf-8')).hexdigest()

        if not raw_text:
            st.error("Please upload a file or enter a topic.")

        return {
            "file": file,
            "raw_text": raw_text,
            "manual_topic": manual_topic,
            "file_hash": file_hash,
            "num_mcqs": state.get("num_mcqs", 5)
        }

    # ------------------- Generate Quiz Node -------------------
    def generate_quiz_node(state: QuizState) -> QuizState:
        import time

        raw_text = state.get("raw_text", "").strip()
        manual_topic = state.get("manual_topic", "").strip()
        file_hash = state.get("file_hash", "manual_topic_no_hash")
        num_mcqs = state.get("num_mcqs", 5)

        if not raw_text or not llm:
            st.warning("LLM not initialized or empty input. Skipping quiz generation.")
            return {"raw_text": raw_text, "context_text": "", "num_mcqs": num_mcqs, "quiz_data": []}

        query = manual_topic if manual_topic else raw_text[:100]

        # ---------------- RAG Pipeline ----------------
        context_text = run_rag_pipeline(raw_text, query, file_hash)
        if not context_text:
            st.warning("RAG pipeline returned empty context. Using first 1000 chars as fallback.")
            context_text = raw_text[:1000]

        # ---------------- LLM Quiz Generation ----------------
        prompt = f"""
            Generate {num_mcqs} multiple-choice questions (MCQs) from the following text.
            generate mcqs in that language which user tells you by default generate quiz in English language
            Format each question clearly with options and mark the correct answer at the end.

            CRITICAL INSTRUCTIONS:
            1. Do NOT use phrases like "provided text," "given text," "as in the text," or "According to the text."
            2. Instead, refer to the actual topic or subject matter.
            3. Start directly with the first question, no extra introductory text.

            Example format:
            Q1. What is AI?
            A) Option 1
            B) Option 2
            C) Option 3
            D) Option 4
            Answer: B

        Text:
        {context_text}
        """

        try:
            start_time = time.time()
            result = llm.invoke(prompt)
            output_text = getattr(result, "content", None) or getattr(result, "output_text", "")
        except Exception as e:
            st.error(f"LLM Error: {e}")
            return {"raw_text": raw_text, "context_text": context_text, "num_mcqs": num_mcqs, "quiz_data": []}

        output_text = re.sub(r'(?i)question\s*\d*[:.]', lambda m: f"Q", output_text)
        output_text = output_text.replace("Option ", "").replace("Answer:", "Answer:")

        pattern = r"""Q\d*[\.\)]?\s*([\s\S]*?)
                      \s*A[\)\.:]\s*([\s\S]*?)
                      \s*B[\)\.:]\s*([\s\S]*?)
                      \s*C[\)\.:]\s*([\s\S]*?)
                      \s*D[\)\.:]\s*([\s\S]*?)
                      \s*Answer[:\s]*([ABCD])
                    """
        matches = re.findall(pattern, output_text, re.IGNORECASE | re.VERBOSE)

        quiz_data = []
        for q in matches:
            question_text, A, B, C, D, ans = q
            clean = lambda s: re.sub(r'\s+', ' ', s.strip())
            quiz_data.append({
                "question": clean(question_text),
                "options": {"A": clean(A), "B": clean(B), "C": clean(C), "D": clean(D)},
                "answer": ans.strip().upper()
            })

        return {"raw_text": raw_text, "context_text": context_text, "num_mcqs": num_mcqs, "quiz_data": quiz_data}

    # ------------------- Display Quiz Node -------------------
    def display_quiz_node(state: QuizState) -> QuizState:
        quiz_data = state.get("quiz_data", [])
        if not quiz_data:
            return {"quiz_data": []}

        st.subheader("✅ Quiz Generated!")
        for i, q in enumerate(quiz_data):
            st.markdown(f"""
            <div style='border-left:4px solid #33CCFF;padding:10px;margin-bottom:10px;'>
                <b>Q{i+1}. {q['question']}</b><br>
                A) {q['options']['A']}<br>
                B) {q['options']['B']}<br>
                C) {q['options']['C']}<br>
                D) {q['options']['D']}<br>
                Answer: {q['answer']}
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.user:
            save_quiz(st.session_state.user["id"], quiz_data)

            # ------------------- Download Button -------------------
            from docx import Document as DocxDocument
            import base64

            doc = DocxDocument()
            for i, q in enumerate(quiz_data):
                doc.add_paragraph(f"Q{i+1}. {q['question']}")
                doc.add_paragraph(f"A) {q['options']['A']}")
                doc.add_paragraph(f"B) {q['options']['B']}")
                doc.add_paragraph(f"C) {q['options']['C']}")
                doc.add_paragraph(f"D) {q['options']['D']}")
                doc.add_paragraph(f"Answer: {q['answer']}")
                doc.add_paragraph("\n")

            bio = BytesIO()
            doc.save(bio)
            bio.seek(0)
            b64 = base64.b64encode(bio.read()).decode()

            download_button_html = f"""
            <a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" 
            download="quiz.docx" 
            style="
                display:inline-block;
                background: rgba(0,0,128,0.7);
                color:white;
                padding:10px 20px;
                border-radius:12px;
                text-decoration:none;
                font-weight:bold;
                box-shadow:0 4px 6px rgba(0,0,0,0.2);
                backdrop-filter: blur(6px);
                margin-top:10px;
            ">
            💾 Download Quiz
            </a>
            """
            st.markdown(download_button_html, unsafe_allow_html=True)

        return {"quiz_data": quiz_data}

    # ------------------- Graph Setup -------------------
    graph.add_node("upload_node", upload_node)
    graph.add_node("extract_text_node", extract_text_node)
    graph.add_node("generate_quiz_node", generate_quiz_node)
    graph.add_node("display_quiz_node", display_quiz_node)

    graph.set_entry_point("upload_node")
    graph.add_edge("upload_node", "extract_text_node")
    graph.add_edge("extract_text_node", "generate_quiz_node")
    graph.add_edge("generate_quiz_node", "display_quiz_node")
    graph.add_edge("display_quiz_node", END)

    # ------------------- Run App with persistent state -------------------
    if "state" not in st.session_state:
        st.session_state.state = {
            "file": None,
            "raw_text": "",
            "context_text": "",
            "manual_topic": "",
            "file_hash": "",
            "quiz_data": [],
            "num_mcqs": 5
        }

    state = st.session_state.state

    # Resume node based on current_step
    current_node = st.session_state.current_step

    if current_node == "upload_node":
        updated = upload_node(state)
    elif current_node == "extract_text_node":
        updated = extract_text_node(state)
    elif current_node == "generate_quiz_node":
        updated = generate_quiz_node(state)
    elif current_node == "display_quiz_node":
        updated = display_quiz_node(state)
    else:
        updated = upload_node(state)

    st.session_state.state = updated  # Persist the updated state

    if st.button("🚀 Generate Quiz"):
        with st.spinner("Generating quiz... please wait"):
            if not st.session_state.use_text and not updated.get("file"):
                st.error("Please upload a file or enter a topic.")
            elif st.session_state.use_text and not updated.get("manual_topic").strip():
                st.error("Please enter a topic.")
            else:
                updated = extract_text_node(updated)
                st.session_state.state = updated
                st.session_state.current_step = "extract_text_node"

                if updated.get("raw_text"):
                    updated = generate_quiz_node(updated)
                    st.session_state.state = updated
                    st.session_state.current_step = "generate_quiz_node"

                    display_quiz_node(updated)
                    st.session_state.state = updated
                    st.session_state.current_step = "display_quiz_node"


else:
    st.warning("Please login or register to access the AI Quiz Generator.")
