import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
from typing import TypedDict, Optional, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import re

# ------------------ Backend / LLM setup ------------------
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
except Exception:
    st.error("Warning: GOOGLE_API_KEY not found. LLM functions will fail.")
    llm = None

st.set_page_config(page_title="🧠 AI Quiz Generator", layout="centered")

# ------------------ CSS + Styling (Animations and Reduced Height retained) ------------------
st.markdown("""
<style>
    /* General Setup & Variables */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    :root {
        --primary-accent: #33CCFF; 
        --secondary-dark: #1A2838; 
        --background-main: #0D1621; 
        --card-bg-light: #162231;
        --text-color: #F0F5FA;
        --muted-text: #9FB8C9;

        /* CORRECTED ANSWER COLOR (Dark Blue/Navy Mix) */
        --success-color: #1F3F60;
        --success-text: #E6F2FF;
        --success-glow: rgba(51, 102, 153, 0.7); 

        --shadow-glow: 0 0 20px rgba(51, 204, 255, 0.4);
    }

    .stApp {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-color);
        background: var(--background-main);
        animation: backgroundPulse 15s infinite alternate;
        min-height: 100vh; 
    }

    /* INITIAL VIEWPORT FIT & HEIGHT REDUCTION (Retained) */
    header { visibility: hidden; height: 0; }
    .block-container {
        padding-top: 0.5rem;  
        padding-bottom: 0.5rem; 
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .header-wrap {
        margin-bottom: 15px;
        padding: 10px 20px 5px 20px;
    }
    .title { font-size: 2.5em; margin-bottom: 2px; }
    .subtitle { font-size: 0.9em; margin-top: 0px; }
    h3 { margin-top: 0px !important; margin-bottom: 10px !important; font-size: 1.2em; }
    .stFileUploader, .stNumberInput, .stRadio {
        padding: 8px 15px; 
        margin-bottom: 10px;
    }
    div[data-testid="stFileUploader"], div[data-testid="stNumberInput"] {
        margin-bottom: 5px;
    }
    /* END: INITIAL VIEWPORT FIT */


    /* Quiz Card Styling (ANIMATED) */
    .quiz-card {
        background-color: var(--card-bg-light);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 35px;
        border-left: 4px solid var(--primary-accent);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);

        /* ANMIMATION PROPERTIES */
        opacity: 0; 
        transition: all 0.4s ease-out;
        animation-name: slideInCard; /* Defined below */
        animation-fill-mode: forwards;
        animation-duration: 0.7s;
    }

    .quiz-card:hover {
        transform: scale(1.00) translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.8), 0 0 15px rgba(51, 204, 255, 0.6);
    }

    .quiz-question {
        font-size: 1.3em;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 20px;
        text-shadow: 0 0 5px rgba(51, 204, 255, 0.3);
    }

    .quiz-option {
        padding: 12px 18px;
        background-color: var(--secondary-dark);
        border-radius: 8px;
        margin-bottom: 8px;
        transition: all 0.3s;
        font-size: 1.0em;
        border: 1px solid transparent;
        animation: floatingPulse 6s infinite ease-in-out; 
    }

    /* Correct Answer Display (Dark Blue/Navy with Animation) */
    .quiz-answer {
        margin-top: 25px;
        padding: 15px;
        /* Dark Blue/Navy Gradient Retained */
        background: linear-gradient(90deg, #1A314A 0%, #1F3F60 100%); 
        color: var(--success-text); 
        font-weight: 700;
        border-radius: 10px;
        border: 1px solid rgba(51, 102, 153, 0.5); 
        box-shadow: 0 4px 15px var(--success-glow); 

        /* ANMIMATION PROPERTIES */
        opacity: 0;
        animation: answerReveal 0.6s cubic-bezier(0.68, -0.55, 0.27, 1.55) forwards; 
        animation-delay: 0.2s;
    }

    /* Keyframe Animations */
    @keyframes backgroundPulse {
        0% { background-color: #0D1621; }
        50% { background-color: #0F1A28; }
        100% { background-color: #0D1621; }
    }

    @keyframes textFlicker {
        0%, 100% { opacity: 1; text-shadow: 0 0 10px var(--primary-accent); }
        50% { opacity: 0.9; text-shadow: none; }
    }

    /* Quiz Card Slide-in Animation */
    @keyframes slideInCard {
        from { opacity: 0; transform: translateY(30px) rotateX(-5deg); }
        to { opacity: 1; transform: translateY(0) rotateX(0); }
    }

    @keyframes floatingPulse {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-3px); }
    }

    /* Answer Reveal Pop Animation */
    @keyframes answerReveal {
        0% { opacity: 0; transform: scale(0.5) translateY(20px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }

</style>
""", unsafe_allow_html=True)

# ---------- Header (REDUCED HEIGHT) ----------
st.markdown('<div class="header-wrap container">', unsafe_allow_html=True)
st.markdown("<div class='title'>🧠 AI Quiz Generator</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle small-muted'>Upload a PDF or DOCX and get professional MCQs with a single click.</div>",
    unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ------------------ LangGraph State Schema (UNCHANGED) ------------------
class QuizState(TypedDict, total=False):
    file: Optional[any]
    text: str
    quiz_data: List[Dict]
    num_mcqs: int


graph = StateGraph(QuizState)


# ---------- Upload file (REDUCED HEIGHT) ----------
def upload_file(state: QuizState) -> QuizState:
    with st.container():
        st.markdown("### 1. Document & Settings", unsafe_allow_html=False)

        file = st.file_uploader("📂 Upload PDF or DOCX Source Document", type=["pdf", "docx"])
        state["file"] = file

        current_num_mcqs = st.session_state.get("mcq_number_input", state.get("num_mcqs", 5))

        st.markdown('<div class="controls">', unsafe_allow_html=True)
        state["num_mcqs"] = st.number_input(
            "Enter number of MCQs to generate:",
            min_value=1,
            max_value=50,
            value=current_num_mcqs,
            step=1,
            key="mcq_number_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if state["file"] is None:
        st.info("Please upload a file to generate quiz.")
        return {}

    return state


# ---------- Extract text (UNCHANGED) ----------
def extract_text(state: QuizState) -> QuizState:
    file = state.get("file")
    if file is None:
        return {}

    text = ""
    file.seek(0)

    if file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += (page.extract_text() or "") + " "
    elif file.name.endswith(".docx"):
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    if not text.strip():
        st.warning("No text could be extracted from this file.")
        return {}

    return {"text": text, "num_mcqs": state.get("num_mcqs", 5)}


# ---------- Generate quiz (UNCHANGED) ----------
def generate_quiz(state: QuizState) -> QuizState:
    text = state.get("text", "")
    num_mcqs = state.get("num_mcqs")
    if not text or num_mcqs is None or llm is None:
        return {}

    prompt = f"""
You are a professional quiz creator.
Generate {num_mcqs} multiple-choice questions (MCQs) from the following text.
Each question should have:
- A question line starting with Q1., Q2., etc.
- Four options labeled A, B, C, D.
- The correct answer letter at the end.

Format exactly like this, ensuring all components are present and followed by a newline:
Q1. <question>
A) <option>
B) <option>
C) <option>
D) <option>
Answer: <A/B/C/D>

Text:
---
{text[:8000]}
---
"""
    with st.spinner("🚀 Generating professional quiz... This may take a moment..."):
        result = llm.invoke(prompt)
        quiz_text = getattr(result, "content", None) or getattr(result, "output_text", "")

    if not quiz_text.strip():
        st.error("Failed to generate quiz. The model did not return valid content.")
        return {}

    questions = []
    pattern = r"Q\d+\.\s*(.*?)\s*A\)\s*(.*?)\s*B\)\s*(.*?)\s*C\)\s*(.*?)\s*D\)\s*(.*?)\s*Answer:\s*([ABCD])"
    matches = re.findall(pattern, quiz_text, re.DOTALL | re.IGNORECASE)

    for q in matches:
        question_text, A, B, C, D, ans = q
        questions.append({
            "question": question_text.strip(),
            "options": {"A": A.strip(), "B": B.strip(), "C": C.strip(), "D": D.strip()},
            "answer": ans.strip().upper()
        })

    if not questions:
        st.error("⚠️ Failed to parse any valid MCQs from the LLM output. Try adjusting the input text.")
        return {}

    return {"quiz_data": questions}


# ---------- Display quiz (ANIMATED) ----------
def display_quiz(state: QuizState) -> QuizState:
    quiz_data = state.get("quiz_data", [])
    if not quiz_data:
        return {}

    st.subheader("Quiz Generated!")
    st.markdown('<div class="container">', unsafe_allow_html=True)


    for i, q in enumerate(quiz_data):
        # Apply stagger delay for the slide-in animation
        card_style = f"animation-delay: {i * 0.2}s;"

        st.markdown(f"""
        <div class='quiz-card' style="{card_style}">
            <div class='quiz-question'>Q{i + 1}. {q['question']}</div>
            <div class='quiz-options'>
                <div class='quiz-option'>A) {q['options']['A']}</div>
                <div class='quiz-option'>B) {q['options']['B']}</div>
                <div class='quiz-option'>C) {q['options']['C']}</div>
                <div class='quiz-option'>D) {q['options']['D']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    # Download Area (UNCHANGED)
    st.markdown('<div class="download-area">', unsafe_allow_html=True)
    st.markdown("### Export Your Quiz", unsafe_allow_html=False)

    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        file_format = st.radio(
            "Select Format:",
            ["Text", "Word (.docx)"],
            key="download_format",
            horizontal=True
        )

    with col_d2:
        if file_format == "Text":
            quiz_text = "\n\n".join([
                f"Q{i + 1}. {q['question']}\n" +
                "\n".join([f"{k}) {v}" for k, v in q['options'].items()]) +
                f"\nAnswer: {q['answer']}"
                for i, q in enumerate(quiz_data)
            ])
            st.download_button(
                "⬇️ Download as Text File",
                quiz_text,
                "AI_Generated_Quiz.txt",
                "text/plain"
            )
        else:
            doc = Document()
            doc.add_heading("AI Generated Quiz", 0)
            for i, q in enumerate(quiz_data):
                doc.add_paragraph(f"Q{i + 1}. {q['question']}")
                for k, v in q["options"].items():
                    doc.add_paragraph(f"{k}) {v}")
                doc.add_paragraph(f"Answer: {q['answer']}\n")
            buf = BytesIO()
            doc.save(buf)
            buf.seek(0)
            st.download_button(
                "⬇️ Download as Word Document",
                buf.getvalue(),
                "AI_Generated_Quiz.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return {}


# ------------------ Graph wiring (UNCHANGED) ------------------
graph.add_node("upload_file", upload_file)
graph.add_node("extract_text", extract_text)
graph.add_node("generate_quiz", generate_quiz)
graph.add_node("display_quiz", display_quiz)
graph.set_entry_point("upload_file")
graph.add_edge("upload_file", "extract_text")
graph.add_edge("extract_text", "generate_quiz")
graph.add_edge("generate_quiz", "display_quiz")
graph.add_edge("display_quiz", END)

app = graph.compile()
initial_state: QuizState = {}
app.invoke(initial_state)