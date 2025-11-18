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

google_api_key = os.environ.get("GOOGLE_API_KEY")

try:
    llm = ChatGoogleGenerativeAI(
      model="gemini-2.5-flash",
     google_api_key=google_api_key
    )
except Exception:
    st.error("⚠️ GOOGLE_API_KEY not found. LLM features disabled.")
    llm = None

st.set_page_config(page_title="🧠 AI Quiz Generator", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800;900&display=swap');

:root {
    --primary-accent: #33CCFF;
    --secondary-dark: #0B1A2B;
    --background-main: #162231;
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

/* H1 STYLE */
h1 {
    text-align: center;
    font-size: 3.8em; 
    font-weight: 900;
    background: linear-gradient(90deg, #33CCFF, #00FFE0); 
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    color: #33CCFF;
    margin-bottom: 5px;
    text-shadow: 
        0 0 10px rgba(51, 204, 255, 0.9),
        0 0 25px rgba(0, 255, 224, 0.7),
        0 0 50px rgba(51, 204, 255, 0.5);
    transition: all 0.4s ease-in-out; 
}
h1:hover {
    text-shadow: 
        0 0 15px rgba(51, 204, 255, 1),
        0 0 40px rgba(0, 255, 224, 1),
        0 0 70px rgba(51, 204, 255, 0.8);
    transform: scale(1.03); 
}

p { 
    text-align: center; 
    color: var(--muted-text);
    font-size: 1.1em;
}

/* BUBBLE ANIMATION */
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

/* QUIZ CARD */
.quiz-card {
    background-color: var(--card-bg-light);
    padding: 25px 30px;
    border-radius: 20px;
    margin: 25px auto;
    border-left: 6px solid var(--primary-accent);
    box-shadow: 
        0 0 15px rgba(0, 0, 0, 0.8), 
        0 0 25px rgba(51, 204, 255, 0.2), 
        0 0 40px rgba(0, 255, 224, 0.1);
    transition: transform 0.3s ease, box-shadow 0.4s ease-in-out;
}
.quiz-card:hover {
    transform: translateY(-8px); 
    box-shadow: 
        0 0 20px rgba(0, 0, 0, 0.9), 
        0 0 40px rgba(51, 204, 255, 0.4), 
        0 0 60px rgba(0, 255, 224, 0.3);
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

/* BUTTONS */
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


/* ================================= */
/*       RESPONSIVE MOBILE FIX       */
/* ================================= */
@media (max-width: 768px) {

    h1 {
        font-size: 2.2em !important;
        text-shadow: 
            0 0 6px rgba(51, 204, 255, 0.8),
            0 0 20px rgba(0, 255, 224, 0.6),
            0 0 35px rgba(51, 204, 255, 0.4);
    }

    p {
        font-size: 0.9em !important;
    }

    .quiz-card {
        width: 95% !important;
        padding: 15px !important;
        margin: 15px auto !important;
    }

    .quiz-question {
        font-size: 1.1em !important;
    }

    .quiz-option {
        font-size: 0.9em !important;
        padding: 10px 14px !important;
    }

    .stButton button, 
    .stDownloadButton button {
        width: 100% !important;
        font-size: 1em !important;
    }

    .bubble {
        opacity: 0.15 !important;
        animation-duration: 18s !important;
    }
}

@media (max-width: 480px) {
    h1 { font-size: 1.8em !important; }
    .quiz-card { padding: 12px !important; }
    .quiz-question { font-size: 1em !important; }
}
</style>

<div id="bubbles"></div>

<script>
const colors = ['#33CCFF','#00FFE0','#FF33CC','#FFDD33','#33FFAA'];
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

st.markdown("<h1>🧠 AI Quiz Generator</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a PDF/DOCX or type a topic to generate MCQs (Documents are saved locally for history)</p>",
            unsafe_allow_html=True)

# ------------------------ YOUR ORIGINAL LOGIC BELOW -----------------------------

class QuizState(TypedDict, total=False):
    file: Optional[any]
    raw_text: str
    context_text: str
    manual_topic: str
    file_hash: str
    quiz_data: List[Dict]
    num_mcqs: int

graph = StateGraph(QuizState)

def upload_node(state: QuizState) -> QuizState:
    if 'use_text' not in st.session_state:
        st.session_state.use_text = False

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


def generate_quiz_node(state: QuizState) -> QuizState:
    raw_text = state.get("raw_text", "").strip()
    manual_topic = state.get("manual_topic", "").strip()
    file_hash = state.get("file_hash", "manual_topic_no_hash")
    num_mcqs = state.get("num_mcqs", 5)

    if not raw_text or not llm:
        return {"raw_text": raw_text, "context_text": "", "num_mcqs": num_mcqs, "quiz_data": []}

    query = manual_topic if manual_topic else raw_text[:100]

    context_text = run_rag_pipeline(raw_text, query, file_hash)

    if not context_text:
        st.error("RAG pipeline failed.")
        return {"raw_text": raw_text, "context_text": "", "num_mcqs": num_mcqs, "quiz_data": []}

    prompt = f""" Generate {num_mcqs} multiple-choice questions (MCQs) from the following text. 
    generate mcqs in that language which user tells you by default generate quiz in English language 
    Format each question clearly with options and mark the correct answer at the end. 
    CRITICAL INSTRUCTIONS:
     1. Do NOT use phrases like "provided text," "given text," "as in the text," or "According to the text." 
     2. Instead, refer to the actual topic or subject matter. 
     3. Start directly with the first question, no extra introductory text. 
     Example format: Q1. What is AI? 
     A) Option 1 
     B) Option 2 
     C) Option 3 
     D) Option 4 
     Answer: B

    Text:
    {context_text}
    """

    try:
        result = llm.invoke(prompt)
        output_text = getattr(result, "content", None) or getattr(result, "output_text", "")
    except Exception:
        output_text = ""

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
            "options": {
                "A": clean(A),
                "B": clean(B),
                "C": clean(C),
                "D": clean(D),
            },
            "answer": ans.strip().upper()
        })

    return {"raw_text": raw_text, "context_text": context_text, "num_mcqs": num_mcqs, "quiz_data": quiz_data}


def display_quiz_node(state: QuizState) -> QuizState:
    quiz_data = state.get("quiz_data", [])
    if not quiz_data:
        return {"quiz_data": []}

    st.subheader("✅ Quiz Generated!")

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

    fmt = st.radio("Download format:", ["Word (.docx)"], horizontal=True)

    if fmt == "Word (.docx)":
        doc = Document()
        doc.add_heading("AI Generated Quiz", 0)
        doc.add_paragraph("--- ANSWER KEY ---")

        for i, q in enumerate(quiz_data):
            doc.add_paragraph(f"Q{i + 1}. {q['question']}")
            for k, v in q["options"].items():
                doc.add_paragraph(f"{k}) {v}")
            doc.add_paragraph(f"Correct Answer: {q['answer']}\n")

        buf = BytesIO()
        doc.save(buf)
        buf.seek(0)
        st.download_button("⬇️ Download as Word", buf, "AI_Quiz.docx")

    return {"quiz_data": quiz_data}


graph.add_node("upload_node", upload_node)
graph.add_node("extract_text_node", extract_text_node)
graph.add_node("generate_quiz_node", generate_quiz_node)
graph.add_node("display_quiz_node", display_quiz_node)

graph.set_entry_point("upload_node")
graph.add_edge("upload_node", "extract_text_node")
graph.add_edge("extract_text_node", "generate_quiz_node")
graph.add_edge("generate_quiz_node", "display_quiz_node")
graph.add_edge("display_quiz_node", END)

app = graph.compile()

state = {"file": None, "raw_text": "", "context_text": "", "manual_topic": "", "file_hash": "", "quiz_data": [],
         "num_mcqs": 5}
updated = upload_node(state)

if st.button("🚀 Generate Quiz"):
    if not st.session_state.use_text and not updated.get("file"):
        st.error("Please upload a file or check the box to manually enter text.")
    elif st.session_state.use_text and not updated.get("manual_topic").strip():
        st.error("Please enter a topic.")
    else:
        with st.spinner("Generating quiz..."):
            updated = extract_text_node(updated)
            if updated.get("raw_text"):
                updated = generate_quiz_node(updated)
                display_quiz_node(updated)
