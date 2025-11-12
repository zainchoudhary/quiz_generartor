import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
import re
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------     LLM Setup ------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ------------------ Page Config ------------------
st.set_page_config(page_title="🧠 AI Quiz Generator", layout="centered")

# ------------------ Initialize Session State ------------------
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
if "text" not in st.session_state:
    st.session_state.text = ""
if "num_mcqs" not in st.session_state:
    st.session_state.num_mcqs = 5
if "file" not in st.session_state:
    st.session_state.file = None

# ------------------ CSS & Bubbles ------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

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

h1 { text-align: center; font-size: 2.8em; font-weight: 700; color: var(--primary-accent); margin-bottom:5px;}
p { text-align: center; color: var(--muted-text); font-size:1.1em;}

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

.quiz-question { font-size:1.35em; font-weight:600; margin-bottom:15px; color: var(--primary-accent);}
.quiz-option {
    padding: 12px 20px; background-color: var(--secondary-dark);
    border-radius:12px; margin-bottom:10px; cursor:pointer;
    transition: background 0.3s ease, transform 0.2s ease, color 0.3s ease;
    color: var(--text-color);
}
.quiz-option:hover { background-color:#1F3B5A; color:#33CCFF; transform:translateX(5px);}

/* Button Styles */
.stDownloadButton button, .stButton button {
    background: linear-gradient(135deg,#0B1A2B,#162231) !important;
    color: #33CCFF !important;
    font-weight:700; font-size:1.05em; padding:12px 28px !important;
    border-radius:20px !important; border:2px solid #33CCFF !important;
    box-shadow:0 5px 20px rgba(0,0,0,0.5); transition: all 0.3s ease;
}
.stDownloadButton button:hover, .stButton button:hover {
    box-shadow:0 10px 30px rgba(0,0,0,0.7);
    background: linear-gradient(135deg,#162231,#0B1A2B);
    color: #00FFE0 !important; transform: scale(1.05);
}
</style>

<div id="bubbles"></div>
<script>
const colors = ['#33CCFF','#00FFE0','#FF33CC','#FFDD33','#33FFAA'];
for(let i=0;i<30;i++){
    const bubble = document.createElement('div');
    bubble.className='bubble';
    let size=Math.random()*50+10;
    bubble.style.width=size+'px';
    bubble.style.height=size+'px';
    bubble.style.left=Math.random()*100+'vw';
    bubble.style.background=colors[Math.floor(Math.random()*colors.length)];
    bubble.style.animationDuration=(Math.random()*15+15)+'s';
    bubble.style.animationDelay=(Math.random()*5)+'s';
    document.getElementById('bubbles').appendChild(bubble);
}
</script>
""", unsafe_allow_html=True)

st.markdown("<h1>🧠 AI Quiz Generator</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a PDF/DOCX or type a topic to generate MCQs</p>", unsafe_allow_html=True)

# ------------------ Input Section ------------------
use_text = st.checkbox("Or type a topic manually")
if use_text:
    st.session_state.text = st.text_area("Enter topic or text", height=150, value=st.session_state.text)
else:
    st.session_state.file = st.file_uploader("📂 Upload PDF or DOCX", type=["pdf", "docx"])

st.session_state.num_mcqs = st.number_input("Number of MCQs", 1, 50, st.session_state.num_mcqs, 1)


# ------------------ Extract Text Function ------------------
def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    elif file.name.lower().endswith(".docx"):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""


# ------------------ Generate Quiz Function ------------------
def generate_quiz():
    text = st.session_state.text.strip()
    num_mcqs = st.session_state.num_mcqs
    if not text or not llm:
        st.warning("No text or LLM available")
        return

    prompt = f"""
        Generate {num_mcqs} multiple-choice questions (MCQs) from the following text.
        Format each question clearly with options and mark the correct answer at the end.
        don't give extra information just start from the first question.
        also dont used such type of words like "provided text" and "given text" anf " as in the text?" and "According to the text" or "as mentioned in the text"
        etc , means it should not feel that yuo generate mcqs from provided text, response automatically by guessing which is asked by the user and used that topic name instead of say provided text or any other words like it

        Example format:
        Q1. What is AI?
        A) Option 1
        B) Option 2
        C) Option 3
        D) Option 4
        Answer: B


    Text:
    {text[:4000]}
    """
    try:
        result = llm.invoke(prompt)
        output_text = getattr(result, "content", None) or getattr(result, "output_text", "")
    except Exception:
        output_text = ""

    # Parse quiz
    output_text = re.sub(r'(?i)question\s*\d*[:.]', "Q", output_text)
    output_text = output_text.replace("Option ", "").replace("Answer:", "Answer:")

    pattern = r"""Q\d*[\.\)]?\s*([\s\S]*?)      # Question
                  \s*A[\)\.:]\s*([\s\S]*?)      # Option A
                  \s*B[\)\.:]\s*([\s\S]*?)      # Option B
                  \s*C[\)\.:]\s*([\s\S]*?)      # Option C
                  \s*D[\)\.:]\s*([\s\S]*?)      # Option D
                  \s*Answer[:\s]*([ABCD])       # Correct Answer
               """
    matches = re.findall(pattern, output_text, re.IGNORECASE | re.VERBOSE)
    quiz_data = []
    for q in matches:
        question_text, A, B, C, D, ans = q
        clean = lambda s: re.sub(r'\s+', ' ', s)
        quiz_data.append({
            "question": clean(question_text),
            "options": {"A": clean(A), "B": clean(B), "C": clean(C), "D": clean(D)},
            "answer": ans.strip().upper()
        })
    st.session_state.quiz_data = quiz_data


# ------------------ Display Quiz Function ------------------
def display_quiz():
    quiz_data = st.session_state.quiz_data
    if not quiz_data:
        st.warning("No quiz generated yet.")
        return
    st.subheader("✅ Quiz Generated!")
    for i, q in enumerate(quiz_data):
        st.markdown(f"<div class='quiz-card'><div class='quiz-question'>Q{i + 1}. {q['question']}</div>"
                    f"<div class='quiz-option'>A) {q['options']['A']}</div>"
                    f"<div class='quiz-option'>B) {q['options']['B']}</div>"
                    f"<div class='quiz-option'>C) {q['options']['C']}</div>"
                    f"<div class='quiz-option'>D) {q['options']['D']}</div></div>", unsafe_allow_html=True)
        with st.expander("💡 Show Answer", expanded=False):
            st.markdown(f"✅ **Answer: {q['answer']}**")
    st.markdown("---")
    # Download button
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
    st.download_button("⬇️ Download as Word", buf, "AI_Quiz.docx")


# ------------------ Generate Button ------------------
if st.button("🚀 Generate Quiz"):
    with st.spinner("Generating quiz... please wait"):
        if st.session_state.file and not use_text:
            st.session_state.text = extract_text(st.session_state.file)
        generate_quiz()

# Always display after generation
if st.session_state.quiz_data:
    display_quiz()
