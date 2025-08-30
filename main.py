# studymate_streamlit.py
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- Global Variables -----------------
doc_chunks = []
vectorizer = None
tfidf_matrix = None
full_text = ""

# ----------------- Functions -----------------
def extract_text_from_pdf(pdf_file):
    """Extracts text from uploaded PDF file"""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def create_vector_store(text):
    """Split text into small chunks and create TF-IDF vectors"""
    global doc_chunks, vectorizer, tfidf_matrix
    chunk_size = 100  # smaller chunks = more precise matching
    doc_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    vectorizer = TfidfVectorizer().fit(doc_chunks)
    tfidf_matrix = vectorizer.transform(doc_chunks)

def process_pdf(pdf_file):
    """Process PDF and prepare for Q&A"""
    global full_text
    full_text = extract_text_from_pdf(pdf_file)
    if not full_text:
        return None, "⚠️ PDF has no extractable text."
    create_vector_store(full_text)
    summary = full_text[:500] + "..." if len(full_text) > 500 else full_text
    return full_text, summary

def answer_question(question):
    """Answer question using TF-IDF similarity"""
    global tfidf_matrix, vectorizer, doc_chunks, full_text
    if tfidf_matrix is None:
        return "⚠️ Please upload and process a PDF first."

    # Special handling for "title" question
    if "title" in question.lower():
        first_lines = full_text.splitlines()
        for line in first_lines:
            if line.strip():
                return line.strip()

    # TF-IDF similarity for general questions
    q_vec = vectorizer.transform([question])
    sim_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = sim_scores.argsort()[-3:][::-1]  # top 3 chunks
    context = " ".join([doc_chunks[i] for i in top_idx])
    return context

# ----------------- Streamlit UI -----------------
st.set_page_config(
    page_title="StudyMate 📘",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Dark blue background
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0a2e4d;
}
h1, h2, h3, h4, h5, h6, p, span, label {
    color: white;
}
.stButton>button {
    background-color: #0a2e4d;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("📘 StudyMate - PDF Q&A")
st.write("Ask anything about your PDF. The assistant searches the document semantically to answer.")

# Sidebar for instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a PDF.  
2. Wait for the summary to appear.  
3. Ask any question related to the PDF.  
4. For title, it will return the first meaningful line.
""")

# ----------------- PDF Upload -----------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file:
    text, summary = process_pdf(uploaded_file)
    if summary:
        st.success("✅ PDF processed successfully!")
        st.subheader("Summary of PDF:")
        st.write(summary)

# ----------------- Q&A -----------------
question = st.text_input("Ask a Question:")
if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        answer = answer_question(question)
        st.subheader("Answer:")
        st.write(answer)
