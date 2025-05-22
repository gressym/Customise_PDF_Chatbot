import streamlit as st
import fitz  # PyMuPDF
import docx
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import re

# ==== CONFIG ====
GROQ_API_KEY = "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"
MODEL_NAME = "llama3-70b-8192"

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)

# ==== PDF Extraction ====
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ==== DOCX Extraction ====
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# ==== Split text into chunks ~1000 words ====
def chunk_text(text, max_chunk_size=1000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = " ".join(words[i:i+max_chunk_size])
        chunks.append(chunk)
    return chunks

# ==== Simple keyword overlap search ====
def preprocess_text(text):
    # lowercase, remove punctuation, split into words
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return set(text.split())

def get_most_relevant_chunk(chunks, question):
    question_words = preprocess_text(question)
    max_overlap = 0
    best_chunk = chunks[0]

    for chunk in chunks:
        chunk_words = preprocess_text(chunk)
        overlap = len(question_words.intersection(chunk_words))
        if overlap > max_overlap:
            max_overlap = overlap
            best_chunk = chunk

    return best_chunk

# ==== LangGraph QA Node ====
def qa_node(state: dict) -> dict:
    question = state["question"]
    context = state["context"]

    prompt = f"""Use the following context to answer the question:
Context: {context}

Question: {question}
Answer:"""

    response = llm([HumanMessage(content=prompt)])
    state["answer"] = response.content
    return state

# ==== Build LangGraph ====
graph = StateGraph(dict)
graph.add_node("qa", qa_node)
graph.set_entry_point("qa")
graph.set_finish_point("qa")
chatbot = graph.compile()

# ==== Streamlit UI ====
st.set_page_config(page_title="Doc/PDF QA Chatbot (no embeddings)", layout="wide")
st.title("📄 Document Chatbot (PDF + Word) using Groq + LangGraph")

uploaded_file = st.file_uploader("Upload a PDF or Word Document", type=["pdf", "docx"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    with st.spinner(f"Extracting text from .{file_type}..."):
        if file_type == "pdf":
            doc_text = extract_text_from_pdf(uploaded_file)
        elif file_type == "docx":
            doc_text = extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

    st.success("✅ Document loaded! Ask your question:")
    question = st.text_input("Your question")

    if question:
        chunks = chunk_text(doc_text)
        relevant_chunk = get_most_relevant_chunk(chunks, question)

        state = {"question": question, "context": relevant_chunk}
        with st.spinner("Thinking..."):
            result = chatbot.invoke(state)
        st.markdown("### 🧠 Answer:")
        st.write(result["answer"])
