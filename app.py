import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from PyPDF2 import PdfReader
import torch

# -------------------------------
# LOAD MODELS
# -------------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# -------------------------------
# LOAD DOCUMENTS
# -------------------------------
def load_docs(folder):
    texts = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())

        elif file.endswith(".pdf"):
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()
            texts.append(text)

    return texts

# -------------------------------
# CHUNKING
# -------------------------------
def chunk_text(texts):
    chunks = []
    for text in texts:
        for i in range(0, len(text), 300):
            chunks.append(text[i:i+300])
    return chunks

# -------------------------------
# CREATE FAISS INDEX
# -------------------------------
def create_index(chunks):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index

# -------------------------------
# SEARCH FUNCTION
# -------------------------------
def search(query, index, chunks):
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb), k=3)
    return [chunks[i] for i in I[0]]

# -------------------------------
# GENERATE ANSWER
# -------------------------------
def generate_answer(question, context):
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits).item()
    end = torch.argmax(outputs.end_logits).item()

    if end < start:
        return "⚠️ Answer not found clearly in documents"

    answer_ids = inputs["input_ids"][0][start:end+1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    answer = answer.strip().replace("Employee Handbook", "")

    if len(answer) == 0:
        return "⚠️ No clear answer found"

    return answer

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Enterprise RAG")

st.title("📚 Enterprise Knowledge Base Q&A")
st.markdown("### 💬 Ask questions about company policies")

# Initialize
if "index" not in st.session_state:
    docs = load_docs(".")

    if len(docs) == 0:
        st.error("⚠️ No documents found! Add .txt or .pdf files.")
    else:
        chunks = chunk_text(docs)
        index = create_index(chunks)

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.success("✅ Documents loaded successfully!")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# INPUT FORM (ENTER + BUTTON)
# -------------------------------
with st.form("query_form"):
    query = st.text_input("Ask a question:")
    submit = st.form_submit_button("🔍 Get Answer")

if submit and query:
    with st.spinner("Searching documents..."):
        results = search(query, st.session_state.index, st.session_state.chunks)
        context = "\n".join(results)[:1000]
        answer = generate_answer(query, context)

        st.session_state.history.append((query, answer, results))

# -------------------------------
# DISPLAY CHAT
# -------------------------------
for q, a, res in reversed(st.session_state.history):
    st.markdown(f"**🧑 You:** {q}")
    st.markdown(f"**🤖 AI:** {a}")

    st.markdown("**📄 Sources:**")
    for r in res:
        st.write("-", r[:120])

    st.markdown("---")