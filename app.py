import streamlit as st
import json
import openai
import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Ashby Admin Q&A", layout="wide")
st.title("📘 Ask Ashby Admin Docs")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

with open("ashby_chunks_with_links.json", "r") as f:
    chunks = json.load(f)

chunk_texts = [c["text"] for c in chunks]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(chunk_texts)

query = st.text_input("Ask a question:")
if query:
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    top_indices = similarities.argsort()[-3:][::-1]
    top_chunks = [chunks[i] for i in top_indices]

    context = "\n\n".join([c["text"] + f"\nSource: {c['url']}" for c in top_chunks])

    prompt = f"Answer the following question using the context provided. Be concise.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    st.subheader("Answer")
    st.write(response.choices[0].message.content.strip())
    st.markdown("#### Sources")
    for c in top_chunks:
        st.markdown(f"- [{c['url']}]({c['url']})")
