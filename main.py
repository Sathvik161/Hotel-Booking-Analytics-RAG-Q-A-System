import streamlit as st
import pandas as pd
import numpy as np
import faiss
import requests
import sqlite3
import os
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
import uvicorn
from threading import Thread

# ==========================
# 🔑 LOAD API KEYS FROM .env
# ==========================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY or not GROQ_API_KEY:
    st.error("❌ API keys not found! Please check your .env file.")
    st.stop()

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# ==========================
# 🎯 STREAMLIT UI SETUP
# ==========================
st.title("📊 Hotel Booking Analytics & RAG-based Q&A")

# File upload
uploaded_file = st.file_uploader("📂 Upload cleaned dataset (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Data loaded successfully!")

# ==========================
# 🔄 DATA PREPROCESSING
# ==========================
if uploaded_file:
    st.subheader("🔄 Data Preprocessing")
    
    # Handle missing values
    df.fillna({'children': 0, 'agent': 0, 'company': 0}, inplace=True)
    df.dropna(subset=['country'], inplace=True)
    
    # Convert to datetime
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    
    # Save cleaned dataset
    df.to_csv("cleaned_hotel_bookings.csv", index=False)
    st.write("✅ Data cleaned and saved!")

# ==========================
# 📊 DATABASE SETUP
# ==========================
analytics_conn = sqlite3.connect("analytics.db", check_same_thread=False)
analytics_cursor = analytics_conn.cursor()
analytics_cursor.execute('''
CREATE TABLE IF NOT EXISTS analytics_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT UNIQUE,
    metric_value TEXT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')
analytics_conn.commit()

rag_conn = sqlite3.connect("rag_storage.db", check_same_thread=False)
rag_cursor = rag_conn.cursor()
rag_cursor.execute("CREATE TABLE IF NOT EXISTS booking_data (id INTEGER PRIMARY KEY, text TEXT)")
rag_conn.commit()

# ==========================
# 📈 GENERATE ANALYTICS
# ==========================
if uploaded_file:
    st.subheader("📊 Generating Analytics")

    analytics_queries = {
        "Revenue Trends": "Analyze revenue trends over time.",
        "Cancellation Rate": "Calculate the percentage of canceled bookings.",
        "Geographical Distribution": "Find top booking locations.",
        "Booking Lead Time": "Analyze lead time between booking and check-in.",
    }

    def get_mixtral_analysis(prompt):
        api_url = "https://api.groq.com/v1/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "mixtral", "messages": [{"role": "user", "content": prompt}]}
        response = requests.post(api_url, headers=headers, json=data)
        return response.json().get("choices", [{}])[0].get("text", "No response")

    for metric, prompt in analytics_queries.items():
        result = get_mixtral_analysis(prompt)
        analytics_cursor.execute("INSERT OR REPLACE INTO analytics_results (metric_name, metric_value) VALUES (?, ?)",
                                 (metric, result))
        analytics_conn.commit()

    st.write("✅ Analytics generated and stored!")

# ==========================
# 🔍 RAG MODEL SETUP
# ==========================
if uploaded_file:
    st.subheader("🔍 Setting Up RAG Model")

    df["text"] = df.apply(lambda row: " | ".join([f"{col}: {row[col]}" for col in df.columns]), axis=1)
    
    dimension = 768
    index = faiss.IndexFlatL2(dimension)

    def get_google_embeddings(text):
        response = genai.embed_content("models/embedding-001", text)
        return np.array(response["embedding"])

    embeddings = np.array([get_google_embeddings(text) for text in df["text"]], dtype=np.float32)
    index.add(embeddings)

    for i, text in enumerate(df["text"]):
        rag_cursor.execute("INSERT INTO booking_data (id, text) VALUES (?, ?)", (i, text))
    rag_conn.commit()

    st.write("✅ RAG model initialized!")

# ==========================
# 🚀 FASTAPI BACKEND
# ==========================
app = FastAPI()

@app.post("/analytics")
def get_analytics():
    analytics_cursor.execute("SELECT metric_name, metric_value FROM analytics_results")
    results = analytics_cursor.fetchall()
    if not results:
        raise HTTPException(status_code=404, detail="No analytics found")
    return {metric: value for metric, value in results}

def retrieve_relevant_data(query, top_k=3):
    query_embedding = get_google_embeddings(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    retrieved_texts = []
    for idx in indices[0]:
        rag_cursor.execute("SELECT text FROM booking_data WHERE id=?", (int(idx),))
        retrieved_texts.append(rag_cursor.fetchone()[0])
    
    return retrieved_texts

def get_mixtral_answer(query):
    relevant_data = retrieve_relevant_data(query)
    prompt = f"Based on these booking records: {relevant_data}\n\nAnswer: {query}"
    
    api_url = "https://api.groq.com/v1/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "mixtral", "messages": [{"role": "user", "content": prompt}]}
    
    response = requests.post(api_url, headers=headers, json=data)
    return response.json().get("choices", [{}])[0].get("text", "No response")

@app.post("/ask")
def ask_question(query: str):
    answer = get_mixtral_answer(query)
    return {"query": query, "answer": answer}

def start_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run FastAPI in background
Thread(target=start_api, daemon=True).start()

# ==========================
# 🎯 STREAMLIT FRONTEND
# ==========================
st.subheader("📊 Analytics")
if st.button("Fetch Analytics"):
    response = requests.post("http://localhost:8000/analytics").json()
    st.json(response)

st.subheader("💬 Ask Questions")
query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    response = requests.post("http://localhost:8000/ask", params={"query": query}).json()
    st.write(f"🤖 Answer: {response['answer']}")

st.success("✅ System Ready!")
