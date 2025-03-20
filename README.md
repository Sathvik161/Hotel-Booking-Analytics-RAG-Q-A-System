# Hotel Booking Analytics & RAG Q&A System 🚀

This project provides hotel booking **analytics** and **question-answering** using **LLM (Groq Mixtral)** and **RAG (FAISS + Google Generative AI Embeddings)**.

## Features

- ✅ **Data Preprocessing** - Cleans and structures hotel booking data.
- ✅ **Analytics Reports** - Revenue trends, cancellation rates, booking patterns.
- ✅ **RAG Model** - Uses FAISS for retrieval & Mixtral for answer generation.
- ✅ **FastAPI Backend** - Serves analytics & Q&A via API.
- ✅ **Streamlit UI** - Interactive web interface.

## 🛠️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/hotel-booking-analytics-rag.git
cd hotel-booking-analytics-rag
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys

Create a `.env` file and add:

```ini
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

This will **automatically start FastAPI** and open the **Streamlit UI**.

## API Endpoints

| Method | Endpoint         | Description                   |
| ------ | ---------------- | ----------------------------- |
| `POST` | `/analytics`     | Get stored analytics          |
| `POST` | `/ask?query=...` | Ask booking-related questions |

## Sample Queries & Expected Results

See [`test_queries.json`](test_queries.json) for examples.

## Implementation Report

See [`report.pdf`](report.pdf) for system design & challenges.

## License

MIT License. Free to use & modify!
