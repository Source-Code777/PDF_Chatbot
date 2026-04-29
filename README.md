# 📄 PDF Chatbot (RAG-Based)

A conversational AI system that allows users to upload a PDF and ask questions about it using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

* 📄 Upload any PDF
* 💬 Ask questions in natural language
* 🔍 Hybrid retrieval (BM25 + dense embeddings)
* 🧠 Cross-encoder reranking
* 🖥️ Streamlit UI
* 🐳 Docker support

---

## 🧱 Project Structure

```
PDF_Chatbot/
├── app.py
├── src/
│   ├── core/
│   ├── utils/
│   ├── retrieval/
│   ├── loader.py
│   ├── splitter.py
│   ├── vectorstore.py
│   └── ...
├── data/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup (Local)

### 1. Clone the repository

```
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot
```

---

### 2. Create virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

### 4. Run Ollama (Required)

```
ollama run mistral:7b-instruct
```

---

### 5. Run the app

```
streamlit run app.py
```

---

## 🐳 Run with Docker

### 1. Build Docker image

```
docker build -t pdf-chatbot .
```

---

### 2. Run container

```
docker run -p 8501:8501 pdf-chatbot
```

---

### 3. Open in browser

```
http://localhost:8501
```

---

## ⚠️ Important Note

This project uses a locally running Ollama model.

👉 Make sure Ollama is running before using the chatbot.

---

## 🧠 Future Improvements

* Cloud deployment (Render / Railway)
* Replace local LLM with API (OpenAI / Groq)
* Multi-document support
* Improved UI

---

## 📌 Author

Aasim Athar Alam
