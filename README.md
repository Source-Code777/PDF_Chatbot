# PDF Chatbot (RAG-Based)

A conversational AI system that allows users to upload a PDF and ask questions about its content using **Retrieval-Augmented Generation (RAG)**.

---

## Features

* Upload and process PDF documents
* Ask questions in natural language
* Hybrid retrieval (BM25 + dense embeddings)
* Cross-encoder reranking (local mode)
* Dual mode:

  * **Local (Ollama)**
  * **API (Groq)**
* Streamlit-based user interface
* Docker support for containerized execution

---

## Project Structure

```
PDF_Chatbot/
│
├── app.py
├── src/
│   ├── core/
│   ├── retrieval/
│   ├── utils/
│   ├── loader.py
│   ├── splitter.py
│   ├── vectorstore.py
│   └── ...
│
├── data/
├── Dockerfile
├── requirements.txt
├── requirements-local.txt
└── README.md
```

---

## Local Setup

### 1. Clone the repository

```
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot
```

---

### 2. Create and activate virtual environment

**Windows**

```
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux**

```
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

For **local mode (Ollama + reranker)**:

```
pip install -r requirements-local.txt
```

---

## Running the App

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## Modes

### 1. API Mode (Recommended)

* Uses Groq for fast inference
* Enter your API key in the UI

Get API key:
https://console.groq.com

---

### 2. Local Mode (Ollama)

Requires a local LLM setup.

### Install Ollama

https://ollama.com/download

### Run model

```
ollama pull mistral:7b-instruct
ollama serve
```

---

## Docker Setup (API Mode)

### 1. Build image

```
docker build -t pdf-chatbot .
```

---

### 2. Run container

```
docker run -p 8501:8501 -e GROQ_API_KEY=your_key pdf-chatbot
```

---

### 3. Access app

```
http://localhost:8501
```

---

## Important Notes

* Docker container is configured for **API mode only**
* Local mode requires Ollama running on host machine
* Default Ollama endpoint:

```
http://host.docker.internal:11434
```

* This works on Docker Desktop (Windows/macOS)
* For Linux/cloud, update the URL accordingly

---

## Tech Stack

* **LLM**: Groq / Ollama (Mistral)
* **Embeddings**: BGE / MiniLM
* **Vector DB**: Chroma
* **Retrieval**: BM25 + Dense Retrieval
* **Reranking**: Cross-Encoder (local)
* **Framework**: Streamlit + LangChain

---

## Future Improvements

* Streaming responses (ChatGPT-like UX)
* Better retrieval (hybrid scoring / fusion)
* Multi-document support
* Deployment (Render / Railway / VPS)
* UI enhancements

---

## Author

**Source_Code777**
