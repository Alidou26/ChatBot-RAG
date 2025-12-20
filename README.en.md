<div align="left"> <a href="./README.md">🇫🇷 Français</a> | <a href="./README.en.md">🇬🇧 English</a> </div>

***

<a name="top"></a>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/ChromaDB-20232A?style=for-the-badge&logo=aws-amplify&logoColor=white" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Jina%20Embeddings-v3-4B32C3?style=for-the-badge" alt="Jina Embeddings v3">
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama">
  <h1>AI Assistant for Technical Knowledge Management (RAG + RCA)</h1>

  <p>On‑premise conversational assistant for SOLENT, built on an advanced RAG pipeline (RAG‑Fusion + Self‑RAG) and a root cause analysis (RCA) module connected to Redmine.</p>

</div>

# [Demo Video](https://drive.google.com/file/d/1-ywtd9EggAKQTIJjwxZUoYZfcXjqBXJq/view?usp=sharing)
<div>Click on Demo Video above<div>

<div>If the link does not work, consider copying it and pasting it into your browser’s address bar.<div>

# [Report](https://drive.google.com/file/d/1EJiBMSZ3w7-WeSRf2kRZgYXzPE1OgyUs/view?usp=sharing)

<div>Click on Report above<div>

<div>If the link does not work, consider copying it and pasting it into your browser’s address bar.<div>

***

## Table of Contents

1. [Introduction](#introduction)
2. [Key Features](#features)
3. [Used Technologies](#tech)
4. [Installation and Run](#installation)
5. [RAG / RCA Architecture](#archi)
6. [Future Work and Improvements](#future)

***

## Introduction<a name="introduction"></a>

This project provides a **technical knowledge management** assistant that lets users query documentation, source code, and Redmine tickets via a RAG chat interface and a dedicated root cause analysis (RCA) module.  
The solution is designed for fully on‑premise deployment, with language models, embeddings, and vector database all running locally to comply with SOLENT’s confidentiality constraints.

<div align="right">
  <a href="#top">⬆ Back to top</a>
</div>

***

## Key Features<a name="features"></a>

### RAG Documentation Chat

- Natural language Q&A over technical documentation, source code, and Redmine tickets.
- Display of source passages (chunks) used, with metadata and similarity scores, to ensure full traceability of answers.


### RAG‑Fusion and Self‑RAG

- Deterministic RAG‑Fusion: multiple query generation, parallel search in ChromaDB, result fusion using Reciprocal Rank Fusion (RRF), and strict deduplication of chunks.
- Backend Self‑RAG: generation is gated based on number of chunks, average similarity score, and total context volume, with explicit refusal to answer when context is insufficient.


### Redmine Integration

- Full Redmine connector: retrieval of issues, wiki pages, documents, attachments, versions, members, and metadata through the REST API.
- Automatic ingestion of a Redmine project (issues + related artifacts), then indexing into ChromaDB to be used by both RAG and RCA.


### Root Cause Analysis (RCA)

- Dedicated form where the user can paste an error message or describe a technical incident.
- RCA‑oriented retrieval prioritizing chunks from Redmine issues, code, and technical artifacts, followed by generation of a synthetic report (symptoms, root causes, corrective actions, preventive measures).


### Multi‑source Ingestion

- Ingestion of files (PDF, DOCX, PPTX, TXT, JSON, code, etc.), ZIP archives of projects, and Redmine content.
- Modular parsing and chunking pipeline, with specialized parsers (PDF, code, office documents, structured formats) and dedicated chunkers for both text and code.

<div align="right">
  <a href="#top">⬆ Back to top</a>
</div>

***

## Used Technologies<a name="tech"></a>

<div align="center">
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="60" height="60" alt="Python">
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/fastapi/fastapi-original.svg" width="60" height="60" alt="FastAPI">
  <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png" width="120" alt="Streamlit">
  <img src="https://vectorlogo.zone/logos/docker/docker-official.svg" width="60" height="60" alt="Docker">
</div>

- **Backend**: FastAPI (REST API, RAG/RCA/ingestion endpoints).
- **Frontend**: Streamlit (RAG chat, RCA page, ingestion interface, backend health‑check).
- **Vector database**: ChromaDB, HNSW index with cosine similarity.
- **Embeddings**: Jina AI embeddings v3 (multilingual retrieval‑oriented model, 1024 dimensions).
- **LLM**: local model via Ollama (for example `qwen2.5:3b`, recommended and validated for this project).

<div align="right">
  <a href="#top">⬆ Back to top</a>
</div>

***

## Installation and Run<a name="installation"></a>

### 1. Clone the repository

```bash
git clone https://github.com/Alidou26/ChatBot-RAG.git
```

### 2. Prerequisites

- Python 3.10 or higher (macOS / Linux / Windows).
- A modern web browser to access the Streamlit interface.


### 3. Create and activate virtual environments

From the project root, create a virtual environment in each subfolder (backend and frontend), then activate it from inside each folder.


### macOS / Linux

From the project root, open a terminal.

#### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

Open a second terminal.

#### Frontend
```bash
cd frontend
python3 -m venv venv
source venv/bin/activate
```

### Windows

From the project root, open a terminal.

#### Backend
```bash
cd backend
python3 -m venv venv
venv\Scriptsctivate
```

Open a second terminal.

#### Frontend
```bash
cd frontend
python3 -m venv venv
venv\Scriptsctivate
```

### 4. Install dependencies

Backend:

In the backend terminal:

```bash
pip install -r requirements.txt
```

Frontend:

In the frontend terminal:

```bash
pip install -r requirements.txt
```

These dependencies notably include FastAPI, ChromaDB, LlamaIndex, Jina embeddings, and Streamlit.

### 5. Install Ollama

- macOS: https://ollama.com/download/mac  
- Windows: https://ollama.com/download/windows  

Ollama is used to run LLMs locally.

### 6. Download the recommended LLM model

Once Ollama is installed, open a third terminal:

```bash
ollama pull qwen2.5:3b
```

The pipeline is compatible with other Ollama models, but `qwen2.5:3b` has been validated for this project.

### 7. Configure environment variables

A **`.env` file is already present at the project root**.

**You do not need to create a new `.env` file.**

You only need to **update the existing parameters if needed**, for example:

```env
REDMINE_URL=http://localhost:3000
REDMINE_API_KEY=9fd3becfae03e9af25016b50f623d08224fe38ff
LLM_BASE_URL=http://127.0.0.1:11434 
LLM_MODEL=qwen2.5:3b 
LLM_TEMPERATURE=0 
LLM_TIMEOUT=180
RAG_FUSION_ENABLED=1
RAG_FUSION_RRF_K=60
```

These variables control the connection to the local LLM, the activation of RAG‑Fusion, and the connection to the Redmine project.

### 8. Start the Ollama server

In the third terminal:

```bash
ollama serve
```

The server must remain active for the assistant to work.

### 9. Start the FastAPI API

In the first terminal dedicated to the backend (with the virtual environment activated):

```bash
uvicorn app.main:app --reload
```

- FastAPI API: http://127.0.0.1:8000  
- OpenAPI documentation: http://127.0.0.1:8000/docs  

The API exposes endpoints for RAG chat, RCA, file ingestion, archive ingestion, and Redmine ingestion.

### 10. Start the Streamlit interface

In the second terminal dedicated to the frontend (still with the virtual environment activated):

```bash
streamlit run streamlit_app.py
```

- Streamlit UI: http://localhost:8501  

The sidebar provides access to the RAG chat, RCA analysis, and data ingestion page.

<div align="right">
  <a href="#top">⬆ Back to top</a>
</div>


## Future Work and Improvements<a name="future"></a>

- Integration of advanced image parsing (diagrams, screenshots) via OCR and vision models to enrich the knowledge base.
- Migration toward graph‑structured embeddings to better model relationships between tickets, documents, users, and technical components.
- Evolution into an **AI agent** able to achieve objectives (e.g. ticket prioritization, Redmine report generation) instead of a simple Q&A chatbot.
- Full containerization (Docker) and deployment automation to simplify industrialization of the solution at SOLENT.

<div align="right">
  <a href="#top">⬆ Back to top</a>
</div>



---

## Demo<a name="demo"></a>

<img src="image/c1.png" style="width: 80%; max-width: 600px; height: auto; display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 10px;" alt="Screenshot preview">
<img src="image/c2.png" style="width: 80%; max-width: 600px; height: auto; display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 10px;" alt="Screenshot preview">
<img src="image/c3.png" style="width: 80%; max-width: 600px; height: auto; display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 10px;" alt="Screenshot preview">
<img src="image/c4.png" style="width: 80%; max-width: 600px; height: auto; display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 10px;" alt="Screenshot preview">
<img src="image/c5.png" style="width: 80%; max-width: 600px; height: auto; display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 10px;" alt="Screenshot preview">
<img src="image/c6.png" style="width: 80%; max-width: 600px; height: auto; display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 10px;" alt="Screenshot preview">
<img src="image/c7.png" style="width: 80%; max-width: 600px; height: auto; display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 10px;" alt="Screenshot preview">
<img src="image/c8.png" style="width: 80%; max-width: 600px; height: auto; display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 10px;" alt="Screenshot preview">
<img src="image/c9.png" style="width: 80%; max-width: 600px; height: auto; display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 10px;" alt="Screenshot preview">
<img src="image/c10.png" style="width: 80%; max-width: 600px; height: auto; display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 10px;" alt="Screenshot preview">
<img src="image/c11.png" style="width: 80%; max-width: 600px; height: auto; display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 10px;" alt="Screenshot preview">
<img src="image/c12.png" style="width: 80%; max-width: 600px; height: auto; display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 10px;" alt="Screenshot preview">


<div align="right"> <a href="#top">⬆ Back to top</a> </div>
