
# pyChatbot

## Overview
Offline, multi‑instance chatbot engine built with **Python**, **Flask**, **llama‑cpp**, **FAISS**, **BM25**, and a custom crawling + indexing pipeline.  
Supports per‑instance configuration, persistent storage, hybrid retrieval, session‑aware chat, and local model inference.

---

## Features
- **Local LLM inference** via *llama-cpp* (LLaMA and compatible HF GGUF models).  
- **Hybrid RAG pipeline** using **SentenceTransformer embeddings + FAISS** and **BM25** for dense+sparse ranking.  
- **Website/content ingestion** with recursive crawler and HTML cleaner.  
- **SQLite‑based store** for chunks, embeddings, and metadata.  
- **Instance manager API** to create, load, delete, and configure independent chatbot instances.  
- **Session‑aware chat** with per-user histories and scalable LRU caching.  
- **CORS-enabled Flask backend** for frontend or external integration.

---

## Project Structure
```
pyChatbot/
│── app_protocol.py       # Flask app: APIs, instance management, session handling  
│── pybot_module.py       # Core bot engine: crawling, chunking, BM25, FAISS, llama inference  
│── requirements.txt      # Dependencies  
│── test_pyBot_module.py  # Unit tests  
```

---

## Installation
```bash
pip install -r requirements.txt
```

### Environment notes
- Set local HuggingFace cache via:
```
HF_HOME=D:/huggingface
```
- Configure base storage directory for instance data:
```
BOT_SAVE_DIR=D:/.yourchatbot
```

---

## Running the Server
```bash
python app_protocol.py
```

---

## API Summary
### Create a chatbot instance
`POST /instances`

### Query a chatbot
`POST /chat/<instance_name>`

### Delete an instance
`DELETE /instances/<instance_name>`

---

## How It Works
1. **Crawler** fetches pages up to depth/limits.  
2. **Cleaner + splitter** converts HTML to text chunks.  
3. **BM25 + Transformer embeddings** generated.  
4. **FAISS index** stored on disk.  
5. **llama-cpp** generates responses using retrieved context.  
6. **Session manager** maintains chat continuity.


