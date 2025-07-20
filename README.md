# Semantic Search Web App (Qdrant + Groq + FastAPI)

A modern semantic search and knowledge base web application using Qdrant (vector database), Groq LLM, and FastAPI. Supports text, category, and PDF upload with vectorization.

---

## 🚀 Features
- Semantic search with vector embeddings
- Retrieval-augmented LLM answers (Groq)
- Create and manage Qdrant collections
- Add documents and upload PDFs (auto-chunked and embedded)
- Modern web UI with responsive navigation

---

## 🐳 Quick Start (Docker Compose)

### 1. **Clone the repo and set up your `.env` file**
```
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
GROQ_API_KEY=your_groq_api_key
```

### 2. **Build and start the stack**
```bash
docker-compose up --build
```

- App: [http://localhost:8000](http://localhost:8000)
- Qdrant: [http://localhost:6333](http://localhost:6333)

---

## 📝 Usage
- **Search:** Use the main page to search your knowledge base.
- **Create Collection:** Create a new Qdrant collection for your data.
- **Add Document:** Add text and category to any collection.
- **Upload PDF:** Upload a PDF, which will be chunked, embedded, and indexed for semantic search.

All features are accessible from the hamburger menu.

---

## ⚙️ Development
- The app runs with `--reload` for live code changes.
- Code and templates are mounted into the container.
- To install new Python packages, add them to `requirements.txt` and rebuild:
  ```bash
  docker-compose build app
  docker-compose up
  ```

---

## 🧩 Prerequisites
- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
- Groq API key (get one from [Groq Cloud](https://console.groq.com/))

---

## 📄 PDF Upload Notes
- Large PDFs will be split into chunks and each chunk embedded and indexed.
- For very large files, upload may take several minutes.
- You can adjust chunk and batch size in `app.py` for performance.

---

## 🛠️ Customization
- Change embedding model in `app.py` if desired.
- Add more endpoints or UI features as needed.

---

## 📦 Production
- For production, remove `--reload` from the compose file and set proper secrets.

---

## License
MIT 