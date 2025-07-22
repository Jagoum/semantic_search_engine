"""
FastAPI web application for semantic search using Qdrant and Groq AI.
Features:
- Search existing collections using semantic search and LLM answers
- Create new Qdrant collections
- Add documents to collections
- Upload and index PDF files (extract, chunk, embed, and store in Qdrant)

Requirements:
- Qdrant cloud instance and API key
- Groq AI API key
- .env file with all secrets
- requests, qdrant_client, sentence-transformers, groq, pypdf, fastapi, uvicorn, jinja2, python-multipart, python-dotenv
"""

from fastapi import FastAPI, Request, Form, UploadFile, Cookie
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from groq import Groq
from pydantic import BaseModel
from typing import List
from pypdf import PdfReader
import uuid
import time

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Semantic Search System")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

from datetime import datetime
def datetimeformat(value, format="%Y-%m-%d %H:%M:%S"):
    try:
        return datetime.fromtimestamp(int(value)).strftime(format)
    except Exception:
        return value

templates.env.filters['datetimeformat'] = datetimeformat

# Initialize Qdrant, embedding, and Groq clients
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120.0  # Set timeout to 120 seconds (or higher if needed)
)
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
groq_client = Groq(api_key=GROQ_API_KEY)

class SearchRequest(BaseModel):
    query: str
    collection_name: str = "knowledge_base"

def search_and_query_groq(query: str, collection_name: str = "knowledge_base"):
    """
    Perform semantic search in Qdrant and get an LLM answer from Groq.
    """
    # Check if collection exists
    if collection_name not in [c.name for c in client.get_collections().collections]:
        return [], "Collection not found. Please create and populate the collection first."
    # Generate embedding for query
    query_embedding = embedder.encode(query)
    # Search in Qdrant
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=10
    )
    # Build context from search results
    context = "Based on the following relevant documents:\n\n"
    for i, hit in enumerate(search_result, 1):
        if hit.payload:
            context += f"{i}. {hit.payload.get('text', 'N/A')}\n"
    # Create the full prompt with context
    full_prompt = f"{context}\n\nQuestion: {query}\n\nAnswer based on the documents above:"
    # Query Groq AI
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    ai_response = response.choices[0].message.content
    return search_result, ai_response

def extract_pdf_chunks(file, chunk_size=500) -> List[str]:
    """
    Extract text from a PDF file and split it into chunks of a given size.
    Args:
        file: A file-like object containing the PDF.
        chunk_size: Number of characters per chunk.
    Returns:
        List of text chunks.
    """
    reader = PdfReader(file)
    all_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            all_text += page_text + "\n"
    # Split into chunks (e.g., every 500 characters)
    chunks = [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size) if all_text[i:i+chunk_size].strip()]
    return chunks

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with search interface."""
    collections = [c.name for c in client.get_collections().collections]
    return templates.TemplateResponse("index.html", {"request": request, "collections": collections})

@app.post("/search")
async def search_endpoint(request: Request, query: str = Form(...), collection_name: str = Form("knowledge_base")):
    """Handle search requests."""
    search_results, ai_response = search_and_query_groq(query, collection_name)
    # Format results for display
    formatted_results = []
    for i, hit in enumerate(search_results, 1):
        formatted_results.append({
            "rank": i,
            "score": f"{hit.score:.4f}",
            "text": hit.payload.get('text', 'N/A') if hit.payload else 'N/A',
            "category": hit.payload.get('category', 'N/A') if hit.payload else 'N/A'
        })
    return templates.TemplateResponse(
        "results.html", 
        {
            "request": request,
            "query": query,
            "results": formatted_results,
            "ai_response": ai_response,
            "collection_name": collection_name
        }
    )

@app.get("/api/search")
async def api_search(query: str, collection_name: str = "knowledge_base"):
    """API endpoint for programmatic access."""
    search_results, ai_response = search_and_query_groq(query, collection_name)
    # Format for JSON response
    results = []
    for hit in search_results:
        results.append({
            "score": hit.score,
            "text": hit.payload.get('text', 'N/A') if hit.payload else 'N/A',
            "category": hit.payload.get('category', 'N/A') if hit.payload else 'N/A'
        })
    return {
        "query": query,
        "collection": collection_name,
        "search_results": results,
        "ai_response": ai_response
    }

@app.get("/create-collection", response_class=HTMLResponse)
async def create_collection_form(request: Request):
    """Show form to create a new Qdrant collection."""
    return templates.TemplateResponse("create_collection.html", {"request": request})

@app.post("/create-collection")
async def create_collection(request: Request, collection_name: str = Form(...), vector_size: int = Form(384)):
    """Create a new Qdrant collection with the given name and vector size."""
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name in existing_collections:
        msg = f"Collection '{collection_name}' already exists."
        return templates.TemplateResponse("create_collection.html", {"request": request, "message": msg, "success": False})
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    )
    msg = f"Collection '{collection_name}' created successfully!"
    return templates.TemplateResponse("create_collection.html", {"request": request, "message": msg, "success": True})

@app.get("/add-document", response_class=HTMLResponse)
async def add_document_form(request: Request):
    """Show form to add a document to a collection."""
    collections = [c.name for c in client.get_collections().collections]
    return templates.TemplateResponse("add_document.html", {"request": request, "collections": collections})

@app.post("/add-document")
async def add_document(request: Request, collection_name: str = Form(...), text: str = Form(...), category: str = Form(...)):
    """Add a document (text + category) to a collection, embedding it and assigning a new ID."""
    embedding = embedder.encode(text)
    # Get next available ID
    points = client.scroll(collection_name=collection_name, limit=100, with_payload=False, with_vectors=False)
    if points and points[0]:
        max_id = max(int(point.id) for point in points[0])
        next_id = max_id + 1
    else:
        next_id = 1
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=next_id,
                vector=embedding.tolist(),
                payload={"text": text, "category": category}
            )
        ]
    )
    msg = f"Document added to '{collection_name}' with ID {next_id}."
    collections = [c.name for c in client.get_collections().collections]
    return templates.TemplateResponse("add_document.html", {"request": request, "collections": collections, "message": msg, "success": True})

@app.get("/upload-pdf", response_class=HTMLResponse)
async def upload_pdf_form(request: Request):
    """Show form to upload a PDF and select a collection."""
    collections = [c.name for c in client.get_collections().collections]
    return templates.TemplateResponse("upload_pdf.html", {"request": request, "collections": collections})

@app.post("/upload-pdf", response_class=HTMLResponse)
async def upload_pdf(request: Request, collection_name: str = Form(...), file: UploadFile = Form(...)):
    """
    Handle PDF upload, extract and chunk text, embed each chunk, and store in Qdrant.
    Args:
        collection_name: The collection to store the PDF chunks in.
        file: The uploaded PDF file.
    Returns:
        Rendered template with upload summary.
    """
    collections = [c.name for c in client.get_collections().collections]
    if file is None or not getattr(file, 'filename', '').lower().endswith(".pdf"):
        msg = "Please upload a valid PDF file."
        return templates.TemplateResponse("upload_pdf.html", {"request": request, "collections": collections, "message": msg, "success": False})
    # Read and process PDF
    contents = await file.read()
    import io
    pdf_file = io.BytesIO(contents)
    chunks = extract_pdf_chunks(pdf_file)
    # Get next available ID
    points = client.scroll(collection_name=collection_name, limit=100, with_payload=False, with_vectors=False)
    if points and points[0]:
        max_id = max(int(point.id) for point in points[0])
        next_id = max_id + 1
    else:
        next_id = 1
    # Batch upsert chunks
    batch_size = 20
    batch = []
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk)
        batch.append(
            models.PointStruct(
                id=next_id + i,
                vector=embedding.tolist(),
                payload={"text": chunk, "source": file.filename, "chunk": i}
            )
        )
        if len(batch) == batch_size or i == len(chunks) - 1:
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            batch = []
    msg = f"Uploaded and indexed {len(chunks)} chunks from '{file.filename}' into '{collection_name}'."
    return templates.TemplateResponse("upload_pdf.html", {"request": request, "collections": collections, "message": msg, "success": True, "chunks": len(chunks), "filename": file.filename})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, session_id: str = Cookie(None)):
    collections = [c.name for c in client.get_collections().collections]
    # Generate a new session_id if not present
    if not session_id:
        session_id = str(uuid.uuid4())
    response = templates.TemplateResponse("chat.html", {"request": request, "collections": collections, "history": [], "session_id": session_id})
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return response

@app.post("/chat", response_class=HTMLResponse)
async def chat_post(request: Request, collection_name: str = Form(...), user_message: str = Form(...), history: str = Form("[]"), session_id: str = Cookie(None)):
    import json
    collections = [c.name for c in client.get_collections().collections]
    # Parse chat history
    chat_history = json.loads(history)
    # Vector search for user message
    query_embedding = embedder.encode(user_message)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=5
    )
    # Build context from search results (no explicit reference to documents)
    context = ""
    for i, hit in enumerate(search_result, 1):
        if hit.payload:
            context += f"{hit.payload.get('text', 'N/A')}\n"
    # System prompt to restrict bot to context only
    system_prompt = "You are a helpful assistant. Only answer using the provided information. If you don't know, say you don't know."
    # Build full prompt
    full_prompt = f"{system_prompt}\n\n{context}\n\nUser: {user_message}\nAssistant:"
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=200,
        temperature=0.2,
    )
    ai_response = response.choices[0].message.content
    # Update chat history
    chat_history.append({"user": user_message, "bot": ai_response})
    # Store chat turn in Qdrant chat_history collection
    if not session_id:
        session_id = str(uuid.uuid4())
    timestamp = int(time.time())
    embedding = embedder.encode(user_message)
    # Ensure chat_history collection exists
    if "chat_history" not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name="chat_history",
            vectors_config=models.VectorParams(size=embedding.shape[0], distance=models.Distance.COSINE)
        )
    client.upsert(
        collection_name="chat_history",
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),  # Use a valid UUID string
                vector=embedding.tolist(),
                payload={
                    "session_id": session_id,
                    "user_message": user_message,
                    "bot_response": ai_response,
                    "timestamp": timestamp,
                    "collection": collection_name
                }
            )
        ]
    )
    response = templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "collections": collections,
            "collection_name": collection_name,
            "history": chat_history,
            "session_id": session_id
        }
    )
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return response

@app.get("/chat-history", response_class=HTMLResponse)
async def chat_history_page(request: Request, session_id: str = Cookie(None)):
    # Get all sessions from chat_history collection
    if "chat_history" not in [c.name for c in client.get_collections().collections]:
        sessions = []
    else:
        points = client.scroll(collection_name="chat_history", limit=1000, with_payload=True, with_vectors=False)
        sessions = {}
        for point in points[0]:
            payload = point.payload
            if payload is not None:
                sid = payload.get("session_id")
                if sid not in sessions:
                    sessions[sid] = {
                        "session_id": sid,
                        "first_message": payload.get("user_message", ""),
                        "timestamp": payload.get("timestamp", 0)
                    }
        # Sort by timestamp (descending)
        sessions = sorted(sessions.values(), key=lambda x: -x["timestamp"])
    return templates.TemplateResponse("chat_history.html", {"request": request, "sessions": sessions, "current_session": session_id})

@app.get("/chat-session/{session_id}", response_class=HTMLResponse)
async def chat_session_page(request: Request, session_id: str):
    # Load chat history for a session
    collections = [c.name for c in client.get_collections().collections]
    points = client.scroll(collection_name="chat_history", limit=1000, with_payload=True, with_vectors=False)
    history = []
    collection_name = None
    for point in sorted(points[0], key=lambda p: p.payload.get("timestamp", 0) if p.payload else 0):
        payload = point.payload
        if payload is not None and payload.get("session_id") == session_id:
            history.append({"user": payload.get("user_message", ""), "bot": payload.get("bot_response", "")})
            if not collection_name:
                collection_name = payload.get("collection")
    response = templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "collections": collections,
            "collection_name": collection_name or (collections[0] if collections else None),
            "history": history,
            "session_id": session_id
        }
    )
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 