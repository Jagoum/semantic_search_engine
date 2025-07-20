"""
FastAPI web application for semantic search using Qdrant and Groq AI.
"""

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from groq import Groq
from pydantic import BaseModel

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Semantic Search System")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize clients
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
groq_client = Groq(api_key=GROQ_API_KEY)

class SearchRequest(BaseModel):
    query: str
    collection_name: str = "knowledge_base"

def search_and_query_groq(query: str, collection_name: str = "knowledge_base"):
    """
    Perform semantic search and get AI response.
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
        limit=5
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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with search interface."""
    return templates.TemplateResponse("index.html", {"request": request})

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
    return templates.TemplateResponse("create_collection.html", {"request": request})

@app.post("/create-collection")
async def create_collection(request: Request, collection_name: str = Form(...), vector_size: int = Form(384)):
    # Check if collection already exists
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name in existing_collections:
        msg = f"Collection '{collection_name}' already exists."
        return templates.TemplateResponse("create_collection.html", {"request": request, "message": msg, "success": False})
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    )
    msg = f"Collection '{collection_name}' created successfully!"
    return templates.TemplateResponse("create_collection.html", {"request": request, "message": msg, "success": True})

@app.get("/add-document", response_class=HTMLResponse)
async def add_document_form(request: Request):
    # Get all collections for dropdown
    collections = [c.name for c in client.get_collections().collections]
    return templates.TemplateResponse("add_document.html", {"request": request, "collections": collections})

@app.post("/add-document")
async def add_document(request: Request, collection_name: str = Form(...), text: str = Form(...), category: str = Form(...)):
    # Embed text
    embedding = embedder.encode(text)
    # Get next ID
    points = client.scroll(collection_name=collection_name, limit=100, with_payload=False, with_vectors=False)
    if points and points[0]:
        max_id = max(int(point.id) for point in points[0])
        next_id = max_id + 1
    else:
        next_id = 1
    # Upsert document
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
    # Get all collections for dropdown
    collections = [c.name for c in client.get_collections().collections]
    return templates.TemplateResponse("add_document.html", {"request": request, "collections": collections, "message": msg, "success": True})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 