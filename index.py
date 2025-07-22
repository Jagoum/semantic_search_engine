"""
This script demonstrates how to:
- Connect to a Qdrant vector database and ensure a collection exists for storing vector embeddings.
- Generate embeddings using the BAAI/bge-small-en-v1.5 model (via sentence-transformers).
- Search embeddings in Qdrant using vector similarity.
- Query Groq AI's LLM using vector search results as context to answer questions about building web services with vector embeddings.

Requirements:
- Qdrant cloud instance and API key
- Groq AI API key
- .env file with all secrets (see below)
- requests, qdrant_client, sentence-transformers, groq Python packages

.env file example:
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
"""

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

from qdrant_client import QdrantClient, models

# Initialize Qdrant client
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Ask user for collection name
collection_name = input("Enter the Qdrant collection name to search: ").strip()
if not collection_name:
    print("Collection name is required. Exiting.")
    exit(1)

model_name = "BAAI/bge-small-en-v1.5"  # Embedding model name

# Ensure the collection exists in Qdrant
if collection_name not in [c.name for c in client.get_collections().collections]:
    print(f"\nCollection '{collection_name}' does not exist. Please run index_qdrant.py first to create and populate the collection.")
    exit(1)
else:
    print(f"\nCollection '{collection_name}' exists and is ready for search.")

# --- Embedding model integration ---
from sentence_transformers import SentenceTransformer

# Load the embedding model
embedder = SentenceTransformer(model_name)

# Example: Search for similar documents
# query_text = "How can I use vector search for semantic retrieval?"

query_text = input("Enter your query: ")
query_embedding = embedder.encode(query_text)

search_result = client.search(
    collection_name=collection_name,
    query_vector=query_embedding.tolist(),
    limit=7
)
print(f"\nSearch results for: '{query_text}'")
print("=" * 50)
for i, hit in enumerate(search_result, 1):
    print(f"{i}. Score: {hit.score:.4f}")
    if hit.payload:
        print(f"   Text: {hit.payload.get('text', 'N/A')}")
        print(f"   Category: {hit.payload.get('category', 'N/A')}")
    print()

# --- Groq AI integration ---
from groq import Groq

groq_client = Groq(api_key=GROQ_API_KEY)

# def query_groq(prompt):
#     """
#     Query Groq AI's LLM to answer a prompt.

#     Args:
#         prompt (str): The input prompt/question for the model.

#     Returns:
#         str: The generated response from the model.
#     """
#     response = groq_client.chat.completions.create(
#         model="llama3-8b-8192",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=512,
#         temperature=0.7,
#     )
#     content = response.choices[0].message.content
#     print(content)
#     return content

# # def query_groq_with_context(prompt, search_results, max_tokens=200):
# #     """
# #     Query Groq AI's LLM using vector search results as context.

# #     Args:
# #         prompt (str): The input prompt/question for the model.
# #         search_results (list): Results from vector search in Qdrant.
# #         max_tokens (int): Maximum tokens for the response.

# #     Returns:
# #         str: The generated response from the model.
# #     """
# #     # Build context from search results
# #     context = "Based on the following relevant documents:\n\n"
# #     for i, hit in enumerate(search_results, 1):
# #         if hit.payload:
# #             context += f"{i}. {hit.payload.get('text', 'N/A')}\n"
    
# #     # Create the full prompt with context
# #     full_prompt = f"{context}\n\nQuestion: {prompt}\n\nAnswer based on the documents above:"
    
# #     response = groq_client.chat.completions.create(
# #         model="llama3-8b-8192",
# #         messages=[{"role": "user", "content": full_prompt}],
# #         max_tokens=max_tokens,
# #         temperature=0.3,  # Lower temperature for more focused responses
# #     )
# #     content = response.choices[0].message.content
# #     print("=" * 50)
# #     print("GROQ RESPONSE (using vector search results):")
# #     print("=" * 50)
# #     print(content)
# #     print("=" * 50)
# #     return content

# # # Example prompt for the language model
# prompt = input("Enter your prompt: ")
# Query the Groq AI model using vector search results as context
# query_groq_with_context(prompt, search_result)

# prompt = """
# What tools should I need to use to build a web service using vector embeddings for search?
# """
# print(f"Prompt: {prompt}")