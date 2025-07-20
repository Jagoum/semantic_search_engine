"""
Interactive script to index (add) data to a Qdrant vector database.
- Optionally create a new collection.
- Prompt user for text and category.
- Embed the text and store it in Qdrant with an auto-assigned ID.
- Uses .env for secrets.
"""

from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

collection_name = input("Enter the Qdrant collection name: ").strip()

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Ask if user wants to create a new collection
if collection_name not in [c.name for c in client.get_collections().collections]:
    create = input(f"Collection '{collection_name}' does not exist. Create it? (y/n): ").strip().lower()
    if create == 'y':
        # Ask for vector size (default 384 for BAAI/bge-small-en-v1.5)
        size = input("Enter vector size (default 384): ").strip()
        size = int(size) if size else 384
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print("Exiting.")
        exit(0)
else:
    print(f"Collection '{collection_name}' already exists.")

# Load embedding model
model_name = "BAAI/bge-small-en-v1.5"
embedder = SentenceTransformer(model_name)

# Get current max ID in collection (if any)
def get_next_id():
    points = client.scroll(collection_name=collection_name, limit=100, with_payload=False, with_vectors=False)
    if points and points[0]:
        max_id = max(int(point.id) for point in points[0])
        return max_id + 1
    return 1

print("\nAdd entries to your vector database. Press Enter with empty text to finish.")

while True:
    text = input("Enter text to add (or just Enter to finish): ").strip()
    if not text:
        print("Done.")
        break
    category = input("Enter category: ").strip()
    if not category:
        print("Category is required. Skipping entry.")
        continue
    embedding = embedder.encode(text)
    id = get_next_id()
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=id,
                vector=embedding.tolist(),
                payload={"text": text, "category": category}
            )
        ]
    )
    print(f"Added entry with ID {id} to collection '{collection_name}'.") 