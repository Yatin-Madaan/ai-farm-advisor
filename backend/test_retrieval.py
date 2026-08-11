from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

collection_name = "agriculture_knowledge"

# Test agricultural question
query = """
What are the critical growth stages of maize when water availability is limited?
"""

# Convert query to embedding
query_vector = model.encode(
    query,
    normalize_embeddings=True
).tolist()

# Search Qdrant
results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=5
).points

# Display results
print("\nTOP RETRIEVED RESULTS\n")

for i, result in enumerate(results, 1):
    print(f"\n--- RESULT {i} ---")
    print("Score:", result.score)
    print("Source:", result.payload["source"])
    print("Domain:", result.payload["domain"])
    print("Text:")
    print(result.payload["text"][:1000])
