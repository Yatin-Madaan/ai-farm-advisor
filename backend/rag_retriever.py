from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# =========================
# LOAD EMBEDDING MODEL
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# CONNECT TO QDRANT
# =========================

client = QdrantClient(
    url="http://localhost:6333"
)

collection_name = "agriculture_knowledge"


# =========================
# RETRIEVE KNOWLEDGE
# =========================

def retrieve_context(query, top_k=5):

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    ).points

    context_parts = []

    for i, result in enumerate(results, 1):

        text = result.payload["text"]
        source = result.payload["source"]

        context_parts.append(
            f"""
SOURCE {i}
Source: {source}
Relevance score: {result.score:.3f}

{text}
"""
        )

    return "\n".join(context_parts)


# =========================
# TEST
# =========================

query = """
What are the critical growth stages of maize when water availability is limited?
"""

context = retrieve_context(query, top_k=5)

print("\nRETRIEVED CONTEXT")
print("=================\n")
print(context)
