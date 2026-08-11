from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from google import genai
import os


# =========================
# CONFIGURATION
# =========================

COLLECTION_NAME = "agriculture_knowledge"
QDRANT_URL = "http://localhost:6333"
TOP_K = 5


# =========================
# LOAD EMBEDDING MODEL
# =========================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================
# CONNECT TO QDRANT
# =========================

qdrant = QdrantClient(
    url=QDRANT_URL
)


# =========================
# CONNECT TO GEMINI
# =========================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set"
    )

gemini = genai.Client(
    api_key=api_key
)


# =========================
# USER QUESTION
# =========================

question = """
What are the critical growth stages of maize
when water availability is limited?
"""


# =========================
# CREATE QUERY EMBEDDING
# =========================

query_vector = embedding_model.encode(
    question,
    normalize_embeddings=True
).tolist()


# =========================
# RETRIEVE RELEVANT KNOWLEDGE
# =========================

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=TOP_K
).points


# =========================
# BUILD CONTEXT
# =========================

context_parts = []

for i, result in enumerate(results, 1):

    context_parts.append(
        f"""
SOURCE {i}
Source: {result.payload["source"]}
Relevance score: {result.score:.3f}

{result.payload["text"]}
"""
    )


context = "\n".join(context_parts)


# =========================
# CREATE RAG PROMPT
# =========================

prompt = f"""
You are an agricultural decision-support assistant.

Answer the user's question using the agricultural
knowledge provided in the CONTEXT below.

Important rules:

1. Use the provided context as your primary knowledge source.
2. Do not invent agricultural facts that are not supported
   by the context.
3. If the context does not contain enough information,
   clearly say that the available knowledge is insufficient.
4. Give a concise, practical answer.
5. Mention the relevant source when appropriate.

USER QUESTION:
{question}

CONTEXT:
{context}
"""


# =========================
# GENERATE ANSWER
# =========================

response = gemini.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=prompt
)


# =========================
# DISPLAY RESULT
# =========================

print("\n==============================")
print("RAG ANSWER")
print("==============================\n")

print(response.text)

print("\n==============================")
print("SOURCES RETRIEVED")
print("==============================\n")

for i, result in enumerate(results, 1):

    print(
        f"{i}. {result.payload['source']} "
        f"(score: {result.score:.3f})"
    )
