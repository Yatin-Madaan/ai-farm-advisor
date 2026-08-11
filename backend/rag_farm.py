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
# FARM CONTEXT
# =========================

farm = {
    "crop": "maize",
    "growth_stage": "flowering",
    "days_after_planting": 66,
    "temperature": 33.5,
    "et0": 5.16,
    "rain_probability": 10,
    "rainfall": 0,
    "irrigation": "yes"
}


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
# BUILD DYNAMIC RAG QUERY
# =========================

question = f"""
Agricultural water management for {farm["crop"]}
at the {farm["growth_stage"]} growth stage,
{farm["days_after_planting"]} days after planting.

Current weather:
Temperature: {farm["temperature"]} °C
ET0: {farm["et0"]} mm/day
Rain probability: {farm["rain_probability"]}%
Rainfall: {farm["rainfall"]} mm

Find relevant agricultural knowledge about:
crop water requirements, critical growth stages,
water stress, irrigation and irrigation scheduling.
"""


print("\nRAG QUERY")
print("=========\n")
print(question)


# =========================
# CREATE QUERY EMBEDDING
# =========================

query_vector = embedding_model.encode(
    question,
    normalize_embeddings=True
).tolist()


# =========================
# RETRIEVE KNOWLEDGE
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
# GEMINI PROMPT
# =========================

prompt = f"""
You are an agricultural decision-support assistant.

Use the retrieved agricultural knowledge as your
primary evidence.

FARM INFORMATION:
Crop: {farm["crop"]}
Growth stage: {farm["growth_stage"]}
Days after planting: {farm["days_after_planting"]}

WEATHER:
Temperature: {farm["temperature"]} °C
ET0: {farm["et0"]} mm/day
Rain probability: {farm["rain_probability"]}%
Rainfall: {farm["rainfall"]} mm
Irrigation available: {farm["irrigation"]}

RETRIEVED AGRICULTURAL KNOWLEDGE:
{context}

TASK:

Explain the water-management considerations for this
specific farm situation.

Rules:

1. Use the retrieved knowledge as the primary source.
2. Do not invent specific agricultural facts.
3. Clearly distinguish information from the retrieved
   knowledge from interpretation of the current weather.
4. Do not give an exact irrigation amount unless the
   retrieved knowledge supports it.
5. If the retrieved knowledge is insufficient, say so.
6. Keep the recommendation practical and concise.
7. Mention the relevant source(s).
"""


# =========================
# GENERATE ANSWER
# =========================

response = gemini.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=prompt
)


# =========================
# DISPLAY ANSWER
# =========================

print("\n==============================")
print("FARM-SPECIFIC RAG ANSWER")
print("==============================\n")

print(response.text)


# =========================
# DISPLAY SOURCES
# =========================

print("\n==============================")
print("RETRIEVED SOURCES")
print("==============================\n")

for i, result in enumerate(results, 1):

    print(
        f"{i}. {result.payload['source']} "
        f"(score: {result.score:.3f})"
    )
