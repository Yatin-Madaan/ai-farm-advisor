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
    raise RuntimeError("GEMINI_API_KEY is not set")

gemini = genai.Client(
    api_key=api_key
)


# =========================
# BUILD QUERY
# =========================

query = f"""
Agricultural water management for {farm["crop"]}
at the {farm["growth_stage"]} growth stage,
{farm["days_after_planting"]} days after planting.

Weather:
Temperature: {farm["temperature"]} °C
ET0: {farm["et0"]} mm/day
Rain probability: {farm["rain_probability"]}%
Rainfall: {farm["rainfall"]} mm

Find agricultural knowledge about crop water
requirements, critical growth stages, water stress,
irrigation and irrigation scheduling.
"""


# =========================
# EMBEDDING
# =========================

query_vector = embedding_model.encode(
    query,
    normalize_embeddings=True
).tolist()


# =========================
# RETRIEVAL
# =========================

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=TOP_K
).points


# =========================
# BUILD NUMBERED CONTEXT
# =========================

context_parts = []

for i, result in enumerate(results, 1):

    source = result.payload.get("source", "Unknown source")
    text = result.payload.get("text", "")

    context_parts.append(
        f"""
[EVIDENCE {i}]
Source: {source}
Relevance score: {result.score:.3f}

{text}
"""
    )

context = "\n".join(context_parts)


# =========================
# GEMINI PROMPT
# =========================

prompt = f"""
You are an agricultural decision-support assistant.

You must answer using the EVIDENCE provided below.

FARM DATA
---------
Crop: {farm["crop"]}
Growth stage: {farm["growth_stage"]}
Days after planting: {farm["days_after_planting"]}

WEATHER
-------
Temperature: {farm["temperature"]} °C
ET0: {farm["et0"]} mm/day
Rain probability: {farm["rain_probability"]}%
Rainfall: {farm["rainfall"]} mm
Irrigation available: {farm["irrigation"]}

EVIDENCE
--------
{context}

INSTRUCTIONS
------------

1. Use the evidence as the primary agricultural knowledge source.

2. Do not invent agricultural facts, irrigation amounts,
   thresholds or schedules that are not supported by the evidence.

3. Clearly distinguish between:
   - Evidence from the agricultural sources
   - Interpretation of the current farm/weather conditions

4. Do not claim that a statement came from a source unless
   that source actually supports it.

5. If the evidence is insufficient for a specific recommendation,
   explicitly say that the available evidence is insufficient.

6. Do not provide an exact irrigation amount unless the evidence
   supports calculating one.

7. Give a concise practical assessment.

8. After each important evidence-based recommendation,
   include the corresponding evidence number in brackets,
   for example [Evidence 1].

9. Finish with a section called:
   LIMITATIONS

   Explain briefly what cannot be determined from the available
   farm data and retrieved evidence.
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

print("\n")
print("=" * 60)
print("FARM ADVISORY")
print("=" * 60)

print(f"\nCrop: {farm['crop']}")
print(f"Growth stage: {farm['growth_stage']}")
print(f"DAP: {farm['days_after_planting']}")

print(f"\nTemperature: {farm['temperature']} °C")
print(f"ET0: {farm['et0']} mm/day")
print(f"Rain probability: {farm['rain_probability']}%")
print(f"Rainfall: {farm['rainfall']} mm")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60 + "\n")

print(response.text)


# =========================
# DISPLAY RETRIEVED EVIDENCE
# =========================

print("\n" + "=" * 60)
print("RETRIEVED EVIDENCE")
print("=" * 60)

for i, result in enumerate(results, 1):

    source = result.payload.get(
        "source",
        "Unknown source"
    )

    print(
        f"\n[Evidence {i}] "
        f"{source} "
        f"(relevance: {result.score:.3f})"
    )
