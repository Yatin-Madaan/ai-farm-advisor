from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from google import genai
import os


# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_NAME = "agriculture_knowledge"
QDRANT_URL = "http://localhost:6333"
TOP_K = 5

app = FastAPI(
    title="Agricultural RAG API",
    version="1.0"
)


# ============================================================
# LOAD MODELS AND CLIENTS
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Connecting to Qdrant...")

qdrant = QdrantClient(
    url=QDRANT_URL
)

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set"
    )

print("Connecting to Gemini...")

gemini = genai.Client(
    api_key=api_key
)

print("RAG API ready.")


# ============================================================
# INPUT DATA MODEL
# ============================================================

class FarmData(BaseModel):

    field: str | None = None

    field_id: str | None = None

    crop: str

    variety: str | None = None

    growth_stage: str | None = None

    reported_growth_stage: str | None = None

    days_after_planting: int | None = None

    area_ha: float | None = None

    irrigation: str | None = None

    planting_date: str | None = None

    farm_weather_daily: list[dict[str, Any]] = []

    farm_weather_hourly: list[dict[str, Any]] = []


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "service": "Agricultural RAG API"
    }


# ============================================================
# ADVISORY
# ============================================================

@app.post("/advisory")
def advisory(farm: FarmData):

    # --------------------------------------------------------
    # GET CURRENT WEATHER
    # --------------------------------------------------------

    if farm.farm_weather_daily:

        weather = farm.farm_weather_daily[0]

    else:

        weather = {}


    temperature = weather.get(
        "temperature_max"
    )

    rainfall = weather.get(
        "rain_mm"
    )

    rain_probability = weather.get(
        "rain_probability"
    )

    et0 = weather.get(
        "et0"
    )

    wind_speed = weather.get(
        "wind_speed"
    )


    # --------------------------------------------------------
    # BUILD FARM-SPECIFIC RAG QUERY
    # --------------------------------------------------------

    query = f"""
Agricultural management for {farm.crop}
variety {farm.variety}
at the {farm.growth_stage} growth stage,
{farm.days_after_planting} days after planting.

Farm information:
Field: {farm.field}
Field area: {farm.area_ha} hectares
Irrigation available: {farm.irrigation}
Planting date: {farm.planting_date}

Current weather:
Maximum temperature: {temperature} °C
Rainfall: {rainfall} mm
Rain probability: {rain_probability}%
ET0: {et0} mm/day
Wind speed: {wind_speed}

Find relevant agricultural knowledge concerning:
crop water requirements, critical growth stages,
water stress, irrigation, irrigation scheduling,
and crop-specific management.
"""


    # --------------------------------------------------------
    # CREATE QUERY EMBEDDING
    # --------------------------------------------------------

    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()


    # --------------------------------------------------------
    # SEARCH QDRANT
    # --------------------------------------------------------

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K
    ).points


    # --------------------------------------------------------
    # BUILD RETRIEVED EVIDENCE
    # --------------------------------------------------------

    context_parts = []

    evidence_list = []


    for i, result in enumerate(results, 1):

        source = result.payload.get(
            "source",
            "Unknown source"
        )

        text = result.payload.get(
            "text",
            ""
        )

        score = result.score


        context_parts.append(
            f"""
[EVIDENCE {i}]

Source: {source}

Relevance score: {score:.3f}

{text}
"""
        )


        evidence_list.append({

            "evidence_id": i,

            "source": source,

            "relevance_score": round(
                score,
                3
            )

        })


    context = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an agricultural decision-support assistant.

Use the retrieved agricultural evidence as your
primary knowledge source.

FARM INFORMATION
----------------

Field: {farm.field}
Field ID: {farm.field_id}
Crop: {farm.crop}
Variety: {farm.variety}
Growth stage: {farm.growth_stage}
Days after planting: {farm.days_after_planting}
Area: {farm.area_ha} hectares
Irrigation available: {farm.irrigation}
Planting date: {farm.planting_date}

CURRENT WEATHER
---------------

Temperature: {temperature} °C
Rainfall: {rainfall} mm
Rain probability: {rain_probability}%
ET0: {et0} mm/day
Wind speed: {wind_speed}

RETRIEVED AGRICULTURAL EVIDENCE
--------------------------------

{context}


INSTRUCTIONS
------------

1. Use the retrieved evidence as the primary
   agricultural knowledge source.

2. Do not invent agricultural facts, irrigation
   amounts, thresholds or schedules that are not
   supported by the evidence.

3. Clearly distinguish evidence-based information
   from interpretation of the current weather.

4. Do not claim that a statement came from a source
   unless that source supports it.

5. If the evidence is insufficient for a specific
   recommendation, explicitly state that.

6. Do not provide an exact irrigation amount unless
   the evidence supports calculating one.

7. Give a concise and practical assessment.

8. After important evidence-based recommendations,
   cite the evidence number, for example:
   [Evidence 1]

9. Include these sections:

   ASSESSMENT

   RECOMMENDATIONS

   LIMITATIONS
"""


    # --------------------------------------------------------
    # GENERATE GEMINI RESPONSE
    # --------------------------------------------------------

    response = gemini.models.generate_content(

        model="gemini-3.1-flash-lite",

        contents=prompt

    )


    # --------------------------------------------------------
    # RETURN RESULT TO N8N
    # --------------------------------------------------------

    return {

        "status": "success",

        "farm": {

            "field": farm.field,

            "field_id": farm.field_id,

            "crop": farm.crop,

            "variety": farm.variety,

            "growth_stage": farm.growth_stage,

            "days_after_planting":
                farm.days_after_planting

        },

        "weather": {

            "temperature":
                temperature,

            "rainfall":
                rainfall,

            "rain_probability":
                rain_probability,

            "et0":
                et0,

            "wind_speed":
                wind_speed

        },

        "advisory":
            response.text,

        "retrieved_evidence":
            evidence_list

    }
