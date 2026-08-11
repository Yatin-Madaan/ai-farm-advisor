# AI Farm Advisor 🌾

An AI-powered agricultural decision-support system that combines farm information, weather data, retrieval-augmented generation (RAG), and AI reasoning to produce structured field recommendations.

## Overview

AI Farm Advisor is an automation-based agricultural advisory prototype built with **n8n**, a **FastAPI AI/RAG backend**, **Qdrant**, **Sentence Transformers**, and **Google Gemini**.

The system collects farm and crop information from a farmer, retrieves location-specific weather data, processes weather indicators, and sends the combined farm and weather context to an AI advisory backend.

The generated response is then formatted into a structured agricultural advisory.

## System Architecture

![AI Farm Advisor System Architecture](docs/architecture.png)

```text
Farmer
   │
   ▼
n8n Recommendation Form
   │
   ├── Farm & Crop Information
   │
   ▼
Location Lookup
   │
   ▼
Weather API
   │
   ▼
Weather Processing
   │
   ├── Temperature
   ├── Rainfall
   ├── Rain Probability
   ├── Wind Speed
   ├── ET₀
   ├── Heat Risk
   ├── Rain Risk
   ├── Spray Suitability
   └── Water Demand
   │
   ▼
Farm Context + Weather Context
   │
   ▼
FastAPI AI Advisory Backend
   │
   ├── Sentence Transformers
   │
   ├── Qdrant Vector Database
   │
   └── Google Gemini
   │
   ▼
Structured AI Advisory
   │
   ├── Field Assessment
   ├── Agricultural Recommendations
   ├── Weather-aware Recommendations
   ├── Evidence References
   └── Limitations
   │
   ▼
n8n Output / Database
