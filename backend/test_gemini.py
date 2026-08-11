from google import genai
import os

# Get API key from environment
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Simple test
response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Explain in one sentence what RAG means in AI."
)

print(response.text)
