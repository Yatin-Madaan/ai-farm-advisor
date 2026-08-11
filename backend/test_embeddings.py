from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Maize requires careful water management during crop development."

embedding = model.encode(text)

print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])
