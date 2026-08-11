from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid

# Load dataset
dataset = load_dataset("Rady10/Agriculture-Rag-Data-Index")
records = dataset["train"]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

collection_name = "agriculture_knowledge"

batch_size = 100

for start in range(0, len(records), batch_size):

    batch = records[start:start + batch_size]

    texts = batch["text"]

    # Generate embeddings
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    points = []

    for i, embedding in enumerate(embeddings):

        payload = {
            "text": batch["text"][i],
            "source": batch["source"][i],
            "domain": batch["domain"][i]
        }

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=payload
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(
        f"Uploaded {min(start + batch_size, len(records))}"
        f"/{len(records)} records"
    )

print("Ingestion complete.")
