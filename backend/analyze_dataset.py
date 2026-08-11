from datasets import load_dataset
from collections import Counter

dataset = load_dataset("Rady10/Agriculture-Rag-Data-Index")
train = dataset["train"]

texts = train["text"]
sources = train["source"]
domains = train["domain"]

print("Total records:", len(train))

# Text lengths
lengths = [len(text) for text in texts]

print("\nText length:")
print("Minimum:", min(lengths))
print("Maximum:", max(lengths))
print("Average:", sum(lengths) / len(lengths))

# Empty / very short records
short_records = sum(1 for text in texts if len(text.strip()) < 100)

print("\nRecords shorter than 100 characters:", short_records)

# Sources
print("\nSources:")
for source, count in Counter(sources).most_common():
    print(f"{source}: {count}")

# Domains
print("\nDomains:")
for domain, count in Counter(domains).most_common():
    print(f"{domain}: {count}")
