from datasets import load_dataset

dataset = load_dataset("Rady10/Agriculture-Rag-Data-Index")

print(dataset)

train = dataset["train"]

print("Number of records:", len(train))
print("Columns:", train.column_names)

for i in range(3):
    print("\n--- RECORD", i, "---")
    print(train[i])
