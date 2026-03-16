import json
with open(r"C:\Users\Mohamed\Desktop\better_call_maro\rag_project\llama_training_data\full_dataset_fixed.json", encoding="utf-8") as f:
    data = json.load(f)
remaining = sum(1 for e in data if "Consult the relevant article to restructure" in e.get("output",""))
print(f"Remaining generic entries: {remaining}")
print(f"Total entries: {len(data)}")