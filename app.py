from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print("Downloading model... This may take a while on first run.")
model = SentenceTransformer(model_name)
print("Done. Model is cached.")