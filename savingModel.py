from sentence_transformers import SentenceTransformer

print("Downloading the Model.")

Model = SentenceTransformer('intfloat/multilingual-e5-small')
Model.save('./localEmbeddingModel')
print("Model is now successfully saved in './LocalEmbeddingModel' folder!")


