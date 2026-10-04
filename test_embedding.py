from sentence_transformers import SentenceTransformer
import os

# آدرس پوشه‌ای که در مرحله قبل مدل را در آن ذخیره کردیم
model_path = r"D:\New folder\Projects\RAG\EmbeddingModel"

print("Loading model from LOCAL folder (Total Offline)...")

try:
    # وقتی آدرس پوشه بدهی، دیگر سراغ اینترنت نمی‌رود
    model = SentenceTransformer(model_path)
    print("Success! Model loaded without internet.")

    sentences = ["این یک تست کاملا آفلاین است."]
    embeddings = model.encode(sentences)
    print(f"Embedding done! Shape: {embeddings.shape}")

except Exception as e:
    print(f"Error: {e}")
