import numpy as np
from sentence_transformers import SentenceTransformer, util

model_path = r"D:\New folder\Projects\RAG\EmbeddingModel"
model = SentenceTransformer(model_path)

# ۲. تعریف جملات برای تست هوش مدل
sentences = [
    "می‌خواهم برای هفته آینده مرخصی بگیرم.",     
    "درخواست استراحت و مرخصی دارم.",           
    "فردا هوا آفتابی و گرم خواهد بود."             
]

print("Converting sentences to vectors...")
embeddings = model.encode(sentences)

sim_1_2 = util.cos_sim(embeddings[0], embeddings[1])
sim_1_3 = util.cos_sim(embeddings[0], embeddings[2])

print("\n--- نتیجه تحلیل مدل ---")
print(f"جمله ۱: {sentences[0]}")
print(f"جمله ۲: {sentences[1]}")
print(f"جمله ۳: {sentences[2]}")

print("-" * 30)
print(f"میزان شباهت جمله ۱ و ۲: {sim_1_2.item():.4f} (باید عدد بالایی باشد)")
print(f"میزان شباهت جمله ۱ و ۳: {sim_1_3.item():.4f} (باید عدد پایینی باشد)")

if sim_1_2 > sim_1_3:
    print("\n✅ نتیجه عالی: مدل تفاوت معنایی را به درستی درک کرد!")
else:
    print("\n❌ نتیجه عجیب: مدل در تشخیص معنا دچار خطا شد.")
