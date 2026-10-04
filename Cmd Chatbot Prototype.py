import chromadb
from sentence_transformers import SentenceTransformer
import ollama

print("Loading embedding model...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="regulation_chunks")

def ask_rag(question):
    print(f"\n🔍 در حال جستجو برای: '{question}'...")
    
    question_embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )


    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    context_text = ""
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        context_text += f"--- منبع {i+1} (ماده {meta.get('Article_number', 'نامشخص')}، بخش {meta.get('Section', 'نامشخص')}) ---\n{doc}\n\n"

    prompt = f"""شما یک دستیار هوشمند و دقیق برای پاسخگویی به سوالات بر اساس آیین‌نامه هستید.
لطفاً فقط و فقط با استفاده از متن آیین‌نامه زیر به سوال کاربر پاسخ دهید. به زبان فارسی روان پاسخ دهید.
اگر جواب سوال در متن زیر وجود ندارد، صراحتاً بگو "بر اساس اطلاعات آیین‌نامه، پاسخی برای این سوال یافت نشد" و از خودت چیزی نساز.

متن آیین‌نامه:
{context_text}

سوال کاربر: {question}
"""

    print("🧠 در حال تولید پاسخ توسط Qwen...\n")
    response = ollama.chat(model='qwen2.5-coder:7b', messages=[
        {'role': 'user', 'content': prompt}
    ])

    # چاپ پاسخ نهایی
    print("="*60)
    print("🤖 پاسخ چت‌بات:")
    print(response['message']['content'])
    print("="*60)
    print("📚 منابع استفاده شده (از دیتابیس):")
    for meta in metadatas:
        print(f"- ماده: {meta.get('Article_number', 'نامشخص')} | بخش: {meta.get('Section', 'نامشخص')} | صفحه: {meta.get('Page_start', 'نامشخص')}")
    print("="*60)

if __name__ == "__main__":
    user_question = "شرایط مرخصی تحصیلی برای دانشجویان چیست؟"
    ask_rag(user_question)
