import json

def analyze_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"تعداد کل چانک‌ها: {len(data)}")
    
    # بررسی نویز و خطاهای ساختاری
    errors = []
    for idx, item in enumerate(data):
        # بررسی کلیدهای تکراری یا تایپی
        if 'Parent_article' not in item and 'Parent_ticle' not in item:
            errors.append(f"ردیف {idx}: فاقد کلید Parent_article است.")
        
        # بررسی فیلد Full_Chunk
        if not item.get('Full_Chunk') or len(item['Full_Chunk']) < 20:
            errors.append(f"ردیف {idx}: متن Full_Chunk کوتاه یا خالی است.")
            
    if errors:
        print(f"تعداد خطاها: {len(errors)}")
        for err in errors[:3000]: print(f"- {err}") # چاپ ۵ خطای اول
    else:
        print("داده‌ها از نظر ساختار اصلی تمیز هستند!")

# فرض کنیم فایل شما data.json است
analyze_json(r"D:\New folder\Projects\RAG\regulationChunk.json")
