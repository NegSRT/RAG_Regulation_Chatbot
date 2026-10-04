import os
from pathlib import Path

# مسیر پیش‌فرض کش Hugging Face در ویندوز
cache_path = Path(os.environ.get('USERPROFILE')) / ".cache" / "huggingface" / "hub"

print(f"مدل‌های شما احتمالا در این مسیر هستند:\n{cache_path}")

# لیست کردن پوشه‌های موجود برای پیدا کردن مدل
if cache_path.exists():
    folders = os.listdir(cache_path)
    print("\nپوشه‌های پیدا شده:")
    for f in folders:
        print(f"- {f}")
else:
    print("\nمسیر کش پیدا نشد.")
