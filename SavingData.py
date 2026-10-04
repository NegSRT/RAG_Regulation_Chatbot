import json
import chromadb
from sentence_transformers import SentenceTransformer

#Getting Embedding model from the local folder
print("Loading embedding model...")
model = SentenceTransformer('./localEmbeddingModel')

#Creating Database space
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")

# Loading the JSON data file
collection = client.get_or_create_collection(name="regulation_chunks")

print("Loading JSON data...")

with open('regulationChunk.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

#making lists to append
documents = []
metadatas = []
ids = []

#Extracting data
for item in data:
    part_name = item.get('Article_number', '')
    article_name = item.get('Section', '')
    chunk_text = item.get('Full_Chunk', '')

    #Enriching the texts from the info
    enriched_text = f"موضوع: {part_name} - {article_name}\nمتن اصلی: {chunk_text}"

    #Adding to Documents list
    documents.append(enriched_text)
    
    # ui metadata list
    metadatas.append({
        "Part": part_name,
        "Article": article_name,
        "Page": str(item.get('Page_start', ''))
    })
    ids.append(str(item['id']))

# Encoding data information chunks
print(f"Creating embeddings for {len(documents)} chunks...")
embeddings = model.encode(documents).tolist()

# Saving in ChromaDB
print("Saving to ChromaDB...")

collection.add(
    #The data information
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents
)

print("Vector database was successfully made!")
print(f"the record numbers {len(collection)}")