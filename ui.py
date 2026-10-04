import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
import pandas as pd
import os
from datetime import datetime


#Saving feedback in excel
def save_feedback_to_excel(question, response, feedback_type, sources):
    file_name = "feedback_log.xlsx"
    
    #Acquiring the data
    new_data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "User Question": [question],
        "Model Response": [response],
        "Feedback Type": [feedback_type],  # 0 = Dislike, 1 = Like
        "Sources": [" | ".join(sources) if sources else "بدون منبع"]
    }
    
    df_new = pd.DataFrame(new_data)  # Appending new data row
    
    # Adding new data to an excel that has existed before
    if os.path.exists(file_name):
        try:
            df_existing = pd.read_excel(file_name)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final.to_excel(file_name, index=False)
        # In case if any error happens
        except Exception as e:
            st.error(f"خطا در نوشتن فایل اکسل: {e}")
    else:
        df_new.to_excel(file_name, index=False)




# Responsive page configuration
st.set_page_config(page_title="چت‌بات آیین‌نامه", page_icon="📚", layout="wide")
st.title("📚 چت‌بات هوشمند آیین‌نامه")
st.caption("سوالات خود را بر اساس آیین‌نامه بپرسید...")

# Making of the setting sidebar
with st.sidebar:
    st.header("⚙️ تنظیمات سیستم")
    st.write("---")
    
    if st.button("🗑️ پاک کردن تاریخچه گفتگو", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_sources = []
        st.toast("تاریخچه گفتگو با موفقیت پاک شد!", icon="✨")




#Loading Models
@st.cache_resource
def load_system():
    #asks for search
    embed_model = SentenceTransformer('./localEmbeddingModel')
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="regulation_chunks")
    return embed_model, collection

embedding_model, chroma_collection = load_system()




if "messages" not in st.session_state:
    st.session_state.messages = []
    
# to avoid saving the sources model in the excel if the source does not exist
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []

# showing history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# The deduction part
def stream_generator(final_prompt, sources_list):
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:7b",
        "prompt": final_prompt,
        "stream": True 
    }
    
    try:
        response = requests.post(ollama_url, json=payload, stream=True)
        response.raise_for_status()
        
        full_generated_text = ""
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                word = chunk.get("response", "")
                full_generated_text += word
                yield word 
                
        if sources_list and "یافت نشد" not in full_generated_text:
            sources_text = "\n\n**منابع استخراج شده:**\n"
            for s in set(sources_list):
                sources_text += f"- {s}\n"
            yield sources_text 
            
    except Exception as e:
        yield f"\n\nخطا در ارتباط با مدل Ollama: {e}"


if prompt := st.chat_input("سوال خود را اینجا بنویسید..."):
    # showing user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # retrieval mode
    with st.chat_message("assistant"):
        with st.spinner("در حال جستجو در آیین‌نامه..."):
            question_embedding = embedding_model.encode(prompt).tolist()
            
            results = chroma_collection.query(
                query_embeddings=[question_embedding],
                n_results=5
            )
            
            contexts = []
            sources = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i]
                    contexts.append(doc)
                    
                    article = meta.get('Article', 'نامشخص')
                    part = meta.get('Part', 'نامشخص')
                    page = meta.get('Page', 'نامشخص')
                    sources.append(f"ماده {part}، بخش {article} (صفحه {page})")
            
            # ذخیره منابع در Session State برای سیستم بازخورد
            st.session_state.current_sources = list(set(sources))
            
            context_text = "\n\n---\n\n".join(contexts)
            
            system_prompt = f"""شما یک دستیار هوشمند، دقیق و فارسی‌زبان برای پاسخگویی به سوالات بر اساس آیین‌نامه هستید.
            لطفاً با استفاده از متن‌های زیر به سوال کاربر پاسخ دهید.
            اگر پاسخ در متن زیر نیست، فقط بگو "بر اساس اطلاعات آیین‌نامه، پاسخی برای این سوال یافت نشد." و از خودت چیزی نساز.
            اگر ازت خواستن ماده ای رو تعریف کنی یا بگی ماده‌ی فلان چیست، از همون آیین نامه متن خود ماده رو نشون بده.
            متن آیین‌نامه:
            {context_text}
            """
            
            final_prompt_to_send = f"{system_prompt}\n\nسوال کاربر: {prompt}"

        response_placeholder = st.empty()
        full_response = ""
        
        for new_word in stream_generator(final_prompt_to_send, sources):
            full_response += new_word
            response_placeholder.markdown(full_response + "▌") 
            
        response_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})




#The feedback system

if len(st.session_state.messages) >= 2 and st.session_state.messages[-1]["role"] == "assistant":
    st.write("") 
    col1, col2, col3 = st.columns([1, 1, 10]) 
    
    # extracting the question and answer
    last_user_question = st.session_state.messages[-2]["content"]
    last_assistant_response = st.session_state.messages[-1]["content"]
    last_used_sources = st.session_state.current_sources
    
    with col1:
        if st.button("👍", key=f"good_{len(st.session_state.messages)}", help="پاسخ مفید بود"):
            save_feedback_to_excel(last_user_question, last_assistant_response, 1, last_used_sources)
            st.toast("ممنون از بازخورد مثبت شما!", icon="🙏")
            
    with col2:
        if st.button("👎", key=f"bad_{len(st.session_state.messages)}", help="پاسخ اشتباه یا ناقص بود"):
            save_feedback_to_excel(last_user_question, last_assistant_response, 0, last_used_sources)
            st.toast("ممنون! این بازخورد به بهبود سیستم کمک می‌کند.", icon="🛠️")