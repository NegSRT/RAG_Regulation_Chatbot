import ollama

response = ollama.chat(model='qwen2.5:7b', messages=[
  {
    'role': 'user',
    'content': 'سلام، تو کی هستی؟',
  },
])

print(response['message']['content'])
