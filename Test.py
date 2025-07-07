# Create a simple test script to verify BOt.py works
from Bot import OllamaRAG
import ollama

rag = OllamaRAG(model_name="llama3.2")
docs = rag.query("Is ollama working?")
context = rag.format_retrieved_context(docs)

rag_prompt = f"""
Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, say you don't know.

Context:
{context}

Question: {"Is ollama working?"}

Answer:
"""
# Add the RAG prompt to messages
rag.messages.append({"role": "user", "content": rag_prompt})
# Get response from Ollama
response = rag.ollama.chat(model=rag.model_name, messages=rag.messages)
answer = response.message.content
print("Bot:", answer)