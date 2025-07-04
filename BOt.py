import ollama
import os
from typing import List, Dict, Any
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import query
class OllamaRAG:
    def __init__(
        self, 
        model_name: str = "llama3.2",
    ):
        self.model_name = model_name
        # Initialize messages for chat
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately. If the answer isn't in the context, say you don't know."},
        ]
        

    def query(self,query_text, k: int = 10) -> List[str]:
        docs=query.query_faiss(query_text, k)
        return docs
    
    def format_retrieved_context(self, docs: list) -> str:
        """Format retrieved tuples into a context string"""
        context = "Here is relevant information:\n\n"
    
        for i, (chunk_text, (filename, chunk_index), distance) in enumerate(docs):
            context += f"Document {i+1} (from {filename}, chunk {chunk_index}):\n"
            context += chunk_text.strip() + "\n"
            context += f"Similarity Score: {distance:.4f}\n"
            context += "-" * 40 + "\n"

        return context
    
    def chat(self):
        """Interactive chat interface with RAG capabilities"""
        print(f"Initialized Ollama RAG chatbot with model: {self.model_name}")
        print("Type 'exit' to quit")
        
        # Initial greeting
        response = ollama.chat(model=self.model_name, messages=self.messages)
        print("Bot:", response.message.content)
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() == 'exit':
                print("Exiting chat...")
                break
            # Retrieve relevant context from vector database
            docs = self.query(user_input)
            context = self.format_retrieved_context(docs)
            
            # Create RAG prompt with the retrieved context
            rag_prompt = f"""
Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, say you don't know.

Context:
{context}

Question: {user_input}

Answer:
"""
            
            # Add the RAG prompt to messages
            self.messages.append({"role": "user", "content": rag_prompt})
            
            # Get response from Ollama
            response = ollama.chat(model=self.model_name, messages=self.messages)
            answer = response.message.content
            print("Bot:", answer)
            
            # Update message history
            # Remove the RAG prompt and replace with original question and answer
            self.messages.pop()  # Remove the RAG prompt
            self.messages.append({"role": "user", "content": user_input})
            self.messages.append({"role": "assistant", "content": answer})


def ollama_chat():
    """Main function to start the RAG-enabled Ollama chat"""
    rag = OllamaRAG(model_name="llama3.2")
    rag.chat()


# Run the chat interface if executed as a script
if __name__ == "__main__":
    ollama_chat()