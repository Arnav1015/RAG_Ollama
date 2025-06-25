import ollama
import os
from typing import List, Dict, Any
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class OllamaRAG:
    def __init__(
        self, 
        model_name: str = "llama3.2",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db"
    ):
        self.model_name = model_name
        
        # Force CPU usage for embeddings to avoid CUDA issues
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        self.persist_directory = persist_directory
        
        # Create vectorstore or load existing one
        if os.path.exists(persist_directory):
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize messages for chat
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately. If the answer isn't in the context, say you don't know."},
        ]
    
    def add_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 20) -> None:
        """
        Load a PDF document, split it into chunks, and add to the vector database.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks (default: 1000)
            chunk_overlap: Overlap between chunks (default: 20)
        """
        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                print(f"Error: PDF file not found at {pdf_path}")
                return
                
            # Load the PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Configure text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " "]
            )
            
            # Split documents into chunks
            texts = text_splitter.split_documents(documents)
            
            # Add metadata to each chunk
            for i, text in enumerate(texts):
                if not text.metadata:
                    text.metadata = {}
                text.metadata["source"] = pdf_path
                text.metadata["chunk"] = i
            
            # Add to vector database
            self.vectorstore.add_documents(texts)
            self.vectorstore.persist()
            
            print(f"Successfully added PDF {pdf_path} to the vector database")
            print(f"Added {len(texts)} chunks from the PDF")
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
    
    def add_text(self, text: str, metadata: Dict[str, Any] = None) -> None:
        """Add text to the vector database after splitting into chunks"""
        chunks = self.text_splitter.split_text(text)
        
        # Add chunks to the vectorstore
        metadatas = [metadata] * len(chunks) if metadata else None
        self.vectorstore.add_texts(chunks, metadatas=metadatas)
        self.vectorstore.persist()
        print(f"Added {len(chunks)} chunks to the vector database")
    
    def add_file(self, file_path: str) -> None:
        """Add content from a file to the vector database"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Error: File not found at {file_path}")
                return
                
            # Check file extension to determine how to process it
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Process PDF files
            if file_extension == '.pdf':
                self.add_pdf(file_path)
            # Process text files
            elif file_extension in ['.txt', '.md', '.csv', '.json']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                metadata = {"source": file_path}
                self.add_text(content, metadata)
                print(f"Successfully added {file_path} to the vector database")
            else:
                print(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            print(f"Error adding file {file_path}: {e}")
    
    def query(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents for a query"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
    
    def format_retrieved_context(self, docs: List) -> str:
        """Format retrieved documents into a context string"""
        context = "Here is relevant information:\n\n"
        
        for i, doc in enumerate(docs):
            context += f"Document {i+1}:\n{doc.page_content}\n"
            if doc.metadata and "source" in doc.metadata:
                context += f"Source: {doc.metadata['source']}\n"
            context += "\n" + "-"*40 + "\n"
        
        return context
    
    def chat(self):
        """Interactive chat interface with RAG capabilities"""
        print(f"Initialized Ollama RAG chatbot with model: {self.model_name}")
        print("Type 'exit' to quit")
        print("Type 'add file [path]' to add a text file")
        print("Type 'add pdf [path]' to add a PDF file")
        
        # Initial greeting
        response = ollama.chat(model=self.model_name, messages=self.messages)
        print("Bot:", response.message.content)
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() == 'exit':
                print("Exiting chat...")
                break
                
            # Command to add a text file to the knowledge base
            if user_input.lower().startswith('add file '):
                file_path = user_input[9:].strip()
                self.add_file(file_path)
                continue
                
            # Command to add a PDF file to the knowledge base
            if user_input.lower().startswith('add pdf '):
                pdf_path = user_input[8:].strip()
                self.add_pdf(pdf_path)
                continue
            
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