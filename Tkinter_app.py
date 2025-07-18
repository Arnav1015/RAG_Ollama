import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
import os
import time
import traceback
import ollama  
import re 
import json
import pandas as pd
import chromadb

from Vector_Database_ChromaDB import ChromaVectorStore
from Embedd import process_file, get_embedding_model, process_folder

# Print ChromaDB version
print(f"ChromaDB version: {chromadb.__version__}")

# Soothing color palette
soothing_colors = {
    'background': '#f5f7fa',       # Soft light gray-blue
    'text_area': "#ffffff",        # Pure white 
    'text_color': '#3a4a5a',       # Soft dark blue-gray
    'accent': '#7eb6bd',           # Calm teal
    'accent_hover': '#6ea5ad',     # Slightly deeper teal
    'secondary': '#d3e5eb',        # Very soft blue
    'highlight': '#f2d9c2',        # Warm beige
    'success': "#61dc75",          # Soft green
    'border': '#c0d6df'            # Muted blue-gray
}

class ChromaRAG:
    """RAG system using ChromaDB for vector storage and retrieval"""
    
    def __init__(self, model_name: str = "llama3.2", collection_name: str = "hybrid_data"):
        self.model_name = model_name
        self.vector_store = ChromaVectorStore(collection_name=collection_name, persist_directory="chroma_db")
        # Initialize messages for chat
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately. If the answer isn't in the context, say you don't know."},
        ]
    
    def query(self, query_text: str, k: int = 5) -> list:
        """Query the ChromaDB vector store"""
        try:
            # Get embedding model and encode query
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.encode([query_text])[0].tolist()
            
            # Query the vector store
            results = self.vector_store.query(query_embedding, top_k=k)
            
            # Format results similar to the old system
            docs = []
            if results and 'documents' in results and results['documents']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # Extract filename and chunk info from metadata
                    filename = metadata.get('filename', 'unknown')
                    chunk_id = metadata.get('chunk', i)
                    docs.append((doc, (filename, chunk_id), distance))
            
            return docs
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []
    
    def format_retrieved_context(self, docs: list) -> str:
        """Format retrieved documents into a context string"""
        context = "Here is relevant information:\n\n"
        
        for i, (chunk_text, (filename, chunk_index), distance) in enumerate(docs):
            context += f"Document {i+1} (from {filename}, chunk {chunk_index}):\n"
            context += chunk_text.strip() + "\n"
            context += f"Similarity Score: {distance:.4f}\n"
            context += "-" * 40 + "\n"
        
        return context

class ChatWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('AI-RAG Assistant')
        self.root.geometry('1200x700')
        self.root.configure(bg=soothing_colors['background'])
        
        # Initialize RAG system
        self.rag_system = ChromaRAG(model_name="llama3.2")
        self.structured_data = None
        self.current_status = "Status: Ready"
        self.is_minimized = False
        
        # Setup window state monitoring
        self.setup_window_state_monitoring()
        
        # Create the UI
        self.create_widgets()
        
        # Configure styles
        self.configure_styles()
    
    def setup_window_state_monitoring(self):
        """Setup window state monitoring for minimized status"""
        def on_window_state_change(event):
            if event.widget == self.root:
                if self.root.state() == 'iconic':  # Minimized
                    self.is_minimized = True
                    self.root.title(f'AI-RAG Assistant - {self.current_status}')
                else:  # Normal or maximized
                    self.is_minimized = False
                    self.root.title('AI-RAG Assistant')
        
        self.root.bind('<Map>', on_window_state_change)
        self.root.bind('<Unmap>', on_window_state_change)
    
    def update_status_with_title(self, status_text):
        """Update status and window title if minimized"""
        self.current_status = status_text
        self.status_label.configure(text=status_text)
        
        if self.is_minimized:
            self.root.title(f'AI-RAG Assistant - {status_text}')
    
    def configure_styles(self):
        """Configure ttk styles for consistent theming"""
        style = ttk.Style()
        
        # Configure button style with better visibility
        style.configure("Custom.TButton",
                       background=soothing_colors['accent'],
                       foreground=soothing_colors['text_color'],  # Dark text for visibility
                       font=('Segoe UI', 10, 'bold'),
                       relief='flat',
                       borderwidth=0,
                       padding=(10, 5))
        
        # Configure button hover and focus states
        style.map("Custom.TButton",
                 background=[('active', soothing_colors['accent_hover']),
                            ('pressed', soothing_colors['accent_hover']),
                            ('focus', soothing_colors['accent'])],
                 foreground=[('active', 'white'),
                            ('pressed', 'white'),
                            ('focus', soothing_colors['text_color'])],
                 relief=[('pressed', 'flat'),
                        ('focus', 'solid')])
        
        # Configure frame style
        style.configure("Custom.TFrame",
                       background=soothing_colors['background'],
                       relief='flat',
                       borderwidth=0)
        
        # Configure label style
        style.configure("Custom.TLabel",
                       background=soothing_colors['background'],
                       foreground=soothing_colors['text_color'],
                       font=('Segoe UI', 10))
        
        # Configure labelframe style
        style.configure("Custom.TLabelframe",
                       background=soothing_colors['background'],
                       foreground=soothing_colors['text_color'],
                       font=('Segoe UI', 11, 'bold'),
                       relief='solid',
                       borderwidth=1)
        
        style.configure("Custom.TLabelframe.Label",
                       background=soothing_colors['background'],
                       foreground=soothing_colors['text_color'],
                       font=('Segoe UI', 11, 'bold'))
    
    def create_widgets(self):
        """Create all UI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, style="Custom.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Chat section (left side)
        chat_frame = ttk.Frame(main_frame, style="Custom.TFrame")
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Text area for chat history
        self.text_area = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=80,
            height=25,
            bg=soothing_colors['text_area'],
            fg=soothing_colors['text_color'],
            font=('Segoe UI', 11),
            relief='solid',
            borderwidth=2,
            insertbackground=soothing_colors['text_color']
        )
        self.text_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.text_area.insert(tk.END, "Welcome to the RAG-Enabled AI Assistant!\n\n")
        
        # Input field for user query
        input_frame = ttk.Frame(chat_frame, style="Custom.TFrame")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Your Query:", style="Custom.TLabel").pack(anchor=tk.W)
        
        self.input_field = tk.Text(
            input_frame,
            height=4,
            wrap=tk.WORD,
            bg=soothing_colors['text_area'],
            fg=soothing_colors['text_color'],
            font=('Segoe UI', 11),
            relief='solid',
            borderwidth=2,
            insertbackground=soothing_colors['text_color']
        )
        self.input_field.pack(fill=tk.X, pady=(5, 0))
        
        # Button frame for chat controls
        button_frame = ttk.Frame(chat_frame, style="Custom.TFrame")
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Submit button
        self.submit_button = ttk.Button(
            button_frame,
            text="Submit Query",
            command=self.handle_query,
            style="Custom.TButton"
        )
        self.submit_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Clear button
        self.clear_button = ttk.Button(
            button_frame,
            text="Clear Chat",
            command=self.clear_chat_history,
            style="Custom.TButton"
        )
        self.clear_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Export button
        self.export_button = ttk.Button(
            button_frame,
            text="Export to Excel",
            command=self.export_to_excel,
            style="Custom.TButton",
            state=tk.DISABLED
        )
        self.export_button.pack(side=tk.LEFT)
        
        # Status label
        self.status_label = ttk.Label(
            chat_frame,
            text="Status: Ready",
            style="Custom.TLabel",
            foreground=soothing_colors['success']
        )
        self.status_label.pack(anchor=tk.W)
        
        # Controls section (right side)
        controls_frame = ttk.LabelFrame(
            main_frame,
            text="RAG Controls",
            style="Custom.TLabelframe"
        )
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Add file button
        self.add_file_button = ttk.Button(
            controls_frame,
            text="Add Document",
            command=self.add_document,
            style="Custom.TButton"
        )
        self.add_file_button.pack(pady=5, padx=10, fill=tk.X)
        
        # Add folder button
        self.add_folder_button = ttk.Button(
            controls_frame,
            text="Add Folder",
            command=self.add_folder,
            style="Custom.TButton"
        )
        self.add_folder_button.pack(pady=5, padx=10, fill=tk.X)
        
        # Document info
        ttk.Label(controls_frame, text="Document Information:", style="Custom.TLabel").pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        self.doc_info = scrolledtext.ScrolledText(
            controls_frame,
            wrap=tk.WORD,
            width=30,
            height=15,
            bg=soothing_colors['secondary'],
            fg=soothing_colors['text_color'],
            font=('Segoe UI', 10),
            relief='solid',
            borderwidth=2,
            state=tk.DISABLED
        )
        self.doc_info.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        # Exit button
        self.exit_button = ttk.Button(
            controls_frame,
            text="Exit",
            command=self.root.quit,
            style="Custom.TButton"
        )
        self.exit_button.pack(pady=5, padx=10, fill=tk.X)
        
        # Bind Enter key to submit query
        self.input_field.bind('<Control-Return>', lambda event: self.handle_query())
    
    def handle_query(self):
        """Handle user query submission"""
        user_query = self.input_field.get("1.0", tk.END).strip()
        if not user_query:
            self.update_status_with_title("Status: Please enter a query.")
            return
        
        # Update chat history with user query
        self.text_area.insert(tk.END, f"\nYou: {user_query}\n\n")
        self.text_area.see(tk.END)
        
        # Clear input field
        self.input_field.delete("1.0", tk.END)
        
        # Update status
        self.update_status_with_title("Status: Processing query...")
        
        # Disable submit button during processing
        self.submit_button.configure(state=tk.DISABLED)
        
        # Process query in a separate thread
        def process_query():
            try:
                # Get the RAG response
                docs = self.rag_system.query(user_query)
                context = self.rag_system.format_retrieved_context(docs)
                
                # Create RAG prompt
                rag_prompt = f"""
You are an AI assistant tasked with answering questions based solely on the provided context. 

Please follow these guidelines:
- Use only the information provided in the context.
- When applicable, cite or reference the part of the context that supports your answer.
- If the user responds with a follow-up like "yes", "tell me more", or asks for clarification, you may continue the conversation naturally based on your previous answer — but indicate that you're stepping out of context-based answering.
- If the answer is not explicitly stated or clearly inferable, inform the user and then proceed to give your best answer.
- You may summarize or rephrase content from the context, but do not introduce any new information.
- DO not Halucinate

IMPORTANT INSTRUCTION FOR STRUCTURED DATA or EXCELL SHEET:
When the query asks for creating excel sheet, 
ALWAYS format your response as a list of dictionaries that can be exported to Excel, following these rules:

1. ALWAYS wrap structured data in triple backticks with python syntax highlighting an change the number of coloum and value accordingly:
```python
[
    {{"column1": "value1", "column2": "value2"}},
    {{"column1": "value3", "column2": "value4"}}
]

Context:
{context}

Question:
{user_query}

Answer:
"""
                # Add the RAG prompt to messages and get response
                self.rag_system.messages.append({"role": "user", "content": rag_prompt})
                
                # Using ollama directly
                response = ollama.chat(model=self.rag_system.model_name, messages=self.rag_system.messages)
                answer = response.message.content
                
                # Update message history
                self.rag_system.messages.pop()  # Remove the RAG prompt
                self.rag_system.messages.append({"role": "user", "content": user_query})
                self.rag_system.messages.append({"role": "assistant", "content": answer})
                
                # Update UI in main thread
                self.root.after(0, self.handle_response, answer)
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                print(f"Error in process_query: {error_message}")
                traceback.print_exc()
                self.root.after(0, self.handle_response, error_message)
        
        # Start processing in a separate thread
        thread = threading.Thread(target=process_query)
        thread.daemon = True
        thread.start()
    
    def extract_structured_data(self, response_text):
        """Extract structured data from response text"""
        # Pattern to match Python code blocks with lists/dictionaries
        pattern = r'```python\s*\n([\s\S]*?)\n\s*```'
        
        match = re.search(pattern, response_text)
        if match:
            code_block = match.group(1).strip()
            try:
                # Replace single quotes with double quotes for JSON parsing
                code_block = code_block.replace("'", '"')
                # Parse the data as JSON
                data = json.loads(code_block)
            
                # Check if it's a list of dictionaries or a dictionary
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    return data
                elif isinstance(data, dict):
                    return [data]
            except json.JSONDecodeError:
                # If JSON parsing fails, try using ast.literal_eval
                try:
                    import ast
                    data = ast.literal_eval(code_block)
                
                    # Check if it's a list of dictionaries or a dictionary
                    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                        return data
                    elif isinstance(data, dict):
                        return [data]
                except (SyntaxError, ValueError):
                    pass
        return None
    
    def handle_response(self, response):
        """Handle AI response and update UI"""
        # Extract structured data if it exists
        structured_data = self.extract_structured_data(response)
        
        if structured_data:
            # Store the structured data for later use
            self.structured_data = structured_data
            
            # Enable the export button
            self.export_button.configure(state=tk.NORMAL)
            
            # Show notification about structured data
            self.text_area.insert(tk.END, "AI: 📊 Structured data detected! Use the 'Export to Excel' button to save this data.\n\n")
            
            # Clean the response by removing the code block for display
            clean_response = re.sub(r'```python\s*\n[\s\S]*?\n\s*```', 
                                  '[Structured data ready for export]', 
                                  response)
            self.text_area.insert(tk.END, f"{clean_response}\n\n")
        else:
            # No structured data, disable the export button
            self.export_button.configure(state=tk.DISABLED)
            
            # Normal response display
            self.text_area.insert(tk.END, f"AI: {response}\n\n")
        
        self.text_area.see(tk.END)
        
        # Update status and re-enable submit button
        self.update_status_with_title("Status: Query processed successfully.")
        self.submit_button.configure(state=tk.NORMAL)
    
    def add_document(self):
        """Add a document to the RAG system"""
        file_path = filedialog.askopenfilename(
            title="Open Document",
            filetypes=[
                ("All Supported", "*.txt *.md *.csv *.json *.xlsx *.pdf *.jpg *.jpeg *.png"),
                ("Text Files", "*.txt *.md"),
                ("Data Files", "*.csv *.xlsx *.json"),
                ("PDF Files", "*.pdf"),
                ("Images", "*.jpg *.jpeg *.png"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            def process_document():
                try:
                    # Update status
                    self.root.after(0, lambda: self.update_status_with_title(f"Status: Processing {os.path.basename(file_path)}..."))
                    
                    # Check file size first
                    file_size = os.path.getsize(file_path)
                    if file_size > 50 * 1024 * 1024:  # 50MB limit
                        self.root.after(0, lambda: messagebox.showwarning("File Too Large", "File is too large. Please use a smaller file."))
                        return
                    
                    # Force garbage collection before processing
                    import gc
                    gc.collect()
                    
                    # Use process_file from Embedd.py to handle the document
                    existing_files = self.rag_system.vector_store.get_all_filenames()
                    
                    try:
                        process_file(file_path, self.rag_system.vector_store, existing_files)
                        
                        # Update document info
                        self.doc_info.configure(state=tk.NORMAL)
                        self.doc_info.insert(tk.END, f"Added document: {os.path.basename(file_path)}\n")
                        self.doc_info.configure(state=tk.DISABLED)
                        self.doc_info.see(tk.END)
                        
                        self.root.after(0, lambda: self.update_status_with_title(f"Status: Added {os.path.basename(file_path)}"))
                        
                        # Force garbage collection after processing
                        gc.collect()
                        
                    except Exception as process_error:
                        error_msg = f"Status: Error processing document - {str(process_error)}"
                        self.root.after(0, lambda: self.update_status_with_title(error_msg))
                        print(f"Process file error: {process_error}")
                        traceback.print_exc()
                        
                except Exception as e:
                    error_msg = f"Status: Error adding document - {str(e)}"
                    self.root.after(0, lambda: self.update_status_with_title(error_msg))
                    print(f"Add document error: {e}")
                    traceback.print_exc()
            
            # Process document in a separate thread
            thread = threading.Thread(target=process_document)
            thread.daemon = True
            thread.start()
    
    def add_folder(self):
        """Add all supported documents from a folder to the RAG system"""
        folder_path = filedialog.askdirectory(title="Select Folder with Documents")
        
        if folder_path:
            def process_folder_documents():
                try:
                    # Update status
                    self.root.after(0, lambda: self.update_status_with_title(f"Status: Processing folder {os.path.basename(folder_path)}..."))
                    
                    # Use process_folder from Embedd.py to handle all files in the folder
                    process_folder(folder_path, self.rag_system.vector_store)
                    
                    # Update document info
                    self.doc_info.configure(state=tk.NORMAL)
                    self.doc_info.insert(tk.END, f"Added folder: {os.path.basename(folder_path)}\n")
                    self.doc_info.configure(state=tk.DISABLED)
                    self.doc_info.see(tk.END)
                    
                    self.root.after(0, lambda: self.update_status_with_title(f"Status: Added folder {os.path.basename(folder_path)}"))
                    
                except Exception as e:
                    error_msg = f"Status: Error adding folder - {str(e)}"
                    self.root.after(0, lambda: self.update_status_with_title(error_msg))
                    print(f"Add folder error: {e}")
                    traceback.print_exc()
            
            # Process folder in a separate thread
            thread = threading.Thread(target=process_folder_documents)
            thread.daemon = True
            thread.start()
    
    def clear_chat_history(self):
        """Clear the chat history and reset the RAG system's messages"""
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, "Chat history cleared.\n\n")
        
        # Reset RAG system messages
        self.rag_system.messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately. If the answer isn't in the context, say you don't know."},
        ]
        
        # Clear structured data and disable export button
        self.structured_data = None
        self.export_button.configure(state=tk.DISABLED)
        
        self.update_status_with_title("Status: Chat history cleared.")
        self.input_field.delete("1.0", tk.END)
        self.input_field.focus()
    
    def export_to_excel(self):
        """Export structured data to Excel"""
        if not self.structured_data:
            messagebox.showwarning("No Data", "No structured data available to export.")
            return
        
        # Ask user for file location
        file_path = filedialog.asksaveasfilename(
            title="Save Excel File",
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Convert to pandas DataFrame
            df = pd.DataFrame(self.structured_data)
            
            # Export to Excel
            df.to_excel(file_path, index=False)
            
            self.update_status_with_title(f"Status: Data exported to {file_path}")
            messagebox.showinfo("Export Successful", f"Data has been exported to {file_path}")
        except Exception as e:
            error_msg = f"Status: Error exporting data - {str(e)}"
            self.update_status_with_title(error_msg)
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = ChatWindow()
    app.run()