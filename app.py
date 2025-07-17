from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QGridLayout, QWidget,
    QPushButton, QTableWidget, QTableWidgetItem, QInputDialog, QTextEdit, 
    QFileDialog, QHBoxLayout, QVBoxLayout, QGroupBox, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal
import sys
import os
import time
import traceback
import ollama  
import re 
import json
import pandas as pd

from Vector_Database_ChromaDB import ChromaVectorStore
from Embedd import process_file, get_embedding_model

# Soothing color palette
soothing_colors = {
    'background': '#f5f7fa',       # Soft light gray-blue
    'text_area': "#e3e3e3",        # Pure white 
    'text_color': '#3a4a5a',       # Soft dark blue-gray
    'accent': '#7eb6bd',           # Calm teal
    'accent_hover': '#6ea5ad',     # Slightly deeper teal
    'secondary': '#d3e5eb',        # Very soft blue
    'highlight': '#f2d9c2',        # Warm beige
    'success': "#61dc75",          # Soft green
    'border': '#c0d6df'            # Muted blue-gray
}

# Text area style
setStyleQte = f"""
    QTextEdit {{
        background-color: {soothing_colors['text_area']};
        color: {soothing_colors['text_color']};
        font-family: 'Segoe UI', 'Helvetica';
        font-size: 12pt;
        font-weight: 400;
        border: 2px solid {soothing_colors['border']};
        border-radius: 8px;
        padding: 12px;
    }}
    
    QTextEdit#doc_info {{
        background-color: {soothing_colors['secondary']};
        color: {soothing_colors['text_color']};
        font-size: 11pt;
    }}
"""

# Button and control styles
setStyletui = f"""
    QPushButton {{
        background-color: {soothing_colors['accent']};
        color: white;
        font-family: 'Segoe UI', 'Arial';
        font-size: 11pt;
        font-weight: 600;
        border: 1px solid {soothing_colors['accent']};
        border-radius: 8px;
        padding: 8px 16px;
        margin: 5px 8px 5px 0px;
    }}
    
    QPushButton:hover {{
        background-color: {soothing_colors['accent_hover']};
    }}
    
    QPushButton:pressed {{
        background-color: {soothing_colors['accent_hover']};
        margin: 6px 7px 4px 1px;
    }}
    
    QGroupBox {{
        background-color: {soothing_colors['background']};
        font-family: 'Segoe UI', 'Arial';
        font-size: 12pt;
        font-weight: bold;
        border: 2px solid {soothing_colors['border']};
        border-radius: 8px;
        margin-top: 25px;
        padding-top: 15px;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        background-color: {soothing_colors['secondary']};
        padding: 6px 12px;
        color: {soothing_colors['text_color']};
        border: 1px solid {soothing_colors['border']};
        border-radius: 6px;
        left: 20px;
    }}
    
    QLabel {{
        font-family: 'Segoe UI', 'Arial';
        font-size: 11pt;
        font-weight: 500;
        color: {soothing_colors['text_color']};
    }}
"""

# Main window style
main_window_style = f"""
    QMainWindow, QWidget {{
        background-color: {soothing_colors['background']};
    }}
"""

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

class OllamaThread(QThread):
    """Thread for running Ollama queries without freezing the UI"""
    response_signal = pyqtSignal(str)
    
    def __init__(self, rag_system, query_text):
        super().__init__()
        self.rag_system = rag_system
        self.query_text = query_text
        
    def run(self):
        try:
            # Get the RAG response
            docs = self.rag_system.query(self.query_text)
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
{self.query_text}

Answer:r reference the part of the context that supports your answer.

"""
            # Add the RAG prompt to messages and get response
            self.rag_system.messages.append({"role": "user", "content": rag_prompt})
            
            # Useing ollama directly
            response = ollama.chat(model=self.rag_system.model_name, messages=self.rag_system.messages)
            answer = response.message.content
            
            # Update message history
            self.rag_system.messages.pop()  # Remove the RAG prompt
            self.rag_system.messages.append({"role": "user", "content": self.query_text})
            self.rag_system.messages.append({"role": "assistant", "content": answer})
            
            # Emit the response signal
            self.response_signal.emit(answer)
            
        except Exception as e:
            error_message = f"Error in OllamaThread: {str(e)}\n{traceback.format_exc()}"
            print(error_message)
            self.response_signal.emit(f"Error: {str(e)}")


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AI-RAG Assistant')
        self.setMinimumSize(1000, 600)
        
        # Apply main window style first
        self.setStyleSheet(main_window_style)
        
        # Initialize RAG system
        self.rag_system = ChromaRAG(model_name="llama3.2")
        self.ollama_thread = None

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # Chat section (left side)
        self.chat_section = QVBoxLayout()
        
        # Text area for chat history
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setPlaceholderText("Chat history will appear here...")
        self.text_area.setStyleSheet(setStyleQte)
        self.chat_section.addWidget(self.text_area)
        self.text_area.setText("Welcome to the RAG-Enabled AI Assistant!\n\n")
        
        # Input field for user query
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type your query here...")
        self.input_field.setStyleSheet(setStyleQte)
        self.input_field.setFixedHeight(100)
        self.chat_section.addWidget(self.input_field)
        
        # Button layout for chat controls
        chat_buttons = QHBoxLayout()
        
        # Button to submit query
        self.submit_button = QPushButton("Submit Query")
        self.submit_button.setStyleSheet(setStyletui)
        self.submit_button.clicked.connect(self.handle_query)
        chat_buttons.addWidget(self.submit_button)
        
        # Button to clear chat history
        self.clear_button = QPushButton("Clear Chat")
        self.clear_button.setStyleSheet(setStyletui)
        self.clear_button.clicked.connect(self.clear_chat_history)
        chat_buttons.addWidget(self.clear_button)
        
        #Export to Excel button
        self.export_button = QPushButton("Export to Excel")
        self.export_button.setStyleSheet(setStyletui)
        self.export_button.clicked.connect(self.export_to_excel)
        self.export_button.setEnabled(False)  # Initially disabled
        chat_buttons.addWidget(self.export_button)
        
        self.chat_section.addLayout(chat_buttons)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet(f"font-weight: bold; color: {soothing_colors['success']};")
        self.chat_section.addWidget(self.status_label)
        
        # Add chat section to main layout
        self.main_layout.addLayout(self.chat_section, 9)
        
        # Controls section (right side)
        controls_box = QGroupBox("RAG Controls")
        controls_layout = QVBoxLayout()
        controls_box.setLayout(controls_layout)
        
        # Add file button
        self.add_file_button = QPushButton("Add Document")
        self.add_file_button.clicked.connect(self.add_document)
        controls_layout.addWidget(self.add_file_button)
        
        # Add folder button
        self.add_folder_button = QPushButton("Add Folder")
        self.add_folder_button.clicked.connect(self.add_folder)
        controls_layout.addWidget(self.add_folder_button)
        
        # Document info
        self.doc_info = QTextEdit()
        self.doc_info.setAcceptDrops(True)
        self.doc_info.setObjectName("doc_info")  # Add this line to apply specific styling
        self.doc_info.setReadOnly(True)
        self.doc_info.setPlaceholderText("Document information will appear here...")
        controls_layout.addWidget(self.doc_info)
        
        # Exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        controls_layout.addWidget(self.exit_button)
        
        # Add controls to main layout
        self.main_layout.addWidget(controls_box, 3)
        
        # Set the stylesheet for the main window
        self.setStyleSheet("background-color: #f0f0f0; font-family: Arial;")
    
    
    def handle_query(self):
        user_query = self.input_field.toPlainText().strip()
        if not user_query:
            self.status_label.setText("Status: Please enter a query.")
            return
        
        # Update chat history with user query
        self.text_area.append(f"\n<span style='color:#8C6057; font-weight:bold;'>You:</span> {user_query}\n")
        self.text_area.moveCursor(self.text_area.textCursor().End)
        
        # Clear input field
        self.input_field.clear()
        
        # Update status
        self.status_label.setText("Status: Processing query...")
        
        # Process query in a separate thread
        try:
            # Start thread for Ollama query
            self.ollama_thread = OllamaThread(self.rag_system, user_query)
            self.ollama_thread.response_signal.connect(self.handle_response)
            self.ollama_thread.start()
        except Exception as e:
            self.text_area.append(f"Error: {str(e)}")
            self.status_label.setText("Status: Error processing query.")
            self.text_area.moveCursor(self.text_area.textCursor().End)
    
    def extract_structured_data(self, response_text):    
        # Pattern to match Python code blocks with lists/dictionaries
        pattern = r'```python\s*\n([\s\S]*?)\n\s*```'
        
        match = re.search(pattern, response_text)  # Changed from self.response_text to response_text
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
        # Extract structured data if it exists
        structured_data = self.extract_structured_data(response)
        
        if structured_data:
            # Store the structured data for later use
            self.structured_data = structured_data
            
            # Enable the export button
            self.export_button.setEnabled(True)
            
            # Show notification about structured data
            data_notification = """<div style='background-color:#f0f7ee; padding:10px; border-left:4px solid #8bbf9f; margin:5px 0;'>
            <b>📊 Structured data detected!</b> Use the "Export to Excel" button to save this data.
            </div>"""
            
            # Add AI response with formatting
            self.text_area.append(f"\n<span style='color:{soothing_colors['accent']}; font-weight:bold;'>AI:</span>")
            self.text_area.append(data_notification)
            
            # Clean the response by removing the code block for display
            import re
            clean_response = re.sub(r'```python\s*\n[\s\S]*?\n\s*```', 
                                  '<i>[Structured data ready for export]</i>', 
                                  response)
            self.text_area.append(f"{clean_response}\n")
        else:
            # No structured data, disable the export button
            if hasattr(self, 'export_button'):
                self.export_button.setEnabled(False)
            
            # Normal response display
            self.text_area.append(f"\n<span style='color:{soothing_colors['accent']}; font-weight:bold;'>AI:</span>")
            self.text_area.append(f"{response}\n")
        
        self.text_area.moveCursor(self.text_area.textCursor().End)
        
        # Update status
        self.status_label.setText("Status: Query processed successfully.")
    
    def add_document(self):
        """Add a document to the RAG system"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", "", "All Supported (*.txt *.md *.csv *.json *.xlsx *.pdf *.jpg *.jpeg *.png);;Text Files (*.txt *.md);;Data Files (*.csv *.xlsx *.json);;PDF Files (*.pdf);;Images (*.jpg *.jpeg *.png)"
        )
        
        if file_path:
            try:
                # Update status
                self.status_label.setText(f"Status: Processing {os.path.basename(file_path)}...")
                
                # Use process_file from Embedd.py to handle the document
                existing_files = self.rag_system.vector_store.get_all_filenames()
                process_file(file_path, self.rag_system.vector_store, existing_files)
                
                # Update document info
                self.doc_info.append(f"Added document: {os.path.basename(file_path)}")
                self.status_label.setText(f"Status: Added {os.path.basename(file_path)}")
            except Exception as e:
                self.status_label.setText(f"Status: Error adding document - {str(e)}")
    
    def add_folder(self):
        """Add all supported documents from a folder to the RAG system"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder with Documents"
        )
        
        if folder_path:
            try:
                # Update status
                self.status_label.setText(f"Status: Processing folder {os.path.basename(folder_path)}...")
                
                # Use process_folder from Embedd.py to handle all files in the folder
                from Embedd import process_folder
                process_folder(folder_path, self.rag_system.vector_store)
                
                # Update document info
                self.doc_info.append(f"Added folder: {os.path.basename(folder_path)}")
                self.status_label.setText(f"Status: Added folder {os.path.basename(folder_path)}")
            except Exception as e:
                self.status_label.setText(f"Status: Error adding folder - {str(e)}")

    def add_pdf(self):
        """Add a PDF document to the RAG system (legacy method - now handled by add_document)"""
        # Redirect to add_document since it now handles all file types
        self.add_document()
    
    def clear_chat_history(self):
        """Clear the chat history and reset the RAG system's messages"""
        self.text_area.clear()
        self.text_area.setText("Chat history cleared.\n\n")
        
        # Reset RAG system messages
        self.rag_system.messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately. If the answer isn't in the context, say you don't know."},
        ]
        
        # Clear structured data and disable export button
        if hasattr(self, 'structured_data'):
            delattr(self, 'structured_data')
        self.export_button.setEnabled(False)
        
        self.status_label.setText("Status: Chat history cleared.")
        self.input_field.clear()
        self.input_field.setFocus()

    def export_to_excel(self):
        """Export structured data to Excel"""
        if not hasattr(self, 'structured_data') or not self.structured_data:
            QMessageBox.warning(self, "No Data", "No structured data available to export.")
            return
        
        # Ask user for file location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel File", "", "Excel Files (*.xlsx)"
        )
        
        if not file_path:
            return
        
        if not file_path.endswith('.xlsx'):
            file_path += '.xlsx'
        
        try:
            import pandas as pd
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(self.structured_data)
            
            # Export to Excel
            df.to_excel(file_path, index=False)
            
            self.status_label.setText(f"Status: Data exported to {file_path}")
            QMessageBox.information(self, "Export Successful", f"Data has been exported to {file_path}")
        except Exception as e:
            self.status_label.setText(f"Status: Error exporting data - {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide style options
    app.setStyle("Fusion")  # Fusion style for a cleaner look
    
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())