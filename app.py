from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QGridLayout, QWidget,
    QPushButton, QTableWidget, QTableWidgetItem, QInputDialog, QTextEdit, 
    QFileDialog, QHBoxLayout, QVBoxLayout, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal
import sys
import os
import time
import traceback
import ollama  

from Bot import OllamaRAG

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
        self.rag_system = OllamaRAG(model_name="llama3.2")
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
        
        # Add PDF button
        self.add_pdf_button = QPushButton("Add PDF")
        self.add_pdf_button.clicked.connect(self.add_pdf)
        controls_layout.addWidget(self.add_pdf_button)
        
        # Document info
        self.doc_info = QTextEdit()
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
        
        # Update chat history with user query - formatted for readability
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
    
    def handle_response(self, response):
        # Update chat history with AI response - formatted for readability
        self.text_area.append(f"\n<span style='color:{soothing_colors['accent']}; font-weight:bold;'>AI:</span>")
        self.text_area.append(f"{response}\n")
        self.text_area.moveCursor(self.text_area.textCursor().End)
        
        # Update status
        self.status_label.setText("Status: Query processed successfully.")
    
    def add_document(self):
        """Add a text document to the RAG system"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Text File", "", "Text Files (*.txt *.md *.csv *.json)"
        )
        
        if file_path:
            try:
                # Call file processing method from OllamaRAG
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add the file to the query system
                import query
                query.add_document_to_index(content, os.path.basename(file_path))
                
                # Update document info
                self.doc_info.append(f"Added document: {os.path.basename(file_path)}")
                self.status_label.setText(f"Status: Added {os.path.basename(file_path)}")
            except Exception as e:
                self.status_label.setText(f"Status: Error adding document - {str(e)}")
    
    def add_pdf(self):
        """Add a PDF document to the RAG system"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF File", "", "PDF Files (*.pdf)"
        )
        
        if file_path:
            try:
                # Update status
                self.status_label.setText(f"Status: Adding PDF {os.path.basename(file_path)}...")
                
                # Call PDF processing method
                import query
                query.add_pdf_to_index(file_path)
                
                # Update document info
                self.doc_info.append(f"Added PDF: {os.path.basename(file_path)}")
                self.status_label.setText(f"Status: Added {os.path.basename(file_path)}")
            except Exception as e:
                self.status_label.setText(f"Status: Error adding PDF - {str(e)}")
    
    def clear_chat_history(self):
        """Clear the chat history and reset the RAG system's messages"""
        self.text_area.clear()
        self.text_area.setText("Chat history cleared.\n\n")
        
        # Reset RAG system messages
        self.rag_system.messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately. If the answer isn't in the context, say you don't know."},
        ]
        
        self.status_label.setText("Status: Chat history cleared.")
        self.input_field.clear()
        self.input_field.setFocus()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide style options
    app.setStyle("Fusion")  # Use Fusion style for a cleaner look
    
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())