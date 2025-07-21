
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
