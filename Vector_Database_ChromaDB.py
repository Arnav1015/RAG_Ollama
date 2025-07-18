import chromadb
from chromadb.config import Settings
from typing import List, Dict
import os
import sys
import gc
import traceback
import json

class ChromaVectorStore:
    def __init__(self, collection_name: str = "default", persist_directory: str = "chroma_db"):
        try:
            print(f"DEBUG: Starting ChromaDB initialization...")
            print(f"DEBUG: Python version: {sys.version}")
            print(f"DEBUG: ChromaDB version: {chromadb.__version__}")
            print(f"DEBUG: Current working directory: {os.getcwd()}")
            print(f"DEBUG: Persist directory: {persist_directory}")
            
            # Clean start
            if os.path.exists(persist_directory):
                import shutil
                print(f"DEBUG: Removing existing database directory...")
                shutil.rmtree(persist_directory)
            
            os.makedirs(persist_directory, exist_ok=True)
            print(f"DEBUG: Created fresh directory: {persist_directory}")
            
            # Initialize ChromaDB client with detailed settings
            print(f"DEBUG: Creating ChromaDB client...")
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    persist_directory=persist_directory,
                    is_persistent=True
                )
            )
            print(f"DEBUG: ChromaDB client created successfully")
            print(f"DEBUG: Client type: {type(self.client)}")
            
            # Test client functionality
            print(f"DEBUG: Testing client functionality...")
            try:
                client_info = self.client.heartbeat()
                print(f"DEBUG: Client heartbeat successful: {client_info}")
            except Exception as e:
                print(f"DEBUG: Client heartbeat failed: {e}")
            
            # Create collection
            print(f"DEBUG: Creating collection '{collection_name}'...")
            self.collection = self.client.create_collection(name=collection_name)
            print(f"DEBUG: Collection created successfully")
            print(f"DEBUG: Collection type: {type(self.collection)}")
            print(f"DEBUG: Collection name: {self.collection.name}")
            
            # Test collection functionality
            print(f"DEBUG: Testing collection functionality...")
            try:
                count = self.collection.count()
                print(f"DEBUG: Collection count: {count}")
            except Exception as e:
                print(f"DEBUG: Collection count failed: {e}")
            
            print(f"DEBUG: ChromaDB initialization completed successfully")
            
        except Exception as e:
            print(f"ERROR: ChromaDB initialization failed: {e}")
            traceback.print_exc()
            raise

    def add_documents(self, ids, documents, embeddings, metadatas=None):
        """Add documents to the collection with batch processing"""
        try:
            # Process in smaller batches to avoid memory issues
            batch_size = 100
            
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size] if metadatas else None
                
                # Convert to proper format
                batch_embeddings = [
                    emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                    for emb in batch_embeddings
                ]
                
                # Upsert batch
                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                
                # Force garbage collection between batches
                import gc
                gc.collect()
                
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

    def _get_memory_usage(self):
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return f"{process.memory_info().rss / 1024 / 1024:.2f} MB"
        except ImportError:
            return "N/A (psutil not available)"
        except Exception as e:
            return f"Error: {e}"

    def get_all_filenames(self):
        try:
            print(f"DEBUG: Getting all filenames...")
            result = self.collection.get()
            print(f"DEBUG: Get result type: {type(result)}")
            
            if result and 'metadatas' in result:
                filenames = set(meta.get('filename', '') for meta in result['metadatas'] if meta.get('filename'))
                print(f"DEBUG: Found {len(filenames)} unique filenames")
                return filenames
            return set()
        except Exception as e:
            print(f"ERROR getting filenames: {e}")
            traceback.print_exc()
            return set()

    def query(self, query_embedding: List[float], top_k: int = 5, where: Dict = None):
        try:
            print(f"DEBUG: Querying with embedding length: {len(query_embedding)}")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )
            print(f"DEBUG: Query successful, results type: {type(results)}")
            return results
        except Exception as e:
            print(f"ERROR querying ChromaDB: {e}")
            traceback.print_exc()
            return None

    def delete_by_ids(self, ids: List[str]):
        try:
            print(f"DEBUG: Deleting {len(ids)} documents...")
            self.collection.delete(ids=ids)
            print(f"DEBUG: Delete successful")
        except Exception as e:
            print(f"ERROR deleting documents: {e}")
            traceback.print_exc()

    def reset(self):
        try:
            print(f"DEBUG: Resetting ChromaDB...")
            self.client.reset()
            print(f"DEBUG: Reset successful")
        except Exception as e:
            print(f"ERROR resetting ChromaDB: {e}")
            traceback.print_exc()
