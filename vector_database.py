import faiss
import numpy as np

import os

class VectorDatabase:
    def __init__(self, dimension, index_type='flat', metric='l2', database_file=None):
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric # 'l2' or 'cosine'
        self.index = self._create_index()
        self.database_file = database_file
        database_path = "material/my_faiss_index.index"

        if database_file:# Chaeck if folder exists
            folder = os.path.dirnaAme(database_file)
            os.makedirs(folder, exist_ok=True)

        self.index = self._create_index()

        if database_file and os.path.exists(database_file):
            self.load_database(database_file)

    def _create_index(self):
        use_cosine = self.metric == 'cosine'
        metric_type = faiss.METRIC_INNER_PRODUCT if use_cosine else faiss.METRIC_L2
        
        if self.index_type == 'flat':
            return faiss.IndexFlatIP(self.dimension) if use_cosine else faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'ivfflat':
            quantizer = faiss.IndexFlatIP(self.dimension) if use_cosine else faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, metric_type)
            return index
        else:
            raise ValueError("Unsupported index type")

    def _normalize(self, vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-10)

    def add_vectors(self, vectors, vector_ids):
        assert len(vectors) == len(vector_ids), "Number of vectors and IDs must match"
        vectors_np = np.array(vectors, dtype=np.float32)

        if self.metric == 'cosine':
            vectors_np = self._normalize(vectors_np)

        if self.index_type == 'ivfflat' and not self.index.is_trained:
            self.index.train(vectors_np)

        self.index.add_with_ids(vectors_np, np.array(vector_ids, dtype=np.int64))

        if self.database_file:
            self.save_database()

    def search_similar_vectors(self, query_vector, k=5):
        query_vector = np.array([query_vector], dtype=np.float32)
        if self.metric == 'cosine':
            query_vector = self._normalize(query_vector)
        distances, indices = self.index.search(query_vector, k)
        return distances[0], indices[0]

    def save_database(self):
        print(f"Saving index to: {self.database_file}")
        faiss.write_index(self.index, self.database_file)

    def load_database(self, database_file):
        if os.path.exists(database_file):
            self.index = faiss.read_index(database_file)
        else:
            raise FileNotFoundError(f"Database file '{database_file}' not found.")

# Create a database with IVFFlat index and cosine similarity, and save to material/
db = VectorDatabase(
    dimension=512,
    index_type='ivfflat',
    metric='cosine',
    database_file='material/my_faiss_index.index'
)

# random vectors
num_vectors = 10000
dimension = 512
vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
vector_ids = [i for i in range(num_vectors)]  # FAISS needs int64 IDs

# Add vectors to the database
db.add_vectors(vectors, vector_ids)

# Search for similar vectors
query_vector = np.random.rand(dimension).astype(np.float32)
distances, indices = db.search_similar_vectors(query_vector)

print("Similar vectors indices:", indices)
print("Similar vectors distances:", distances)

