import os
import pickle

class VectorStore:
    def __init__(self, name):
        self.name = name
        self.vectors = {}

    def insert_vector(self, id, vector, metadata=None):
        self.vectors[id] = {"vector": vector, "metadata": metadata}

    def save(self):
        with open(f"{self.name}.pkl", "wb") as f:
            pickle.dump(self.vectors, f)

    def load(self):
        if os.path.exists(f"{self.name}.pkl"):
            with open(f"{self.name}.pkl", "rb") as f:
                self.vectors = pickle.load(f)

    def query(self, query_vector, top_k=5):
        # Dummy query function, replace with actual similarity search
        return list(self.vectors.items())[:top_k]
