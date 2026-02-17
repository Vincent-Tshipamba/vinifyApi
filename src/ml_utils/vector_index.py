import faiss
import numpy as np


class VectorIndex:

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add(self, embeddings: np.ndarray, metadata: list):
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query: np.ndarray, top_k: int):
        scores, indices = self.index.search(query, top_k)
        return scores[0], [self.metadata[i] for i in indices[0]]
