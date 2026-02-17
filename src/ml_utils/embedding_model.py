import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)

    def encode(self, texts, batch_size=32, to_numpy=True):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=to_numpy,
            normalize_embeddings=True
        )
