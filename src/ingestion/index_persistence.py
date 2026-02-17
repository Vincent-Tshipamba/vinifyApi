import faiss
import pickle


class IndexPersistence:

    @staticmethod
    def save(index, metadata, path: str):
        faiss.write_index(index, f"{path}.index")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump(metadata, f)

    @staticmethod
    def load(path: str):
        index = faiss.read_index(f"{path}.index")
        with open(f"{path}.meta", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
