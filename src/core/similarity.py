import re
from difflib import SequenceMatcher


class Similarity:

    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def lexical_similarity(cls, a: str, b: str) -> float:
        return SequenceMatcher(
            None,
            cls.normalize(a),
            cls.normalize(b)
        ).ratio() * 100
