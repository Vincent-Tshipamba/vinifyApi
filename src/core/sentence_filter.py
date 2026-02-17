import re
from config import HARD_MIN_SENTENCE_WORDS, GENERIC_ACADEMIC_PATTERNS

class SentenceFilter:

    @staticmethod
    def is_too_short(sentence: str) -> bool:
        return len(sentence.split()) < HARD_MIN_SENTENCE_WORDS

    @staticmethod
    def is_figure_or_table_caption(sentence: str) -> bool:
        return bool(re.search(r"^(figure|fig\.|tableau|table)\s*\d+", sentence.lower()))

    @staticmethod
    def is_page_reference(sentence: str) -> bool:
        return bool(re.search(r"\b(p\.?|pp\.?)\s*\d+|\bpage\s*\d+", sentence.lower()))

    @staticmethod
    def is_mostly_uppercase(sentence: str) -> bool:
        letters = [c for c in sentence if c.isalpha()]
        if not letters:
            return False
        return sum(c.isupper() for c in letters) / len(letters) > 0.6

    @staticmethod
    def has_low_lexical_diversity(sentence: str, threshold: float) -> bool:
        words = sentence.lower().split()
        if len(words) < 8:
            return True
        return (len(set(words)) / len(words)) < threshold

    @staticmethod
    def contains_citation(sentence: str) -> bool:
        return bool(re.search(r"\([A-Za-z\s,]+\d{4}\)|\[\d+\]", sentence))

    @staticmethod
    def is_generic_academic_sentence(sentence: str) -> bool:
        for pattern in GENERIC_ACADEMIC_PATTERNS:
            if re.search(pattern, sentence.lower()):
                return True
        return False

    @classmethod
    def should_ignore(cls, sentence: str, diversity_threshold: float) -> bool:
        return (
            cls.is_page_reference(sentence)
            or cls.is_mostly_uppercase(sentence)
            or cls.contains_citation(sentence)
            or cls.has_low_lexical_diversity(sentence, diversity_threshold)
        )
