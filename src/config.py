EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

SEMANTIC_THRESHOLD = 70.0
LEXICAL_THRESHOLD_SHORT = 60.0
LEXICAL_THRESHOLD_LONG = 45.0

MIN_SENTENCE_WORDS = 8
LEXICAL_DIVERSITY_THRESHOLD = 0.55

TOP_K_RESULTS = 5

HARD_MIN_SENTENCE_WORDS = 6  # seuil absolu

GENERIC_ACADEMIC_PATTERNS = [
    r"^ensuite[, ]",
    r"^dans ce chapitre",
    r"^ce travail",
    r"^ce mémoire",
    r"^ce rapport",
    r"^figure\s+\d+",
    r"^tableau\s+\d+",
    r"^chapitre\s+\d+",
    r"^annexe\s+\d+"
]
