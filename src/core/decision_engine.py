class DecisionEngine:

    def __init__(
        self,
        semantic_threshold: float,
        lexical_short: float,
        lexical_long: float
    ):
        self.semantic_threshold = semantic_threshold
        self.lexical_short = lexical_short
        self.lexical_long = lexical_long

    def is_plagiarism(
        self,
        semantic_score: float,
        lexical_score: float,
        sentence_length: int
    ) -> bool:

        if semantic_score < self.semantic_threshold:
            return False

        if sentence_length < 20:
            return lexical_score >= self.lexical_short

        return lexical_score >= self.lexical_long
