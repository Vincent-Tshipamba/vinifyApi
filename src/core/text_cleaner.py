import re


class TextCleaner:

    @staticmethod
    def remove_reference_sections(text: str) -> str:
        markers = [
            r"\bbibliographie\b",
            r"\bréférences\b",
            r"\breferences\b",
            r"\bwebographie\b",
            r"\bsources\b",
            r"\bannexes\b"
        ]

        for marker in markers:
            match = re.search(marker, text, flags=re.IGNORECASE)
            if match:
                return text[:match.start()]

        return text
