import spacy
from core.text_cleaner import TextCleaner
from core.sentence_filter import SentenceFilter
from config import MIN_SENTENCE_WORDS, LEXICAL_DIVERSITY_THRESHOLD


class CorpusIngestor:

    def __init__(self):
        self.nlp = spacy.load("fr_core_news_sm")

    def ingest(self, documents: list):
        sentences = []
        metadata = []

        for doc in documents:
            clean_text = TextCleaner.remove_reference_sections(doc["content"])
            parsed = self.nlp(clean_text)

            for sent in parsed.sents:
                sentence = sent.text.strip()

                if len(sentence.split()) < MIN_SENTENCE_WORDS:
                    continue

                if SentenceFilter.should_ignore(sentence, LEXICAL_DIVERSITY_THRESHOLD):
                    continue

                sentences.append(sentence)
                metadata.append({
                    "text": sentence,
                    "source_id": doc["id"],
                    "source_name": doc["name"]
                })

        return sentences, metadata
