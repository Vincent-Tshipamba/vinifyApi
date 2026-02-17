import html
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
from sentence_transformers import SentenceTransformer, util
import torch


STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "et", "ou", "en", "dans", "sur", "sous",
    "pour", "par", "avec", "sans", "au", "aux", "ce", "cet", "cette", "ces", "se", "sa", "son", "ses",
    "que", "qui", "quoi", "dont", "ne", "pas", "plus", "moins", "tres", "the", "a", "an", "and", "or",
    "to", "of", "for", "in", "on", "at", "is", "are", "be", "as", "by", "from", "http", "https", "www",
    "html", "markup", "language", "processing", "natural",
}

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL.to(_DEVICE)


def analyze_against_corpus(target_text: str, source_documents: List[Dict]) -> Dict:
    target_sentences = split_sentences(target_text)
    similarities: List[Dict] = []
    seen_keys: Set[str] = set()
    matched_indices: Set[int] = set()
    next_id = 1

    for source_document in source_documents:
        source_text = (source_document.get("content") or "").strip()
        if not source_text:
            continue

        source_sentences = split_sentences(source_text)
        if not source_sentences:
            continue

        for target_index, target_sentence in enumerate(target_sentences):
            if len(target_sentence) < 50:
                continue

            best_match = find_best_match_with_transformer(target_sentence, source_sentences)
            if not best_match:
                continue

            if best_match["score"] < 62:
                continue

            dedupe_key = f"{target_sentence}|{best_match['source_phrase']}|{source_document.get('id')}"
            if dedupe_key in seen_keys:
                continue

            seen_keys.add(dedupe_key)
            matched_indices.add(target_index)

            similarities.append(
                {
                    "id": next_id,
                    "similarity_percentage": round(best_match["score"], 2),
                    "source_document_id": source_document.get("id"),
                    "source_document_name": source_document.get("name") or f"Document #{source_document.get('id', 'N/A')}",
                    "source_phrase": best_match["source_phrase"],
                    "plagiarized_text": target_sentence,
                }
            )
            next_id += 1

    plagiarized_chars = sum(len(target_sentences[idx]) for idx in matched_indices if idx < len(target_sentences))
    base_length = max(len(target_text), 1)
    plagiarism_percentage = min(100, round((plagiarized_chars / base_length) * 100, 2))

    return {
        "full_text": target_text,
        "highlighted_text": build_highlighted_text(target_text, similarities),
        "plagiarism_percentage": plagiarism_percentage,
        "similarities": {
            "similarities": similarities,
            "excerpted_text": [],
        },
    }


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.?!;:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def normalize_linguistically(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("â€™", "'").replace("`", "'").replace("Â´", "'")
    text = strip_accents(text)
    text = re.sub(r"[^\w\s']+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
    return text.strip()


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def tokenize_words(text: str) -> List[str]:
    if not text:
        return []
    words = [w for w in re.split(r"\s+", text) if len(w) > 1]
    return [stem_word(word) for word in words]


def remove_stop_words(words: List[str]) -> List[str]:
    filtered = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    return filtered if len(filtered) >= 4 else words


def stem_word(word: str) -> str:
    suffixes = ["ements", "ement", "ations", "ation", "ments", "ment", "euses", "euse", "eaux", "aux", "ees", "es"]
    for suffix in suffixes:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def select_ngram_size(word_count: int) -> int:
    if word_count >= 15:
        return 4
    if word_count >= 7:
        return 3
    return 2


def build_word_ngrams(words: List[str], n: int) -> List[str]:
    if len(words) < n:
        return []
    chunks = []
    for i in range(0, len(words) - n + 1):
        chunks.append(" ".join(words[i : i + n]))
    return list(dict.fromkeys(chunks))


def within_tolerance(left: int, right: int, tolerance: float) -> bool:
    max_value = max(left, right)
    min_value = min(left, right)
    if max_value == 0:
        return False
    return (min_value / max_value) >= (1 - tolerance)


def find_best_match_with_transformer(target_sentence: str, source_sentences: List[str]) -> Optional[Dict]:
    best_score = 0.0
    best_phrase = None

    target_normalized = normalize_linguistically(target_sentence)
    target_words = tokenize_words(target_normalized)
    target_filtered_words = remove_stop_words(target_words)
    target_length = len(target_normalized)
    target_word_count = len(target_filtered_words)

    if target_length == 0 or target_word_count < 6:
        return None

    n = select_ngram_size(target_word_count)
    target_ngrams = build_word_ngrams(target_filtered_words, n)
    if len(target_ngrams) < 2:
        return None

    candidate_rows: List[Tuple[str, List[str], int, int]] = []

    for source_sentence in source_sentences:
        source_normalized = normalize_linguistically(source_sentence)
        source_words = remove_stop_words(tokenize_words(source_normalized))
        source_length = len(source_normalized)
        source_word_count = len(source_words)

        if source_length == 0 or source_word_count < 3:
            continue
        if not within_tolerance(target_length, source_length, 0.20):
            continue
        if not within_tolerance(target_word_count, source_word_count, 0.20):
            continue

        source_ngrams = build_word_ngrams(source_words, n)
        if not source_ngrams:
            continue

        # Gate lexical overlap before embedding comparison to reduce false positives.
        intersection = len(set(target_ngrams).intersection(set(source_ngrams)))
        required_intersection = max(2, int((len(target_ngrams) * 0.25) + 0.9999))
        if intersection < required_intersection:
            continue

        candidate_rows.append((source_sentence, source_words, source_length, source_word_count))

    if not candidate_rows:
        return None

    source_texts = [row[0] for row in candidate_rows]
    target_embedding = _MODEL.encode([target_sentence], convert_to_tensor=True, device=_DEVICE)
    source_embeddings = _MODEL.encode(source_texts, convert_to_tensor=True, device=_DEVICE)
    cosine_scores = util.cos_sim(target_embedding, source_embeddings)[0]

    for index, raw_score in enumerate(cosine_scores):
        score = float(raw_score.item()) * 100.0
        if score > best_score:
            best_score = score
            best_phrase = source_texts[index]

    if best_phrase is None:
        return None

    return {
        "score": best_score,
        "source_phrase": best_phrase,
    }


def build_highlighted_text(target_text: str, similarities: List[Dict]) -> str:
    highlighted = html.escape(target_text or "")
    similarities_sorted = sorted(similarities, key=lambda item: len((item.get("plagiarized_text") or "").strip()), reverse=True)

    for item in similarities_sorted:
        excerpt = (item.get("plagiarized_text") or "").strip()
        if not excerpt:
            continue

        safe_excerpt = html.escape(excerpt)
        replacement = f"<span class=\"plagiarized\" data-id=\"{html.escape(str(item.get('id')))}\">{safe_excerpt}</span>"

        exact_pattern = re.escape(safe_excerpt)
        highlighted, count = re.subn(exact_pattern, replacement, highlighted, count=1, flags=re.UNICODE)
        if count > 0:
            continue

        tokens = [token for token in re.split(r"\s+", safe_excerpt.strip()) if token]
        if len(tokens) < 2:
            continue
        token_pattern = r"\s+".join(re.escape(token) for token in tokens)
        highlighted = re.sub(token_pattern, replacement, highlighted, count=1, flags=re.UNICODE)

    return highlighted.replace("\n", "<br>")
