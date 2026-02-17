import html
import re
from flask import request, jsonify
from core.similarity import Similarity
from core.decision_engine import DecisionEngine
from core.text_cleaner import TextCleaner
from core.sentence_filter import SentenceFilter
from config import *


def build_highlighted_text(target_text: str, similarities: list) -> str:
    highlighted = html.escape(target_text or "")

    similarities_sorted = sorted(
        similarities,
        key=lambda item: len((item.get("plagiarized_text") or "").strip()),
        reverse=True
    )

    for item in similarities_sorted:
        excerpt = (item.get("plagiarized_text") or "").strip()
        if not excerpt:
            continue

        safe_excerpt = html.escape(excerpt)
        replacement = (
            f"<span class=\"plagiarized\" "
            f"data-id=\"{html.escape(str(item.get('id')))}\">"
            f"{safe_excerpt}</span>"
        )

        exact_pattern = re.escape(safe_excerpt)
        highlighted, count = re.subn(
            exact_pattern,
            replacement,
            highlighted,
            count=1,
            flags=re.UNICODE
        )

        if count > 0:
            continue

        tokens = [t for t in re.split(r"\s+", safe_excerpt.strip()) if t]
        if len(tokens) < 2:
            continue

        token_pattern = r"\s+".join(re.escape(t) for t in tokens)
        highlighted = re.sub(
            token_pattern,
            replacement,
            highlighted,
            count=1,
            flags=re.UNICODE
        )

    return highlighted.replace("\n", "<br>")


def register_routes(app, embedding_model, vector_index, nlp):

    decision_engine = DecisionEngine(
        SEMANTIC_THRESHOLD,
        LEXICAL_THRESHOLD_SHORT,
        LEXICAL_THRESHOLD_LONG
    )

    @app.route("/check-plagiarism", methods=["POST"])
    def check_plagiarism():
        app.logger.info(
            "Request received | /check-plagiarism | method=%s | content_type=%s",
            request.method,
            request.content_type,
        )

        payload = request.get_json() or {}
        text = payload.get("document_text", "").strip()
        corpus_size = len(payload.get("source_documents") or [])
        app.logger.info(
            "Payload parsed | document_text_chars=%s | source_documents=%s",
            len(text),
            corpus_size,
        )

        if not text:
            return jsonify({"similarities": {}}), 200

        clean_text = TextCleaner.remove_reference_sections(text)
        sentences = [s.text.strip() for s in nlp(clean_text).sents]

        similarities = []
        excerpted_text = []

        embeddings = embedding_model.encode(sentences)

        match_id = 1

        for i, sentence in enumerate(sentences):

            if SentenceFilter.should_ignore(sentence):
                continue

            emb = embeddings[i].reshape(1, -1)

            scores, candidates = vector_index.search(
                emb,
                TOP_K_RESULTS
            )

            for score, candidate in zip(scores, candidates):

                semantic_score = score * 100

                lexical_score = Similarity.lexical_similarity(
                    sentence,
                    candidate["text"]
                )

                if decision_engine.is_plagiarism(
                    semantic_score,
                    lexical_score,
                    len(sentence.split())
                ):
                    match = {
                        "id": match_id,
                        "plagiarized_text": sentence,
                        "source_text": candidate["text"],
                        "source_name": candidate["source_name"],
                        "semantic_score": round(semantic_score, 2),
                        "lexical_score": round(lexical_score, 2)
                    }

                    similarities.append(match)
                    excerpted_text.append(sentence)
                    match_id += 1
                    break

        highlighted = build_highlighted_text(text, similarities)

        plagiarism_percentage = (
            (len(similarities) / len(sentences)) * 100
            if sentences else 0
        )

        response = {
            "full_text": text,
            "highlighted_text": highlighted,
            "plagiarism_percentage": round(plagiarism_percentage, 2),
            "similarities": similarities,
            "excerpted_text": excerpted_text
        }

        return jsonify({"similarities": response}), 200
