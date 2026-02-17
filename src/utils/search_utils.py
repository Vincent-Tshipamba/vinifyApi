from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
from dotenv import load_dotenv
import os
import re
import spacy
import torch
from typing import Dict, List

load_dotenv()

try:
    nlp_spacy = spacy.load('fr_core_news_sm')
except OSError:
    print("Le modele spaCy 'fr_core_news_sm' n'est pas trouve. Telechargement en cours...")
    spacy.cli.download('fr_core_news_sm')
    nlp_spacy = spacy.load('fr_core_news_sm')

model = SentenceTransformer('all-MiniLM-L6-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

EN_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'in', 'on', 'at', 'for', 'to', 'from',
    'with', 'without', 'of', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'this', 'that',
    'these', 'those', 'it', 'its', 'as', 'into', 'about', 'than', 'such'
}
FR_STOPWORDS = set(nlp_spacy.Defaults.stop_words) if nlp_spacy is not None else set()
ALL_STOPWORDS = FR_STOPWORDS.union(EN_STOPWORDS)

IGNORED_PHRASES = [
    'Republique Democratique Du Congo',
    'ENSEIGNEMENT SUPERIEUR ET UNIVERSITAIRE',
    "INSTITUT SUPERIEUR D'INFORMATIQUE, PROGRAMMATION ET ANALYSE",
    'Tous droits reserves',
    'Copyright',
    'Version actuelle',
    'HyperText Markup Language',
    'Natural Language Processing',
]


def preprocess_text(text: str) -> str:
    """
    First-step preprocessing applied to user and corpus texts:
    - lowercase
    - remove punctuation
    - remove stopwords
    """
    if not isinstance(text, str):
        return ''

    lowered = text.lower()
    no_punct = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    tokens = [token for token in re.split(r"\s+", no_punct) if token]
    filtered = [token for token in tokens if token not in ALL_STOPWORDS]
    return ' '.join(filtered).strip()

def remove_bibliography(text: str) -> str:
    patterns = [
        r'\bBibliographie\b',
        r'\bReferences\b',
        r'\bRéférences\b',
        r'\bWebographie\b'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return text[:match.start()]
    return text



def extract_with_context(full_text: str, target_sentence: str, data_id=None, window: int = 1):
    if nlp_spacy is None:
        return '', ''

    doc = nlp_spacy(full_text)
    sentences = [sent.text for sent in doc.sents]

    target_sentence_cleaned = target_sentence.strip()
    target_sentence_lower = target_sentence_cleaned.lower()

    index = -1
    for i, sentence in enumerate(sentences):
        if target_sentence_lower == sentence.strip().lower():
            index = i
            break

    if index == -1:
        for i, sentence in enumerate(sentences):
            if SequenceMatcher(None, target_sentence_lower, sentence.strip().lower()).ratio() > 0.95:
                index = i
                break

    if index == -1:
        return '', ''

    start = max(0, index - window)
    end = min(len(sentences), index + window + 1)
    context_text = ' '.join(sentences[start:end])

    if data_id is not None:
        span_tag = f"<span class='plagiarized' data-id='{data_id}'>{target_sentence_cleaned}</span>"
    else:
        span_tag = f"<span class='plagiarized'>{target_sentence_cleaned}</span>"

    highlighted = re.sub(re.escape(target_sentence_cleaned), span_tag, context_text, 1, flags=re.IGNORECASE)
    if span_tag not in highlighted:
        return context_text, context_text

    return context_text, highlighted


def is_ignorable(sentence: str) -> bool:
    return any(phrase.lower() in sentence.lower() for phrase in IGNORED_PHRASES)

def is_page_reference(sentence: str) -> bool:
    return bool(re.search(r'\b(Page|Pages|page|pages|p\.?|pp\.?)\s*\d+', sentence.lower()))


def extract_footnotes(full_text: str) -> Dict[str, str]:
    footnote_matches = re.findall(r'\[(\d+)\]\s*(.+)', full_text)
    if not footnote_matches:
        return {}
    return {num: content for num, content in footnote_matches}


def is_cited_by_footnote(sentence: str, footnotes: Dict[str, str]) -> bool:
    matches = re.findall(r'\[(\d+)\]|\((\d+)\)', sentence)
    for match in matches:
        num = match[0] if match[0] else match[1]
        if num and num in footnotes:
            if any(keyword in footnotes[num].lower() for keyword in ['http', 'www', 'doi', 'isbn', 'edition', 'press', 'university', 'source', 'auteur']):
                return True
    return False


def is_cited_with_nlp(sentence_text: str) -> bool:
    if nlp_spacy is None:
        return False

    doc = nlp_spacy(sentence_text)
    if re.search(r'\([A-Za-z\s,]+\s*\d{4}\)|\(\d{4}\)|\[\d+\]', sentence_text):
        return True

    for token in doc:
        if token.lemma_.lower() in ['selon', "d'apres", 'declarer', 'affirmer', 'citer', 'rapporter', 'indiquer', 'montrer']:
            for next_token in doc[token.i + 1:]:
                if next_token.ent_type_ in {'PER', 'ORG', 'LOC', 'MISC'} or next_token.pos_ == 'PROPN':
                    return True
                if next_token.is_punct and next_token.text in ['.', ';', '!', '?']:
                    break
                if next_token.pos_ == 'VERB':
                    break
    return False


def split_into_semantic_chunks(text: str, max_chunk_words: int = 75, min_chunk_words: int = 15) -> List[str]:
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    segments: List[str] = []

    for para in paragraphs:
        if nlp_spacy is None:
            temp_sentences = re.split(r'(?<=[.!?])\s+', para)
            sentences = [s.strip() for s in temp_sentences if len(s.strip().split()) >= 3]
        else:
            doc = nlp_spacy(para)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) >= 3]

        current_chunk_sentences: List[str] = []
        current_chunk_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if sentence_words > max_chunk_words:
                words = sentence.split()
                for i in range(0, sentence_words, max_chunk_words):
                    part = ' '.join(words[i:i + max_chunk_words])
                    if current_chunk_word_count >= min_chunk_words:
                        segments.append(' '.join(current_chunk_sentences))
                        current_chunk_sentences = [part]
                        current_chunk_word_count = len(part.split())
                    else:
                        current_chunk_sentences.append(part)
                        current_chunk_word_count += len(part.split())
                continue

            if (current_chunk_word_count + sentence_words > max_chunk_words) and (current_chunk_word_count >= min_chunk_words):
                segments.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_chunk_word_count = sentence_words
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_word_count += sentence_words

        if current_chunk_sentences:
            segments.append(' '.join(current_chunk_sentences))

    return segments


def compare_with_corpus(text: str, source_documents: List[Dict], threshold: float = 62.0) -> Dict:
    original_text = remove_bibliography(text or '')

    # Very first step for analysis: preprocess user document.
    preprocessed_document_text = preprocess_text(original_text)

    all_results = []
    excerpts = []
    seen_phrases = set()
    highlighted_text = original_text
    footnotes = extract_footnotes(original_text)

    if not preprocessed_document_text:
        return {
            'full_text': original_text,
            'highlighted_text': original_text,
            'plagiarism_percentage': 0,
            'similarities': {
                'similarities': [],
                'excerpted_text': []
            }
        }

    if not source_documents:
        return {
            'full_text': original_text,
            'highlighted_text': original_text,
            'plagiarism_percentage': 0,
            'similarities': {
                'similarities': [],
                'excerpted_text': []
            }
        }

    source_sentence_rows = []
    for doc in source_documents:
        content = (doc.get('content') or '').strip()
        if not content:
            continue

        # Very first step for corpus input: preprocess whole corpus document.
        preprocessed_content = preprocess_text(content)
        if not preprocessed_content:
            continue

        if nlp_spacy is None:
            candidates = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]
        else:
            d = nlp_spacy(content)
            candidates = [s.text.strip() for s in d.sents if s.text.strip()]

        for sentence in candidates:
            processed_sentence = preprocess_text(sentence)
            if len(processed_sentence.split()) < 5:
                continue
            if len(processed_sentence) < 30:
                continue
            source_sentence_rows.append({
                'source_sentence': sentence,
                'source_sentence_processed': processed_sentence,
                'source_document_id': doc.get('id'),
                'source_document_name': doc.get('name') or f"Document #{doc.get('id', 'N/A')}",
            })

    if not source_sentence_rows:
        return {
            'full_text': original_text,
            'highlighted_text': original_text,
            'plagiarism_percentage': 0,
            'similarities': {
                'similarities': [],
                'excerpted_text': []
            }
        }

    source_sentences = [row['source_sentence_processed'] for row in source_sentence_rows]
    source_embeddings = model.encode(source_sentences, convert_to_tensor=True, device=device)

    segments = split_into_semantic_chunks(original_text)

    for segment in segments:
        if nlp_spacy is None:
            target_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', segment) if s.strip()]
        else:
            doc_segment = nlp_spacy(segment)
            target_sentences = [s.text for s in doc_segment.sents]

        valid_targets = []
        valid_targets_processed = []
        for sentence in target_sentences:
            processed_sentence = preprocess_text(sentence)

            if is_ignorable(sentence):
                continue
            if sentence in seen_phrases:
                continue
            if len(processed_sentence.split()) < 5:
                continue
            if len(processed_sentence) < 30:
                continue
            if is_cited_with_nlp(sentence):
                continue
            if is_cited_by_footnote(sentence, footnotes):
                continue
            if is_page_reference(sentence):
                continue


            valid_targets.append(sentence)
            valid_targets_processed.append(processed_sentence)

        if not valid_targets:
            continue

        target_embeddings = model.encode(valid_targets_processed, convert_to_tensor=True, device=device)
        sim_matrix = util.pytorch_cos_sim(target_embeddings, source_embeddings).cpu().numpy() * 100

        for i, sentence in enumerate(valid_targets):
            best_idx = int(sim_matrix[i].argmax())
            best_similarity = float(sim_matrix[i][best_idx])

            if best_similarity < threshold:
                continue

            src = source_sentence_rows[best_idx]
            data_id = len(all_results)

            if sentence in highlighted_text:
                span_tag = f"<span class='plagiarized' data-id='{data_id}'>{sentence}</span>"
                highlighted_text = highlighted_text.replace(sentence, span_tag, 1)

            all_results.append({
                'id': data_id,
                'source_document_id': src['source_document_id'],
                'source_document_name': src['source_document_name'],
                'source_phrase': src['source_sentence'],
                'similarity_percentage': round(best_similarity, 2),
                'plagiarized_text': sentence,
            })
            seen_phrases.add(sentence)

            context, highlighted = extract_with_context(original_text, sentence, data_id=data_id)
            if context and highlighted:
                excerpts.append({
                    'id': data_id,
                    'context': context,
                    'highlighted': highlighted,
                })

    unique_results = {r['plagiarized_text']: r for r in all_results}
    final_results = list(unique_results.values())

    plagiarized_word_count = sum(len(r['plagiarized_text'].split()) for r in final_results)
    total_word_count = len(original_text.split())
    plagiarism_percentage = round((plagiarized_word_count / total_word_count) * 100, 2) if total_word_count > 0 else 0

    return {
        'full_text': original_text,
        'highlighted_text': highlighted_text,
        'plagiarism_percentage': plagiarism_percentage,
        'similarities': {
            'similarities': final_results,
            'excerpted_text': excerpts
        }
    }


def compare_with_search_results(text: str, source_documents: List[Dict] = None) -> Dict:
    """Compat legacy: conserve le nom historique mais travaille maintenant sur le corpus."""
    return compare_with_corpus(text, source_documents or [])
