from sentence_transformers import SentenceTransformer, util
import requests
import re
import openai
import os
from dotenv import load_dotenv
from difflib import SequenceMatcher
import spacy
import torch

# Charge les variables d'environnement depuis le fichier.env
load_dotenv()

try:
    nlp_spacy = spacy.load('fr_core_news_sm')
except OSError:
    print("Le modèle spaCy 'fr_core_news_sm' n'est pas trouvé. Téléchargement en cours...")
    spacy.cli.download("fr_core_news_sm")
    nlp_spacy = spacy.load('fr_core_news_sm')

model = SentenceTransformer('all-MiniLM-L6-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def extract_with_context(full_text, target_sentence, data_id=None, window=1):
    """
    Extrait une phrase cible avec son contexte (phrases avant et après).
    C'est utile pour afficher autour de la phrase plagiée et donner une meilleure idée
    de où elle se trouve dans le texte original.
    Le 'data_id' permet de créer une balise HTML unique pour la phrase, ce qui est pratique
    pour la lier à des détails dans l'interface utilisateur.
    """
    if nlp_spacy is None:
        return "", "" # Ne peut pas traiter sans le modèle spaCy

    doc = nlp_spacy(full_text)
    sentences = [sent.text for sent in doc.sents]

    try:
        index = -1
        # Cherche l'index de la phrase qu'on veut extraire avec une correspondance exacte ou très proche
        # Utiliser lower() pour une comparaison insensible à la casse et strip() pour enlever les espaces inutiles
        target_sentence_cleaned = target_sentence.strip()
        target_sentence_lower = target_sentence_cleaned.lower()
        
        # Tenter de trouver la phrase exacte ou une correspondance très proche
        for i, s in enumerate(sentences):
            if target_sentence_lower == s.strip().lower():
                index = i
                break
        
        # Si pas de correspondance exacte, tenter une correspondance floue (ratio élevé)
        if index == -1:
            for i, s in enumerate(sentences):
                # Utiliser SequenceMatcher sur les versions nettoyées et en minuscules
                if SequenceMatcher(None, target_sentence_lower, s.strip().lower()).ratio() > 0.95:
                    index = i
                    break
        
        if index == -1:
            return "", "" # Si toujours pas trouvée, retourner vide

    except StopIteration:
        return "", ""

    start = max(0, index - window)
    end = min(len(sentences), index + window + 1)
    context_sentences = sentences[start:end]
    context_text = ' '.join(context_sentences)

    if data_id is not None:
        span_tag = f"<span class='plagiarized' data-id='{data_id}'>{target_sentence_cleaned}</span>"
    else:
        span_tag = f"<span class='plagiarized'>{target_sentence_cleaned}</span>"

    # Pour surligner, nous devons trouver la phrase exacte ou la plus proche dans le CONTEXT_TEXT
    # et non dans le full_text.
    # Construire une regex pour chercher la phrase cible dans le contexte
    # re.escape() pour échapper les caractères spéciaux de la phrase
    # \s* pour gérer les espaces blancs multiples autour de la phrase ou entre les mots (à ajuster si nécessaire)
    # re.IGNORECASE pour l'insensibilité à la casse
    
    # Stratégie 1: Tentative de remplacement exact de la phrase cible dans le contexte.
    # On va utiliser la phrase telle qu'elle est extraite à l'origine (target_sentence_cleaned)
    # pour le remplacement afin de maintenir la casse et la ponctuation d'origine si possible.
    highlighted = re.sub(re.escape(target_sentence_cleaned), span_tag, context_text, 1, flags=re.IGNORECASE)

    # Vérifier si le remplacement a eu lieu.
    # Si la version surlignée ne contient pas la balise span, cela signifie que le remplacement a échoué.
    if span_tag not in highlighted:
        # Tenter une correspondance plus souple si la première échoue.
        # Par exemple, en normalisant les espaces.
        normalized_context = re.sub(r'\s+', ' ', context_text).strip()
        normalized_target = re.sub(r'\s+', ' ', target_sentence_cleaned).strip()

        # Essayer de remplacer sur la version normalisée si la phrase nettoyée est trouvée
        if normalized_target.lower() in normalized_context.lower():
            # Trouver l'index de la phrase normalisée dans le contexte normalisé
            idx_start = normalized_context.lower().find(normalized_target.lower())
            if idx_start != -1:
                idx_end = idx_start + len(normalized_target)
                
                # Reconstruire le highlighted_text en utilisant le contexte original
                # pour préserver la mise en forme des espaces/retours à la ligne du contexte original
                original_text_segment = normalized_context[idx_start:idx_end] # Non, il faut retrouver la portion dans le TEXTE ORIGINAL du contexte
                
                # C'est la partie la plus délicate. Trouver la correspondance exacte dans le contexte original.
                # On peut itérer sur les phrases du contexte pour trouver celle qui correspond le mieux.
                # Pour éviter la complexité excessive et le risque d'introduire "Information SQL",
                # si re.sub échoue la première fois, et que la phrase normalisée est trouvée,
                # on va re-surligner en utilisant la phrase trouvée dans le contexte.
                
                # Une approche plus simple pour la robustesse si la première re.sub échoue:
                # Chercher la phrase originale dans le contexte original avec une correspondance insensible à la casse
                # et insérer le span_tag. C'est ce que re.sub est censé faire.
                # Le problème est souvent que re.escape(target_sentence_cleaned) est trop strict.
                
                # Revenir à une approche plus simple si la première re.sub échoue :
                # Si le span_tag n'est pas dans highlighted, ça veut dire que la regex exacte n'a pas matché.
                # On peut alors juste retourner le contexte non surligné pour éviter l'erreur "Information SQL".
                return context_text, context_text # Retourne le contexte non surligné si le remplacement échoue.
        else:
            return context_text, context_text # Retourne le contexte non surligné si la phrase n'est pas trouvée même normalisée.

    return context_text, highlighted

# Liste des titres de sections qui indiquent une bibliographie ou des références.
REFERENCE_SECTIONS = [
    "Bibliographie", "Références bibliographiques", "Webographie",
    "Ouvrages", "Travaux scientifiques", "Dictionnaires"
]
# Pattern Regex pour détecter ces sections dans le texte.
REFERENCE_PATTERN = r'^(' + '|'.join([re.escape(s) for s in REFERENCE_SECTIONS]) + r')\s*$'

# Phrases courantes à ignorer lors de la détection de plagiat.
IGNORED_PHRASES = [
    "République Démocratique Du Congo",
    "ENSEIGNEMENT SUPÉRIEUR ET UNIVERSITAIRE",
    "INSTITUT SUPÉRIEUR D'INFORMATIQUE, PROGRAMMATION ET ANALYSE",
    "Tous droits réservés",
    "©Copyright",
    "Version actuelle"
]

def extract_referenced_sources(text):
    """
    Extrait les entrées de la bibliographie ou des sections de référence.
    L'idée est de trouver la dernière section de référence et de prendre tout le texte
    qui suit comme étant la liste des sources. Les lignes très courtes sont ignorées.
    """

    # Compile le pattern pour trouver les titres de section, insensible à la casse et sur plusieurs lignes
    pattern = re.compile(r'^(' + '|'.join(re.escape(s) for s in REFERENCE_SECTIONS) + r')\s*$', re.IGNORECASE | re.MULTILINE)
    matches = list(pattern.finditer(text))
    
    if not matches:
        return []

    # On prend tout le bloc de texte qui suit la dernière section de référence trouvée
    start_index = matches[-1].end()
    ref_block = text[start_index:].strip()

    # Découpe le bloc en lignes et ne garde que les lignes assez longues pour être des références valides
    return [line.strip() for line in ref_block.splitlines() if len(line.strip()) > 3]

def is_covered_by_bibliography(sentence, bibliography_refs, threshold=0.5):
    """
    Vérifie si une phrase ressemble fortement à l'une des entrées de la bibliographie.
    La fonction 'SequenceMatcher' est utilisée pour calculer une similarité "floue".
    Le 'threshold' (0.5 ici) détermine à quel point la similarité doit être élevée pour
    considérer la phrase comme "couverte" par la bibliographie.
    """
    for ref in bibliography_refs:
        ratio = SequenceMatcher(None, sentence.lower(), ref.lower()).ratio()
        if ratio > threshold:
            return True
    return False

def is_ignorable(sentence):
    """
    Détermine si une phrase doit être ignorée en cherchant si elle contient une des
    phrases listées dans 'IGNORED_PHRASES'. Simple et efficace pour les "faux positifs"
    connus.
    """
    return any(phrase.lower() in sentence.lower() for phrase in IGNORED_PHRASES)

def extract_footnotes(full_text):
    """
    Extrait les notes de bas de page du texte.
    Actuellement, ça cherche les notes au format '[Nombre] Contenu'.
    Si tu as d'autres formats de notes (comme '(1)'), il faudra ajuster cette expression régulière.
    """
    # Trouve les notes du type [1] Auteur, etc.
    footnote_matches = re.findall(r'\[(\d+)\]\s*(.+)', full_text)
    
    if not footnote_matches:
        return {}
    
    return {num: content for num, content in footnote_matches}

def is_cited_by_footnote(sentence, footnotes):
    """
    Vérifie si une phrase contient une référence à une note de bas de page et si
    cette note semble être une source (contient des mots-clés comme "http", "doi", etc.).
    C'est une vérification plus poussée pour s'assurer que la note est bien une citation.
    """
    # Cherche les numéros de citation entre crochets ou parenthèses
    matches = re.findall(r'\[(\d+)\]|\((\d+)\)', sentence)
    for match in matches:
        # Récupère le numéro de la citation
        num = match[0] if match[0] else match[1]
        if num and num in footnotes:
            # Vérifie si le contenu de la note contient des mots-clés qui indiquent une source
            if any(keyword in footnotes[num].lower() for keyword in ["http", "www", "doi", "isbn", "éditeur", "édition", "press", "university", "source", "auteur"]):
                return True
    return False

# Détection de citation plus robuste avec spaCy
def is_cited_with_nlp(sentence_text):
    doc = nlp_spacy(sentence_text)

    if re.search(r'\([A-Za-z\s,]+\s*\d{4}\)|\(\d{4}\)|\[\d+\]', sentence_text):
        return True

    for token in doc:
        # Vérifier les lemmes des verbes introducteurs
        if token.lemma_.lower() in ["selon", "d'après", "déclarer", "affirmer", "citer", "rapporter", "indiquer", "montrer"]:
            for next_token in doc[token.i+1:]:
                if next_token.ent_type_ in {"PER", "ORG", "LOC", "MISC"} or next_token.pos_ == "PROPN":
                    return True
                if next_token.is_punct and next_token.text in ['.', ';', '!', '?']:
                    break
                if next_token.pos_ == "VERB":
                    break
    return False

def compare_with_search_results(text):
    serper_api_key = os.getenv('SERPER_API_KEY')
    if not serper_api_key:
        return {"error": "API Key SERPER manquante."}

    all_results = []
    excerpts = []
    seen_phrases = set()
    highlighted_text = text
    footnotes = extract_footnotes(text)
    print(f"Foot notes : {footnotes}")

    match = re.search(
        r'^(Bibliographie|Références bibliographiques)\s*$',
        text, flags=re.MULTILINE|re.IGNORECASE
    )
    main_text = text[:match.start()] if match else text
    references = extract_referenced_sources(text)

    segments = split_into_semantic_chunks(main_text)

    for segment in segments:
        print(f"Segment to analyse : {segment}");
        print("==============================================")
        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": segment[:200]}

        try:
            response = requests.post("https://google.serper.dev/search", json=payload, headers=headers)
            response.raise_for_status()
            search_results = response.json().get('organic',)
            if not search_results:
                continue
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête Serper : {e}")
            continue

        doc_segment = nlp_spacy(segment)
        sentences = [sent.text for sent in doc_segment.sents]

        # --- Début des modifications pour la gestion de la mémoire ---
        # Collecter tous les textes à encoder
        texts_to_embed = sentences + [result.get('snippet', '') for result in search_results if result.get('snippet', '')]
        texts_to_embed = [t for t in texts_to_embed if t]

        if not texts_to_embed:
            continue

        # Définir une taille de sous-lot. Ajustez cette valeur si l'erreur de mémoire persiste.
        sub_batch_size = 1000 
        all_embeddings = []

        # Boucle pour encoder par sous-lots
        for i in range(0, len(texts_to_embed), sub_batch_size):
            sub_batch_texts = texts_to_embed[i:i + sub_batch_size]
            try:
                sub_batch_embeddings = model.encode(sub_batch_texts, convert_to_tensor=True, device=device)
                all_embeddings.append(sub_batch_embeddings)
            except RuntimeError as e:
                print(f"Erreur de mémoire lors de l'encodage du sous-lot. Réduire la taille du sous-lot ou revoir la taille du segment. Erreur: {e}")
                all_embeddings = None
                break

        if all_embeddings is None:
            continue

        all_embeddings = torch.cat(all_embeddings, dim=0)
        # Séparation des embeddings du document et des snippets
        num_sentences = len(sentences)
        sentence_embeddings_matrix = all_embeddings[:num_sentences]
        snippet_embeddings_matrix = all_embeddings[num_sentences:]

        for i, sentence in enumerate(sentences):
            if i >= len(sentence_embeddings_matrix):
                continue
            
            if is_ignorable(sentence):
                continue
            
            if sentence in seen_phrases:
                continue
            
            if len(sentence.split()) < 5 or len(sentence) < 40:
                continue
            
            if len(snippet_embeddings_matrix) > 0:
                # Utilisation de util.pytorch_cos_sim
                similarities = util.pytorch_cos_sim(sentence_embeddings_matrix[i], snippet_embeddings_matrix).cpu().numpy().flatten() * 100
            else:
                similarities = []

            for j, similarity in enumerate(similarities):
                if similarity > 60 and not is_cited_with_nlp(sentence) and not is_cited_by_footnote(sentence, footnotes) and not is_covered_by_bibliography(sentence, references):
                    # Trouver le résultat Serper correspondant
                    corresponding_result = search_results[j]
                    title = corresponding_result.get('title', '')
                    link = corresponding_result.get('link', '')

                    if sentence in highlighted_text:
                        data_id = len(all_results)
                        span_tag = f"<span class='plagiarized' data-id='{data_id}'>{sentence}</span>"
                        highlighted_text = highlighted_text.replace(sentence, span_tag)

                        all_results.append({
                            'id': data_id,
                            'title': title,
                            'link': link,
                            'similarity_percentage': round(similarity, 2),
                            'plagiarized_text': sentence
                        })
                        
                        seen_phrases.add(sentence)

                        context, highlighted = extract_with_context(text, sentence, data_id=data_id)
                        if context and highlighted:
                            excerpts.append({
                                'id': data_id,
                                'context': context,
                                'highlighted': highlighted
                            })

    unique_results = {r['plagiarized_text']: r for r in all_results}
    final_results = list(unique_results.values())
    
    plagiarized_word_count = sum(len(r['plagiarized_text'].split()) for r in final_results)
    total_word_count = len(text.split())
    plagiarism_percentage = round((plagiarized_word_count / total_word_count) * 100, 2) if total_word_count > 0 else 0

    print("________________________")
    print(f"Results of comparison : \n {all_results}")
    return {
        "full_text": text,
        "highlighted_text": highlighted_text,
        "plagiarism_percentage": plagiarism_percentage,
        "similarities": {
            "similarities": final_results,
            "excerpted_text": excerpts
        }
    }

def split_into_semantic_chunks(text: str, max_chunk_words: int = 75, min_chunk_words: int = 15) -> list[str]:
    if not text.strip():
        return []

    sentences = []
    if nlp_spacy is None:
        print("Avertissement : spaCy non disponible. Utilisation d'une division basique par phrase.")
        temp_sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in temp_sentences if s.strip()]
    else:
        doc = nlp_spacy(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    segments = []

    for para in paragraphs:
        sentences = []
        if nlp_spacy is None:
            temp_sentences = re.split(r'(?<=[.!?])\s+', para)
            sentences = [s.strip() for s in temp_sentences if len(s.strip().split()) >= 3]
        else:
            doc = nlp_spacy(para)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) >= 3]

        current_chunk_sentences = []
        current_chunk_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split()) # Compte les mots de la phrase actuelle

            if sentence_words > max_chunk_words:
                # Découpe la phrase en sous-parties
                words = sentence.split()
                for i in range(0, sentence_words, max_chunk_words):
                    part = " ".join(words[i:i+max_chunk_words])
                    if current_chunk_word_count >= min_chunk_words:
                        segments.append(" ".join(current_chunk_sentences))
                        current_chunk_sentences = [part]
                        current_chunk_word_count = len(part.split())
                    else:
                        current_chunk_sentences.append(part)
                        current_chunk_word_count += len(part.split())
                continue
            
            if (current_chunk_word_count + sentence_words > max_chunk_words) and (current_chunk_word_count >= min_chunk_words):
                segments.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_chunk_word_count = sentence_words
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_word_count += sentence_words

        if current_chunk_sentences:
            segments.append(" ".join(current_chunk_sentences))

    return segments

def detect_ai_generated_text(text):
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        return {"error": "OPENAI_API_KEY environment variable not set."}

    client = openai.OpenAI(api_key=openai_api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Tu es un détecteur d'IA. Réponds uniquement avec un pourcentage de probabilité que ce texte ait été généré par une IA."},
                {"role": "user", "content": f"Analyse ce texte et donne un score de 0 à 100 pour savoir s'il a été généré par une IA : {text}"}
            ],
        )

        result = response.choices.message.content
        probability = float(result.replace('%', '').strip())

        return {
            "text": text,
            "ai_generated_probability": probability,
            "interpretation": "Plus le score est élevé, plus le texte est susceptible d'avoir été généré par une IA."
        }
    except Exception as e:
        return {"error": f"Une erreur est survenue lors de la détection de l'IA : {str(e)}"}