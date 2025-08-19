from google import genai
import logging
import os
from dotenv import load_dotenv
import json
import PyPDF2
import docx
from google.genai import types
import fitz

# Charge les variables d'environnement depuis le fichier.env
load_dotenv()

def extract_text_from_file(file_path):
    """
    Extrait le texte d'un fichier en fonction de son extension.
    Prend en charge les fichiers .txt, .pdf et .docx.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_extension == '.pdf':
        try:
            doc = fitz.open(file_path)
            full_text = []
            for page in doc:
                full_text.append(page.get_text())
            return "\n".join(full_text)
        except Exception as e:
            # En cas d'échec (PDF corrompu, etc.), on peut logguer l'erreur
            # ou tenter une autre méthode moins performante.
            print(f"Erreur d'extraction avec PyMuPDF: {e}. Tentative avec PyPDF2.")
            # Optionnel: Revenir à l'ancienne méthode si la nouvelle échoue
            try:
                text = ""
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            except Exception as e:
                raise ValueError(f"Impossible d'extraire le texte du PDF. Erreur: {e}")

    elif file_extension == '.docx':
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    else:
        raise ValueError(f"Type de fichier non supporté: {file_extension}")

def analyze_document_with_gemini(document_text):
    """
    Analyse un document texte avec l'API Gemini pour détecter le plagiat.

    Args:
        document_text (str): Le texte complet du document à analyser.

    Returns:
        dict: Un dictionnaire contenant les résultats de l'analyse,
            ou un dictionnaire d'erreur.
    """
    # Prompting pour le modèle Gemini
    prompt_text = """
    Tu es un expert en détection de plagiat. Ton rôle est d'analyser le document ci-dessous pour trouver des phrases ou des passages très similaires à du contenu existant sur le web.

    **Instructions strictes :**
    1.  **Recherche web obligatoire :** Utilise tes outils de recherche pour identifier les sources potentielles.
    2.  **Exclusivité du résultat :** La seule chose que tu dois renvoyer est un tableau JSON, et rien d'autre. Pas d'explications, pas de texte d'introduction, rien avant ou après le JSON.
    3.  **Ne considère un passage comme plagié que s’il contient au moins **7 mots consécutifs**.
    4.  **Contenu à ignorer :** Ne signale pas comme plagiat : 
        - Les titres, les listes de références, les citations claires (ex: "Selon [auteur], ...", ou les notes de bas de page).
        - Les phrases génériques et très courtes (moins de 5 mots).
        - Les définitions générales, les slogans connus, ni les termes techniques universels.
        - Les termes techniques, les noms de concepts ou les acronymes courants (ex: "Programmation Orientée Objet", "API REST", "IA", "HTML").
        - Les phrases d'introduction ou de conclusion standard.
        - Ignore complètement les expressions génériques et académiques courantes, par exemple :
            - "Massive Open Online Courses"
            - "Intelligence Artificielle"
            - "Programmation Orientée Objet"
            - "Gestion de Projet"
            - "Technologies de l’Information et de la Communication"
            - (et toute autre expression du même type, considérée comme standard)
    5.  **Retour verbatim :** Le texte que tu identifies comme plagié dans le document (`plagiarized_text`) doit être renvoyé **mot pour mot, caractère pour caractère**, exactement comme il apparaît dans le document d'origine. Ne le modifie pas, même pas un article, une majuscule ou un point.
    6.  **Format de l'objet JSON :** Chaque objet dans le tableau doit avoir les clés suivantes exactement comme elles sont écrites :
        - `plagiarized_text` (string) : **Le passage exact du document qui est plagié, sans aucune modification.**
        - `title` (string) : Le titre de la source web.
        - `link` (string) : L'URL complète de la source web.
        - `similarity_percentage` (number) : Un pourcentage de similarité estimé (par exemple, 95).
        - `context` (string) : Le paragraphe ou la phrase complète du document original où se trouve le `plagiarized_text`.

    **Exemple de format de sortie si plagiat détecté :**
    [
    {
        "plagiarized_text": "L'éducation est la clé pour libérer l'esprit humain et favoriser le progrès social.",
        "title": "Un article sur l'importance de l'éducation",
        "link": "https://example.com/article-education",
        "similarity_percentage": 98,
        "context": "Dans la section sur les fondements de la société, il a été souligné que l'éducation est la clé pour libérer l'esprit humain et favoriser le progrès social, en ouvrant de nouvelles perspectives pour les générations futures."
    }
    ]

    **Exemple de format de sortie si aucun plagiat n'est détecté :**
    []

    **Document à analyser ci-dessous. Tu dois traiter son contenu et renvoyer le JSON en tant que réponse finale.**
    ---
    """
    
    # Créer un objet client centralisé
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    except Exception as e:
        return {"error": f"Impossible d'initialiser le client Gemini: {str(e)}"}

    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[document_text],
            config=types.GenerateContentConfig(
                system_instruction=prompt_text,
                temperature=0,
                response_mime_type="application/json",
                # tools=[types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())]
            )
        )

        if response and response.text:
            plagiarism_data = json.loads(response.text)
            return {"results": plagiarism_data}
        else:
            return {"error": "Aucune réponse de l'API Gemini."}

    except Exception as e:
        print(f"Erreur lors de l'appel à l'API Gemini: {e}")
        return {"error": str(e)}

def main_plagiarism_checker(document_text):
    """
    Fonction principale pour l'analyse de plagiat.
    Elle fait appel à la nouvelle fonction Gemini.
    """
    
    # Appeler la fonction Gemini pour obtenir les résultats de plagiat
    gemini_response = analyze_document_with_gemini(document_text)
    
    if "error" in gemini_response:
        return gemini_response # Retourne l'erreur si l'API a échoué
    
    results = gemini_response["results"]
    
    if not results:
        return {
            "full_text": document_text,
            "highlighted_text": document_text,
            "plagiarism_percentage": 0,
            "similarities": {
                "similarities": [],
                "excerpted_text": []
            }
        }
        
    highlighted_text = document_text
    plagiarized_word_count = 0
    excerpted_text = []
    
    final_similarities = []

    for i, item in enumerate(results):
        # Création de l'objet pour la liste finale.
        # On y ajoute un ID directement.
        plagiarism_data = {
            'id': i,
            'plagiarized_text': item.get('plagiarized_text', ''),
            'title': item.get('title', 'Titre non disponible'),
            'link': item.get('link', '#'),
            'similarity_percentage': item.get('similarity_percentage', 0),
        }
        final_similarities.append(plagiarism_data)

        # On extrait les données pour le surlignage et les extraits
        plagiarized_text = plagiarism_data['plagiarized_text']
        context = item.get('context', '')
        
        # Le reste de la logique pour les extraits et le surlignage
        if plagiarized_text and context:
            # Créer la balise surlignée avec l'ID
            span_tag = f"<span class='plagiarized' data-id='{i}'>{plagiarized_text}</span>"
            
            # Utiliser le contexte directement fourni par Gemini
            highlighted_context = context.replace(plagiarized_text, span_tag, 1)

            # Ajouter l'extrait formaté à la liste
            excerpted_text.append({
                'id': i,
                'context': context,
                'highlighted': highlighted_context
            })
            
            # Surligner le texte complet du document
            if plagiarized_text in highlighted_text:
                highlighted_text = highlighted_text.replace(plagiarized_text, span_tag, 1)
                plagiarized_word_count += len(plagiarized_text.split())

    total_word_count = len(document_text.split())
    plagiarism_percentage = round((plagiarized_word_count / total_word_count) * 100, 2) if total_word_count > 0 else 0
    
    return {
        "full_text": document_text,
        "highlighted_text": highlighted_text,
        "plagiarism_percentage": plagiarism_percentage,
        "similarities": {
            "similarities": final_similarities,
            "excerpted_text": excerpted_text
        }
    }