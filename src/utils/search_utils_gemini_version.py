from google import genai
import logging
import os
from dotenv import load_dotenv
import re
import base64
import json
import mimetypes
import pathlib
import requests
import PyPDF2
import docx
from google.genai import types

from google.genai.types import (
    File
)

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
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
            return text
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
    Tu es un expert en détection de plagiat. Ton rôle est d'analyser le document ci-dessous et d'identifier les 
    phrases ou paragraphes qui sont très similaires à du contenu existant publiquement sur internet. 
    Pour chaque cas de plagiat que tu trouves, tu dois fournir un résultat au format JSON.

    **Consignes importantes :**
    1. **Utilise la recherche sur le web :** Utilise tes outils de recherche pour identifier 
    les sources potentielles de plagiat pour chaque passage que tu suspectes.
    2. **Sois précis :** Fournis le texte exact qui semble plagié.
    3. **Identifie la source :** Fournis l'URL de la source web où tu as trouvé le passage similaire.
    4. **Estime la similarité :** Donne une estimation en pourcentage de la similarité.
    5. **Gère les exceptions :** Ne considère pas comme plagiat les phrases d'introduction, 
    les titres, les listes de référence, les citations correctement formatées (ex. : "Selon [auteur], ...", et les citations en notes en bas de pages), ou 
    les phrases génériques et très courantes. Concentre-toi sur les textes qui semblent être copiés/collés sans attribution.

    **Format de sortie JSON :**
    Le format JSON doit être une liste d'objets. Chaque objet doit contenir les clés suivantes :
    - "plagiarized_text": Le texte exact qui semble plagié.
    - "title": Le titre de la page web source.
    - "link": L'URL de la source web trouvée.
    - "similarity_percentage": Un pourcentage de similarité estimé entre le texte plagié et sa source présumée.

    Si aucun plagiat n'est détecté, renvoie un tableau vide [].
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
    
    for i, item in enumerate(results):
        plagiarized_text = item.get("plagiarized_text", "")
        # Pour le surlignage, remplace le texte plagié par un tag HTML
        if plagiarized_text in highlighted_text:
            data_id = i
            span_tag = f"<span class='plagiarized' data-id='{data_id}'>{plagiarized_text}</span>"
            highlighted_text = highlighted_text.replace(plagiarized_text, span_tag, 1) # Remplace seulement la première occurrence
            
            plagiarized_word_count += len(plagiarized_text.split())
            
            # Créer l'extrait avec le texte surligné (cela est simplifié, car Gemini ne donne pas le contexte)
            excerpted_text.append({
                'id': data_id,
                'context': plagiarized_text, # ou le trouver dans le texte original
                'highlighted': span_tag
            })
            
    total_word_count = len(document_text.split())
    plagiarism_percentage = round((plagiarized_word_count / total_word_count) * 100, 2) if total_word_count > 0 else 0
    
    return {
        "full_text": document_text,
        "highlighted_text": highlighted_text,
        "plagiarism_percentage": plagiarism_percentage,
        "similarities": {
            "similarities": results,
            "excerpted_text": excerpted_text
        }
    }