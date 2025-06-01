from sentence_transformers import SentenceTransformer, util
import requests
import openai
import os
from dotenv import load_dotenv

load_dotenv()

import os
import requests
from sentence_transformers import SentenceTransformer, util

def compare_with_search_results(text):
    serper_api_key = os.getenv('SERPER_API_KEY')
    if not serper_api_key:
        return {"error": "API Key SERPER manquante."}
    
    query = text[:200]  # Limiter la requête pour éviter les erreurs (ex: 200 premiers caractères)
    
    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
    payload = {"q": query}
    
    try:
        response = requests.post("https://google.serper.dev/search", json=payload, headers=headers)
        response.raise_for_status()  # Vérifier les erreurs HTTP
        search_results = response.json().get('organic', [])
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur API Serper : {str(e)}"}
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Diviser le texte en phrases
    pdf_sentences = text.split('. ')  

    # Encoder toutes les phrases du texte en une seule fois pour l'optimisation
    pdf_embeddings = model.encode(pdf_sentences, convert_to_tensor=True)

    similarity_results = []
    highlighted_text = text  

    for result in search_results:
        search_text = result.get('snippet', '')
        search_title = result.get('title', '')
        search_link = result.get('link', '')

        if not search_text:
            continue  

        search_embedding = model.encode(search_text, convert_to_tensor=True)

        # Comparer chaque phrase avec le snippet trouvé sur Google
        for i, sentence in enumerate(pdf_sentences):
            similarity = util.pytorch_cos_sim(pdf_embeddings[i], search_embedding).item() * 100  

            if similarity > 60:  # Seuil de plagiat à 60%
                highlighted_text = highlighted_text.replace(sentence, f"**{sentence}**")
                
                similarity_results.append({
                    'title': search_title,
                    'link': search_link,
                    'similarity_percentage': round(similarity, 2),
                    'plagiarized_text': sentence
                })

    return {
        "full_text": text,
        "highlighted_text": highlighted_text,
        "similarities": similarity_results
    }


  
def detect_ai_generated_text(text):
    openai_api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=openai_api_key)  

    try:
        response = client.chat.completions.create(
            model="text-moderation-stable",
            store=True,
            messages=[
                {"role": "system", "content": "Tu es un détecteur d'IA. Réponds uniquement avec un pourcentage de probabilité que ce texte ait été généré par une IA."},
                {"role": "user", "content": f"Analyse ce texte et donne un score de 0 à 100 pour savoir s'il a été généré par une IA : {text}"}
            ],
            
        )

        result = response.choices[0].message.content  # Extraire le texte de la réponse
        probability = float(result.replace('%', '').strip())  # Convertir en nombre

        return {
            "text": text,
            "ai_generated_probability": probability,
            "interpretation": "Plus le score est élevé, plus le texte est susceptible d'avoir été généré par une IA."
        }
    except Exception as e:
        return {"error": f"Une erreur est survenue : {str(e)}"}
