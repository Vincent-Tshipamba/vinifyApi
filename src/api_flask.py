from flask import Flask, request, jsonify
from docx import Document
from utils.search_utils_gemini_version import analyze_document_with_gemini, extract_text_from_file, main_plagiarism_checker
from dotenv import load_dotenv
import logging
import os

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['GET'])
def voir_sur_web():
    return jsonify({'salutation': 'hello'})

@app.route('/check-plagiarism', methods=['POST'])
def check_plagiarism():
    logging.info("🔍 Requête reçue par l’API Flask")
    
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400
    
    temp_dir = 'temp_uploads'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)
    
    full_text = None
    try:
        # Étape 1 : Extraire le texte du fichier sauvegardé
        full_text = extract_text_from_file(temp_path)

        # Étape 2 : Lancer le checker principal avec le texte extrait
        response = main_plagiarism_checker(full_text)
        
        logging.info(f"Plagiarism check completed for {file.filename}. Response: {response}")

        # Étape 3 : Supprimer le fichier temporaire
        os.remove(temp_path)
        logging.info(f"Deleted local file {temp_path}.")

        return jsonify(response)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logging.info(f"Deleted local file {temp_path} after error.")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
# @app.route('/check-plagiarism', methods=['POST'])
# def check_plagiarism():
#     print("🔍 Requête reçue par l’API Flask")
#     data = request.get_json()
#     text = data.get('text')
#     logging.debug(f"Received text: {text}")

#     if not text:
#         logging.error("Missing text")
#         return jsonify({'error': 'Missing text'}), 400

#     try:
#         if not text.strip():
#             logging.error("No text provided")
#             return jsonify({'error': 'No text provided'}), 400

#         # Vérification de similarité
#         similarities = compare_with_search_results(text)

#         if "error" in similarities:
#             logging.error(f"Erreur pendant la détection : {similarities['error']}")
#             return jsonify({'error': similarities["error"]}), 500

#         # Détection de texte généré par l'IA
#         # ai_generated_probability = detect_ai_generated_text(text)
#         label = 'Real'
        
#         if label in ['fake', 'Fake', 'FAKE']:
#             is_ai_generated = True
#         else:
#             is_ai_generated = False

#         return jsonify({
#             'similarities': similarities,
#             'is_ai_generated': is_ai_generated,
#             'ai_generated_label': label,
#             'ai_generated_probability': 0,
#         })

#     except Exception as e:
#         logging.error(f"An error occurred: {str(e)}")
#         return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/extract-text/docx', methods=['POST'])
def extract_docx():
    file = request.files.get('file')
    print("Fichier reçu :", file.filename)
    if not file:
        return jsonify({"error": "Aucun fichier envoyé."}), 400

    try:
        document = Document(file)
        text = '\n'.join([para.text for para in document.paragraphs if para.text.strip()])
        return jsonify({"text": text.strip()}), 200
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'extraction : {str(e)}"}), 500


# @app.route('/detect-ai', methods=['POST'])
# def detect_ai():
#     data = request.get_json()
#     text = data.get('text')
#     if not text:
#         return jsonify({"error": "Texte manquant"}), 400
    
#     result = detect_ai_generated_text(text)
#     return jsonify({
#         "label": result["label"],
#         "score": round(result["score"], 2)
#     })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
