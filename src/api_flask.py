from flask import Flask, request, jsonify
from utils.search_utils import compare_with_search_results, detect_ai_generated_text
from dotenv import load_dotenv
import logging

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['GET'])
def voir_sur_web():
    return jsonify({'salutation': 'hello'})

@app.route('/check-plagiarism', methods=['POST'])
def check_plagiarism():
    data = request.get_json()
    text = data.get('text')
    logging.debug(f"Received text: {text}")

    if not text:
        logging.error("Missing text")
        return jsonify({'error': 'Missing text'}), 400

    try:
        if not text.strip():
            logging.error("No text provided")
            return jsonify({'error': 'No text provided'}), 400

        # Vérification de similarité
        similarities = compare_with_search_results(text)

        # Détection de texte généré par l'IA
        ai_generated_probability = detect_ai_generated_text(text)

        return jsonify({
            'similarities': similarities,
            'ai_generated_probability': ai_generated_probability
        })

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
