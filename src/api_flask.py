# from flask import Flask, request, jsonify
# from utils.search_utils import compare_with_search_results
# from dotenv import load_dotenv
# import logging
# import os

# load_dotenv()

# app2 = Flask(__name__)
# logging.basicConfig(level=logging.DEBUG)

# plagiarism_engine = AcademicPlagiarismEngine(
#     embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
#     semantic_threshold=70.0
# )


# @app.route('/', methods=['GET'])
# def voir_sur_web():
#     return jsonify({'salutation': 'hello'})


# @app.route('/check-plagiarism', methods=['POST'])
# def check_plagiarism():
#     logging.info('Request received by Flask API (/check-plagiarism)')

#     try:
#         payload = request.get_json(silent=True) or {}

#         document_text = (
#             payload.get('document_text')
#             or payload.get('text')
#             or payload.get('content')
#             or ''
#         )
#         source_documents = payload.get('source_documents') or []

#         if not isinstance(document_text, str) or not document_text.strip():
#             return jsonify({'error': 'Missing document_text'}), 400

#         if not isinstance(source_documents, list) or len(source_documents) == 0:
#             return jsonify({'error': 'Missing source_documents (corpus)'}), 400

#         result = plagiarism_engine.compare_document_against_corpus(
#             document_text=document_text.strip(),
#             corpus_documents=source_documents
#         )
#         return jsonify({'similarities': response}), 200

#     except Exception as e:
#         logging.exception('check_plagiarism failed')
#         return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


# @app.route('/extract-text/docx', methods=['POST'])
# def extract_docx():
#     file = request.files.get('file')
#     if not file:
#         return jsonify({'error': 'Aucun fichier envoye.'}), 400

#     try:
#         document = Document(file)
#         text = '\n'.join([para.text for para in document.paragraphs if para.text.strip()])
#         return jsonify({'text': text.strip()}), 200
#     except Exception as e:
#         return jsonify({'error': f'Erreur lors de l extraction: {str(e)}'}), 500


# if __name__ == '__main__':
#     host = os.getenv('FLASK_HOST', '127.0.0.1')
#     port = int(os.getenv('FLASK_PORT', '5050'))
#     debug = os.getenv('FLASK_DEBUG', '1') == '1'
#     app.run(host=host, port=port, debug=debug, use_reloader=False)
