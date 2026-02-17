from flask import Flask
import spacy
import logging
from ml_utils.embedding_model import EmbeddingModel
from ml_utils.vector_index import VectorIndex
from api.routes import register_routes
from config import EMBEDDING_MODEL_NAME

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Force visible logs in console for incoming Laravel requests.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
app.logger.setLevel(logging.INFO)

nlp = spacy.load("fr_core_news_sm")
embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)

# ⚠️ L’index FAISS doit être chargé ici (persisté)
vector_index = VectorIndex(dimension=384)

register_routes(app, embedding_model, vector_index, nlp)

# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5050
    debug = True
    app.run(host=host, port=port, debug=debug, use_reloader=False)
