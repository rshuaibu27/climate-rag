from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        print(f"Loading embedding model: {self.MODEL_NAME}...")
        self.model = SentenceTransformer(self.MODEL_NAME)
        print("Ready.")

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_query(self, query):
        return self.embed(query)[0]