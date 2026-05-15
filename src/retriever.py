import chromadb
import json
from src.embedder import Embedder

COLLECTION_NAME = "climate_papers"

class Retriever:
    def __init__(self, persist_dir="./chroma_store"):
        self.embedder = Embedder()
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def index_papers(self, papers):
        existing_ids = set(self.collection.get()["ids"])

        ids, embeddings, metadatas, documents = [], [], [], []

        for paper in papers:
            if paper["id"] in existing_ids:
                continue

            text = paper["title"] + ". " + paper["abstract"]
            embedding = self.embedder.embed(text)[0].tolist()

            ids.append(paper["id"])
            embeddings.append(embedding)
            documents.append(text)
            metadatas.append({
                "title": paper["title"],
                "authors": ", ".join(paper["authors"][:3]),
                "year": str(paper.get("year", "Unknown")),
                "source": paper["source"]
            })

        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Indexed {len(ids)} new papers.")
        else:
            print("All papers already indexed.")

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedder.embed_query(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["metadatas", "distances"]
        )

        papers = []
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
            paper = dict(metadata)
            paper["score"] = round(1 - distance, 4)
            papers.append(paper)

        return papers