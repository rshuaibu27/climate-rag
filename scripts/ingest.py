import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import json
import time

def fetch_climate_papers(query, limit=25):
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": limit,
        "filter": "has_abstract:true",
        "select": "id,title,abstract_inverted_index,authorships,publication_year"
    }

    response = requests.get(url, params=params, headers={"User-Agent": "climate-rag-project/1.0"})
    data = response.json()

    papers = []
    for paper in data.get("results", []):
        # OpenAlex stores abstracts in an inverted index format — we need to reconstruct it
        inverted = paper.get("abstract_inverted_index")
        if not inverted:
            continue

        # Reconstruct abstract from inverted index
        abstract = reconstruct_abstract(inverted)

        authors = [
            a["author"]["display_name"]
            for a in paper.get("authorships", [])
            if a.get("author")
        ]

        papers.append({
            "id": paper["id"],
            "title": paper["title"],
            "abstract": abstract,
            "authors": authors,
            "year": paper.get("publication_year"),
            "source": "openalex"
        })

    return papers


def reconstruct_abstract(inverted_index):
    """
    OpenAlex stores abstracts as an inverted index: {"word": [position1, position2], ...}
    This function reconstructs the original abstract string from that format.
    """
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


queries = [
    "sea level rise climate change",
    "global warming temperature projections",
    "Arctic ice melting climate",
    "climate change biodiversity loss",
    "carbon emissions reduction pathways",
]

all_papers = []
for query in queries:
    print(f"Fetching: {query}")
    papers = fetch_climate_papers(query, limit=20)
    all_papers.extend(papers)
    print(f"  Got {len(papers)} papers")
    time.sleep(1)

# Remove duplicates by id
seen = set()
unique_papers = []
for p in all_papers:
    if p["id"] not in seen:
        seen.add(p["id"])
        unique_papers.append(p)

print(f"\nFetched {len(unique_papers)} unique papers")

with open("data/papers.json", "w") as f:
    json.dump(unique_papers, f, indent=2)

print("Saved to data/papers.json")

from src.retriever import Retriever

print("\nIndexing papers into ChromaDB...")
retriever = Retriever()
retriever.index_papers(unique_papers)
print("Done! Papers are indexed and ready to search.")