# Climate Research Navigator

A RAG-powered research assistant that answers climate science questions 
grounded in real academic papers and IPCC findings — not hallucinated generalities.

Built to explore how retrieval-augmented generation can make scientific 
literature more accessible to students and researchers.

## How it works
User question
↓
Embed query into a vector
↓
Search ChromaDB for the closest matching papers
↓
Pass retrieved papers + question to LLM
↓
Grounded, cited answer

## Stack
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector store:** ChromaDB
- **LLM:** Llama 3.2 via Ollama (runs locally)
- **UI:** Streamlit
- **Data:** OpenAlex API (100 real climate science papers)

## Running locally

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Install and start Ollama: https://ollama.com, then `ollama pull llama3.2`
4. Fetch and index papers: `python scripts/ingest.py`
5. Launch the app: `streamlit run app.py`

## Known limitations

- **Boolean logic gap:** The embedding model can't enforce hard constraints 
  like "after 2020" or "specifically about coral reefs" — retrieval finds 
  semantically similar papers, not exact matches
- **Context window:** Very long conversations cause the LLM to lose 
  earlier context as the history grows
- **Dataset size:** 100 papers is enough to demo the concept but a 
  production system would index thousands