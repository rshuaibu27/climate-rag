import ollama
from src.retriever import Retriever

SYSTEM_PROMPT = """You are a climate science research assistant helping students and researchers understand climate science literature.

You will be given a set of real academic papers and IPCC findings relevant to the user's question.

Your job is to:
1. Answer the question using ONLY the provided papers — do not use outside knowledge
2. Cite papers by their title and year when you reference them
3. Be honest if the retrieved papers don't fully answer the question
4. Use clear, precise language appropriate for students and researchers

If the papers don't contain enough information to answer confidently, say so."""


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.conversation_history = []

    def chat(self, query):
        # Step 1: Retrieve relevant papers
        papers = self.retriever.retrieve(query, top_k=5)

        # Step 2: Format them into context
        context = self._format_papers(papers)

        # Step 3: Build augmented message
        user_message = f"""Question: {query}

Relevant research:
{context}

Answer based on the research above, citing papers where relevant."""

        # Step 4: Add to history and call LLM
        self.conversation_history.append({"role": "user", "content": user_message})

        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history
        )

        reply = response["message"]["content"]
        self.conversation_history.append({"role": "assistant", "content": reply})

        return reply, papers

    def _format_papers(self, papers):
        lines = []
        for i, p in enumerate(papers, 1):
            lines.append(
                f"{i}. {p['title']} ({p['year']}) — {p['authors']}\n"
                f"   Relevance score: {p['score']}\n"
            )
        return "\n".join(lines)

    def reset(self):
        self.conversation_history = []