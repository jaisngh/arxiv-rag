"""RAG (Retrieval-Augmented Generation) pipeline using Ollama."""

import ollama
from typing import Optional

from .config import Config
from .database import db
from .embeddings import embedder


class RAGPipeline:
    """RAG pipeline for answering questions using arXiv papers."""
    
    def __init__(self, llm_model: str = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            llm_model: Ollama LLM model for generation
        """
        self.llm_model = llm_model or Config.OLLAMA_LLM_MODEL
        self.client = ollama.Client(host=Config.OLLAMA_HOST)
        self.top_k = Config.TOP_K_RESULTS
        
    def ensure_model(self) -> bool:
        """
        Ensure the LLM model is available, pulling if necessary.
        
        Returns:
            True if model is ready, False otherwise
        """
        try:
            self.client.show(self.llm_model)
            return True
        except ollama.ResponseError:
            try:
                print(f"Pulling LLM model: {self.llm_model}...")
                self.client.pull(self.llm_model)
                print(f"Successfully pulled {self.llm_model}")
                return True
            except Exception as e:
                print(f"Failed to pull model {self.llm_model}: {e}")
                return False
                
    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Retrieve relevant papers for a query.
        
        Args:
            query: Natural language query
            top_k: Number of results to retrieve
            
        Returns:
            List of relevant paper dictionaries
        """
        k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = embedder.generate(query)
        
        # Search for similar papers
        results = db.search_similar(query_embedding, k)
        
        return results
        
    def _build_context(self, papers: list[dict]) -> str:
        """Build context string from retrieved papers."""
        context_parts = []
        
        for i, paper in enumerate(papers, 1):
            authors = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                authors += " et al."
                
            context_parts.append(
                f"[Paper {i}]\n"
                f"Title: {paper['title']}\n"
                f"Authors: {authors}\n"
                f"arXiv ID: {paper['arxiv_id']}\n"
                f"Abstract: {paper['abstract']}\n"
            )
            
        return "\n---\n".join(context_parts)
        
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the LLM."""
        return f"""You are a helpful research assistant with expertise in analyzing scientific papers. 
Use the following research papers to answer the user's question. Base your answer on the provided papers.
If the papers don't contain enough information to fully answer the question, acknowledge this limitation.
Always cite which paper(s) you're referencing in your answer using their arXiv IDs.

RESEARCH PAPERS:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer based on the papers above:"""

    def query(
        self,
        question: str,
        top_k: int = None,
        stream: bool = False,
    ):
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            top_k: Number of papers to retrieve
            stream: Whether to stream the response
            
        Returns:
            Generated answer (or generator if streaming)
        """
        # Retrieve relevant papers
        papers = self.retrieve(question, top_k)
        
        if not papers:
            return {
                "answer": "I couldn't find any relevant papers in the database. Please try indexing some papers first.",
                "sources": [],
            }
            
        # Build context and prompt
        context = self._build_context(papers)
        prompt = self._build_prompt(question, context)
        
        if stream:
            return self._stream_response(prompt, papers)
        else:
            return self._generate_response(prompt, papers)
            
    def _generate_response(self, prompt: str, papers: list[dict]) -> dict:
        """Generate a complete response."""
        response = self.client.generate(
            model=self.llm_model,
            prompt=prompt,
        )
        
        return {
            "answer": response["response"],
            "sources": [
                {
                    "arxiv_id": p["arxiv_id"],
                    "title": p["title"],
                    "similarity": p.get("similarity", 0),
                }
                for p in papers
            ],
        }
        
    def _stream_response(self, prompt: str, papers: list[dict]):
        """Stream the response token by token."""
        stream = self.client.generate(
            model=self.llm_model,
            prompt=prompt,
            stream=True,
        )
        
        for chunk in stream:
            yield {
                "token": chunk["response"],
                "done": chunk.get("done", False),
                "sources": [
                    {
                        "arxiv_id": p["arxiv_id"],
                        "title": p["title"],
                        "similarity": p.get("similarity", 0),
                    }
                    for p in papers
                ] if chunk.get("done", False) else None,
            }


# Singleton instance
rag = RAGPipeline()

