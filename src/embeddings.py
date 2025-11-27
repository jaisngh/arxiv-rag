"""Embeddings module using Ollama for vector generation."""

import ollama
from typing import Union

from .config import Config


class EmbeddingGenerator:
    """Generates embeddings using Ollama's embedding models."""
    
    def __init__(self, model: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model: Ollama model name for embeddings (default: nomic-embed-text)
        """
        self.model = model or Config.OLLAMA_EMBED_MODEL
        self.client = ollama.Client(host=Config.OLLAMA_HOST)
        
    def ensure_model(self) -> bool:
        """
        Ensure the embedding model is available, pulling if necessary.
        
        Returns:
            True if model is ready, False otherwise
        """
        try:
            # Try to get model info
            self.client.show(self.model)
            return True
        except ollama.ResponseError:
            # Model not found, try to pull it
            try:
                print(f"Pulling embedding model: {self.model}...")
                self.client.pull(self.model)
                print(f"Successfully pulled {self.model}")
                return True
            except Exception as e:
                print(f"Failed to pull model {self.model}: {e}")
                return False
                
    def generate(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        response = self.client.embed(model=self.model, input=text)
        return response["embeddings"][0]
        
    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        response = self.client.embed(model=self.model, input=texts)
        return response["embeddings"]
        
    def generate_for_paper(self, title: str, abstract: str) -> list[float]:
        """
        Generate embedding for a paper by combining title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            Embedding vector for the combined text
        """
        combined_text = f"Title: {title}\n\nAbstract: {abstract}"
        return self.generate(combined_text)


# Singleton instance
embedder = EmbeddingGenerator()

