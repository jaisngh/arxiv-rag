"""Paper ingestion module for fetching and indexing arXiv papers."""

from typing import Optional

from .arxiv_fetcher import ArxivFetcher, Paper, POPULAR_CATEGORIES
from .embeddings import embedder
from .database import db


class PaperIngestor:
    """Handles fetching papers from arXiv and indexing them in the database."""
    
    def __init__(self):
        self.fetcher = ArxivFetcher()
        
    def ingest_paper(self, paper: Paper) -> bool:
        """
        Ingest a single paper into the database.
        
        Args:
            paper: Paper object to ingest
            
        Returns:
            True if paper was ingested, False if it already existed
        """
        # Check if paper already exists
        if db.paper_exists(paper.arxiv_id):
            return False
            
        # Generate embedding
        embedding = embedder.generate_for_paper(paper.title, paper.abstract)
        
        # Insert into database
        db.insert_paper(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            abstract=paper.abstract,
            authors=paper.authors,
            categories=paper.categories,
            published_date=paper.published_date,
            embedding=embedding,
        )
        
        return True
        
    def ingest_from_search(
        self,
        query: str,
        max_papers: int = 50,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """
        Ingest papers from an arXiv search query.
        
        Args:
            query: Search query
            max_papers: Maximum papers to fetch
            progress_callback: Optional callback(current, total, paper_title)
            
        Returns:
            Dict with ingestion statistics
        """
        stats = {"fetched": 0, "ingested": 0, "skipped": 0}
        
        papers = list(self.fetcher.search(query, max_papers))
        total = len(papers)
        
        for i, paper in enumerate(papers):
            stats["fetched"] += 1
            
            if progress_callback:
                progress_callback(i + 1, total, paper.title)
                
            if self.ingest_paper(paper):
                stats["ingested"] += 1
            else:
                stats["skipped"] += 1
                
        return stats
        
    def ingest_from_category(
        self,
        category: str,
        max_papers: int = 50,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """
        Ingest papers from a specific arXiv category.
        
        Args:
            category: arXiv category code
            max_papers: Maximum papers to fetch
            progress_callback: Optional callback(current, total, paper_title)
            
        Returns:
            Dict with ingestion statistics
        """
        stats = {"fetched": 0, "ingested": 0, "skipped": 0}
        
        papers = list(self.fetcher.fetch_by_category(category, max_papers))
        total = len(papers)
        
        for i, paper in enumerate(papers):
            stats["fetched"] += 1
            
            if progress_callback:
                progress_callback(i + 1, total, paper.title)
                
            if self.ingest_paper(paper):
                stats["ingested"] += 1
            else:
                stats["skipped"] += 1
                
        return stats


# Singleton instance
ingestor = PaperIngestor()

