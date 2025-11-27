"""ArXiv paper fetcher module."""

import arxiv
from typing import Generator
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Paper:
    """Data class representing an arXiv paper."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published_date: datetime


class ArxivFetcher:
    """Fetches papers from arXiv API."""
    
    def __init__(self):
        self.client = arxiv.Client()
        
    def search(
        self,
        query: str,
        max_results: int = 50,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    ) -> Generator[Paper, None, None]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search query (supports arXiv query syntax)
            max_results: Maximum number of results to return
            sort_by: Sort criterion (Relevance, LastUpdatedDate, SubmittedDate)
            sort_order: Sort order (Ascending, Descending)
            
        Yields:
            Paper objects matching the search criteria
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        
        for result in self.client.results(search):
            yield Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title.replace("\n", " "),
                abstract=result.summary.replace("\n", " "),
                authors=[author.name for author in result.authors],
                categories=list(result.categories),
                published_date=result.published,
            )
            
    def fetch_by_category(
        self,
        category: str,
        max_results: int = 50,
    ) -> Generator[Paper, None, None]:
        """
        Fetch papers from a specific arXiv category.
        
        Args:
            category: arXiv category (e.g., 'cs.AI', 'cs.LG', 'physics.gen-ph')
            max_results: Maximum number of results to return
            
        Yields:
            Paper objects in the specified category
        """
        query = f"cat:{category}"
        yield from self.search(query, max_results)
        
    def fetch_recent(
        self,
        categories: list[str] = None,
        max_results: int = 50,
    ) -> Generator[Paper, None, None]:
        """
        Fetch recent papers, optionally filtered by categories.
        
        Args:
            categories: List of arXiv categories to filter by
            max_results: Maximum number of results to return
            
        Yields:
            Recent paper objects
        """
        if categories:
            query = " OR ".join(f"cat:{cat}" for cat in categories)
        else:
            query = "all"
            
        yield from self.search(
            query,
            max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )


# Common arXiv categories for reference
POPULAR_CATEGORIES = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning", 
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision",
    "cs.NE": "Neural and Evolutionary Computing",
    "stat.ML": "Machine Learning (Statistics)",
    "cs.RO": "Robotics",
    "cs.IR": "Information Retrieval",
    "cs.HC": "Human-Computer Interaction",
    "physics.gen-ph": "General Physics",
    "math.OC": "Optimization and Control",
    "quant-ph": "Quantum Physics",
}

