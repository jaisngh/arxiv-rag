"""Database module for PostgreSQL with pgvector support."""

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from typing import Optional

from .config import Config


class Database:
    """PostgreSQL database handler with pgvector support."""
    
    def __init__(self):
        self.conn: Optional[psycopg2.extensions.connection] = None
        
    def connect(self) -> None:
        """Establish database connection."""
        self.conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            dbname=Config.POSTGRES_DB,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD,
        )
        register_vector(self.conn)
        
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def init_schema(self) -> None:
        """Initialize database schema with pgvector extension."""
        if not self.conn:
            self.connect()
            
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create papers table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS papers (
                    id SERIAL PRIMARY KEY,
                    arxiv_id VARCHAR(50) UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    abstract TEXT NOT NULL,
                    authors TEXT[],
                    categories TEXT[],
                    published_date TIMESTAMP,
                    embedding vector({Config.EMBEDDING_DIM}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for vector similarity search (using HNSW for better performance)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS papers_embedding_idx 
                ON papers 
                USING hnsw (embedding vector_cosine_ops)
            """)
            
            self.conn.commit()
            
    def paper_exists(self, arxiv_id: str) -> bool:
        """Check if a paper already exists in the database."""
        if not self.conn:
            self.connect()
            
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM papers WHERE arxiv_id = %s)",
                (arxiv_id,)
            )
            return cur.fetchone()[0]
            
    def insert_paper(
        self,
        arxiv_id: str,
        title: str,
        abstract: str,
        authors: list[str],
        categories: list[str],
        published_date,
        embedding: list[float],
    ) -> int:
        """Insert a new paper with its embedding."""
        if not self.conn:
            self.connect()
            
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO papers (arxiv_id, title, abstract, authors, categories, published_date, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    abstract = EXCLUDED.abstract,
                    authors = EXCLUDED.authors,
                    categories = EXCLUDED.categories,
                    published_date = EXCLUDED.published_date,
                    embedding = EXCLUDED.embedding
                RETURNING id
                """,
                (arxiv_id, title, abstract, authors, categories, published_date, embedding)
            )
            paper_id = cur.fetchone()[0]
            self.conn.commit()
            return paper_id
            
    def search_similar(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for papers similar to the query embedding using cosine similarity."""
        if not self.conn:
            self.connect()
            
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT 
                    arxiv_id,
                    title,
                    abstract,
                    authors,
                    categories,
                    published_date,
                    1 - (embedding <=> %s) as similarity
                FROM papers
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k)
            )
            return [dict(row) for row in cur.fetchall()]
            
    def get_paper_count(self) -> int:
        """Get the total number of papers in the database."""
        if not self.conn:
            self.connect()
            
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM papers")
            return cur.fetchone()[0]
            
    def get_all_papers(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Get all papers with pagination."""
        if not self.conn:
            self.connect()
            
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT arxiv_id, title, abstract, authors, categories, published_date
                FROM papers
                ORDER BY published_date DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset)
            )
            return [dict(row) for row in cur.fetchall()]


# Singleton instance
db = Database()

