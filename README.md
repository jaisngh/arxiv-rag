# 📚 arXiv RAG

A Retrieval-Augmented Generation (RAG) application for searching and querying arXiv research papers using natural language. Built with PostgreSQL + pgvector for vector storage, Ollama for local LLM inference, and Streamlit for the UI.

## Features

- **Paper Ingestion**: Fetch papers from arXiv by search query or category
- **Vector Search**: Semantic search using pgvector with HNSW indexing
- **RAG Pipeline**: Answer questions using retrieved paper context
- **Local LLM**: Runs entirely locally using Ollama
- **Beautiful UI**: Modern Streamlit interface

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   arXiv     │────▶│ nomic-embed-│────▶│ PostgreSQL +│
│   API       │     │    text     │     │ pgvector db │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                  ┌─────────────┐       ┌─────────────┐                     
                  │    User     │ ────▶ │  Llama 3.2  │ 
                  │    Query    │       │     (3B)    │
                  └─────────────┘       └─────────────┘              
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Answer &   │
                                        │ Cited Papers│
                                        └─────────────┘
```

## Prerequisites

- **Docker** (with Docker Compose)
- **Ollama** (for embeddings and LLM)

## Quick Start

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Start Ollama and pull required models:**
```bash
ollama serve &                    # Start Ollama (runs in background)
ollama pull nomic-embed-text      # Embedding model (~274MB)
ollama pull llama3.2:3b           # LLM model (~2GB)
```

### 2. Create docker-compose.yml

Copy the `docker-compose.yml` file from the repository:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: arxiv-rag-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: arxiv_rag
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d arxiv_rag"]
      interval: 10s
      timeout: 5s
      retries: 5

  app:
    build:
      context: https://github.com/jaisngh/arxiv-rag.git#main
    container_name: arxiv-rag-app
    restart: unless-stopped
    ports:
      - "8501:8501"
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_PORT: 5432
      POSTGRES_DB: arxiv_rag
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      OLLAMA_HOST: http://host.docker.internal:11434
      OLLAMA_EMBED_MODEL: nomic-embed-text
      OLLAMA_LLM_MODEL: llama3.2:3b
      EMBEDDING_DIM: 768
      TOP_K_RESULTS: 5
    depends_on:
      postgres:
        condition: service_healthy
    extra_hosts:
      - "host.docker.internal:host-gateway"

volumes:
  postgres_data:
```

### 3. Start the Application

```bash
docker compose up -d
```

This will:
- Pull and start PostgreSQL 16 with pgvector extension pre-installed
- Build the app image from the GitHub repository
- Start the Streamlit application

The app will be available at **http://localhost:8501**

## Usage

### Ingest Papers

1. Go to **📥 Ingest Papers**
2. Enter a search query (e.g., "transformer attention mechanisms") or select a category
3. Choose the number of papers to fetch
4. Click **Fetch & Index**

### Query Papers

1. Go to **🔍 Query Papers**
2. Enter your question in natural language
3. Adjust the number of sources if needed
4. Click **Search & Answer**

## Docker Commands

```bash
docker compose up -d              # Start all services
docker compose up -d --build      # Rebuild after repo updates
docker compose down               # Stop all services
docker compose down -v            # Stop and delete all data
docker compose logs -f app        # View app logs
docker compose restart app        # Restart just the app
```

## Configuration

Environment variables can be customized in docker-compose.yml:

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `postgres` | PostgreSQL host (Docker service name) |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `arxiv_rag` | Database name |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | `postgres` | Database password |
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | Ollama API URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_LLM_MODEL` | `llama3.2:3b` | LLM for generation |
| `EMBEDDING_DIM` | `768` | Embedding dimensions |
| `TOP_K_RESULTS` | `5` | Default number of results |

## Alternative LLM Models

You can use different Ollama models based on your hardware:

| Model | Size | RAM Required | Notes |
|-------|------|--------------|-------|
| `llama3.2:1b` | 1.3B | ~2GB | Fastest, less accurate |
| `llama3.2:3b` | 3B | ~4GB | Good balance (default) |
| `llama3.1:8b` | 8B | ~8GB | Better quality |
| `mistral` | 7B | ~8GB | Great for reasoning |
| `phi3:mini` | 3.8B | ~4GB | Fast and capable |

To use a different model:
```bash
ollama pull <model-name>
# Update OLLAMA_LLM_MODEL in docker-compose.yml
```

## Troubleshooting

### Ollama connection failed
- Ensure Ollama is running: `ollama serve`
- Verify models are pulled: `ollama list`
- Check Ollama is accessible: `curl http://localhost:11434`

### Database errors
- Restart containers: `docker compose restart`
- Check logs: `docker compose logs postgres`

### Rebuild after code updates
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

## Local Development (Without Docker)

For development or if you prefer not to use Docker:

### Prerequisites

- **Python 3.12+**
- **uv** (Python package manager)
- **PostgreSQL 16** with pgvector extension
- **Ollama**

### Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install PostgreSQL with pgvector (macOS)
brew install postgresql@16
brew services start postgresql@16

# Install pgvector extension
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make && make install
cd .. && rm -rf pgvector

# Create database
createdb arxiv_rag

# Install Ollama and models
brew install ollama
ollama serve &
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

### Run the App

```bash
# Clone the repository
git clone https://github.com/jaisngh/arxiv-rag.git
cd arxiv-rag

# Install dependencies
uv sync

# Create .env file (optional - defaults work for local PostgreSQL)
cat > .env << EOF
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=arxiv_rag
POSTGRES_USER=$(whoami)
POSTGRES_PASSWORD=
OLLAMA_HOST=http://localhost:11434
EOF

# Run the app
uv run streamlit run app.py
```

The app will be available at **http://localhost:8501**

---

## Project Structure

```
arxiv-rag/
├── app.py              # Streamlit UI
├── src/
│   ├── config.py       # Configuration settings
│   ├── database.py     # PostgreSQL/pgvector operations
│   ├── arxiv_fetcher.py # arXiv API client
│   ├── embeddings.py   # Ollama embedding generation
│   ├── ingest.py       # Paper ingestion pipeline
│   └── rag.py          # RAG pipeline
├── Dockerfile          # Container build instructions
├── docker-compose.yml  # Multi-container orchestration
└── pyproject.toml      # Python dependencies
```