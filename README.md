# 📚 arXiv RAG

A Retrieval-Augmented Generation (RAG) application for searching and querying arXiv research papers using natural language. Built with PostgreSQL + pgvector for vector storage, Ollama for embeddings and LLM inference, and Streamlit for the UI.

## Features

- **Paper Ingestion**: Fetch papers from arXiv by search query or category
- **Vector Search**: Semantic search using pgvector with HNSW indexing
- **RAG Pipeline**: Answer questions using retrieved paper context
- **Local LLM**: Runs entirely locally using Ollama
- **Beautiful UI**: Modern Streamlit interface

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   arXiv     │────▶│   Ollama    │────▶│  PostgreSQL │
│   API       │     │ (Embeddings)│     │  + pgvector │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Ollama    │
                    │    (LLM)    │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Streamlit  │
                    │     UI      │
                    └─────────────┘
```

## Prerequisites

1. **PostgreSQL with pgvector**
2. **Ollama**
3. **uv** (Python package manager)

## Quick Start

### 1. Install PostgreSQL with pgvector

**macOS (Homebrew):**
```bash
brew install postgresql@16
brew services start postgresql@16

# Install pgvector extension
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

**Or using Docker:**
```bash
docker run -d \
  --name postgres-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=arxiv_rag \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 2. Install Ollama

**macOS:**
```bash
brew install ollama
ollama serve  # Start Ollama server (run in background)
```

**Pull required models:**
```bash
ollama pull nomic-embed-text  # Embedding model
ollama pull llama3.2:3b       # LLM model (lightweight, runs on most machines)
```

### 3. Setup the Application

```bash
# Clone and enter directory
cd arxiv-rag

# Create environment file
cp env.example .env
# Edit .env with your settings if needed

# Install dependencies with uv
uv sync

# Create the database (if not using Docker)
createdb arxiv_rag
```

### 4. Run the Application

```bash
uv run streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Configuration

All settings can be configured via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `arxiv_rag` | Database name |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | `postgres` | Database password |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_LLM_MODEL` | `llama3.2:3b` | LLM for generation |
| `EMBEDDING_DIM` | `768` | Embedding dimensions |
| `TOP_K_RESULTS` | `5` | Default number of results |

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

The system will:
1. Generate an embedding for your query
2. Find the most similar papers using cosine similarity
3. Build a context from retrieved papers
4. Generate an answer using the local LLM

## Project Structure

```
arxiv-rag/
├── app.py              # Streamlit UI
├── src/
│   ├── __init__.py
│   ├── config.py       # Configuration settings
│   ├── database.py     # PostgreSQL/pgvector operations
│   ├── arxiv_fetcher.py # arXiv API client
│   ├── embeddings.py   # Ollama embedding generation
│   ├── ingest.py       # Paper ingestion pipeline
│   └── rag.py          # RAG pipeline
├── Dockerfile          # Container build instructions
├── docker-compose.yml  # Multi-container orchestration
├── pyproject.toml      # Project dependencies
├── env.example         # Environment template
└── README.md
```

## Alternative LLM Models

You can use different Ollama models based on your hardware:

| Model | Size | RAM Required | Notes |
|-------|------|--------------|-------|
| `llama3.2:1b` | 1.3B | ~2GB | Fastest, less accurate |
| `llama3.2:3b` | 3B | ~4GB | Good balance (default) |
| `llama3.1:8b` | 8B | ~8GB | Better quality |
| `mistral` | 7B | ~8GB | Great for reasoning |
| `phi3:mini` | 3.8B | ~4GB | Fast and capable |

To change the model:
```bash
ollama pull <model-name>
# Update OLLAMA_LLM_MODEL in .env
```

## Docker Deployment (Linux Server)

Deploy on any Linux server by pulling directly from the Git repository.

### 1. Install Ollama on the Host

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

### 2. Create docker-compose.yml

Create a `docker-compose.yml` file on your server:

```yaml
version: '3.8'

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

**Update the repository URL** in the `build.context` line to your actual repo.

### 3. Start the Application

```bash
docker compose up -d

# View logs
docker compose logs -f app
```

The app will be available at `http://your-server-ip:8501`

### 4. Docker Commands

```bash
docker compose up -d              # Start all services
docker compose up -d --build      # Rebuild and start (after repo updates)
docker compose down               # Stop all services
docker compose down -v            # Stop and delete data
docker compose logs -f            # View logs
docker compose restart app        # Restart just the app
```

### Build Directly from Git

Build the image directly without cloning the repo:

```bash
docker build -t arxiv-rag https://github.com/jaisngh/arxiv-rag.git#main
```

---

## Troubleshooting

### Database connection failed
- Ensure PostgreSQL is running: `brew services list`
- Check credentials in `.env`
- Verify database exists: `psql -l`

### Ollama model not found
- Ensure Ollama is running: `ollama serve`
- Pull required models: `ollama pull nomic-embed-text`

### pgvector extension error
- Install pgvector: follow instructions at https://github.com/pgvector/pgvector
- Enable extension: `CREATE EXTENSION vector;`

## License

MIT License

