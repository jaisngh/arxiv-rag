"""Streamlit UI for the arXiv RAG application."""

import streamlit as st
from datetime import datetime

from src.config import Config
from src.database import db
from src.embeddings import embedder
from src.rag import rag
from src.ingest import ingestor
from src.arxiv_fetcher import POPULAR_CATEGORIES


# Page configuration
st.set_page_config(
    page_title="arXiv RAG",
    page_icon="books",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');
    
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-tertiary: #21262d;
        --accent-cyan: #58a6ff;
        --accent-purple: #bc8cff;
        --accent-green: #3fb950;
        --accent-orange: #d29922;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --border-color: #30363d;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0a0f14 100%);
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-family: 'Outfit', sans-serif;
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .paper-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .paper-card:hover {
        border-color: var(--accent-cyan);
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.1);
    }
    
    .paper-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .paper-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--accent-cyan);
        margin-bottom: 0.75rem;
    }
    
    .paper-abstract {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .similarity-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--accent-green) 0%, #238636 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stat-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--accent-cyan);
    }
    
    .stat-label {
        font-family: 'Outfit', sans-serif;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    .answer-box {
        background: var(--bg-secondary);
        border: 1px solid var(--accent-purple);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .source-tag {
        display: inline-block;
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        color: var(--accent-orange);
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 0.5rem;
        margin-bottom: 0.25rem;
    }
    
    div[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    .stTextInput > div > div > input {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        border-radius: 8px;
    }
    
    .stTextArea > div > div > textarea {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        border-radius: 8px;
    }
    
    .stSelectbox > div > div {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def check_services():
    """Check if required services are available."""
    issues = []
    
    # Check database
    try:
        db.connect()
        db.init_schema()
    except Exception as e:
        issues.append(f"Database: {str(e)}")
        
    # Check Ollama embedding model
    try:
        if not embedder.ensure_model():
            issues.append(f"Embedding model '{Config.OLLAMA_EMBED_MODEL}' not available")
    except Exception as e:
        issues.append(f"Ollama embeddings: {str(e)}")
        
    # Check Ollama LLM model
    try:
        if not rag.ensure_model():
            issues.append(f"LLM model '{Config.OLLAMA_LLM_MODEL}' not available")
    except Exception as e:
        issues.append(f"Ollama LLM: {str(e)}")
        
    return issues


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.markdown('<p class="main-header" style="font-size: 1.8rem;">arXiv RAG</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header" style="font-size: 0.9rem; margin-bottom: 1rem;">Research Paper Intelligence</p>', unsafe_allow_html=True)
        
        st.divider()
        
        page = st.radio(
            "Navigation",
            ["Query Papers", "Ingest Papers", "Database Stats"],
            label_visibility="collapsed",
        )
        
        st.divider()
        
        st.markdown("### Settings")
        st.text(f"Embed Model: {Config.OLLAMA_EMBED_MODEL}")
        st.text(f"LLM Model: {Config.OLLAMA_LLM_MODEL}")
        st.text(f"Top-K Results: {Config.TOP_K_RESULTS}")
        
        return page


def render_query_page():
    """Render the query/RAG page."""
    st.markdown('<h1 class="main-header">Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about scientific papers in your knowledge base</p>', unsafe_allow_html=True)
    
    # Check paper count
    try:
        paper_count = db.get_paper_count()
    except:
        paper_count = 0
        
    if paper_count == 0:
        st.warning("No papers in the database yet. Go to 'Ingest Papers' to add some papers first.")
        return
        
    st.info(f"{paper_count} papers indexed and ready for querying")
    
    # Query input
    query = st.text_area(
        "Your Question",
        placeholder="e.g., What are the latest approaches for improving transformer efficiency?",
        height=100,
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        top_k = st.number_input("Sources", min_value=1, max_value=10, value=5)
    with col2:
        search_button = st.button("Search & Answer", use_container_width=True)
        
    if search_button and query:
        with st.spinner("Searching relevant papers..."):
            # Get relevant papers
            papers = rag.retrieve(query, top_k)
            
        if not papers:
            st.error("No relevant papers found for your query.")
            return
            
        # Show sources first
        with st.expander("Retrieved Sources", expanded=True):
            for paper in papers:
                similarity_pct = paper.get("similarity", 0) * 100
                st.markdown(f"""
                <div class="paper-card">
                    <div class="paper-title">{paper['title']}</div>
                    <div class="paper-meta">
                        arXiv:{paper['arxiv_id']} · 
                        <span class="similarity-badge">{similarity_pct:.1f}% match</span>
                    </div>
                    <div class="paper-abstract">{paper['abstract'][:400]}...</div>
                </div>
                """, unsafe_allow_html=True)
                
        # Generate answer
        with st.spinner("Generating answer..."):
            result = rag.query(query, top_k)
            
        st.markdown("### Answer")
        st.markdown(f"""
        <div class="answer-box">
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)
        
        # Source citations
        st.markdown("**Sources cited:**")
        for source in result['sources']:
            st.markdown(f'<span class="source-tag">arXiv:{source["arxiv_id"]}</span>', unsafe_allow_html=True)


def render_ingest_page():
    """Render the paper ingestion page."""
    st.markdown('<h1 class="main-header">Ingest Papers</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Fetch and index papers from arXiv into your knowledge base</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Search Query", "By Category"])
    
    with tab1:
        st.markdown("### Search arXiv")
        search_query = st.text_input(
            "Search Query",
            placeholder="e.g., transformer attention mechanism, large language models",
        )
        max_papers_search = st.slider("Maximum papers to fetch", 10, 200, 50)
        
        if st.button("Fetch & Index", key="search_ingest"):
            if search_query:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, title):
                    progress_bar.progress(current / total)
                    status_text.text(f"Processing {current}/{total}: {title[:60]}...")
                    
                with st.spinner("Fetching papers from arXiv..."):
                    stats = ingestor.ingest_from_search(
                        search_query,
                        max_papers_search,
                        progress_callback=update_progress,
                    )
                    
                progress_bar.progress(1.0)
                status_text.empty()
                
                st.success(f"""
                Ingestion complete!
                - Fetched: {stats['fetched']} papers
                - New papers indexed: {stats['ingested']}
                - Already existed: {stats['skipped']}
                """)
            else:
                st.warning("Please enter a search query")
                
    with tab2:
        st.markdown("### Browse by Category")
        
        category = st.selectbox(
            "Select Category",
            options=list(POPULAR_CATEGORIES.keys()),
            format_func=lambda x: f"{x} - {POPULAR_CATEGORIES[x]}",
        )
        max_papers_cat = st.slider("Maximum papers to fetch", 10, 200, 50, key="cat_slider")
        
        if st.button("Fetch & Index", key="cat_ingest"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, title):
                progress_bar.progress(current / total)
                status_text.text(f"Processing {current}/{total}: {title[:60]}...")
                
            with st.spinner(f"Fetching papers from {category}..."):
                stats = ingestor.ingest_from_category(
                    category,
                    max_papers_cat,
                    progress_callback=update_progress,
                )
                
            progress_bar.progress(1.0)
            status_text.empty()
            
            st.success(f"""
            Ingestion complete!
            - Fetched: {stats['fetched']} papers
            - New papers indexed: {stats['ingested']}
            - Already existed: {stats['skipped']}
            """)


def render_stats_page():
    """Render the database statistics page."""
    st.markdown('<h1 class="main-header">Database Statistics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Overview of your indexed research papers</p>', unsafe_allow_html=True)
    
    try:
        paper_count = db.get_paper_count()
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return
        
    # Stats cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{paper_count}</div>
            <div class="stat-label">Total Papers</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{Config.EMBEDDING_DIM}</div>
            <div class="stat-label">Embedding Dimensions</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{Config.TOP_K_RESULTS}</div>
            <div class="stat-label">Default Top-K</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    # Recent papers
    st.markdown("### Recent Papers")
    
    if paper_count > 0:
        papers = db.get_all_papers(limit=20)
        
        for paper in papers:
            authors = ", ".join(paper["authors"][:3]) if paper["authors"] else "Unknown"
            if paper["authors"] and len(paper["authors"]) > 3:
                authors += " et al."
            
            pub_date = paper["published_date"].strftime("%Y-%m-%d") if paper["published_date"] else "N/A"
            
            st.markdown(f"""
            <div class="paper-card">
                <div class="paper-title">{paper['title']}</div>
                <div class="paper-meta">
                    arXiv:{paper['arxiv_id']} · {pub_date} · {authors}
                </div>
                <div class="paper-abstract">{paper['abstract'][:300]}...</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No papers indexed yet. Go to 'Ingest Papers' to add some papers.")


def main():
    """Main application entry point."""
    init_session_state()
    
    # Check services on first run
    if not st.session_state.initialized:
        with st.spinner("Initializing services..."):
            issues = check_services()
            
        if issues:
            st.error("Service Issues Detected")
            for issue in issues:
                st.error(f"• {issue}")
            st.info("""
            **Setup Requirements:**
            1. PostgreSQL with pgvector extension running
            2. Ollama running locally with required models
            
            Check the README for setup instructions.
            """)
            st.stop()
            
        st.session_state.initialized = True
        
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "Query Papers":
        render_query_page()
    elif page == "Ingest Papers":
        render_ingest_page()
    else:
        render_stats_page()


if __name__ == "__main__":
    main()

