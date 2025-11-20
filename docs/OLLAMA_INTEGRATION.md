# Ollama Integration Guide

**Complete guide to using local semantic search with Ollama embeddings and Qdrant vector database.**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Architecture](#architecture)
7. [Troubleshooting](#troubleshooting)
8. [Performance](#performance)

---

## Overview

This fork adds **100% local semantic code search** capabilities using:

- **Ollama** - Local embedding generation (no API costs)
- **nomic-embed-code** - State-of-the-art code embeddings (768 dimensions)
- **Qdrant** - High-performance vector database
- **Hybrid Search** - Combines BM25 (keyword) + Semantic (vector) with RRF fusion

### Benefits

- 🆓 **Zero Cost** - No external API dependencies
- 🏠 **100% Local** - Works completely offline
- 🎯 **Better Search** - Understands code semantically
- 🔒 **Privacy** - No data sent to external services
- ⚡ **Fast** - Sub-second search with caching

---

## Prerequisites

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### 2. Pull Embedding Model

```bash
# Pull nomic-embed-code model (required)
ollama pull nomic-embed-code

# Verify installation
ollama list | grep nomic-embed-code
```

### 3. Start Ollama Service

```bash
# macOS
brew services start ollama

# Linux/Manual
ollama serve

# Verify running
curl http://localhost:11434/api/tags
```

### 4. Install Qdrant

**Option A: Docker (Recommended)**
```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/.indexes/qdrant:/qdrant/storage \
  qdrant/qdrant:latest

# Verify
curl http://localhost:6333/collections
```

**Option B: Binary**
```bash
# Download from https://github.com/qdrant/qdrant/releases
# Run binary
./qdrant
```

---

## Installation

### Install Code-Index-MCP with Ollama Support

```bash
# Clone this fork
git clone https://github.com/YOUR_USERNAME/code-index-mcp.git
cd code-index-mcp

# Install dependencies
pip install -e .

# Or with uv (recommended)
uv pip install -e .
```

### Verify Installation

```bash
# Check Ollama connectivity
python -c "import ollama; print(ollama.Client().list())"

# Check Qdrant connectivity
python -c "from qdrant_client import QdrantClient; print(QdrantClient('http://localhost:6333').get_collections())"
```

---

## Configuration

### Environment Variables

Create `.env` file in your project root:

```bash
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=nomic-embed-code
OLLAMA_BATCH_SIZE=10
OLLAMA_TIMEOUT=30

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=code-embeddings
QDRANT_VECTOR_DIM=768

# Search Configuration
SEARCH_MODE=hybrid  # Options: bm25, semantic, hybrid
BM25_WEIGHT=0.5
SEMANTIC_WEIGHT=0.5
SEMANTIC_SEARCH_ENABLED=true

# Embedding Provider
EMBEDDING_PROVIDER=ollama
```

### Claude Desktop Configuration

Update `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "code-index-ollama": {
      "command": "uvx",
      "args": ["--from", "/path/to/code-index-mcp", "code-index-mcp"],
      "env": {
        "OLLAMA_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "nomic-embed-code",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_COLLECTION": "my-project-code",
        "SEARCH_MODE": "hybrid",
        "SEMANTIC_SEARCH_ENABLED": "true"
      }
    }
  }
}
```

---

## Usage

### 1. Index Your Codebase

```python
from code_index_mcp.services import SemanticSearchService

# Initialize service
service = SemanticSearchService(ctx)

# Index entire project
result = service.index_project(
    project_path="/path/to/your/project",
    file_patterns=["*.py", "*.js", "*.ts"]
)

print(f"Indexed {result['indexed_files']} files")
print(f"Total chunks: {result['total_chunks']}")
```

### 2. Search with Natural Language

```python
from code_index_mcp.search import HybridSearchStrategy

# Search for authentication logic
results = hybrid_search.search(
    pattern="where is user login authentication handled",
    base_path="/path/to/project"
)

# Results include both keyword and semantic matches
for file_path, matches in results.items():
    print(f"\n{file_path}:")
    for line_num, content in matches:
        print(f"  Line {line_num}: {content[:80]}...")
```

### 3. Search Modes

**BM25 Only (Keyword Search)**
```bash
SEARCH_MODE=bm25
```
- Fast keyword matching
- Exact string matching
- Best for known function names

**Semantic Only (Vector Search)**
```bash
SEARCH_MODE=semantic
```
- Natural language queries
- Understands code semantics
- Best for "what does this do?" queries

**Hybrid (Recommended)**
```bash
SEARCH_MODE=hybrid
BM25_WEIGHT=0.5
SEMANTIC_WEIGHT=0.5
```
- Best of both worlds
- RRF fusion algorithm
- Optimal for most use cases

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                  MCP Server (FastMCP)                   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Code Indexer Core                     │
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │ Tree-sitter      │  │ Search Router    │            │
│  │ Parser           │  │ (Hybrid Mode)    │            │
│  └──────────────────┘  └──────────────────┘            │
└─────────────────────────────────────────────────────────┘
           │                       │                  │
           ▼                       ▼                  ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ BM25 Search      │  │ Semantic Search  │  │ Hybrid Fusion    │
│ (ripgrep/ag)     │  │ (Ollama+Qdrant)  │  │ (RRF Algorithm)  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Embedding Provider   │
                   │ Factory (Plugin)     │
                   └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ OllamaEmbedProvider  │
                   │ (nomic-embed-code)   │
                   └──────────────────────┘
```

### Key Components

1. **EmbeddingProviderFactory** - Pluggable embedding providers
2. **OllamaEmbeddingProvider** - Ollama client integration
3. **CodeChunker** - Tree-sitter-based code splitting
4. **QdrantStore** - Vector database wrapper
5. **HybridSearchEngine** - RRF fusion algorithm
6. **SemanticSearchService** - MCP service integration

---

## Troubleshooting

### Ollama Not Available

**Error:** `Ollama provider not available`

**Solutions:**
1. Check Ollama is running: `curl http://localhost:11434/api/tags`
2. Pull model: `ollama pull nomic-embed-code`
3. Check environment: `echo $OLLAMA_URL`
4. Verify port: `lsof -i :11434`

### Qdrant Connection Failed

**Error:** `Qdrant connection failed`

**Solutions:**
1. Check Qdrant is running: `curl http://localhost:6333/collections`
2. Check Docker: `docker ps | grep qdrant`
3. Restart Qdrant: `docker restart qdrant`
4. Check firewall: Ensure port 6333 is open

### Slow Embedding Generation

**Issue:** Indexing takes too long

**Solutions:**
1. Reduce batch size: `OLLAMA_BATCH_SIZE=5`
2. Use GPU if available (Ollama auto-detects)
3. Index incrementally (file-by-file)
4. Increase Ollama timeout: `OLLAMA_TIMEOUT=60`

### Empty Search Results

**Issue:** Semantic search returns no results

**Solutions:**
1. Check index exists: `service.get_index_stats()`
2. Verify indexing completed: Check `vectors_count > 0`
3. Lower score threshold: `score_threshold=0.3`
4. Try hybrid mode instead of semantic-only

---

## Performance

### Expected Benchmarks

| Metric | Target | Notes |
|--------|--------|-------|
| **Indexing Speed** | 1,000 files in < 60s | Including embedding generation |
| **Search Latency (BM25)** | < 100ms | Sub-100ms as documented |
| **Search Latency (Semantic)** | < 500ms | Depends on Ollama/Qdrant |
| **Search Latency (Hybrid)** | < 600ms | Combined overhead |
| **Embedding Generation** | ~50ms per text | nomic-embed-code on M-series Mac |
| **Batch Embedding (10 texts)** | ~300ms | Batch processing optimization |

### Optimization Tips

1. **Batch Processing** - Always use `embed_batch()` for multiple texts (10x faster)
2. **Caching** - Embeddings cached in Qdrant, no regeneration needed
3. **Fallback Strategy** - BM25 kicks in if semantic search times out
4. **Index Freshness** - File watcher auto-reindexes only changed files

### Storage Requirements

- **768-dim vectors** = ~3KB per code chunk
- **10,000 chunks** = ~30MB storage
- **SQLite index** = ~10MB (BM25)
- **Total for 10K chunks** = ~40MB

---

## FAQ

**Q: Can I use other embedding models?**
A: Yes! Extend `BaseEmbeddingProvider` and add to `EmbeddingProviderFactory`.

**Q: Does this work offline?**
A: Yes, completely! All processing is local.

**Q: Can I use this with OpenAI embeddings?**
A: Yes, but you'll need to implement `OpenAIEmbeddingProvider`.

**Q: How do I delete old embeddings?**
A: Use `vector_store.clear_collection()` or delete by file.

**Q: Can I search multiple projects?**
A: Yes, use different collection names per project.

---

## Next Steps

1. **Index your codebase** - Start with a small project
2. **Experiment with search modes** - Try BM25, semantic, and hybrid
3. **Tune weights** - Adjust BM25_WEIGHT and SEMANTIC_WEIGHT
4. **Monitor performance** - Check `get_index_stats()` regularly
5. **Contribute back** - Share improvements via PR!

---

**Need Help?** Open an issue on GitHub or check the main README.md for more information.
