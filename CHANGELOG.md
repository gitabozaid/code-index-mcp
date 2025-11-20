# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - Ollama Integration

### Added

#### Embedding Provider Abstraction
- `BaseEmbeddingProvider` abstract class for pluggable embedding providers
- `OllamaEmbeddingProvider` implementation using nomic-embed-code model
- `EmbeddingProviderFactory` for creating embedding provider instances
- Support for batch embedding generation
- Graceful availability checking with helpful setup instructions

#### Code Chunking
- `CodeChunker` for semantic code splitting using Tree-sitter
- Support for 7 languages (Python, JavaScript, TypeScript, Go, Rust, Java, C++)
- Extraction of functions, classes, and methods as separate chunks
- Graceful fallback to full-file chunks if parsing fails
- Language-specific Tree-sitter queries

#### Vector Database Integration
- `QdrantStore` wrapper for Qdrant vector database
- Collection management and automatic creation
- Embedding storage with rich metadata
- Similarity search with score thresholding
- File-based filtering and deletion
- Collection statistics and health checking

#### Hybrid Search
- `SemanticSearchStrategy` using Ollama embeddings and Qdrant
- `HybridSearchEngine` combining BM25 + Semantic with RRF fusion
- `HybridSearchStrategy` wrapper for drop-in compatibility
- Configurable weights for BM25 and semantic components
- Reciprocal Rank Fusion (RRF) algorithm implementation
- Graceful fallback if either search method unavailable

#### Configuration Management
- `.env.example` with all configuration parameters
- `SemanticSearchConfig` helper for environment variables
- Support for Ollama, Qdrant, and search mode configuration
- Configurable weights for hybrid search
- Default values for all settings

#### MCP Server Integration
- `SemanticSearchService` for file/project indexing
- Support for individual file indexing and reindexing
- Batch project indexing with file pattern filtering
- Index statistics and health checking
- Clean service-oriented design following project patterns

#### Testing
- Integration tests for `OllamaEmbeddingProvider`
- Tests for `CodeChunker` with multiple languages
- Tests for `QdrantStore` integration
- Hybrid search import tests
- Tests can run with/without Ollama and Qdrant (skipif decorators)

#### Documentation
- Comprehensive OLLAMA_INTEGRATION.md guide
- Installation and setup instructions
- Configuration examples
- Usage examples with code snippets
- Architecture diagrams
- Troubleshooting guide
- Performance benchmarks
- FAQ section

### Changed

- Added ollama>=0.3.0 dependency
- Added qdrant-client>=1.7.0 dependency
- Added numpy>=1.26.0 dependency
- Added tree-sitter language parsers (Python, Go, Rust, C++)
- Updated requirements.txt and pyproject.toml
- Updated search module exports to include semantic and hybrid strategies
- Updated services module exports to include SemanticSearchService
- Updated utils module exports to include SemanticSearchConfig

### Dependencies

#### New Dependencies
- `ollama>=0.3.0` - Ollama Python client for embeddings
- `qdrant-client>=1.7.0` - Qdrant vector database client
- `numpy>=1.26.0` - Numerical operations for vectors
- `tree-sitter-python>=0.21.0` - Python parser
- `tree-sitter-go>=0.21.0` - Go parser
- `tree-sitter-rust>=0.21.0` - Rust parser
- `tree-sitter-cpp>=0.21.0` - C++ parser

### Fixed
- N/A (initial feature addition)

### Removed
- N/A (initial feature addition)

### Security
- All processing is local (no external API calls)
- No data sent to third-party services
- Full privacy for sensitive codebases

---

## [2.9.2] - Upstream Version

Base version from johnhuang316/code-index-mcp before forking.

### Features
- BM25 search with multiple tools (ripgrep, ag, ugrep, grep)
- Tree-sitter code parsing
- File watching with automatic reindexing
- SQLite-based indexing
- MCP server implementation
- Multiple language support

---

## Future Roadmap

### Planned Features
- [ ] OpenAI embedding provider
- [ ] Voyage AI embedding provider
- [ ] FastEmbed local embeddings
- [ ] Support for more embedding models
- [ ] Advanced chunking strategies
- [ ] Relevance feedback learning
- [ ] Query expansion
- [ ] Multi-vector search
- [ ] Embedding caching improvements

### Performance Improvements
- [ ] Parallel embedding generation
- [ ] Incremental indexing optimization
- [ ] Query result caching
- [ ] Batch processing improvements

### Documentation
- [ ] Video tutorials
- [ ] More usage examples
- [ ] Integration guides for IDEs
- [ ] Benchmarking suite

---

**Contributors:** abozaid, johnhuang316 (upstream)
**License:** MIT
**Repository:** https://github.com/gitabozaid/code-index-mcp
