"""Semantic search strategy using embeddings and vector similarity."""

import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .base import SearchStrategy
from ..embeddings import EmbeddingProviderFactory
from ..vector_store import QdrantStore

logger = logging.getLogger(__name__)


class SemanticSearchStrategy(SearchStrategy):
    """
    Semantic search strategy using Ollama embeddings and Qdrant vector database.

    This strategy converts the search query to an embedding and performs
    similarity search against indexed code chunks.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize semantic search strategy.

        Args:
            config: Configuration dictionary with:
                - embedding_provider: 'ollama' (default)
                - ollama_model: 'nomic-embed-code' (default)
                - ollama_url: 'http://localhost:11434' (default)
                - qdrant_url: 'http://localhost:6333' (default)
                - qdrant_collection: 'code-embeddings' (default)
                - vector_dim: 768 (default)
        """
        self.config = config or {}
        self.embedding_provider = None
        self.vector_store = None
        self._initialized = False

        # Initialize components
        self._initialize()

    def _initialize(self):
        """Initialize embedding provider and vector store."""
        try:
            # Create embedding provider
            embedding_config = {
                'embedding_provider': self.config.get('embedding_provider', 'ollama'),
                'model': self.config.get('ollama_model', 'nomic-embed-code'),
                'url': self.config.get('ollama_url', 'http://localhost:11434'),
                'dimension': self.config.get('vector_dim', 768),
            }

            self.embedding_provider = EmbeddingProviderFactory.create_from_env(embedding_config)

            if not self.embedding_provider:
                logger.warning("Embedding provider not available. Semantic search disabled.")
                return

            # Create vector store
            self.vector_store = QdrantStore(
                url=self.config.get('qdrant_url', 'http://localhost:6333'),
                collection_name=self.config.get('qdrant_collection', 'code-embeddings'),
                vector_dim=self.config.get('vector_dim', 768),
                distance_metric='cosine'
            )

            if not self.vector_store.is_available():
                logger.warning("Qdrant not available. Semantic search disabled.")
                self.vector_store = None
                return

            self._initialized = True
            logger.info("Semantic search strategy initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize semantic search: {e}")
            self.embedding_provider = None
            self.vector_store = None

    @property
    def name(self) -> str:
        """Strategy name."""
        return "semantic"

    def is_available(self) -> bool:
        """Check if semantic search is available."""
        return (
            self._initialized and
            self.embedding_provider is not None and
            self.vector_store is not None and
            self.embedding_provider.is_available() and
            self.vector_store.is_available()
        )

    def search(
        self,
        pattern: str,
        base_path: str,
        case_sensitive: bool = True,
        context_lines: int = 0,
        file_pattern: Optional[str] = None,
        fuzzy: bool = False,
        regex: bool = False
    ) -> Dict[str, List[Tuple[int, str]]]:
        """
        Execute semantic search.

        Args:
            pattern: Natural language or code query
            base_path: Project root path (not used for semantic search)
            case_sensitive: Ignored for semantic search
            context_lines: Ignored for semantic search
            file_pattern: Optional file path filter
            fuzzy: Ignored for semantic search
            regex: Ignored for semantic search

        Returns:
            Dictionary mapping file paths to (line_number, content) tuples
        """
        if not self.is_available():
            logger.warning("Semantic search not available. Returning empty results.")
            return {}

        try:
            # Generate query embedding
            logger.debug(f"Generating embedding for query: {pattern[:50]}...")
            query_embedding = self.embedding_provider.embed_text(pattern)

            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                limit=self.config.get('semantic_limit', 10),
                score_threshold=self.config.get('score_threshold', 0.5),
                file_pattern=file_pattern
            )

            # Convert to SearchStrategy format
            output = {}
            for result in results:
                file_path = result['file_path']
                start_line = result['start_line']
                content = result['content']

                # Make path relative to base_path
                try:
                    rel_path = str(Path(file_path).relative_to(base_path))
                except ValueError:
                    rel_path = file_path

                if rel_path not in output:
                    output[rel_path] = []

                output[rel_path].append((start_line, content))

            logger.info(f"Semantic search found {len(output)} files with matches")
            return output

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {}
