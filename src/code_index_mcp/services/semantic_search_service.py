"""Semantic search service for MCP server."""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .base_service import BaseService
from ..embeddings import EmbeddingProviderFactory
from ..vector_store import QdrantStore
from ..chunking import CodeChunker
from ..utils import SemanticSearchConfig

logger = logging.getLogger(__name__)


class SemanticSearchService(BaseService):
    """Service for semantic code search with embeddings."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.config = SemanticSearchConfig.load_from_env()
        self.embedding_provider = None
        self.vector_store = None
        self.code_chunker = None
        self._initialized = False

        # Initialize if semantic search is enabled
        if self.config.get('semantic_enabled', False):
            self._initialize()

    def _initialize(self):
        """Initialize semantic search components."""
        try:
            # Initialize embedding provider
            embedding_config = SemanticSearchConfig.get_embedding_config()
            self.embedding_provider = EmbeddingProviderFactory.create_from_env(embedding_config)

            if not self.embedding_provider:
                logger.warning("Embedding provider not available. Semantic search disabled.")
                return

            # Initialize vector store
            qdrant_config = SemanticSearchConfig.get_qdrant_config()
            self.vector_store = QdrantStore(**qdrant_config)

            if not self.vector_store.is_available():
                logger.warning("Qdrant not available. Semantic search disabled.")
                self.vector_store = None
                return

            # Initialize code chunker
            self.code_chunker = CodeChunker()

            self._initialized = True
            logger.info("Semantic search service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize semantic search: {e}")
            self.embedding_provider = None
            self.vector_store = None
            self.code_chunker = None

    def is_available(self) -> bool:
        """Check if semantic search is available."""
        return (
            self._initialized and
            self.embedding_provider is not None and
            self.vector_store is not None and
            self.embedding_provider.is_available() and
            self.vector_store.is_available()
        )

    def index_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Index a file for semantic search.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            Indexing result with statistics
        """
        if not self.is_available():
            return {'success': False, 'error': 'Semantic search not available'}

        try:
            # Chunk the file
            chunks = self.code_chunker.chunk_file(file_path, content)
            logger.debug(f"Chunked {file_path} into {len(chunks)} chunks")

            # Generate embeddings
            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_provider.embed_batch(texts)

            # Store in Qdrant
            count = self.vector_store.add_embeddings(embeddings, chunks)

            return {
                'success': True,
                'file_path': file_path,
                'chunks_indexed': count,
                'chunks': len(chunks)
            }

        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            return {'success': False, 'error': str(e)}

    def index_project(self, project_path: str, file_patterns: Optional[list] = None) -> Dict[str, Any]:
        """
        Index entire project for semantic search.

        Args:
            project_path: Root path of the project
            file_patterns: List of file patterns to include (e.g., ['*.py', '*.js'])

        Returns:
            Indexing statistics
        """
        if not self.is_available():
            return {'success': False, 'error': 'Semantic search not available'}

        # Default patterns if none provided
        if not file_patterns:
            file_patterns = [
                '*.py', '*.js', '*.ts', '*.tsx', '*.jsx',
                '*.go', '*.rs', '*.java', '*.cpp', '*.cc', '*.h'
            ]

        project = Path(project_path)
        indexed_files = 0
        total_chunks = 0
        errors = []

        try:
            for pattern in file_patterns:
                for file_path in project.rglob(pattern):
                    if file_path.is_file():
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            result = self.index_file(str(file_path), content)

                            if result['success']:
                                indexed_files += 1
                                total_chunks += result.get('chunks', 0)
                            else:
                                errors.append(f"{file_path}: {result.get('error', 'Unknown error')}")

                        except Exception as e:
                            errors.append(f"{file_path}: {str(e)}")
                            logger.warning(f"Failed to read file {file_path}: {e}")

            return {
                'success': True,
                'indexed_files': indexed_files,
                'total_chunks': total_chunks,
                'errors': errors[:10]  # Limit error list
            }

        except Exception as e:
            logger.error(f"Project indexing failed: {e}")
            return {'success': False, 'error': str(e)}

    def reindex_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Reindex a file (delete old embeddings, add new ones).

        Args:
            file_path: Path to the file
            content: Updated file content

        Returns:
            Reindexing result
        """
        if not self.is_available():
            return {'success': False, 'error': 'Semantic search not available'}

        try:
            # Delete old embeddings
            self.vector_store.delete_by_file(file_path)

            # Index new content
            return self.index_file(file_path, content)

        except Exception as e:
            logger.error(f"Failed to reindex file {file_path}: {e}")
            return {'success': False, 'error': str(e)}

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the semantic search index."""
        if not self.is_available():
            return {'available': False}

        try:
            collection_info = self.vector_store.get_collection_info()
            return {
                'available': True,
                'embedding_provider': self.embedding_provider.get_model_name(),
                'embedding_dimension': self.embedding_provider.get_dimension(),
                'collection_name': collection_info.get('name', ''),
                'vectors_count': collection_info.get('vectors_count', 0),
                'points_count': collection_info.get('points_count', 0),
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {'available': True, 'error': str(e)}
