"""Configuration helper for Ollama and Qdrant settings."""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SemanticSearchConfig:
    """Configuration manager for semantic search settings."""

    @staticmethod
    def load_from_env() -> Dict[str, Any]:
        """
        Load semantic search configuration from environment variables.

        Returns:
            Dictionary with all configuration parameters
        """
        config = {
            # Ollama settings
            'ollama_url': os.getenv('OLLAMA_URL', 'http://localhost:11434'),
            'ollama_model': os.getenv('OLLAMA_MODEL', 'nomic-embed-code'),
            'ollama_batch_size': int(os.getenv('OLLAMA_BATCH_SIZE', '10')),
            'ollama_timeout': int(os.getenv('OLLAMA_TIMEOUT', '30')),

            # Qdrant settings
            'qdrant_url': os.getenv('QDRANT_URL', 'http://localhost:6333'),
            'qdrant_collection': os.getenv('QDRANT_COLLECTION', 'code-embeddings'),
            'vector_dim': int(os.getenv('QDRANT_VECTOR_DIM', '768')),

            # Search settings
            'search_mode': os.getenv('SEARCH_MODE', 'hybrid'),  # bm25, semantic, hybrid
            'bm25_weight': float(os.getenv('BM25_WEIGHT', '0.5')),
            'semantic_weight': float(os.getenv('SEMANTIC_WEIGHT', '0.5')),
            'semantic_enabled': os.getenv('SEMANTIC_SEARCH_ENABLED', 'true').lower() == 'true',

            # Embedding provider
            'embedding_provider': os.getenv('EMBEDDING_PROVIDER', 'ollama'),

            # Index directory
            'index_dir': os.getenv('INDEX_DIR', './.indexes'),
        }

        logger.debug(f"Loaded semantic search config: {config}")
        return config

    @staticmethod
    def is_semantic_enabled() -> bool:
        """Check if semantic search is enabled."""
        return os.getenv('SEMANTIC_SEARCH_ENABLED', 'true').lower() == 'true'

    @staticmethod
    def get_search_mode() -> str:
        """
        Get configured search mode.

        Returns:
            'bm25', 'semantic', or 'hybrid'
        """
        mode = os.getenv('SEARCH_MODE', 'hybrid').lower()
        if mode not in ['bm25', 'semantic', 'hybrid']:
            logger.warning(f"Invalid SEARCH_MODE: {mode}. Defaulting to 'hybrid'")
            return 'hybrid'
        return mode

    @staticmethod
    def get_embedding_config() -> Dict[str, Any]:
        """Get embedding provider configuration."""
        return {
            'embedding_provider': os.getenv('EMBEDDING_PROVIDER', 'ollama'),
            'model': os.getenv('OLLAMA_MODEL', 'nomic-embed-code'),
            'url': os.getenv('OLLAMA_URL', 'http://localhost:11434'),
            'batch_size': int(os.getenv('OLLAMA_BATCH_SIZE', '10')),
            'timeout': int(os.getenv('OLLAMA_TIMEOUT', '30')),
            'dimension': int(os.getenv('QDRANT_VECTOR_DIM', '768')),
        }

    @staticmethod
    def get_qdrant_config() -> Dict[str, Any]:
        """Get Qdrant configuration."""
        return {
            'url': os.getenv('QDRANT_URL', 'http://localhost:6333'),
            'collection_name': os.getenv('QDRANT_COLLECTION', 'code-embeddings'),
            'vector_dim': int(os.getenv('QDRANT_VECTOR_DIM', '768')),
            'distance_metric': 'cosine'
        }

    @staticmethod
    def get_search_weights() -> tuple[float, float]:
        """
        Get BM25 and semantic search weights.

        Returns:
            Tuple of (bm25_weight, semantic_weight)
        """
        bm25 = float(os.getenv('BM25_WEIGHT', '0.5'))
        semantic = float(os.getenv('SEMANTIC_WEIGHT', '0.5'))

        # Normalize if needed
        total = bm25 + semantic
        if total > 0:
            bm25 = bm25 / total
            semantic = semantic / total

        return (bm25, semantic)
