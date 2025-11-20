"""Configuration helper for semantic search settings with global config support."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global configuration directory
GLOBAL_CONFIG_DIR = Path.home() / ".config" / "code-index-mcp"
GLOBAL_ENV_FILE = GLOBAL_CONFIG_DIR / ".env"
GLOBAL_IGNORE_FILE = GLOBAL_CONFIG_DIR / "ignore-patterns.txt"


def _load_global_env() -> Dict[str, str]:
    """
    Load environment variables from global config file.

    Returns:
        Dictionary of environment variables from global config
    """
    env_vars = {}

    if not GLOBAL_ENV_FILE.exists():
        logger.debug(f"Global config file not found: {GLOBAL_ENV_FILE}")
        return env_vars

    try:
        with open(GLOBAL_ENV_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()

        logger.debug(f"Loaded {len(env_vars)} variables from global config")
    except Exception as e:
        logger.error(f"Failed to load global config: {e}")

    return env_vars


def _get_config_value(key: str, default: str = '', global_env: Optional[Dict[str, str]] = None) -> str:
    """
    Get configuration value with priority: os.environ > global config > default.

    Args:
        key: Configuration key
        default: Default value if not found
        global_env: Cached global environment variables

    Returns:
        Configuration value
    """
    # Priority 1: OS environment variables
    if key in os.environ:
        return os.environ[key]

    # Priority 2: Global config file
    if global_env is None:
        global_env = _load_global_env()

    if key in global_env:
        return global_env[key]

    # Priority 3: Default value
    return default


class SemanticSearchConfig:
    """Configuration manager for semantic search settings."""

    @staticmethod
    def load_from_env() -> Dict[str, Any]:
        """
        Load semantic search configuration from global config and environment variables.

        Priority: OS environment > global config file > defaults

        Returns:
            Dictionary with all configuration parameters
        """
        # Load global environment once for efficiency
        global_env = _load_global_env()

        config = {
            # Embedding provider selection
            'embedding_provider': _get_config_value('EMBEDDING_PROVIDER', 'ollama', global_env),

            # Ollama settings
            'ollama_url': _get_config_value('OLLAMA_URL', 'http://localhost:11434', global_env),
            'ollama_model': _get_config_value('OLLAMA_MODEL', 'nomic-embed-text', global_env),
            'ollama_batch_size': int(_get_config_value('OLLAMA_BATCH_SIZE', '10', global_env)),
            'ollama_timeout': int(_get_config_value('OLLAMA_TIMEOUT', '30', global_env)),

            # OpenAI settings
            'openai_api_key': _get_config_value('OPENAI_API_KEY', '', global_env),
            'openai_model': _get_config_value('OPENAI_MODEL', 'text-embedding-3-small', global_env),
            'openai_dimension': int(_get_config_value('OPENAI_DIMENSION', '1536', global_env)),
            'openai_batch_size': int(_get_config_value('OPENAI_BATCH_SIZE', '2048', global_env)),

            # Qdrant settings
            'qdrant_url': _get_config_value('QDRANT_URL', 'http://localhost:6333', global_env),
            'qdrant_collection': _get_config_value('QDRANT_COLLECTION', 'code-embeddings', global_env),
            'vector_dim': int(_get_config_value('QDRANT_VECTOR_DIM', '1536', global_env)),

            # Search settings
            'search_mode': _get_config_value('SEARCH_MODE', 'hybrid', global_env),  # bm25, semantic, hybrid
            'bm25_weight': float(_get_config_value('BM25_WEIGHT', '0.5', global_env)),
            'semantic_weight': float(_get_config_value('SEMANTIC_WEIGHT', '0.5', global_env)),
            'semantic_enabled': _get_config_value('SEMANTIC_SEARCH_ENABLED', 'true', global_env).lower() == 'true',

            # Index directory
            'index_dir': _get_config_value('INDEX_DIR', './.indexes', global_env),
        }

        logger.debug(f"Loaded semantic search config (provider: {config['embedding_provider']})")
        return config

    @staticmethod
    def is_semantic_enabled() -> bool:
        """Check if semantic search is enabled."""
        global_env = _load_global_env()
        enabled = _get_config_value('SEMANTIC_SEARCH_ENABLED', 'true', global_env)
        return enabled.lower() == 'true'

    @staticmethod
    def get_search_mode() -> str:
        """
        Get configured search mode.

        Returns:
            'bm25', 'semantic', or 'hybrid'
        """
        global_env = _load_global_env()
        mode = _get_config_value('SEARCH_MODE', 'hybrid', global_env).lower()
        if mode not in ['bm25', 'semantic', 'hybrid']:
            logger.warning(f"Invalid SEARCH_MODE: {mode}. Defaulting to 'hybrid'")
            return 'hybrid'
        return mode

    @staticmethod
    def get_embedding_config() -> Dict[str, Any]:
        """
        Get embedding provider configuration based on selected provider.

        Returns:
            Configuration dictionary for the selected embedding provider
        """
        global_env = _load_global_env()
        provider = _get_config_value('EMBEDDING_PROVIDER', 'ollama', global_env).lower()

        if provider == 'openai':
            return {
                'embedding_provider': 'openai',
                'api_key': _get_config_value('OPENAI_API_KEY', '', global_env),
                'model': _get_config_value('OPENAI_MODEL', 'text-embedding-3-small', global_env),
                'dimension': int(_get_config_value('OPENAI_DIMENSION', '1536', global_env)),
                'batch_size': int(_get_config_value('OPENAI_BATCH_SIZE', '2048', global_env)),
            }
        else:  # Default to Ollama
            return {
                'embedding_provider': 'ollama',
                'model': _get_config_value('OLLAMA_MODEL', 'nomic-embed-text', global_env),
                'url': _get_config_value('OLLAMA_URL', 'http://localhost:11434', global_env),
                'batch_size': int(_get_config_value('OLLAMA_BATCH_SIZE', '10', global_env)),
                'timeout': int(_get_config_value('OLLAMA_TIMEOUT', '30', global_env)),
                'dimension': int(_get_config_value('QDRANT_VECTOR_DIM', '768', global_env)),
            }

    @staticmethod
    def get_qdrant_config() -> Dict[str, Any]:
        """Get Qdrant configuration from global config or environment."""
        global_env = _load_global_env()
        return {
            'url': _get_config_value('QDRANT_URL', 'http://localhost:6333', global_env),
            'collection_name': _get_config_value('QDRANT_COLLECTION', 'code-embeddings', global_env),
            'vector_dim': int(_get_config_value('QDRANT_VECTOR_DIM', '1536', global_env)),
            'distance_metric': 'cosine'
        }

    @staticmethod
    def get_search_weights() -> tuple[float, float]:
        """
        Get BM25 and semantic search weights from global config or environment.

        Returns:
            Tuple of (bm25_weight, semantic_weight)
        """
        global_env = _load_global_env()
        bm25 = float(_get_config_value('BM25_WEIGHT', '0.5', global_env))
        semantic = float(_get_config_value('SEMANTIC_WEIGHT', '0.5', global_env))

        # Normalize if needed
        total = bm25 + semantic
        if total > 0:
            bm25 = bm25 / total
            semantic = semantic / total

        return (bm25, semantic)

    @staticmethod
    def get_ignore_patterns() -> list[str]:
        """
        Load ignore patterns from global config file.

        Returns:
            List of gitignore-style patterns to exclude from indexing
        """
        patterns = []

        if not GLOBAL_IGNORE_FILE.exists():
            logger.debug(f"Global ignore patterns file not found: {GLOBAL_IGNORE_FILE}")
            return patterns

        try:
            with open(GLOBAL_IGNORE_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        patterns.append(line)

            logger.debug(f"Loaded {len(patterns)} ignore patterns from global config")
        except Exception as e:
            logger.error(f"Failed to load ignore patterns: {e}")

        return patterns
