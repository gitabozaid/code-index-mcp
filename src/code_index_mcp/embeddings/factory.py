"""Factory for creating embedding providers."""

import logging
from typing import Dict, Any, Optional

from .base_provider import BaseEmbeddingProvider
from .ollama_provider import OllamaEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingProviderFactory:
    """Factory for creating embedding provider instances."""

    PROVIDERS = {
        'ollama': OllamaEmbeddingProvider,
        'openai': OpenAIEmbeddingProvider,
        # Future providers can be added here:
        # 'voyage': VoyageEmbeddingProvider,
        # 'fastembed': FastEmbedProvider,
    }

    @classmethod
    def create(cls, provider_type: str, config: Dict[str, Any]) -> BaseEmbeddingProvider:
        """
        Create an embedding provider instance.

        Args:
            provider_type: Type of provider ('ollama', etc.)
            config: Provider-specific configuration

        Returns:
            Initialized embedding provider

        Raises:
            ValueError: If provider type is unknown
        """
        provider_class = cls.PROVIDERS.get(provider_type.lower())

        if not provider_class:
            available = ', '.join(cls.PROVIDERS.keys())
            raise ValueError(
                f"Unknown embedding provider: {provider_type}. "
                f"Available providers: {available}"
            )

        logger.info(f"Creating embedding provider: {provider_type}")
        return provider_class(config)

    @classmethod
    def create_from_env(cls, config: Dict[str, Any]) -> Optional[BaseEmbeddingProvider]:
        """
        Create provider from configuration with automatic fallback.

        Args:
            config: Configuration dictionary from environment/settings

        Returns:
            Embedding provider instance or None if not configured
        """
        provider_type = config.get('embedding_provider', 'ollama').lower()

        try:
            provider = cls.create(provider_type, config)

            # Verify availability
            if not provider.is_available():
                logger.warning(f"Provider {provider_type} is not available")
                return None

            return provider

        except Exception as e:
            logger.error(f"Failed to create provider {provider_type}: {e}")
            return None
