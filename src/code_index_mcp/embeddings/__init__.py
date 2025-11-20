"""Embedding provider abstraction layer."""

from .base_provider import BaseEmbeddingProvider
from .ollama_provider import OllamaEmbeddingProvider
from .factory import EmbeddingProviderFactory

__all__ = [
    'BaseEmbeddingProvider',
    'OllamaEmbeddingProvider',
    'EmbeddingProviderFactory',
]
