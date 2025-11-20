"""Base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedding provider with configuration."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Numpy array containing the embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (batch processing).

        Args:
            texts: List of input texts to embed

        Returns:
            List of numpy arrays containing embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the embedding dimension for this provider.

        Returns:
            Integer dimension of embedding vectors
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the model name used by this provider.

        Returns:
            String identifier for the model
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available and properly configured.

        Returns:
            True if provider is ready to use, False otherwise
        """
        pass
