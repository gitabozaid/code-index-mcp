"""OpenAI embedding provider using text-embedding-3-small model."""

import logging
from typing import List, Dict, Any
import numpy as np

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available. Install with: pip install openai")

from .base_provider import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using OpenAI with text-embedding-3-small model."""

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSION = 1536  # text-embedding-3-small dimension
    DEFAULT_BATCH_SIZE = 2048  # OpenAI supports up to 2048 texts per request!

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI embedding provider.

        Args:
            config: Configuration dictionary with keys:
                - api_key: OpenAI API key (required)
                - model: Embedding model name (default: text-embedding-3-small)
                - batch_size: Batch processing size (default: 2048, max: 2048)
                - timeout: Request timeout in seconds (default: 30)
        """
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or provide 'api_key' in config."
            )

        self.model = config.get('model', self.DEFAULT_MODEL)
        self.batch_size = min(
            config.get('batch_size', self.DEFAULT_BATCH_SIZE),
            self.DEFAULT_BATCH_SIZE
        )
        self.timeout = config.get('timeout', 30)
        self.dimension = config.get('dimension', self.DEFAULT_DIMENSION)

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=self.timeout
        )

        logger.info(
            f"Initialized OpenAI provider: model={self.model}, "
            f"batch_size={self.batch_size}, dimension={self.dimension}"
        )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text using OpenAI.

        Args:
            text: Input text (code snippet or natural language query)

        Returns:
            Numpy array containing the embedding vector

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text]  # OpenAI expects a list
            )

            if not response.data:
                raise RuntimeError("Empty response from OpenAI API")

            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.

        OpenAI supports up to 2048 texts per request, making batch processing
        extremely efficient compared to sequential processing.

        Args:
            texts: List of input texts

        Returns:
            List of numpy arrays containing embedding vectors

        Note:
            This implementation uses OpenAI's native batch API, which is
            significantly faster than processing texts one by one.
        """
        embeddings = []

        # Process in batches (max 2048 per request)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch  # Send entire batch at once!
                )

                if not response.data:
                    logger.warning(f"Empty response for batch starting at {i}")
                    # Add zero vectors as placeholders
                    for _ in batch:
                        embeddings.append(np.zeros(self.dimension, dtype=np.float32))
                    continue

                # Extract embeddings from response
                batch_embeddings = [
                    np.array(item.embedding, dtype=np.float32)
                    for item in response.data
                ]

                embeddings.extend(batch_embeddings)

                logger.debug(
                    f"Processed batch {i // self.batch_size + 1}: "
                    f"{len(batch)} texts → {len(batch_embeddings)} embeddings"
                )

            except Exception as e:
                logger.error(f"Batch embedding failed for batch starting at {i}: {e}")
                # Add zero vectors as placeholders for failed embeddings
                for _ in batch:
                    embeddings.append(np.zeros(self.dimension, dtype=np.float32))

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

    def get_model_name(self) -> str:
        """Get model identifier."""
        return self.model

    def is_available(self) -> bool:
        """
        Check if OpenAI API is accessible.

        Returns:
            True if OpenAI API is accessible and API key is valid
        """
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not installed")
            return False

        if not self.api_key:
            logger.warning("OpenAI API key not provided")
            return False

        try:
            # Test connectivity with a simple embedding request
            test_response = self.client.embeddings.create(
                model=self.model,
                input=["test"]
            )

            if not test_response.data:
                logger.warning("OpenAI returned empty response")
                return False

            logger.info(f"OpenAI provider is available: {self.model}")
            return True

        except Exception as e:
            logger.warning(f"OpenAI connection failed: {e}")
            logger.info(
                "Check your API key and internet connection. "
                "Get an API key at: https://platform.openai.com/api-keys"
            )
            return False
