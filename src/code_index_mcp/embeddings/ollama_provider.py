"""Ollama embedding provider using nomic-embed-code model."""

import logging
from typing import List, Dict, Any
import numpy as np
import ollama
from ollama import ResponseError

from .base_provider import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using Ollama with nomic-embed-code model."""

    DEFAULT_MODEL = "nomic-embed-code"
    DEFAULT_DIMENSION = 768  # nomic-embed-code dimension
    DEFAULT_URL = "http://localhost:11434"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ollama embedding provider.

        Args:
            config: Configuration dictionary with keys:
                - model: Embedding model name (default: nomic-embed-code)
                - url: Ollama API URL (default: http://localhost:11434)
                - batch_size: Batch processing size (default: 10)
                - timeout: Request timeout in seconds (default: 30)
        """
        self.model = config.get('model', self.DEFAULT_MODEL)
        self.url = config.get('url', self.DEFAULT_URL)
        self.batch_size = config.get('batch_size', 10)
        self.timeout = config.get('timeout', 30)
        self.dimension = config.get('dimension', self.DEFAULT_DIMENSION)

        # Configure Ollama client
        self.client = ollama.Client(host=self.url)

        logger.info(
            f"Initialized Ollama provider: model={self.model}, "
            f"url={self.url}, dimension={self.dimension}"
        )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text using Ollama.

        Args:
            text: Input text (code snippet or natural language query)

        Returns:
            Numpy array containing the embedding vector

        Raises:
            RuntimeError: If embedding generation fails

        Note:
            Uses the new /api/embed endpoint (supersedes /api/embeddings)
            Response format: {'embedding': [...], 'model': '...', ...}
        """
        try:
            # Use ollama.embed() which calls /api/embed endpoint
            response = self.client.embed(
                model=self.model,
                input=text
            )

            # API returns 'embeddings' (array of arrays) or 'embedding' (single array)
            # Try both formats for compatibility
            # Format 1: {'embeddings': [[...]], ...}  (current Ollama API)
            # Format 2: {'embedding': [...], ...}      (future API?)
            embedding = response.get('embedding')
            if embedding is None:
                embeddings = response.get('embeddings', [])
                embedding = embeddings[0] if embeddings else []

            if not embedding:
                raise RuntimeError("Empty embedding returned from Ollama")

            return np.array(embedding, dtype=np.float32)

        except ResponseError as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in embed_text: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts

        Returns:
            List of numpy arrays containing embedding vectors

        Note:
            Ollama Python client handles batch processing internally.
            We still batch to manage memory and provide progress logging.
        """
        embeddings = []

        # Process in batches for efficiency and memory management
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                # Ollama client accepts both single text and list of texts
                # For batch, we call embed() for each text in the batch
                # (Ollama API doesn't have true batch endpoint yet)
                for text in batch:
                    response = self.client.embed(
                        model=self.model,
                        input=text
                    )

                    embedding = response.get('embedding', [])

                    if embedding:
                        embeddings.append(np.array(embedding, dtype=np.float32))
                    else:
                        logger.warning(f"Empty embedding for text: {text[:50]}...")
                        embeddings.append(np.zeros(self.dimension, dtype=np.float32))

                logger.debug(f"Processed batch {i // self.batch_size + 1}: {len(batch)} texts")

            except Exception as e:
                logger.error(f"Batch embedding failed for batch starting at {i}: {e}")
                # Continue processing remaining batches
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
        Check if Ollama is running and model is available.

        Returns:
            True if Ollama is accessible and model is loaded
        """
        try:
            # Test connectivity with a simple embedding request
            test_response = self.client.embed(
                model=self.model,
                input="test"
            )

            # Try both API formats (embeddings array or embedding single)
            embedding = test_response.get('embedding')
            if embedding is None:
                embeddings = test_response.get('embeddings', [])
                embedding = embeddings[0] if embeddings else []

            if not embedding:
                logger.warning(f"Ollama returned empty embedding for model {self.model}")
                return False

            logger.info(f"Ollama provider is available: {self.model}")
            return True

        except ResponseError as e:
            logger.warning(f"Ollama model {self.model} not available: {e}")
            logger.info(f"To pull the model, run: ollama pull {self.model}")
            return False
        except Exception as e:
            logger.warning(f"Ollama connection failed at {self.url}: {e}")
            logger.info("Ensure Ollama is running: brew services start ollama")
            return False
