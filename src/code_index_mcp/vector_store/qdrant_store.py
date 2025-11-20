"""Qdrant vector database wrapper for code embeddings storage."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
        SearchRequest, QueryResponse
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant client not available. Semantic search will not work.")

import numpy as np

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Qdrant vector database wrapper for storing and searching code embeddings.

    Responsibilities:
    - Create and manage Qdrant collections
    - Store code chunk embeddings with metadata
    - Perform similarity search
    - Handle collection versioning (reindex on model changes)
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "code-embeddings",
        vector_dim: int = 768,
        distance_metric: str = "cosine"
    ):
        """
        Initialize Qdrant store.

        Args:
            url: Qdrant server URL
            collection_name: Name of the collection
            vector_dim: Dimension of embedding vectors
            distance_metric: Distance metric ('cosine', 'euclid', 'dot')
        """
        if not QDRANT_AVAILABLE:
            raise RuntimeError("Qdrant client not installed. Run: pip install qdrant-client")

        self.url = url
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.distance_metric = distance_metric

        # Initialize client
        try:
            self.client = QdrantClient(url=url)
            logger.info(f"Connected to Qdrant at {url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise RuntimeError(f"Qdrant connection failed: {e}")

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")

                # Map distance metric string to Qdrant Distance enum
                distance_map = {
                    'cosine': Distance.COSINE,
                    'euclid': Distance.EUCLID,
                    'dot': Distance.DOT
                }
                distance = distance_map.get(self.distance_metric.lower(), Distance.COSINE)

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=distance
                    )
                )
                logger.info(f"Collection created: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def add_embeddings(
        self,
        embeddings: List[np.ndarray],
        chunks: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Add embeddings with metadata to Qdrant.

        Args:
            embeddings: List of embedding vectors
            chunks: List of chunk metadata dictionaries
            batch_size: Batch size for uploading

        Returns:
            Number of embeddings added
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Embeddings ({len(embeddings)}) and chunks ({len(chunks)}) length mismatch")

        points = []
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            # Generate UUID for the point
            point_id = str(uuid.uuid4())

            # Prepare payload with chunk metadata
            payload = {
                'chunk_id': chunk.get('id', f'chunk_{i}'),
                'file_path': chunk.get('file_path', ''),
                'start_line': chunk.get('start_line', 0),
                'end_line': chunk.get('end_line', 0),
                'chunk_type': chunk.get('chunk_type', 'unknown'),
                'symbol_name': chunk.get('symbol_name', ''),
                'content': chunk.get('content', '')[:1000],  # Limit content size
            }

            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)

        # Upload in batches
        total_uploaded = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                total_uploaded += len(batch)
                logger.debug(f"Uploaded batch {i // batch_size + 1}: {len(batch)} points")
            except Exception as e:
                logger.error(f"Failed to upload batch {i // batch_size + 1}: {e}")
                raise

        logger.info(f"Added {total_uploaded} embeddings to Qdrant")
        return total_uploaded

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        file_pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar code chunks.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1 for cosine)
            file_pattern: Optional file path filter

        Returns:
            List of search results with metadata and scores
        """
        try:
            # Build filter if file_pattern provided
            search_filter = None
            if file_pattern:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_pattern)
                        )
                    ]
                )

            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter
            )

            # Format results
            results = []
            for hit in search_result:
                result = {
                    'file_path': hit.payload.get('file_path', ''),
                    'start_line': hit.payload.get('start_line', 0),
                    'end_line': hit.payload.get('end_line', 0),
                    'chunk_type': hit.payload.get('chunk_type', ''),
                    'symbol_name': hit.payload.get('symbol_name', ''),
                    'content': hit.payload.get('content', ''),
                    'score': hit.score,
                    'source': 'semantic'
                }
                results.append(result)

            logger.debug(f"Found {len(results)} semantic search results")
            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise

    def delete_by_file(self, file_path: str) -> int:
        """
        Delete all embeddings for a specific file.

        Args:
            file_path: Path of the file to delete

        Returns:
            Number of points deleted
        """
        try:
            # Delete points with matching file_path
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_path)
                        )
                    ]
                )
            )
            logger.debug(f"Deleted embeddings for file: {file_path}")
            return 1  # Qdrant doesn't return count for delete operations
        except Exception as e:
            logger.error(f"Failed to delete embeddings for {file_path}: {e}")
            raise

    def clear_collection(self):
        """Delete all points in the collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()  # Recreate empty collection
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                'name': self.collection_name,
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {'error': str(e)}

    def is_available(self) -> bool:
        """Check if Qdrant is available."""
        if not QDRANT_AVAILABLE:
            return False

        try:
            # Try to get collections list
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant not available: {e}")
            return False
