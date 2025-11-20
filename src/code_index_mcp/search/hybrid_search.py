"""Hybrid search combining BM25 and semantic search with RRF fusion."""

import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .base import SearchStrategy

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Hybrid search engine combining BM25 (keyword) and semantic (vector) search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both strategies.

    RRF Formula:
        score(doc) = Σ 1 / (k + rank(doc))

    Where:
    - k = 60 (constant, typically 60)
    - rank(doc) = position of document in result list (1-indexed)
    """

    DEFAULT_K = 60  # RRF constant

    def __init__(
        self,
        bm25_strategy: SearchStrategy,
        semantic_strategy: SearchStrategy,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = DEFAULT_K
    ):
        """
        Initialize hybrid search engine.

        Args:
            bm25_strategy: BM25 search strategy (ripgrep, ag, etc.)
            semantic_strategy: Semantic search strategy
            bm25_weight: Weight for BM25 results (0-1)
            semantic_weight: Weight for semantic results (0-1)
            rrf_k: RRF constant (typically 60)
        """
        self.bm25_strategy = bm25_strategy
        self.semantic_strategy = semantic_strategy
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.rrf_k = rrf_k

        logger.info(
            f"Initialized hybrid search: BM25={bm25_strategy.name}, "
            f"Semantic={semantic_strategy.name}, "
            f"weights=({bm25_weight:.2f}, {semantic_weight:.2f})"
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
        Execute hybrid search with RRF fusion.

        Args:
            pattern: Search query
            base_path: Project root path
            case_sensitive: Case sensitivity for BM25
            context_lines: Context lines for BM25
            file_pattern: File pattern filter
            fuzzy: Fuzzy matching for BM25
            regex: Regex mode for BM25

        Returns:
            Merged results from both strategies
        """
        # Execute both searches in parallel (conceptually)
        logger.debug(f"Running hybrid search for: {pattern[:50]}...")

        bm25_results = {}
        semantic_results = {}

        # BM25 search
        try:
            if self.bm25_strategy.is_available():
                bm25_results = self.bm25_strategy.search(
                    pattern=pattern,
                    base_path=base_path,
                    case_sensitive=case_sensitive,
                    context_lines=context_lines,
                    file_pattern=file_pattern,
                    fuzzy=fuzzy,
                    regex=regex
                )
                logger.debug(f"BM25 found {len(bm25_results)} files")
            else:
                logger.warning("BM25 strategy not available")
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")

        # Semantic search
        try:
            if self.semantic_strategy.is_available():
                semantic_results = self.semantic_strategy.search(
                    pattern=pattern,
                    base_path=base_path,
                    file_pattern=file_pattern
                )
                logger.debug(f"Semantic found {len(semantic_results)} files")
            else:
                logger.warning("Semantic strategy not available")
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")

        # Merge results using RRF
        merged = self._merge_with_rrf(bm25_results, semantic_results)

        logger.info(f"Hybrid search merged {len(merged)} files")
        return merged

    def _merge_with_rrf(
        self,
        bm25_results: Dict[str, List[Tuple[int, str]]],
        semantic_results: Dict[str, List[Tuple[int, str]]]
    ) -> Dict[str, List[Tuple[int, str]]]:
        """
        Merge BM25 and semantic results using Reciprocal Rank Fusion (RRF).

        Args:
            bm25_results: Results from BM25 search
            semantic_results: Results from semantic search

        Returns:
            Merged and ranked results
        """
        # Build document rankings
        # Key: (file_path, line_number)
        # Value: RRF score
        scores = defaultdict(float)

        # Add BM25 results
        for file_path, matches in bm25_results.items():
            for rank, (line_num, content) in enumerate(matches, start=1):
                key = (file_path, line_num)
                rrf_score = self.bm25_weight / (self.rrf_k + rank)
                scores[key] += rrf_score

        # Add semantic results
        for file_path, matches in semantic_results.items():
            for rank, (line_num, content) in enumerate(matches, start=1):
                key = (file_path, line_num)
                rrf_score = self.semantic_weight / (self.rrf_k + rank)
                scores[key] += rrf_score

        # Sort by RRF score (descending)
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Reconstruct output format
        # We need to get the content for each (file_path, line_num)
        merged = defaultdict(list)

        # Build content map
        content_map = {}
        for file_path, matches in bm25_results.items():
            for line_num, content in matches:
                content_map[(file_path, line_num)] = content

        for file_path, matches in semantic_results.items():
            for line_num, content in matches:
                if (file_path, line_num) not in content_map:
                    content_map[(file_path, line_num)] = content

        # Build merged results
        for (file_path, line_num), score in sorted_items:
            content = content_map.get((file_path, line_num), "")
            merged[file_path].append((line_num, content))

        return dict(merged)


class HybridSearchStrategy(SearchStrategy):
    """
    Search strategy wrapper for hybrid search.

    This allows hybrid search to be used as a drop-in replacement
    for other search strategies in the existing codebase.
    """

    def __init__(
        self,
        bm25_strategy: SearchStrategy,
        semantic_strategy: SearchStrategy,
        config: Optional[Dict] = None
    ):
        """
        Initialize hybrid search strategy.

        Args:
            bm25_strategy: BM25 search strategy
            semantic_strategy: Semantic search strategy
            config: Configuration with weights and RRF settings
        """
        self.config = config or {}

        bm25_weight = self.config.get('bm25_weight', 0.5)
        semantic_weight = self.config.get('semantic_weight', 0.5)
        rrf_k = self.config.get('rrf_k', 60)

        self.engine = HybridSearchEngine(
            bm25_strategy=bm25_strategy,
            semantic_strategy=semantic_strategy,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            rrf_k=rrf_k
        )

    @property
    def name(self) -> str:
        """Strategy name."""
        return "hybrid"

    def is_available(self) -> bool:
        """Check if at least one search method is available."""
        return (
            self.engine.bm25_strategy.is_available() or
            self.engine.semantic_strategy.is_available()
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
        """Execute hybrid search."""
        return self.engine.search(
            pattern=pattern,
            base_path=base_path,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            file_pattern=file_pattern,
            fuzzy=fuzzy,
            regex=regex
        )
