"""
Search service for the Code Index MCP server.

This service handles code search operations, search tool management,
and search strategy selection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_service import BaseService
from ..utils import FileFilter, ResponseFormatter, ValidationHelper, QueryAnalyzer
from ..search.base import is_safe_regex_pattern


class SearchService(BaseService):
    """Service for managing code search operations."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.file_filter = self._create_file_filter()

    def search_code(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        pattern: str,
        case_sensitive: bool = True,
        context_lines: int = 0,
        file_pattern: Optional[str] = None,
        fuzzy: bool = False,
        regex: Optional[bool] = None,
        start_index: int = 0,
        max_results: Optional[int] = 10
    ) -> Dict[str, Any]:
        """Search for code patterns in the project."""
        self._require_project_setup()

        if regex is None:
            regex = is_safe_regex_pattern(pattern)

        error = ValidationHelper.validate_search_pattern(pattern, regex)
        if error:
            raise ValueError(error)

        if file_pattern:
            error = ValidationHelper.validate_glob_pattern(file_pattern)
            if error:
                raise ValueError(f"Invalid file pattern: {error}")

        pagination_error = ValidationHelper.validate_pagination(start_index, max_results)
        if pagination_error:
            raise ValueError(pagination_error)

        if not self.settings:
            raise ValueError("Settings not available")

        strategy = self.settings.get_preferred_search_tool()
        if not strategy:
            raise ValueError("No search strategies available")

        self._configure_strategy(strategy)

        try:
            results = strategy.search(
                pattern=pattern,
                base_path=self.base_path,
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                file_pattern=file_pattern,
                fuzzy=fuzzy,
                regex=regex
            )
            filtered = self._filter_results(results)
            formatted_results, pagination = self._paginate_results(
                filtered,
                start_index=start_index,
                max_results=max_results
            )
            return ResponseFormatter.search_results_response(
                formatted_results,
                pagination
            )
        except Exception as exc:
            raise ValueError(f"Search failed using '{strategy.name}': {exc}") from exc

    def search_hybrid_code(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        pattern: str,
        case_sensitive: bool = True,
        context_lines: int = 0,
        file_pattern: Optional[str] = None,
        start_index: int = 0,
        max_results: Optional[int] = 10,
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        auto_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Search using hybrid BM25 + semantic search with RRF fusion.

        Combines keyword-based search (BM25) with semantic understanding (OpenAI embeddings)
        and merges results using Reciprocal Rank Fusion (RRF) for optimal ranking.

        Args:
            pattern: Search query (can be keywords or semantic description)
            case_sensitive: Whether BM25 search is case-sensitive
            context_lines: Number of context lines to show (BM25 only)
            file_pattern: Glob pattern to filter files
            start_index: Pagination offset
            max_results: Maximum number of results to return
            bm25_weight: Weight for BM25 results (0.0-1.0, default None = auto or 0.5)
            semantic_weight: Weight for semantic results (0.0-1.0, default None = auto or 0.5)
            auto_weights: Enable automatic weight selection based on query analysis (default False)

        Returns:
            Dict with search results and pagination metadata, including:
            - results: List of matches
            - pagination: Pagination info
            - weights_info: Weight distribution and reasoning (if auto_weights=True)

        Raises:
            ValueError: If configuration is invalid or search fails
        """
        self._require_project_setup()

        # Validate inputs
        error = ValidationHelper.validate_search_pattern(pattern, regex=False)
        if error:
            raise ValueError(error)

        if file_pattern:
            error = ValidationHelper.validate_glob_pattern(file_pattern)
            if error:
                raise ValueError(f"Invalid file pattern: {error}")

        pagination_error = ValidationHelper.validate_pagination(start_index, max_results)
        if pagination_error:
            raise ValueError(pagination_error)

        if not self.settings:
            raise ValueError("Settings not available")

        # Auto-weight selection based on query analysis
        weights_info = None
        if auto_weights:
            # Use QueryAnalyzer to determine optimal weights
            analysis = QueryAnalyzer.analyze(pattern)
            bm25_weight = analysis['bm25_weight']
            semantic_weight = analysis['semantic_weight']
            weights_info = {
                'auto_selected': True,
                'bm25_weight': bm25_weight,
                'semantic_weight': semantic_weight,
                'query_type': analysis['query_type'],
                'reasoning': analysis['reasoning']
            }
        else:
            # Use provided weights or defaults
            if bm25_weight is None:
                bm25_weight = 0.5
            if semantic_weight is None:
                semantic_weight = 0.5
            weights_info = {
                'auto_selected': False,
                'bm25_weight': bm25_weight,
                'semantic_weight': semantic_weight,
                'query_type': 'manual',
                'reasoning': 'Weights manually specified or using defaults'
            }

        # Import hybrid search components
        try:
            from ..search.semantic_search import SemanticSearchStrategy
            from ..search.hybrid_search import HybridSearchStrategy
            from ..utils.config_helper import SemanticSearchConfig
        except ImportError as exc:
            raise ValueError(
                "Hybrid search requires semantic_search and hybrid_search modules"
            ) from exc

        # Load semantic search configuration from global config
        try:
            config = SemanticSearchConfig.load_from_env()
        except Exception as exc:
            raise ValueError(f"Failed to load semantic search configuration: {exc}") from exc

        # Get preferred BM25 strategy
        bm25_strategy = self.settings.get_preferred_search_tool()
        if not bm25_strategy:
            raise ValueError("No BM25 search strategies available")

        # Create semantic search strategy
        try:
            semantic_strategy = SemanticSearchStrategy(config)
        except Exception as exc:
            raise ValueError(f"Failed to create semantic search strategy: {exc}") from exc

        # Check if semantic search is available (Qdrant + embeddings)
        if not semantic_strategy.is_available():
            raise ValueError(
                "Semantic search not available. "
                "Ensure Qdrant is running and embeddings are configured. "
                "Check ~/.config/code-index-mcp/.env for OPENAI_API_KEY and QDRANT_URL."
            )

        # Create hybrid strategy with configured weights
        try:
            hybrid_strategy = HybridSearchStrategy(
                bm25_strategy=bm25_strategy,
                semantic_strategy=semantic_strategy,
                config={
                    'bm25_weight': bm25_weight,
                    'semantic_weight': semantic_weight,
                    'rrf_k': 60  # RRF constant (higher = more balanced ranking)
                }
            )
        except Exception as exc:
            raise ValueError(f"Failed to create hybrid search strategy: {exc}") from exc

        # Configure file exclusions
        self._configure_strategy(hybrid_strategy)

        # Execute hybrid search
        try:
            results = hybrid_strategy.search(
                pattern=pattern,
                base_path=self.base_path,
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                file_pattern=file_pattern,
                fuzzy=False,  # Fuzzy not applicable to semantic search
                regex=False   # Semantic search uses embeddings, not regex
            )

            # Filter, paginate, and format results (same as regular search)
            filtered = self._filter_results(results)
            formatted_results, pagination = self._paginate_results(
                filtered,
                start_index=start_index,
                max_results=max_results
            )

            # Build response with weights info
            response = ResponseFormatter.search_results_response(
                formatted_results,
                pagination
            )

            # Add weights information to response
            if weights_info:
                response['weights_info'] = weights_info

            return response
        except Exception as exc:
            raise ValueError(f"Hybrid search failed: {exc}") from exc

    def refresh_search_tools(self) -> str:
        """Refresh the available search tools."""
        if not self.settings:
            raise ValueError("Settings not available")

        self.settings.refresh_available_strategies()
        config = self.settings.get_search_tools_config()

        available = config['available_tools']
        preferred = config['preferred_tool']
        return f"Search tools refreshed. Available: {available}. Preferred: {preferred}."

    def get_search_capabilities(self) -> Dict[str, Any]:
        """Get information about search capabilities and available tools."""
        if not self.settings:
            return {"error": "Settings not available"}

        config = self.settings.get_search_tools_config()

        capabilities = {
            "available_tools": config.get('available_tools', []),
            "preferred_tool": config.get('preferred_tool', 'basic'),
            "supports_regex": True,
            "supports_fuzzy": True,
            "supports_case_sensitivity": True,
            "supports_context_lines": True,
            "supports_file_patterns": True
        }

        return capabilities

    def _configure_strategy(self, strategy) -> None:
        """Apply shared exclusion configuration to the strategy if supported."""
        configure = getattr(strategy, 'configure_excludes', None)
        if not configure:
            return

        try:
            configure(self.file_filter)
        except Exception:  # pragma: no cover - defensive fallback
            pass

    def _create_file_filter(self) -> FileFilter:
        """Build a shared file filter drawing from project settings."""
        additional_dirs: List[str] = []
        additional_file_patterns: List[str] = []

        settings = self.settings
        if settings:
            try:
                config = settings.get_file_watcher_config()
            except Exception:  # pragma: no cover - fallback if config fails
                config = {}

            for key in ('exclude_patterns', 'additional_exclude_patterns'):
                patterns = config.get(key) or []
                for pattern in patterns:
                    if not isinstance(pattern, str):
                        continue
                    normalized = pattern.strip()
                    if not normalized:
                        continue
                    additional_dirs.append(normalized)
                    additional_file_patterns.append(normalized)

        file_filter = FileFilter(additional_dirs or None)

        if additional_file_patterns:
            file_filter.exclude_files.update(additional_file_patterns)

        return file_filter

    def _filter_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out matches that reside under excluded paths."""
        if not isinstance(results, dict) or not results:
            return results

        if 'error' in results or not self.file_filter or not self.base_path:
            return results

        base_path = Path(self.base_path)
        filtered: Dict[str, Any] = {}

        for rel_path, matches in results.items():
            if not isinstance(rel_path, str):
                continue

            normalized = Path(rel_path.replace('\\', '/'))
            try:
                absolute = (base_path / normalized).resolve()
            except Exception:  # pragma: no cover - invalid path safety
                continue

            try:
                if self.file_filter.should_process_path(absolute, base_path):
                    filtered[rel_path] = matches
            except Exception:  # pragma: no cover - defensive fallback
                continue

        return filtered

    def _paginate_results(
        self,
        results: Dict[str, Any],
        start_index: int,
        max_results: Optional[int]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Apply pagination to search results and format them for responses."""
        total_matches = 0
        for matches in results.values():
            if isinstance(matches, (list, tuple)):
                total_matches += len(matches)

        effective_start = min(max(start_index, 0), total_matches)

        if total_matches == 0 or effective_start >= total_matches:
            pagination = self._build_pagination_metadata(
                total_matches=total_matches,
                returned=0,
                start_index=effective_start,
                max_results=max_results
            )
            return [], pagination

        collected: List[Dict[str, Any]] = []
        current_index = 0

        sorted_items = sorted(
            (
                (path, matches)
                for path, matches in results.items()
                if isinstance(path, str) and isinstance(matches, (list, tuple))
            ),
            key=lambda item: item[0]
        )

        for path, matches in sorted_items:
            sorted_matches = sorted(
                (match for match in matches if isinstance(match, (list, tuple)) and len(match) >= 2),
                key=lambda pair: pair[0]
            )

            for line_number, content, *_ in sorted_matches:
                if current_index >= effective_start:
                    if max_results is None or len(collected) < max_results:
                        collected.append({
                            "file": path,
                            "line": line_number,
                            "text": content
                        })
                    else:
                        break
                current_index += 1
            if max_results is not None and len(collected) >= max_results:
                break

        pagination = self._build_pagination_metadata(
            total_matches=total_matches,
            returned=len(collected),
            start_index=effective_start,
            max_results=max_results
        )
        return collected, pagination

    @staticmethod
    def _build_pagination_metadata(
        total_matches: int,
        returned: int,
        start_index: int,
        max_results: Optional[int]
    ) -> Dict[str, Any]:
        """Construct pagination metadata for search responses."""
        end_index = start_index + returned
        metadata: Dict[str, Any] = {
            "total_matches": total_matches,
            "returned": returned,
            "start_index": start_index,
            "has_more": end_index < total_matches
        }

        if max_results is not None:
            metadata["max_results"] = max_results

        metadata["end_index"] = end_index
        return metadata
