"""
Query Analyzer for Auto-Weight Selection in Hybrid Search.

This module analyzes search queries and recommends optimal BM25/Semantic weights
based on query characteristics and intent.
"""

import re
from typing import Dict, Tuple


class QueryAnalyzer:
    """
    Analyzes search queries and recommends optimal BM25/Semantic weight distribution.

    The analyzer uses heuristics to determine query intent and complexity:
    - Exact identifiers (function/class names) → High BM25 weight
    - Keywords with technical terms → Balanced weights
    - Conceptual/descriptive queries → High Semantic weight
    - Natural language questions → Very high Semantic weight
    """

    # Patterns for query classification
    EXACT_IDENTIFIER_PATTERNS = [
        r'^[A-Z][a-zA-Z0-9_]*$',           # PascalCase (e.g., UserController)
        r'^[a-z_][a-z0-9_]*\(\)$',         # function_name()
        r'^[a-z_][a-z0-9_]*\.[a-z_]+$',    # module.function
        r'^[A-Z_][A-Z0-9_]*$',             # CONSTANT_NAME
    ]

    KEYWORD_INDICATORS = [
        'function', 'class', 'method', 'variable', 'const',
        'def', 'import', 'export', 'return', 'if', 'for', 'while',
    ]

    CONCEPTUAL_INDICATORS = [
        'how', 'why', 'what', 'when', 'where', 'pattern', 'approach',
        'best practice', 'implementation', 'design', 'architecture',
        'strategy', 'technique', 'process', 'flow', 'handling',
    ]

    QUESTION_WORDS = ['how', 'why', 'what', 'when', 'where', 'which', 'who']

    @classmethod
    def analyze(cls, query: str) -> Dict[str, any]:
        """
        Analyze query and return recommended weights with reasoning.

        Args:
            query: The search query string

        Returns:
            Dictionary containing:
            - bm25_weight: Recommended BM25 weight (0.0-1.0)
            - semantic_weight: Recommended Semantic weight (0.0-1.0)
            - query_type: Classification of query type
            - reasoning: Explanation for weight selection

        Examples:
            >>> QueryAnalyzer.analyze("UserController")
            {
                'bm25_weight': 0.9,
                'semantic_weight': 0.1,
                'query_type': 'exact_identifier',
                'reasoning': 'Exact class/function name - prioritizing keyword matching'
            }

            >>> QueryAnalyzer.analyze("how does authentication work")
            {
                'bm25_weight': 0.2,
                'semantic_weight': 0.8,
                'query_type': 'conceptual_question',
                'reasoning': 'Natural language question - prioritizing semantic understanding'
            }
        """
        query_lower = query.lower().strip()
        query_words = query_lower.split()

        # 1. Check for exact identifiers (PascalCase, function names, etc.)
        if cls._is_exact_identifier(query):
            return {
                'bm25_weight': 0.9,
                'semantic_weight': 0.1,
                'query_type': 'exact_identifier',
                'reasoning': 'Exact class/function name - prioritizing keyword matching'
            }

        # 2. Check for natural language questions
        if cls._is_question(query_lower, query_words):
            return {
                'bm25_weight': 0.2,
                'semantic_weight': 0.8,
                'query_type': 'conceptual_question',
                'reasoning': 'Natural language question - prioritizing semantic understanding'
            }

        # 3. Check for conceptual/pattern queries
        if cls._is_conceptual(query_lower, query_words):
            return {
                'bm25_weight': 0.3,
                'semantic_weight': 0.7,
                'query_type': 'conceptual_search',
                'reasoning': 'Conceptual query about patterns/architecture - favoring semantic search'
            }

        # 4. Check for keyword-heavy queries
        if cls._is_keyword_heavy(query_lower, query_words):
            return {
                'bm25_weight': 0.7,
                'semantic_weight': 0.3,
                'query_type': 'keyword_search',
                'reasoning': 'Technical keywords detected - favoring keyword matching'
            }

        # 5. Check for short precise queries (1-2 words)
        if len(query_words) <= 2:
            return {
                'bm25_weight': 0.8,
                'semantic_weight': 0.2,
                'query_type': 'short_precise',
                'reasoning': 'Short query - likely searching for specific terms'
            }

        # 6. Check for long descriptive queries (6+ words)
        if len(query_words) >= 6:
            return {
                'bm25_weight': 0.3,
                'semantic_weight': 0.7,
                'query_type': 'descriptive_search',
                'reasoning': 'Long descriptive query - semantic understanding needed'
            }

        # 7. Default: Balanced approach
        return {
            'bm25_weight': 0.5,
            'semantic_weight': 0.5,
            'query_type': 'balanced',
            'reasoning': 'Standard query - using balanced hybrid search'
        }

    @classmethod
    def _is_exact_identifier(cls, query: str) -> bool:
        """Check if query looks like an exact identifier (class/function name)."""
        query = query.strip()

        # Check against exact identifier patterns
        for pattern in cls.EXACT_IDENTIFIER_PATTERNS:
            if re.match(pattern, query):
                return True

        # Check for quoted strings (indicates exact match intent)
        if query.startswith('"') and query.endswith('"'):
            return True

        return False

    @classmethod
    def _is_question(cls, query_lower: str, query_words: list) -> bool:
        """Check if query is a natural language question."""
        # Starts with question word
        if query_words and query_words[0] in cls.QUESTION_WORDS:
            return True

        # Contains question mark
        if '?' in query_lower:
            return True

        # Contains multiple question words (likely a complex question)
        question_word_count = sum(1 for word in query_words if word in cls.QUESTION_WORDS)
        if question_word_count >= 2:
            return True

        return False

    @classmethod
    def _is_conceptual(cls, query_lower: str, query_words: list) -> bool:
        """Check if query is conceptual/architectural in nature."""
        # Check for conceptual indicator words/phrases
        for indicator in cls.CONCEPTUAL_INDICATORS:
            if indicator in query_lower:
                return True

        # Check for phrases indicating understanding/explanation
        conceptual_phrases = [
            'best practice', 'design pattern', 'error handling',
            'data flow', 'control flow', 'implementation detail',
        ]
        for phrase in conceptual_phrases:
            if phrase in query_lower:
                return True

        return False

    @classmethod
    def _is_keyword_heavy(cls, query_lower: str, query_words: list) -> bool:
        """Check if query is heavy on technical keywords."""
        keyword_count = sum(1 for word in query_words if word in cls.KEYWORD_INDICATORS)

        # If 50%+ of words are technical keywords
        if len(query_words) > 0 and keyword_count / len(query_words) >= 0.5:
            return True

        # Check for common code patterns
        code_patterns = [
            r'\b(function|class|method|const|var|let)\s+\w+',
            r'\w+\(\)',  # function calls
            r'\w+\.\w+',  # dot notation
        ]
        for pattern in code_patterns:
            if re.search(pattern, query_lower):
                return True

        return False

    @classmethod
    def get_weight_explanation(cls, query_type: str) -> str:
        """
        Get detailed explanation for a given query type.

        Args:
            query_type: The type of query (from analyze() result)

        Returns:
            Detailed explanation string
        """
        explanations = {
            'exact_identifier': (
                'This query appears to be an exact identifier (class/function name). '
                'BM25 keyword matching is highly effective for precise identifiers, '
                'so we assign 90% weight to BM25 and 10% to semantic search.'
            ),
            'conceptual_question': (
                'This is a natural language question seeking conceptual understanding. '
                'Semantic search excels at understanding intent and meaning, '
                'so we assign 80% weight to semantic search and 20% to BM25.'
            ),
            'conceptual_search': (
                'This query involves conceptual or architectural patterns. '
                'Semantic understanding is important but keywords still matter, '
                'so we assign 70% weight to semantic search and 30% to BM25.'
            ),
            'keyword_search': (
                'This query contains technical keywords and specific terms. '
                'Keyword matching is prioritized with 70% weight to BM25 '
                'and 30% to semantic search for context.'
            ),
            'short_precise': (
                'This is a short, precise query (1-2 words). '
                'Users typically expect exact matches, '
                'so we assign 80% weight to BM25 and 20% to semantic search.'
            ),
            'descriptive_search': (
                'This is a long, descriptive query (6+ words). '
                'Semantic understanding is crucial for matching intent, '
                'so we assign 70% weight to semantic search and 30% to BM25.'
            ),
            'balanced': (
                'This query doesn\'t clearly favor keyword or semantic matching. '
                'We use a balanced 50/50 weight distribution to leverage both approaches equally.'
            ),
        }

        return explanations.get(query_type, 'Unknown query type.')
