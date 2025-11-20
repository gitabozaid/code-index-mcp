#!/usr/bin/env python3
"""Test script to verify auto-weight selection in hybrid search."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_auto_weights():
    """Test automatic weight selection for different query types."""
    from code_index_mcp.utils.query_analyzer import QueryAnalyzer

    print("=" * 80)
    print("Testing Auto-Weight Selection")
    print("=" * 80)

    # Test cases covering different query types
    test_queries = [
        # 1. Exact identifier (should favor BM25 heavily)
        {
            'query': 'UserController',
            'expected_type': 'exact_identifier',
            'expected_bm25': 0.9,
        },
        # 2. Function call
        {
            'query': 'login()',
            'expected_type': 'exact_identifier',
            'expected_bm25': 0.9,
        },
        # 3. Natural language question (should favor Semantic heavily)
        {
            'query': 'how does authentication work',
            'expected_type': 'conceptual_question',
            'expected_bm25': 0.2,
        },
        # 4. Question with question mark
        {
            'query': 'what is the login flow?',
            'expected_type': 'conceptual_question',
            'expected_bm25': 0.2,
        },
        # 5. Conceptual pattern query
        {
            'query': 'error handling patterns',
            'expected_type': 'conceptual_search',
            'expected_bm25': 0.3,
        },
        # 6. Best practice query
        {
            'query': 'best practice for API design',
            'expected_type': 'conceptual_search',
            'expected_bm25': 0.3,
        },
        # 7. Technical keyword query
        {
            'query': 'function login validation',
            'expected_type': 'keyword_search',
            'expected_bm25': 0.7,
        },
        # 8. Short precise query
        {
            'query': 'JWT token',
            'expected_type': 'short_precise',
            'expected_bm25': 0.8,
        },
        # 9. Long descriptive query
        {
            'query': 'implementation of spaced repetition algorithm for learning words',
            'expected_type': 'descriptive_search',
            'expected_bm25': 0.3,
        },
        # 10. Balanced query
        {
            'query': 'user authentication system',
            'expected_type': 'balanced',
            'expected_bm25': 0.5,
        },
    ]

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_queries, 1):
        query = test_case['query']
        expected_type = test_case['expected_type']
        expected_bm25 = test_case['expected_bm25']
        expected_semantic = 1.0 - expected_bm25

        print(f"\n{'=' * 80}")
        print(f"Test {i}: {query}")
        print(f"{'=' * 80}")

        # Analyze query
        result = QueryAnalyzer.analyze(query)

        # Display results
        print(f"Query Type: {result['query_type']}")
        print(f"BM25 Weight: {result['bm25_weight']:.1f} (expected: {expected_bm25:.1f})")
        print(f"Semantic Weight: {result['semantic_weight']:.1f} (expected: {expected_semantic:.1f})")
        print(f"Reasoning: {result['reasoning']}")

        # Check if results match expectations
        if (result['query_type'] == expected_type and
            result['bm25_weight'] == expected_bm25 and
            result['semantic_weight'] == expected_semantic):
            print("✅ PASSED")
            passed += 1
        else:
            print("❌ FAILED")
            print(f"   Expected type: {expected_type}, got: {result['query_type']}")
            print(f"   Expected BM25: {expected_bm25}, got: {result['bm25_weight']}")
            failed += 1

    # Summary
    print(f"\n{'=' * 80}")
    print(f"Test Summary")
    print(f"{'=' * 80}")
    print(f"✅ Passed: {passed}/{len(test_queries)}")
    print(f"❌ Failed: {failed}/{len(test_queries)}")
    print(f"{'=' * 80}")

    return failed == 0


def test_hybrid_search_with_auto_weights():
    """Test hybrid search with auto-weight selection."""
    from code_index_mcp.project_settings import ProjectSettings
    from code_index_mcp.services.search_service import SearchService

    print(f"\n{'=' * 80}")
    print("Testing Hybrid Search with Auto-Weights")
    print(f"{'=' * 80}")

    # Create mock context
    class MockContext:
        def __init__(self, base_path):
            class MockLifespanContext:
                def __init__(self, base_path):
                    self.base_path = base_path
                    self.settings = ProjectSettings(base_path)
                    self.file_watcher_service = None

            class MockRequestContext:
                def __init__(self, base_path):
                    self.lifespan_context = MockLifespanContext(base_path)

            self.request_context = MockRequestContext(base_path)

    # Initialize context with project path
    project_path = "/Users/abozaid/Desktop/workspace/abozaid/ENG-LEARN"
    ctx = MockContext(project_path)

    # Create search service
    service = SearchService(ctx)

    # Test queries with auto-weights
    test_queries = [
        ("UserController", "Exact identifier - should use 90% BM25"),
        ("how does spaced repetition work", "Natural question - should use 80% Semantic"),
        ("JWT authentication", "Short precise - should use 80% BM25"),
    ]

    for query, description in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print(f"   Description: {description}")
        print("-" * 80)

        try:
            # Search with auto-weights enabled
            result = service.search_hybrid_code(
                pattern=query,
                auto_weights=True,
                max_results=3
            )

            # Display weights info
            if 'weights_info' in result:
                weights = result['weights_info']
                print(f"✅ Auto-Weight Selection:")
                print(f"   Query Type: {weights['query_type']}")
                print(f"   BM25 Weight: {weights['bm25_weight']:.1%}")
                print(f"   Semantic Weight: {weights['semantic_weight']:.1%}")
                print(f"   Reasoning: {weights['reasoning']}")
                print(f"\n📊 Results: {result['pagination']['total_matches']} matches found")

                if result['results']:
                    print("\n📄 Top Results:")
                    for i, match in enumerate(result['results'][:3], 1):
                        print(f"   {i}. {match['file']}:{match['line']}")
            else:
                print("❌ weights_info not found in response")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n{'=' * 80}")
    print("✅ Hybrid Search Auto-Weight Tests Complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    print("\n🚀 Starting Auto-Weight Tests...\n")

    # Test 1: Query Analyzer only
    analyzer_passed = test_auto_weights()

    # Test 2: Full hybrid search integration (optional, requires Qdrant)
    try:
        test_hybrid_search_with_auto_weights()
    except Exception as e:
        print(f"\n⚠️  Hybrid search test skipped (requires Qdrant): {e}")

    # Exit code
    sys.exit(0 if analyzer_passed else 1)
