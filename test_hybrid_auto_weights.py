#!/usr/bin/env python3
"""Test hybrid search with auto-weight selection (full integration)."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_hybrid_search_auto_weights():
    """Test hybrid search with auto-weight selection."""
    from code_index_mcp.project_settings import ProjectSettings
    from code_index_mcp.services.search_service import SearchService

    print("=" * 80)
    print("Testing Hybrid Search with Auto-Weights (Full Integration)")
    print("=" * 80)

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

    # Test queries with different characteristics
    test_cases = [
        # 1. Exact class name - should favor BM25 (90%)
        {
            'query': 'UserWordProgress',
            'description': 'Exact class name - should use 90% BM25',
            'expected_bm25': 0.9,
        },
        # 2. Natural question - should favor Semantic (80%)
        {
            'query': 'how does spaced repetition algorithm work',
            'description': 'Natural question - should use 80% Semantic',
            'expected_bm25': 0.2,
        },
        # 3. Short precise query - should favor BM25 (80%)
        {
            'query': 'JWT authentication',
            'description': 'Short precise - should use 80% BM25',
            'expected_bm25': 0.8,
        },
        # 4. Conceptual pattern - should favor Semantic (70%)
        {
            'query': 'error handling for database',
            'description': 'Conceptual pattern - should use 70% Semantic',
            'expected_bm25': 0.3,
        },
        # 5. Balanced query
        {
            'query': 'user authentication',
            'description': 'Balanced query - should use 50/50',
            'expected_bm25': 0.5,
        },
    ]

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        query = test_case['query']
        description = test_case['description']
        expected_bm25 = test_case['expected_bm25']
        expected_semantic = 1.0 - expected_bm25

        print(f"\n{'=' * 80}")
        print(f"Test {i}: {query}")
        print(f"Description: {description}")
        print(f"{'=' * 80}")

        try:
            # Search with auto-weights enabled
            result = service.search_hybrid_code(
                pattern=query,
                auto_weights=True,
                max_results=5
            )

            # Verify weights_info exists
            if 'weights_info' not in result:
                print("❌ FAILED: weights_info not found in response")
                failed += 1
                continue

            weights = result['weights_info']

            # Display weights info
            print(f"\n📊 Auto-Weight Selection:")
            print(f"   Query Type: {weights['query_type']}")
            print(f"   BM25 Weight: {weights['bm25_weight']:.1%}")
            print(f"   Semantic Weight: {weights['semantic_weight']:.1%}")
            print(f"   Auto-Selected: {weights['auto_selected']}")
            print(f"   Reasoning: {weights['reasoning']}")

            # Verify weights match expectations
            bm25_match = abs(weights['bm25_weight'] - expected_bm25) < 0.0001
            semantic_match = abs(weights['semantic_weight'] - expected_semantic) < 0.0001

            if bm25_match and semantic_match:
                print(f"\n✅ Weights Match Expected!")
            else:
                print(f"\n⚠️  Weights Mismatch:")
                print(f"   Expected: BM25 {expected_bm25:.1%} / Semantic {expected_semantic:.1%}")
                print(f"   Got: BM25 {weights['bm25_weight']:.1%} / Semantic {weights['semantic_weight']:.1%}")

            # Display search results
            print(f"\n📊 Search Results: {result['pagination']['total_matches']} matches found")
            print(f"   Returned: {result['pagination']['returned']} results")

            if result['results']:
                print(f"\n📄 Top Results:")
                for j, match in enumerate(result['results'][:3], 1):
                    print(f"   {j}. {match['file']}:{match['line']}")
                    text = match['text'][:80].replace('\n', ' ')
                    print(f"      {text}...")

            # Mark as passed if weights match
            if bm25_match and semantic_match:
                print(f"\n✅ Test {i} PASSED")
                passed += 1
            else:
                print(f"\n❌ Test {i} FAILED (weight mismatch)")
                failed += 1

        except Exception as e:
            print(f"\n❌ Test {i} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print(f"\n{'=' * 80}")
    print(f"Test Summary")
    print(f"{'=' * 80}")
    print(f"✅ Passed: {passed}/{len(test_cases)}")
    print(f"❌ Failed: {failed}/{len(test_cases)}")
    print(f"{'=' * 80}")

    return failed == 0


def test_manual_vs_auto_weights():
    """Compare manual weights vs auto-weights for same query."""
    from code_index_mcp.project_settings import ProjectSettings
    from code_index_mcp.services.search_service import SearchService

    print(f"\n{'=' * 80}")
    print("Comparison: Manual Weights vs Auto-Weights")
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

    project_path = "/Users/abozaid/Desktop/workspace/abozaid/ENG-LEARN"
    ctx = MockContext(project_path)
    service = SearchService(ctx)

    query = "how does user authentication work"

    # Test 1: Auto-weights
    print(f"\n🔍 Query: '{query}'")
    print(f"\n1️⃣ Auto-Weight Mode:")
    try:
        result_auto = service.search_hybrid_code(
            pattern=query,
            auto_weights=True,
            max_results=3
        )
        weights_auto = result_auto['weights_info']
        print(f"   Weights: BM25 {weights_auto['bm25_weight']:.1%} / Semantic {weights_auto['semantic_weight']:.1%}")
        print(f"   Reasoning: {weights_auto['reasoning']}")
        print(f"   Results: {result_auto['pagination']['total_matches']} matches")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Manual balanced weights
    print(f"\n2️⃣ Manual Balanced (50/50):")
    try:
        result_manual = service.search_hybrid_code(
            pattern=query,
            bm25_weight=0.5,
            semantic_weight=0.5,
            max_results=3
        )
        weights_manual = result_manual['weights_info']
        print(f"   Weights: BM25 {weights_manual['bm25_weight']:.1%} / Semantic {weights_manual['semantic_weight']:.1%}")
        print(f"   Results: {result_manual['pagination']['total_matches']} matches")
    except Exception as e:
        print(f"   Error: {e}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    print("\n🚀 Starting Hybrid Search Auto-Weight Integration Tests...\n")

    # Test 1: Auto-weight selection
    success = test_hybrid_search_auto_weights()

    # Test 2: Comparison manual vs auto
    try:
        test_manual_vs_auto_weights()
    except Exception as e:
        print(f"\n⚠️  Comparison test failed: {e}")

    print(f"\n{'🎉 All tests passed!' if success else '❌ Some tests failed.'}\n")
    sys.exit(0 if success else 1)
