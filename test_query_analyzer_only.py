#!/usr/bin/env python3
"""Simple test for QueryAnalyzer without MCP dependencies."""

import sys
from pathlib import Path

# Add src to path and import directly to avoid MCP dependencies
sys.path.insert(0, str(Path(__file__).parent / "src" / "code_index_mcp" / "utils"))

# Import QueryAnalyzer module directly
import query_analyzer
QueryAnalyzer = query_analyzer.QueryAnalyzer


def test_query_analyzer():
    """Test automatic weight selection for different query types."""

    print("=" * 80)
    print("Testing Auto-Weight Selection (QueryAnalyzer)")
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
        print(f"BM25 Weight: {result['bm25_weight']:.1%} (expected: {expected_bm25:.1%})")
        print(f"Semantic Weight: {result['semantic_weight']:.1%} (expected: {expected_semantic:.1%})")
        print(f"Reasoning: {result['reasoning']}")

        # Check if results match expectations (with floating point tolerance)
        type_match = result['query_type'] == expected_type
        bm25_match = abs(result['bm25_weight'] - expected_bm25) < 0.0001
        semantic_match = abs(result['semantic_weight'] - expected_semantic) < 0.0001

        if type_match and bm25_match and semantic_match:
            print("✅ PASSED")
            passed += 1
        else:
            print("❌ FAILED")
            if not type_match:
                print(f"   Type mismatch: expected {expected_type}, got {result['query_type']}")
            if not bm25_match:
                print(f"   BM25 mismatch: expected {expected_bm25:.4f}, got {result['bm25_weight']:.4f}")
            if not semantic_match:
                print(f"   Semantic mismatch: expected {expected_semantic:.4f}, got {result['semantic_weight']:.4f}")
            failed += 1

    # Summary
    print(f"\n{'=' * 80}")
    print(f"Test Summary")
    print(f"{'=' * 80}")
    print(f"✅ Passed: {passed}/{len(test_queries)}")
    print(f"❌ Failed: {failed}/{len(test_queries)}")
    print(f"{'=' * 80}")

    # Additional demonstration
    print(f"\n{'=' * 80}")
    print("Additional Examples with Detailed Explanations")
    print(f"{'=' * 80}")

    demo_queries = [
        "SpacedRepetitionService",
        "how to implement JWT authentication",
        "database connection error",
    ]

    for query in demo_queries:
        result = QueryAnalyzer.analyze(query)
        explanation = QueryAnalyzer.get_weight_explanation(result['query_type'])

        print(f"\n📝 Query: '{query}'")
        print(f"📊 Weights: BM25 {result['bm25_weight']:.1%} / Semantic {result['semantic_weight']:.1%}")
        print(f"📌 Type: {result['query_type']}")
        print(f"💡 Reasoning: {result['reasoning']}")
        print(f"\n📚 Detailed Explanation:")
        print(f"   {explanation}")

    return failed == 0


if __name__ == "__main__":
    print("\n🚀 Starting QueryAnalyzer Tests...\n")
    success = test_query_analyzer()
    print(f"\n{'🎉 All tests passed!' if success else '❌ Some tests failed.'}\n")
    sys.exit(0 if success else 1)
