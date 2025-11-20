#!/usr/bin/env python3
"""Test script to verify OpenAI embedding setup."""

import sys
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    try:
        from code_index_mcp.embeddings import OpenAIEmbeddingProvider, EmbeddingProviderFactory
        from code_index_mcp.utils.config_helper import SemanticSearchConfig
        from code_index_mcp.vector_store import QdrantStore
        logger.info("✅ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_global_config():
    """Test that global config is being loaded."""
    logger.info("\nTesting global configuration...")
    try:
        from code_index_mcp.utils.config_helper import SemanticSearchConfig, GLOBAL_CONFIG_DIR, GLOBAL_ENV_FILE, _load_global_env

        # Check if global config directory exists
        if GLOBAL_CONFIG_DIR.exists():
            logger.info(f"✅ Global config directory exists: {GLOBAL_CONFIG_DIR}")
        else:
            logger.warning(f"⚠️  Global config directory not found: {GLOBAL_CONFIG_DIR}")

        # Check if .env file exists
        if GLOBAL_ENV_FILE.exists():
            logger.info(f"✅ Global .env file exists: {GLOBAL_ENV_FILE}")
        else:
            logger.warning(f"⚠️  Global .env file not found: {GLOBAL_ENV_FILE}")

        # Load environment from global config
        global_env = _load_global_env()
        logger.info(f"📊 Loaded {len(global_env)} variables from global config")

        # Load full config
        config = SemanticSearchConfig.load_from_env()
        logger.info(f"📊 Embedding provider: {config['embedding_provider']}")
        logger.info(f"📊 Search mode: {config['search_mode']}")
        logger.info(f"📊 Semantic enabled: {config['semantic_enabled']}")

        if config['embedding_provider'] == 'openai':
            logger.info(f"📊 OpenAI model: {config['openai_model']}")
            logger.info(f"📊 OpenAI dimension: {config['openai_dimension']}")
            logger.info(f"📊 OpenAI batch size: {config['openai_batch_size']}")
            has_api_key = bool(config.get('openai_api_key'))
            if has_api_key:
                logger.info(f"✅ OpenAI API key found (length: {len(config['openai_api_key'])})")
            else:
                logger.error("❌ OpenAI API key not found in config")
                return False

        return True
    except Exception as e:
        logger.error(f"❌ Global config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_config():
    """Test embedding provider configuration."""
    logger.info("\nTesting embedding provider configuration...")
    try:
        from code_index_mcp.utils.config_helper import SemanticSearchConfig

        embedding_config = SemanticSearchConfig.get_embedding_config()
        provider = embedding_config.get('embedding_provider')

        logger.info(f"📊 Provider: {provider}")

        if provider == 'openai':
            logger.info(f"📊 Model: {embedding_config.get('model')}")
            logger.info(f"📊 Dimension: {embedding_config.get('dimension')}")
            logger.info(f"📊 Batch size: {embedding_config.get('batch_size')}")

            has_api_key = bool(embedding_config.get('api_key'))
            if has_api_key:
                api_key = embedding_config['api_key']
                logger.info(f"✅ API key loaded (starts with: {api_key[:20]}...)")
            else:
                logger.error("❌ API key not found in embedding config")
                return False

        return True
    except Exception as e:
        logger.error(f"❌ Embedding config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openai_provider():
    """Test OpenAI provider instantiation."""
    logger.info("\nTesting OpenAI provider instantiation...")
    try:
        from code_index_mcp.embeddings import EmbeddingProviderFactory
        from code_index_mcp.utils.config_helper import SemanticSearchConfig

        config = SemanticSearchConfig.get_embedding_config()
        provider_type = config.get('embedding_provider', 'ollama')

        logger.info(f"📊 Creating {provider_type} provider...")
        provider = EmbeddingProviderFactory.create(provider_type, config)

        logger.info(f"✅ Provider created: {provider.__class__.__name__}")
        logger.info(f"📊 Model: {provider.get_model_name()}")
        logger.info(f"📊 Dimension: {provider.get_dimension()}")

        return True
    except Exception as e:
        logger.error(f"❌ Provider instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openai_api_connection():
    """Test OpenAI API connection with a simple embedding."""
    logger.info("\nTesting OpenAI API connection...")
    try:
        from code_index_mcp.embeddings import EmbeddingProviderFactory
        from code_index_mcp.utils.config_helper import SemanticSearchConfig

        config = SemanticSearchConfig.get_embedding_config()

        if config.get('embedding_provider') != 'openai':
            logger.warning("⚠️  Provider is not set to 'openai', skipping API test")
            return True

        provider = EmbeddingProviderFactory.create('openai', config)

        # Check availability
        logger.info("📊 Checking provider availability...")
        if not provider.is_available():
            logger.error("❌ Provider reports it's not available")
            return False

        logger.info("✅ Provider is available")

        # Test simple embedding
        logger.info("📊 Testing simple text embedding...")
        test_text = "def hello_world(): return 'Hello, World!'"

        embedding = provider.embed_text(test_text)

        logger.info(f"✅ Embedding generated successfully")
        logger.info(f"📊 Embedding shape: {embedding.shape}")
        logger.info(f"📊 Embedding dtype: {embedding.dtype}")
        logger.info(f"📊 First 5 values: {embedding[:5]}")

        # Test batch embedding
        logger.info("\n📊 Testing batch embedding (3 texts)...")
        test_batch = [
            "def add(a, b): return a + b",
            "class User: pass",
            "async function fetchData() { return await fetch('/api'); }"
        ]

        batch_embeddings = provider.embed_batch(test_batch)

        logger.info(f"✅ Batch embeddings generated successfully")
        logger.info(f"📊 Number of embeddings: {len(batch_embeddings)}")
        logger.info(f"📊 Each embedding shape: {batch_embeddings[0].shape}")

        return True
    except Exception as e:
        logger.error(f"❌ API connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ignore_patterns():
    """Test that ignore patterns are loaded correctly."""
    logger.info("\nTesting ignore patterns...")
    try:
        from code_index_mcp.utils.config_helper import SemanticSearchConfig

        patterns = SemanticSearchConfig.get_ignore_patterns()
        logger.info(f"✅ Loaded {len(patterns)} ignore patterns")

        # Show first 10 patterns
        logger.info("📊 First 10 patterns:")
        for i, pattern in enumerate(patterns[:10], 1):
            logger.info(f"   {i}. {pattern}")

        return True
    except Exception as e:
        logger.error(f"❌ Ignore patterns test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("OpenAI Embedding Setup Test")
    logger.info("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Global Config", test_global_config),
        ("Embedding Config", test_embedding_config),
        ("OpenAI Provider", test_openai_provider),
        ("OpenAI API Connection", test_openai_api_connection),
        ("Ignore Patterns", test_ignore_patterns),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {name}")

    logger.info("-" * 70)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 All tests passed! OpenAI setup is complete and working.")
        return 0
    else:
        logger.error("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
