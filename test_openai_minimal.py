#!/usr/bin/env python3
"""Minimal test script to verify OpenAI API connection."""

import sys
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_openai_direct():
    """Test OpenAI API directly without MCP dependencies."""
    logger.info("Testing OpenAI API connection directly...")

    try:
        import openai
        import numpy as np

        logger.info("✅ OpenAI library imported successfully")

        # Load API key from global config
        global_config_path = Path.home() / ".config" / "code-index-mcp" / ".env"

        if not global_config_path.exists():
            logger.error(f"❌ Global config not found: {global_config_path}")
            return False

        # Parse .env file manually
        api_key = None
        with open(global_config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('OPENAI_API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
                    break

        if not api_key:
            logger.error("❌ OPENAI_API_KEY not found in config")
            return False

        logger.info(f"✅ API key loaded (starts with: {api_key[:20]}...)")

        # Create OpenAI client
        client = openai.OpenAI(api_key=api_key)
        logger.info("✅ OpenAI client created")

        # Test simple embedding
        logger.info("📊 Testing simple text embedding...")
        test_text = "def hello_world(): return 'Hello, World!'"

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=test_text
        )

        embedding = np.array(response.data[0].embedding, dtype=np.float32)

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

        batch_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=test_batch
        )

        batch_embeddings = [
            np.array(item.embedding, dtype=np.float32)
            for item in batch_response.data
        ]

        logger.info(f"✅ Batch embeddings generated successfully")
        logger.info(f"📊 Number of embeddings: {len(batch_embeddings)}")
        logger.info(f"📊 Each embedding shape: {batch_embeddings[0].shape}")

        # Test with large batch to verify batch size support
        logger.info("\n📊 Testing large batch (100 texts)...")
        large_batch = [f"# Code snippet {i}" for i in range(100)]

        large_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=large_batch
        )

        logger.info(f"✅ Large batch processed successfully")
        logger.info(f"📊 Number of embeddings: {len(large_response.data)}")

        logger.info("\n🎉 OpenAI API connection test PASSED!")
        return True

    except Exception as e:
        logger.error(f"❌ OpenAI API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_class():
    """Test the OpenAI provider class directly."""
    logger.info("\nTesting OpenAI provider class...")

    try:
        from code_index_mcp.embeddings.openai_provider import OpenAIEmbeddingProvider

        # Load config
        global_config_path = Path.home() / ".config" / "code-index-mcp" / ".env"

        if not global_config_path.exists():
            logger.error(f"❌ Global config not found: {global_config_path}")
            return False

        # Parse .env file manually
        config = {}
        with open(global_config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()

        # Create provider config
        provider_config = {
            'api_key': config.get('OPENAI_API_KEY', ''),
            'model': config.get('OPENAI_MODEL', 'text-embedding-3-small'),
            'dimension': int(config.get('OPENAI_DIMENSION', '1536')),
            'batch_size': int(config.get('OPENAI_BATCH_SIZE', '2048')),
        }

        logger.info(f"📊 Creating provider with model: {provider_config['model']}")

        # Create provider
        provider = OpenAIEmbeddingProvider(provider_config)

        logger.info(f"✅ Provider created: {provider.__class__.__name__}")
        logger.info(f"📊 Model: {provider.get_model_name()}")
        logger.info(f"📊 Dimension: {provider.get_dimension()}")

        # Check availability
        if not provider.is_available():
            logger.error("❌ Provider reports it's not available")
            return False

        logger.info("✅ Provider is available")

        # Test embedding
        test_text = "function calculateSum(a, b) { return a + b; }"
        embedding = provider.embed_text(test_text)

        logger.info(f"✅ Provider embedding successful")
        logger.info(f"📊 Embedding shape: {embedding.shape}")

        # Test batch
        batch = ["code 1", "code 2", "code 3"]
        batch_embeddings = provider.embed_batch(batch)

        logger.info(f"✅ Provider batch embedding successful")
        logger.info(f"📊 Batch size: {len(batch_embeddings)}")

        logger.info("\n🎉 Provider class test PASSED!")
        return True

    except Exception as e:
        logger.error(f"❌ Provider class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("OpenAI Integration Test (Minimal)")
    logger.info("=" * 70)

    tests = [
        ("OpenAI Direct API", test_openai_direct),
        ("OpenAI Provider Class", test_provider_class),
    ]

    results = []
    for name, test_func in tests:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Test: {name}")
        logger.info(f"{'=' * 70}")
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
        logger.info("\n🎉 All tests passed! OpenAI integration is working perfectly.")
        return 0
    else:
        logger.error("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
