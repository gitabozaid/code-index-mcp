"""Basic integration tests for Ollama and Qdrant."""

import pytest
import numpy as np
from src.code_index_mcp.embeddings import OllamaEmbeddingProvider, EmbeddingProviderFactory
from src.code_index_mcp.vector_store import QdrantStore
from src.code_index_mcp.chunking import CodeChunker


class TestOllamaProvider:
    """Test Ollama embedding provider."""

    def test_provider_creation(self):
        """Test creating Ollama provider."""
        config = {
            'model': 'nomic-embed-code',
            'url': 'http://localhost:11434',
            'dimension': 768
        }
        provider = OllamaEmbeddingProvider(config)
        assert provider.get_model_name() == 'nomic-embed-code'
        assert provider.get_dimension() == 768

    @pytest.mark.skipif(True, reason="Requires Ollama running")
    def test_embedding_generation(self):
        """Test embedding generation (requires Ollama)."""
        config = {'model': 'nomic-embed-code'}
        provider = OllamaEmbeddingProvider(config)

        if not provider.is_available():
            pytest.skip("Ollama not available")

        embedding = provider.embed_text("def hello_world():")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)


class TestCodeChunker:
    """Test code chunking functionality."""

    def test_language_detection(self):
        """Test language detection from file extension."""
        chunker = CodeChunker()
        assert chunker.detect_language("file.py") == "python"
        assert chunker.detect_language("file.js") == "javascript"
        assert chunker.detect_language("file.ts") == "typescript"
        assert chunker.detect_language("file.go") == "go"

    def test_python_chunking(self):
        """Test chunking Python code."""
        code = '''
def function_one():
    return "hello"

def function_two():
    return "world"

class MyClass:
    def method(self):
        pass
'''
        chunker = CodeChunker()
        chunks = chunker.chunk_file("test.py", code, language="python")

        # Should extract 2 functions + 1 class (or fallback to full file)
        assert len(chunks) >= 1
        assert all('file_path' in chunk for chunk in chunks)
        assert all('content' in chunk for chunk in chunks)

    def test_text_chunking(self):
        """Test simple text chunking."""
        chunker = CodeChunker()
        text = "a" * 1000
        chunks = chunker.chunk_text(text, max_size=400)
        assert len(chunks) > 1
        assert all(len(chunk) <= 400 for chunk in chunks)


class TestQdrantStore:
    """Test Qdrant vector store."""

    @pytest.mark.skipif(True, reason="Requires Qdrant running")
    def test_qdrant_connection(self):
        """Test Qdrant connection (requires Qdrant)."""
        store = QdrantStore(
            url="http://localhost:6333",
            collection_name="test-collection",
            vector_dim=768
        )
        assert store.is_available()

    def test_qdrant_config(self):
        """Test Qdrant store configuration."""
        store = QdrantStore(
            url="http://localhost:6333",
            collection_name="test-collection",
            vector_dim=768,
            distance_metric="cosine"
        )
        assert store.collection_name == "test-collection"
        assert store.vector_dim == 768


class TestHybridSearch:
    """Test hybrid search functionality."""

    def test_imports(self):
        """Test that all modules can be imported."""
        from src.code_index_mcp.search import SemanticSearchStrategy, HybridSearchStrategy
        assert SemanticSearchStrategy is not None
        assert HybridSearchStrategy is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
