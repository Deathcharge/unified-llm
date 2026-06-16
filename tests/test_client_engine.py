"""Test suite for client and engine modules."""

import pytest


class TestClientFunctionality:
    """Test LLM client functionality."""
    
    @pytest.mark.client
    def test_client_generate(self, mock_llm_client):
        """Test client generate method."""
        result = mock_llm_client.generate.return_value
        assert result == {"content": "Response"}
    
    @pytest.mark.client
    def test_client_streaming(self, mock_llm_client):
        """Test client streaming."""
        result = mock_llm_client.stream.return_value
        assert result == ["chunk1", "chunk2"]
    
    @pytest.mark.client
    def test_llm_request_format(self, mock_llm_request):
        """Test LLM request formatting."""
        assert mock_llm_request["prompt"] == "What is machine learning?"
        assert mock_llm_request["max_tokens"] == 1000
        assert mock_llm_request["temperature"] == 0.7


class TestEngineFunctionality:
    """Test LLM engine functionality."""
    
    @pytest.mark.engine
    def test_engine_generate(self, mock_llm_engine):
        """Test engine generate method."""
        result = mock_llm_engine.generate()
        assert result == "Generated response"
    
    @pytest.mark.engine
    def test_engine_streaming(self, mock_llm_engine):
        """Test engine streaming generation."""
        result = mock_llm_engine.stream_generate()
        assert result == ["chunk1", "chunk2"]
    
    @pytest.mark.engine
    def test_llm_response_format(self, mock_llm_response):
        """Test LLM response formatting."""
        assert mock_llm_response["id"] == "response-1"
        assert "Machine learning" in mock_llm_response["content"]
        assert mock_llm_response["tokens_used"] == 150


class TestClientEngineIntegration:
    """Test client and engine integration."""
    
    @pytest.mark.integration
    def test_request_response_cycle(self, mock_llm_request, mock_llm_response):
        """Test request-response cycle."""
        assert mock_llm_request["model"] == mock_llm_response["model"]
    
    @pytest.mark.integration
    def test_streaming_integration(self, streaming_scenario):
        """Test streaming integration."""
        assert streaming_scenario["stream"] is True
        assert streaming_scenario["chunk_size"] == 50


class TestTokenizer:
    """Test tokenizer functionality."""
    
    @pytest.mark.client
    def test_tokenizer_encode(self, mock_tokenizer):
        """Test tokenizer encoding."""
        tokens = mock_tokenizer.encode()
        assert tokens == [1, 2, 3, 4, 5]
    
    @pytest.mark.client
    def test_tokenizer_decode(self, mock_tokenizer):
        """Test tokenizer decoding."""
        text = mock_tokenizer.decode()
        assert text == "Hello world"
    
    @pytest.mark.client
    def test_tokenizer_vocab_size(self, mock_tokenizer):
        """Test tokenizer vocab size."""
        assert mock_tokenizer.vocab_size == 50257
