"""Test suite for training functionality."""

import pytest


class TestTrainingData:
    """Test training data handling."""
    
    @pytest.mark.engine
    def test_training_data_format(self, mock_training_data):
        """Test training data format."""
        assert len(mock_training_data["examples"]) == 3
        assert mock_training_data["num_epochs"] == 3
        assert mock_training_data["batch_size"] == 32
    
    @pytest.mark.engine
    def test_training_examples(self, mock_training_data):
        """Test training examples."""
        examples = mock_training_data["examples"]
        assert "prompt" in examples[0]
        assert "response" in examples[0]


class TestTrainingMetrics:
    """Test training metrics."""
    
    @pytest.mark.performance
    def test_training_metrics_format(self, mock_training_metrics):
        """Test training metrics format."""
        assert "loss" in mock_training_metrics
        assert "accuracy" in mock_training_metrics
        assert "perplexity" in mock_training_metrics
    
    @pytest.mark.performance
    def test_training_metrics_values(self, mock_training_metrics):
        """Test training metrics values."""
        assert 0 <= mock_training_metrics["accuracy"] <= 1
        assert mock_training_metrics["loss"] > 0
        assert mock_training_metrics["epoch"] == 1
