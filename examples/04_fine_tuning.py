"""Example 4: Fine-Tuning"""

from unified_llm.providers import TrainingPipeline

def example_fine_tuning():
    """Fine-tune a model with custom data."""
    training_data = [
        {"prompt": "Q: What is AI?", "response": "A: AI is artificial intelligence"},
        {"prompt": "Q: What is ML?", "response": "A: ML is machine learning"},
        {"prompt": "Q: What is DL?", "response": "A: DL is deep learning"}
    ]
    
    pipeline = TrainingPipeline(
        model="helix-standard",
        num_epochs=3,
        batch_size=16,
        learning_rate=0.001
    )
    
    print("Training model...")
    metrics = pipeline.train(training_data)
    
    print("Training Results:")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")

if __name__ == "__main__":
    print("=== Fine-Tuning Example ===\n")
    try:
        example_fine_tuning()
    except Exception as e:
        print(f"Example failed: {e}")
