# Unified-LLM Examples

This directory contains practical examples demonstrating how to use Unified-LLM for various tasks.

## Examples

### 1. Basic Generation (01_basic_generation.py)
Simple example showing how to generate text using different providers.

```bash
python examples/01_basic_generation.py
```

### 2. Multi-Provider Fallback (02_multi_provider.py)
Demonstrates intelligent provider fallback and selection.

```bash
python examples/02_multi_provider.py
```

### 3. Streaming Responses (03_streaming.py)
Shows how to stream responses for real-time output.

```bash
python examples/03_streaming.py
```

### 4. Fine-Tuning (04_fine_tuning.py)
Example of training a model with custom data.

```bash
python examples/04_fine_tuning.py
```

### 5. Optimization Techniques (05_optimization.py)
Demonstrates quantization and performance optimization.

```bash
python examples/05_optimization.py
```

## Running Examples

All examples require environment variables to be set:

```bash
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
export HELIX_LLM_PROVIDER=anthropic
```

Then run any example:

```bash
python examples/01_basic_generation.py
```

## Requirements

All examples use the same dependencies as the main package. Install with:

```bash
pip install unified-llm
```

## Contributing Examples

To add a new example:

1. Create a new file: `examples/NN_description.py`
2. Add documentation to this README
3. Ensure the example is self-contained and well-commented
4. Test thoroughly before submitting

## Support

For issues or questions, please open an issue on GitHub.
