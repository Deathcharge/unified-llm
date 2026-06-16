# Contributing to Unified-LLM

We welcome contributions! This guide explains how to get started.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/unified-llm.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make changes and commit: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature/your-feature`
6. Submit a pull request

## Development Setup

```bash
git clone https://github.com/Deathcharge/unified-llm.git
cd unified-llm
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov
pytest tests/ -m provider  # Run specific marker
```

## Coding Standards

- Follow PEP 8
- Use type hints
- Write docstrings
- Keep lines under 100 characters
- Use meaningful variable names
- Add tests for new features

## Documentation

Update documentation for new features:

- Update API_REFERENCE.md for new APIs
- Update GETTING_STARTED.md for new patterns
- Add examples for new features
- Update README.md if needed

## Pull Request Process

1. Update tests and documentation
2. Ensure all tests pass: `pytest tests/`
3. Add a clear description of changes
4. Reference any related issues
5. Wait for review and feedback

## Code of Conduct

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions?

Open an issue or contact the maintainers.
