# Contributing to HideSeek

Thank you for your interest in contributing to HideSeek! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs or request features
- Provide clear descriptions and steps to reproduce
- Include system information and error messages

### Development Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit with descriptive messages
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 🧪 Testing Requirements

- All new features must include tests
- Maintain code coverage above 80%
- Test on multiple platforms when possible
- Include integration tests for analysis pipelines

## 📝 Code Standards

### Style Guidelines
- Follow PEP 8 for Python code style
- Use type hints for function parameters and returns
- Include docstrings for all public functions and classes
- Maximum line length: 100 characters

### Scientific Validation
- New analysis algorithms must be scientifically validated
- Include references to academic papers or standards
- Provide test cases with known expected results
- Document mathematical formulations

## 🏗️ Architecture Guidelines

### Adding New Analyzers
```python
class NewAnalyzer:
    def __init__(self, config):
        self.config = config
        
    def analyze(self, image, background=None):
        # Implementation
        return score, metadata
```

### Adding New CLI Commands
- Extend the CLI class with new command methods
- Follow existing patterns for argument parsing
- Include comprehensive help text and examples
- Add error handling and validation

## 📚 Documentation

- Update README.md for new features
- Add docstrings with parameter descriptions
- Include usage examples in docstrings
- Update API documentation

## 🔬 Scientific Contributions

We welcome contributions that:
- Improve analysis accuracy
- Add new camouflage evaluation methods
- Enhance environmental simulation
- Optimize performance
- Extend to new domains (thermal, multispectral, etc.)

## 📧 Contact

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: 99cvteam@gmail.com

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.