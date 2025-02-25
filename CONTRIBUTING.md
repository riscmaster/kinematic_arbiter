# Contributing to Kinematic Arbiter

We love your input! We want to make contributing to Kinematic Arbiter as easy and transparent as possible.

## Development Process

1. Fork the repo and create your branch from `main`
2. Install development dependencies:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
3. Make your changes
4. Run tests:
   ```bash
   colcon test --packages-select kinematic_arbiter
   ```
5. Create a pull request

## Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to all functions/classes
- Keep functions focused and small
- Write meaningful commit messages

## Testing

- Add tests for new functionality
- Ensure all tests pass
- Update documentation as needed

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation with details of any interface changes
3. The PR will be merged once you have the sign-off of at least one maintainer

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
