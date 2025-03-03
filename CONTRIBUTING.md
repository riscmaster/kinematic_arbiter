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

This project uses pre-commit hooks to enforce code style. The configuration is in `.pre-commit-config.yaml`.

### Python Style Guidelines
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions/classes
- Keep functions focused and small
- Write meaningful commit messages

### ROS 2 Guidelines
- Follow the [ROS 2 Developer Guide](https://docs.ros.org/en/humble/Contributing/Developer-Guide.html)
- Use appropriate message types
- Properly handle lifecycle and cleanup

## Testing

- Add tests for new functionality
- Ensure all tests pass before submitting a PR
- Update documentation as needed

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation with details of any interface changes
3. The PR will be merged once you have the sign-off of at least one maintainer

## Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- ROS 2 version and environment details

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
