# CI/CD Setup for etorotrade

This document describes the continuous integration and deployment setup for the etorotrade project.

## Overview

The etorotrade project uses a lightweight CI/CD approach suitable for locally-run projects while maintaining professional development standards.

## GitHub Actions

### Python Tests Workflow

The project includes automated testing via GitHub Actions:

- **Location**: `.github/workflows/python-tests.yml`
- **Triggers**: 
  - Push to master branch
  - Pull requests to master branch
- **Test Coverage**:
  - Unit tests
  - Memory leak tests
  - Priority limiter tests

## Pre-commit Hooks

Pre-commit hooks ensure code quality before commits:

- **Configuration**: `.pre-commit-config.yaml`
- **Checks**:
  - Trailing whitespace removal
  - Debug statement detection
  - Code formatting with isort and black
  - Static analysis with flake8 and mypy
  - Unit test execution

### Installation

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# Run against all files
pre-commit run --all-files
```

## Code Quality Tools

### Linting and Formatting

The project uses several tools for code quality:

- **Black**: Code formatter (config in `pyproject.toml`)
- **isort**: Import sorter (config in `pyproject.toml`)
- **flake8**: Linter (config in `.flake8`)
- **mypy**: Type checker (config in `pyproject.toml`)

Run all checks:
```bash
make lint
```

Auto-fix issues:
```bash
make lint-fix
```

### Test Runner Script

Use `run_tests.sh` for comprehensive testing:

```bash
# Run all tests
./run_tests.sh --all

# Run specific test types
./run_tests.sh --unit
./run_tests.sh --memory
./run_tests.sh --performance
```

## Version Management

### Tagging Releases

Use `tag_version.sh` for standardized version tagging:

```bash
# Tag a new version
./tag_version.sh 1.0.0 "Initial stable release"

# Tag with custom prefix
./tag_version.sh 2.0.0 "Major update" --prefix release/
```

### Version Naming Convention

Follow semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## Docker Support

### Building and Testing

```bash
# Build the Docker image
docker build -t etorotrade .

# Run tests in Docker
docker-compose up tests

# Run the application
docker-compose up etorotrade
```

### Docker Configuration Files

- **Dockerfile**: Defines the application image
- **docker-compose.yml**: Defines services and their relationships
- **.dockerignore**: Excludes unnecessary files from the build

## Development Workflow

### Feature Development

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make changes and ensure tests pass:
   ```bash
   make test
   make lint
   ```

3. Commit with pre-commit hooks:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

4. Push and create pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Convention

Follow conventional commits format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc)
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Maintenance tasks

## Continuous Monitoring

### Health Checks

The application includes health endpoints for monitoring:

```bash
# Run with health monitoring
python scripts/run_enhanced_monitoring.py --health-port 8081
```

### Performance Monitoring

Track performance metrics:
- Response times
- Error rates
- Resource usage
- API call patterns

## Best Practices

1. **Always run tests before committing**
2. **Use pre-commit hooks to catch issues early**
3. **Follow the code style guidelines**
4. **Write meaningful commit messages**
5. **Keep Docker images up to date**
6. **Monitor application health in production**
7. **Tag releases consistently**

## Troubleshooting

### Common Issues

1. **Pre-commit hook failures**:
   ```bash
   # Skip hooks temporarily (not recommended)
   git commit --no-verify
   
   # Fix issues and retry
   pre-commit run --all-files
   ```

2. **Docker build failures**:
   ```bash
   # Clean Docker cache
   docker system prune -a
   
   # Rebuild without cache
   docker-compose build --no-cache
   ```

3. **Test failures**:
   ```bash
   # Run specific test with verbose output
   pytest -vv tests/path/to/test.py::test_function
   
   # Run with debugging
   pytest --pdb tests/path/to/test.py
   ```

## Future Enhancements

- [ ] Add code coverage reporting to CI
- [ ] Implement automated security scanning
- [ ] Add performance benchmarking to CI
- [ ] Set up automated dependency updates
- [ ] Implement blue-green deployments for production