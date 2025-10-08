# CI/CD Setup for etorotrade

This document describes the comprehensive continuous integration and deployment setup for the etorotrade project.

## Overview

The etorotrade project implements a **production-grade CI/CD pipeline** with multi-version testing, security scanning, code quality gates, and performance benchmarking to ensure enterprise-level reliability.

## GitHub Actions

### Comprehensive CI/CD Pipeline

The project features a sophisticated GitHub Actions pipeline with multiple jobs and quality gates:

- **Location**: `.github/workflows/ci.yml`
- **Triggers**: 
  - Push to main, master, develop branches
  - Pull requests to main, master branches
- **Multi-Version Testing**: Python 3.9, 3.10, 3.11, 3.12
- **Pipeline Jobs**:
  - **test**: Core testing with security and quality checks
  - **integration**: Integration tests on main branch pushes
  - **quality-gates**: Code complexity and dependency validation

### Pipeline Features

**Security Scanning**:
- **Bandit**: Security vulnerability detection in Python code
- **Safety**: Known security vulnerability checks in dependencies
- **Artifact Upload**: Security reports saved for review

**Code Quality Gates**:
- **Flake8**: Syntax errors, undefined names, complexity analysis
- **MyPy**: Type checking with missing imports handling
- **Test Coverage**: 60% minimum threshold with XML/HTML reports
- **Performance Benchmarks**: Automated performance validation

**Advanced Quality Checks**:
- **Code Complexity**: Radon analysis for maintainability
- **Dependency Validation**: Import verification for all modules
- **TODO/FIXME Detection**: Code comment analysis
- **Maintainability Index**: Code quality scoring

## Code Quality Tools

### Integrated CI/CD Tools

The pipeline includes comprehensive code quality tools with optimized configuration:

- **Flake8**: 
  - Max line length: 100 characters
  - Max complexity: 10
  - Critical errors halt build, warnings continue
- **MyPy**: Type checking with relaxed missing imports for CI
- **Bandit**: Security analysis with JSON report generation
- **Safety**: Vulnerability scanning with JSON output
- **Pytest**: Test execution with coverage and parallel processing

### Local Development Tools

For local development, use the following quality tools:

```bash
# Code quality checks (matches CI pipeline)
flake8 . --max-line-length=100 --max-complexity=10
mypy yahoofinance/ trade_modules/ --ignore-missing-imports

# Security scanning
bandit -r yahoofinance/ trade_modules/
safety check

# Test execution with coverage
pytest tests/ --cov=yahoofinance --cov=trade_modules --cov-fail-under=60
```

### Test Execution

The CI/CD pipeline runs comprehensive test suites with intelligent exclusions:

```bash
# CI test execution (excludes integration/e2e tests)
pytest tests/ \
  --cov=yahoofinance \
  --cov=trade_modules \
  --cov-report=xml \
  --cov-report=html \
  --cov-fail-under=60 \
  --ignore=tests/integration/ \
  --ignore=tests/e2e/

# Integration tests (main branch only)
pytest tests/integration/ -v --maxfail=3

# Local development testing
pytest tests/unit/          # Unit tests only
pytest tests/benchmarks/    # Performance benchmarks
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

## Performance Monitoring

Monitor key metrics during development:
- **Response times**: API call latency tracking
- **Cache hit rates**: Data retrieval efficiency
- **Resource usage**: Memory and CPU profiling
- **API call patterns**: Rate limiting and optimization

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

## CI/CD Pipeline Benefits

### Quality Assurance
- **Multi-Version Compatibility**: Ensures compatibility across Python 3.9-3.12
- **Security First**: Automated vulnerability detection and dependency scanning
- **Code Quality Gates**: Prevents technical debt with complexity and quality thresholds
- **Test Coverage**: 90%+ test coverage target with comprehensive edge case testing

### Performance Monitoring
- **Automated Benchmarks**: Performance validation in CI pipeline
- **Regression Detection**: Alerts on performance degradation
- **Scalability Testing**: Large dataset processing validation
- **Memory Profiling**: Memory leak detection and optimization

### Development Efficiency
- **Fast Feedback**: Parallel job execution for quick results
- **Artifact Management**: Test reports, coverage, and security scans preserved
- **Branch Protection**: Quality gates prevent broken code in main branches
- **Local Development**: CI commands work identically in local environment

## Recent Improvements (January 2025)

- [x] ✅ Code coverage reporting with XML/HTML reports
- [x] ✅ Automated security scanning (Bandit + Safety)
- [x] ✅ Performance benchmarking in CI pipeline
- [x] ✅ Removed debug test files and unused monitoring modules
- [x] ✅ Streamlined test structure

## Future Enhancements

- [ ] Set up automated dependency updates (Dependabot/Renovate)
- [ ] Implement deployment automation for production releases
- [ ] Add container security scanning for Docker builds
- [ ] Integrate SonarQube for advanced code quality metrics