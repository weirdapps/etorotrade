# CI/CD for etorotrade

This document outlines the Continuous Integration (CI) and Continuous Deployment (CD) setup for the etorotrade project.

## Overview

For a locally-run project with GitHub repository, we've set up a lightweight CI/CD approach that:

1. Automates testing and validation
2. Ensures code quality
3. Standardizes versioning
4. Facilitates easy testing and debugging

## Components

### 1. GitHub Actions

The `.github/workflows/python-tests.yml` file defines a GitHub Actions workflow that automatically runs when code is pushed to the master branch or when pull requests are created. The workflow:

- Sets up a Python environment
- Installs dependencies
- Runs unit tests with pytest
- Runs memory leak detection tests
- Tests the priority rate limiter functionality

This ensures that code changes don't break existing functionality.

### 2. Pre-commit Hooks

The `.pre-commit-config.yaml` file configures pre-commit hooks that run automatically before each commit. These hooks:

- Check for common issues (trailing whitespace, debug statements)
- Format code (isort, black)
- Run static analysis (flake8, mypy)
- Run unit tests

To install and use pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install

# Now hooks will run automatically on git commit
```

### 3. Test Runner Script

The `run_tests.sh` script provides an easy way to run different types of tests:

```bash
# Run all tests
./run_tests.sh --all

# Run only unit tests
./run_tests.sh --unit

# Run memory leak tests
./run_tests.sh --memory
```

Available options:
- `--all`: Run all tests and benchmarks
- `--unit`: Run only unit tests
- `--integration`: Run integration tests
- `--performance`: Run performance benchmarks
- `--memory`: Run memory leak tests
- `--priority`: Run priority limiter tests

### 4. Version Tagging

The `tag_version.sh` script standardizes the way versions are tagged:

```bash
# Create a new version tag
./tag_version.sh 1.0.0 "Initial stable release"
```

This script:
1. Validates that all tests pass
2. Creates a git tag with the specified version
3. Optionally pushes the tag to GitHub
4. Updates the VERSION file

## Best Practices

1. **Regular Testing**: Run `./run_tests.sh --all` frequently during development
2. **Version Tagging**: Tag stable versions using `./tag_version.sh` when significant changes are complete
3. **Pre-commit**: Let pre-commit hooks catch issues before committing
4. **Pull Requests**: Use GitHub pull requests for significant changes
5. **Issue Templates**: Use the provided issue templates when reporting bugs or requesting features

## Future Enhancements

Possible enhancements to this CI/CD setup:

1. Automated deployment to a test environment
2. More extensive integration testing
3. Performance regression tracking
4. Code coverage requirements
5. Security scanning

## Troubleshooting

If you encounter issues with the CI/CD setup:

1. **Pre-commit Hooks Failing**: Fix the issues reported or temporarily bypass with `git commit --no-verify` (not recommended)
2. **GitHub Actions Failing**: Check the GitHub Actions tab for detailed error messages
3. **Test Script Issues**: Make sure `run_tests.sh` has execution permissions (`chmod +x run_tests.sh`)