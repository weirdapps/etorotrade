# Changelog

All notable changes to the etorotrade project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive code quality tools setup:
  - Added .editorconfig for consistent editor settings
  - Added pyproject.toml with black and isort configuration
  - Added .flake8 configuration
  - Added pre-commit configuration
  - Added dev-requirements.txt for development dependencies
  - Created scripts/run_code_checks.py for running all code quality checks
  - Created scripts/setup_dev_environment.py for setting up the development environment
  - Added scripts/lint.sh as a bash alternative for running linters
  - Created Makefile with standardized commands for development tasks
  - Created .github/PULL_REQUEST_TEMPLATE.md for PR quality checklist
- New standardized logging system:
  - Created logging_config.py with modern logging configuration
  - Added environment variable control for logging levels
  - Implemented context-based logging with ticker information
- Standardized error handling:
  - Added error_handling.py with utilities for consistent error handling
  - Added decorators for error context enrichment, retry logic, and safe operations
  - Created an example provider implementation with enhanced error handling
- Circular dependency resolution:
  - Added imports.py with utilities for resolving circular imports
  - Implemented LazyImport for deferred module loading
  - Added DependencyProvider for dependency injection
  - Created utilities for dynamic imports and delayed imports
- Improved development workflow:
  - Added CONTRIBUTING.md with guidelines for contributors
  - Added setup.py for proper package installation
  - Created CHANGELOG.md to track changes
  - Added optimization plan with roadmap for improvements

### Changed
- Refactored logging in trade.py to use new standardized configuration
- Updated yahoofinance.__init__.py to use new logging configuration
- Updated README.md with new setup instructions and environment variables
- Enhanced utils/__init__.py to include new utility modules
- Updated CONTRIBUTING.md with code quality information and workflow improvements
- Consolidated all technical documentation into CLAUDE.md as a single comprehensive reference
- Simplified README.md to be more user-friendly with less technical detail

### Improved
- Added comprehensive tests for error handling utilities
- Added comprehensive tests for import utilities 
- Created example of improved provider implementation with enhanced error handling
- Reorganized test structure by moving scattered test and debug files into proper test directories
- Created dedicated debug/ directory under tests/ for debug scripts
- Updated tests README.md with improved documentation on test structure

### Fixed
- Removed duplicate imports in trade.py
- Fixed incorrect logging configuration in multiple files

## [2.0.0] - Prior to Optimization Project

### Added
- Enhanced async architecture with true async I/O
- Circuit breaker pattern for improved reliability
- Disk-based caching for better performance
- Provider pattern for data access abstraction
- Hybrid data provider combining YahooFinance with YahooQuery

### Changed
- Updated provider pattern to use hybrid provider by default
- Improved error handling with comprehensive error hierarchy
- Enhanced rate limiting for better API utilization

### Fixed
- Fixed issues with Yahoo Finance API changes
- Addressed rate limiting issues with adaptive delay mechanism