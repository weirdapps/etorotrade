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
  - Added OPTIMIZATION_PLAN.md with roadmap for improvements

### Changed
- Refactored logging in trade.py to use new standardized configuration
- Updated yahoofinance.__init__.py to use new logging configuration
- Updated README.md with new setup instructions and environment variables
- Enhanced utils/__init__.py to include new utility modules

### Improved
- Added comprehensive tests for error handling utilities
- Added comprehensive tests for import utilities 
- Created example of improved provider implementation with enhanced error handling

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