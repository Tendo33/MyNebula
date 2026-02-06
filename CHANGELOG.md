# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-02-07

### Fixed
- Fixed Docker build issues (added missing build tools for hdbscan)
- Correctly handle unstarred repositories in sync process

## [0.2.0] - 2026-02-06

### Added
- 3D-to-2D graph visualization transition
- Enhanced clustering with semantic grouping
- Multi-dimensional filtering (language, star count, time range)
- Real-time sync progress tracking
- User-defined Star Lists support

### Changed
- Improved graph node layout and performance
- Refactored backend sync architecture for better reliability

### Removed
- Deprecated 3D graph visualization code
- Unused GitHub API methods (`get_readme`, `get_rate_limit`)
- Debug console logs in production pages

### Added (from previous)
- Async decorators support (`async_timing_decorator`, `async_retry_decorator`, `async_catch_exceptions`)

## [0.1.0] - 2026-01-20

### Added
- Initial release of Python Template
- **Utils Module**
  - `logger_util`: Loguru-based logging configuration and management
  - `json_utils`: JSON read/write and serialization utilities
  - `file_utils`: File system operations (sync and async)
  - `decorator_utils`: Common decorators (timing, retry, catch_exceptions, etc.)
  - `date_utils`: Date and time manipulation utilities
  - `common_utils`: General utility functions (list chunking, dict operations, etc.)
  - `setting`: Pydantic Settings-based configuration management
  - `context`: Thread-safe runtime context storage
- **Models Module**
  - Base Pydantic models for data validation
- **Scripts**
  - `rename_package.py`: Package renaming utility
  - `setup_pre_commit.py`: Git hooks configuration
  - `update_version.py`: Version update utility
  - `run_vulture.py`: Dead code detection
- **Documentation**
  - Settings guide
  - Models guide
  - SDK usage guide
  - Pre-commit guide
- **Configuration**
  - `pyproject.toml` with full project metadata
  - Ruff linting and formatting configuration
  - Pytest and coverage configuration
  - Pre-commit hooks configuration

[Unreleased]: https://github.com/yourusername/nebula/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yourusername/nebula/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourusername/nebula/releases/tag/v0.1.0
