# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

[Unreleased]: https://github.com/yourusername/python-template/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/python-template/releases/tag/v0.1.0
