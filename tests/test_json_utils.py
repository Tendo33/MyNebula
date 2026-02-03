"""Tests for json_utils module.

测试 JSON 工具函数模块。
"""

import json
from pathlib import Path

from python_template.utils import (
    merge_json_files,
    read_json,
    safe_json_dumps,
    safe_json_loads,
    validate_json_schema,
    write_json,
)


class TestReadJson:
    """Tests for read_json function."""

    def test_read_json_valid(self, temp_json_file: Path) -> None:
        """Test reading valid JSON file."""
        result = read_json(temp_json_file)
        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 123

    def test_read_json_nested(self, temp_json_file: Path) -> None:
        """Test reading nested JSON data."""
        result = read_json(temp_json_file)
        assert result["nested"]["key"] == "value"

    def test_read_json_nonexistent(self, temp_dir: Path) -> None:
        """Test reading nonexistent file returns default."""
        result = read_json(temp_dir / "nonexistent.json", default={"fallback": True})
        assert result == {"fallback": True}

    def test_read_json_invalid_content(self, temp_dir: Path) -> None:
        """Test reading invalid JSON returns default."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("{ invalid json }", encoding="utf-8")
        result = read_json(invalid_file, default=None)
        assert result is None


class TestWriteJson:
    """Tests for write_json function."""

    def test_write_json_basic(self, temp_dir: Path) -> None:
        """Test writing JSON file."""
        file_path = temp_dir / "output.json"
        data = {"key": "value", "number": 42}
        result = write_json(data, file_path)
        assert result is True
        assert file_path.exists()

        # Verify content
        content = json.loads(file_path.read_text())
        assert content == data

    def test_write_json_unicode(self, temp_dir: Path) -> None:
        """Test writing JSON with Unicode characters."""
        file_path = temp_dir / "unicode.json"
        data = {"中文": "测试", "emoji": "😀"}
        result = write_json(data, file_path, ensure_ascii=False)
        assert result is True

        content = file_path.read_text(encoding="utf-8")
        assert "中文" in content
        assert "测试" in content

    def test_write_json_creates_dirs(self, temp_dir: Path) -> None:
        """Test write creates parent directories."""
        file_path = temp_dir / "new" / "nested" / "data.json"
        result = write_json({"test": True}, file_path, create_dirs=True)
        assert result is True
        assert file_path.exists()

    def test_write_json_with_indent(self, temp_dir: Path) -> None:
        """Test writing JSON with custom indent."""
        file_path = temp_dir / "indented.json"
        data = {"a": 1, "b": 2}
        write_json(data, file_path, indent=4)

        content = file_path.read_text()
        # Check indentation is present
        assert "    " in content


class TestSafeJsonLoads:
    """Tests for safe_json_loads function."""

    def test_safe_json_loads_valid(self) -> None:
        """Test parsing valid JSON string."""
        json_str = '{"name": "test", "value": 123}'
        result = safe_json_loads(json_str)
        assert result == {"name": "test", "value": 123}

    def test_safe_json_loads_array(self) -> None:
        """Test parsing JSON array."""
        json_str = "[1, 2, 3]"
        result = safe_json_loads(json_str)
        assert result == [1, 2, 3]

    def test_safe_json_loads_invalid(self) -> None:
        """Test parsing invalid JSON returns default."""
        result = safe_json_loads("{ invalid }", default={"error": True})
        assert result == {"error": True}

    def test_safe_json_loads_none_input(self) -> None:
        """Test handling None input."""
        result = safe_json_loads(None, default="fallback")  # type: ignore
        assert result == "fallback"


class TestSafeJsonDumps:
    """Tests for safe_json_dumps function."""

    def test_safe_json_dumps_dict(self) -> None:
        """Test serializing dictionary."""
        data = {"name": "test", "value": 123}
        result = safe_json_dumps(data)
        assert result is not None
        assert "test" in result
        assert "123" in result

    def test_safe_json_dumps_list(self) -> None:
        """Test serializing list."""
        data = [1, 2, 3, "test"]
        result = safe_json_dumps(data)
        assert result is not None
        assert "[1, 2, 3," in result

    def test_safe_json_dumps_unicode(self) -> None:
        """Test serializing with Unicode."""
        data = {"中文": "测试"}
        result = safe_json_dumps(data, ensure_ascii=False)
        assert result is not None
        assert "中文" in result

    def test_safe_json_dumps_with_indent(self) -> None:
        """Test serializing with indentation."""
        data = {"a": 1, "b": 2}
        result = safe_json_dumps(data, indent=2)
        assert result is not None
        assert "\n" in result


class TestMergeJsonFiles:
    """Tests for merge_json_files function."""

    def test_merge_json_files_basic(self, temp_dir: Path) -> None:
        """Test merging multiple JSON files."""
        file1 = temp_dir / "file1.json"
        file2 = temp_dir / "file2.json"

        file1.write_text('{"a": 1, "b": 2}', encoding="utf-8")
        file2.write_text('{"c": 3, "d": 4}', encoding="utf-8")

        result = merge_json_files([file1, file2])
        assert result is not None
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_merge_json_files_with_output(self, temp_dir: Path) -> None:
        """Test merging with output file."""
        file1 = temp_dir / "file1.json"
        file2 = temp_dir / "file2.json"
        output = temp_dir / "merged.json"

        file1.write_text('{"a": 1}', encoding="utf-8")
        file2.write_text('{"b": 2}', encoding="utf-8")

        result = merge_json_files([file1, file2], output_path=output)
        assert result is not None
        assert output.exists()

        merged_content = json.loads(output.read_text())
        assert merged_content == {"a": 1, "b": 2}


class TestValidateJsonSchema:
    """Tests for validate_json_schema function."""

    def test_validate_json_schema_valid(self) -> None:
        """Test validation with all required keys present."""
        data = {"name": "test", "value": 123, "type": "example"}
        result = validate_json_schema(data, required_keys=["name", "value"])
        assert result is True

    def test_validate_json_schema_missing_keys(self) -> None:
        """Test validation with missing required keys."""
        data = {"name": "test"}
        result = validate_json_schema(data, required_keys=["name", "value", "type"])
        assert result is False

    def test_validate_json_schema_empty_required(self) -> None:
        """Test validation with no required keys."""
        data = {"anything": "goes"}
        result = validate_json_schema(data, required_keys=[])
        assert result is True
