import json
from pathlib import Path

from python_template.utils import read_json


def test_load_data_file():
    # 1. 获取当前测试文件的目录 (tests/)
    current_dir = Path(__file__).parent
    print(current_dir)

    # 2. 获取项目根目录 (假设 tests/ 在根目录下)
    project_root = current_dir.parent
    print(project_root)

    # 3. 拼接数据文件的路径
    data_file_path = project_root / "data" / "test.json"
    print(data_file_path)
    # 4. 验证文件是否存在
    assert data_file_path.exists(), f"Data file not found at {data_file_path}"

    # 5. 读取文件内容
    # 方式 A: 使用标准库
    with open(data_file_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["name"] == "test_data"

    # 方式 B: 使用我们的 SDK 工具函数
    data_sdk = read_json(data_file_path)
    print(data_sdk)
    assert data_sdk["value"] == 123


if __name__ == "__main__":
    test_load_data_file()
