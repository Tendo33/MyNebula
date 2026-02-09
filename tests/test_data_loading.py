import json

from nebula.utils import read_json


def test_load_data_file(tmp_path):
    data_file_path = tmp_path / "test.json"
    payload = {
        "name": "test_data",
        "value": 123,
    }

    with open(data_file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    with open(data_file_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["name"] == "test_data"

    data_sdk = read_json(data_file_path)
    assert data_sdk["value"] == 123
