from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_compose_requires_an_explicit_published_application_image():
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "MYNEBULA_IMAGE:?" in compose
    assert "simonsun3/mynebula:1.2.11" not in compose
