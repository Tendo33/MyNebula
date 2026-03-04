from pathlib import Path


def _scan_forbidden_imports(forbidden_tokens: list[str]) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    scan_roots = [repo_root / "src", repo_root / "tests"]
    current_test = Path(__file__).resolve()

    violations: list[str] = []
    for scan_root in scan_roots:
        for path in scan_root.rglob("*.py"):
            if path.resolve() == current_test:
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                if token in text:
                    violations.append(f"{path}: {token}")
    return violations


def test_no_legacy_sync_imports():
    module_path = "nebula.api." + "sync"
    forbidden_tokens = [
        f"from {module_path} import",
        f"import {module_path}",
    ]
    violations = _scan_forbidden_imports(forbidden_tokens)

    assert not violations, "Found legacy sync imports:\n" + "\n".join(violations)


def test_no_legacy_schedule_schema_imports():
    module_path = "nebula.schemas." + "schedule"
    forbidden_tokens = [
        f"from {module_path} import",
        f"import {module_path}",
    ]
    violations = _scan_forbidden_imports(forbidden_tokens)

    assert not violations, "Found legacy schedule schema imports:\n" + "\n".join(
        violations
    )


def test_no_top_level_auth_and_repos_imports():
    forbidden_tokens = [
        "from nebula.api.auth import",
        "import nebula.api.auth",
        "from nebula.api.repos import",
        "import nebula.api.repos",
    ]
    violations = _scan_forbidden_imports(forbidden_tokens)
    assert not violations, "Found top-level auth/repos imports:\n" + "\n".join(
        violations
    )
