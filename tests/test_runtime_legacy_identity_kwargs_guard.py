import ast
from pathlib import Path


LEGACY_IDENTITY_KWARGS = {"config_plan", "hash_inputs"}


def _runtime_python_files(repo_root: Path):
    yield repo_root / "run.py"
    yield from sorted((repo_root / "pilates").rglob("*.py"))


def _find_legacy_kwarg_calls(py_file: Path):
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg in LEGACY_IDENTITY_KWARGS:
                yield keyword.arg, node.lineno


def test_runtime_code_does_not_use_legacy_identity_kwargs():
    repo_root = Path(__file__).resolve().parents[1]
    offenders = []

    for py_file in _runtime_python_files(repo_root):
        for kwarg, line in _find_legacy_kwarg_calls(py_file):
            rel_path = py_file.relative_to(repo_root)
            offenders.append(f"{rel_path}:{line} uses {kwarg}=")

    assert not offenders, (
        "Legacy identity kwargs are prohibited in active runtime code.\n"
        "Use adapter= and identity_inputs= metadata instead.\n"
        + "\n".join(offenders)
    )
