from __future__ import annotations

import ast
import unittest
from pathlib import Path

from deep_research.application.runtime import worker_script_path

PACKAGE_ROOT = Path(__file__).parents[1] / "deep_research"
LAYERS = ("domain", "infrastructure", "agents", "application", "api")
ALLOWED_DEPENDENCIES = {
    "domain": {"domain"},
    "infrastructure": {"domain", "infrastructure"},
    "agents": {"domain", "infrastructure", "agents"},
    "application": {"domain", "infrastructure", "agents", "application"},
    "api": {"domain", "application", "api"},
}
ALLOWED_ROOT_MODULES = {
    "__init__.py",
    "__main__.py",
    "_worker.py",
    "cli.py",
    "web_cli.py",
}


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_package_root_contains_only_entrypoints(self) -> None:
        root_modules = {path.name for path in PACKAGE_ROOT.glob("*.py")}
        self.assertEqual(ALLOWED_ROOT_MODULES, root_modules)

    def test_worker_entrypoint_resolves_after_package_restructure(self) -> None:
        self.assertTrue(worker_script_path().is_file())
        self.assertEqual("_worker.py", worker_script_path().name)

    def test_backend_layers_only_depend_inward(self) -> None:
        violations: list[str] = []
        for layer in LAYERS:
            for path in (PACKAGE_ROOT / layer).glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom) or node.level != 2:
                        continue
                    dependency = (node.module or "").split(".", 1)[0]
                    if dependency in LAYERS and dependency not in ALLOWED_DEPENDENCIES[layer]:
                        violations.append(f"{path.name}: {layer} imports {dependency}")
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
