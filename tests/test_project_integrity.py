from __future__ import annotations

import json
import py_compile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectIntegrityTests(unittest.TestCase):
    def test_source_files_compile(self) -> None:
        for path in (ROOT / "src").glob("*.py"):
            py_compile.compile(str(path), doraise=True)

    def test_requirements_are_one_dependency_per_line(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        package_lines = [line.strip() for line in requirements if line.strip() and not line.strip().startswith("#")]

        self.assertGreaterEqual(len(package_lines), 5)
        for line in package_lines:
            self.assertNotIn(" ", line, msg=f"Requirement should be one dependency per line: {line!r}")

    def test_existing_metrics_json_files_are_valid(self) -> None:
        metrics_files = list((ROOT / "outputs").glob("*/metrics.json"))
        self.assertGreaterEqual(len(metrics_files), 1)
        for path in metrics_files:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("method", data)
            self.assertIn("rows_real", data)
            self.assertIn("rows_synthetic", data)


if __name__ == "__main__":
    unittest.main()
