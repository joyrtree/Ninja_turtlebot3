"""Run pure Python behavioral checks without installing ROS or moving a robot."""
import ast
import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT.parent / ".runtime/site-packages"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT))


def main():
    suite = unittest.TestSuite()
    for file in sorted((ROOT / "test").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(file.stem, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_"):
                suite.addTest(unittest.FunctionTestCase(function))
    for file in ROOT.rglob("*.py"):
        if not any(part in ("build", ".venv", ".git") for part in file.parts):
            ast.parse(file.read_text(encoding="utf-8-sig"), filename=str(file), feature_version=(3, 8))
    print("Python 3.8 grammar check passed (not a ROS/runtime compatibility test)")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
