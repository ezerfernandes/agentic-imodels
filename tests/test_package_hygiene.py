"""Keep research-harness dependencies and dead classes out of the package."""

import ast
from pathlib import Path

PACKAGE_DIR = Path(__file__).parents[1] / "agentic_imodels"
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "argparse",
    "csv",
    "imodelsx",
    "openai",
    "os",
    "pandas",
    "subprocess",
    "sys",
    "time",
}


def test_package_modules_have_no_research_harness_imports_or_unused_classes():
    violations = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".")[0] for alias in node.names}
                violations.extend(
                    f"{path.name}: import {name}" for name in imported & FORBIDDEN_TOP_LEVEL_IMPORTS
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in FORBIDDEN_TOP_LEVEL_IMPORTS:
                    violations.append(f"{path.name}: from {node.module} import ...")
            elif isinstance(node, ast.ClassDef) and node.name.startswith("_Unused"):
                violations.append(f"{path.name}: class {node.name}")
    assert not violations, "package hygiene violations:\n" + "\n".join(sorted(violations))
