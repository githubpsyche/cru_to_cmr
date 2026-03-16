"""Find unused functions/classes in vendored cru_to_cmr package.

Traces reachable symbols from known entry points (the analysis scripts)
through import statements AND function bodies, then reports definitions
in the package that are never referenced.

Usage:
    uv run python scripts/check_unused.py
"""

import ast
import json
import re
import sys
from pathlib import Path

PACKAGE_DIR = Path("cru_to_cmr")
ENTRY_POINTS = [
    Path("analyses/rename_figures.py"),
    Path("analyses/config.py"),
    # Notebooks (code cells extracted automatically)
    Path("analyses/templates/free_recall_fitting.ipynb"),
    Path("analyses/templates/serial_recall_fitting.ipynb"),
    Path("analyses/render_free_recall.ipynb"),
    Path("analyses/render_serial_recall.ipynb"),
    Path("analyses/render_parameter_shifting.ipynb"),
    Path("analyses/templates/parameter_shifting.ipynb"),
    Path("analyses/HealeyKahana2014_cru_to_cmr_Model_Comparison.ipynb"),
    Path("analyses/Logan2021_cru_to_cmr_Model_Comparison.ipynb"),
]


def notebook_to_python(nb_path: Path) -> str:
    """Extract all code cells from a notebook into a single Python string."""
    with open(nb_path) as f:
        nb = json.load(f)
    lines = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            for line in cell.get("source", []):
                # Strip Jupyter magics and shell commands
                stripped = line.lstrip()
                if stripped.startswith(("%", "!")):
                    lines.append("# " + line)
                else:
                    lines.append(line)
            lines.append("\n")
    return "".join(lines)


def module_to_filepath(module: str) -> Path | None:
    """Convert dotted module path to file path."""
    parts = module.split(".")
    path = Path(*parts).with_suffix(".py")
    if path.exists():
        return path
    path = Path(*parts) / "__init__.py"
    if path.exists():
        return path
    return None


def get_definitions(filepath: Path) -> dict[str, int]:
    """Return {name: lineno} for all top-level def/class in a file."""
    tree = ast.parse(filepath.read_text())
    defs = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[node.name] = node.lineno
    return defs


def get_imports_from_file(filepath: Path) -> list[tuple[str, list[str]]]:
    """Return [(module_path, [imported_names])] for all cru_to_cmr imports.

    Handles both `from cru_to_cmr.X import Y` and dynamic string imports
    like `"cru_to_cmr.analyses.spc.plot_spc"`.
    """
    if filepath.suffix == ".ipynb":
        text = notebook_to_python(filepath)
    else:
        text = filepath.read_text()
    tree = ast.parse(text)
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = resolve_module(filepath, node)
            if resolved and resolved.startswith("cru_to_cmr"):
                names = [alias.name for alias in node.names]
                imports.append((resolved, names))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("cru_to_cmr"):
                    # `import cru_to_cmr.X.Y as Alias` — mark all defs in that module reachable
                    mod_file = module_to_filepath(alias.name)
                    if mod_file:
                        defs = get_definitions(mod_file)
                        imports.append((alias.name, list(defs.keys())))

    # Dynamic import strings
    for match in re.finditer(r'"(cru_to_cmr\.[^"]+)"', text):
        dotted = match.group(1)
        parts = dotted.rsplit(".", 1)
        if len(parts) == 2:
            module, name = parts
            imports.append((module, [name]))

    return imports


def resolve_module(filepath: Path, node: ast.ImportFrom) -> str | None:
    """Resolve an ImportFrom node to an absolute module path.

    Handles both absolute (`from cru_to_cmr.X`) and relative (`from ..X`) imports.
    """
    if node.level == 0:
        # Absolute import
        return node.module if node.module and node.module.startswith("cru_to_cmr") else None

    # Relative import: compute package from filepath
    # e.g., cru_to_cmr/analyses/spc.py -> package = cru_to_cmr.analyses
    parts = list(filepath.with_suffix("").parts)
    # Remove filename to get package
    package_parts = parts[:-1]
    # Go up `level - 1` directories (level=1 is current package, level=2 is parent, etc.)
    for _ in range(node.level - 1):
        if package_parts:
            package_parts.pop()

    base = ".".join(package_parts)
    if node.module:
        return f"{base}.{node.module}" if base else node.module
    return base or None


def get_module_import_map(filepath: Path) -> dict[str, str]:
    """Return {local_name: module_path} for imports in a file.

    Handles both absolute and relative imports within cru_to_cmr.
    """
    tree = ast.parse(filepath.read_text())
    mapping = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = resolve_module(filepath, node)
            if resolved and resolved.startswith("cru_to_cmr"):
                for alias in node.names:
                    local = alias.asname if alias.asname else alias.name
                    mapping[local] = resolved
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("cru_to_cmr"):
                    # `import cru_to_cmr.X.Y as Alias` — Alias.foo resolves to module.foo
                    local = alias.asname if alias.asname else alias.name
                    mapping[local] = alias.name
    return mapping


def get_names_used_in_function(tree: ast.AST, func_name: str) -> tuple[set[str], list[tuple[str, str]]]:
    """Return names and attribute-access pairs used inside a function/class body.

    Returns:
        (names, attr_pairs) where:
        - names: set of plain Name references (e.g., "load_data")
        - attr_pairs: list of (obj_name, attr_name) for `obj.attr` patterns
          (e.g., ("TemporalContext", "init") for `TemporalContext.init(...)`)
    """
    names = set()
    attr_pairs = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == func_name:
                for child in ast.walk(node):
                    if isinstance(child, ast.Name):
                        names.add(child.id)
                    elif isinstance(child, ast.Attribute):
                        names.add(child.attr)
                        # Track obj.attr where obj is a simple Name
                        if isinstance(child.value, ast.Name):
                            attr_pairs.append((child.value.id, child.attr))
    return names, attr_pairs


def get_module_level_names(tree: ast.AST) -> tuple[set[str], list[tuple[str, str]]]:
    """Return names and attribute-access pairs used in module-level statements.

    Module-level code runs on import, so any function/class referenced
    at the top level of a module is reachable when the module is imported.
    """
    names = set()
    attr_pairs = []
    for node in ast.iter_child_nodes(tree):
        # Skip function/class definitions — we only want top-level *statements*
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
            elif isinstance(child, ast.Attribute):
                names.add(child.attr)
                if isinstance(child.value, ast.Name):
                    attr_pairs.append((child.value.id, child.attr))
    return names, attr_pairs


def trace_reachable():
    """Trace all reachable (module_file, symbol_name) pairs from entry points."""
    # Set of (filepath, name) that are confirmed reachable
    reachable: set[tuple[Path, str]] = set()
    # Queue of (filepath, name) to process
    queue: list[tuple[Path, str]] = []
    # Cache parsed ASTs
    ast_cache: dict[Path, ast.AST] = {}
    # Cache import maps per file
    import_map_cache: dict[Path, dict[str, str]] = {}

    def parse(fp: Path) -> ast.AST:
        if fp not in ast_cache:
            ast_cache[fp] = ast.parse(fp.read_text())
        return ast_cache[fp]

    def import_map(fp: Path) -> dict[str, str]:
        if fp not in import_map_cache:
            import_map_cache[fp] = get_module_import_map(fp)
        return import_map_cache[fp]

    # Track which module files we've seeded module-level names for
    seeded_modules: set[Path] = set()

    def seed_module_level(filepath: Path):
        """Seed reachability from module-level code in a package file."""
        if filepath in seeded_modules or not filepath.exists():
            return
        seeded_modules.add(filepath)
        tree = parse(filepath)
        imap = import_map(filepath)
        names_used, attr_pairs = get_module_level_names(tree)
        defs = get_definitions(filepath)
        for used_name in names_used:
            if used_name in imap:
                mod_file = module_to_filepath(imap[used_name])
                if mod_file is not None:
                    queue.append((mod_file, used_name))
            if used_name in defs:
                queue.append((filepath, used_name))
        for obj_name, attr_name in attr_pairs:
            if obj_name in imap:
                mod_file = module_to_filepath(imap[obj_name])
                if mod_file is not None:
                    queue.append((mod_file, attr_name))

    # Seed: collect all cru_to_cmr symbols imported by entry points
    for ep in ENTRY_POINTS:
        if not ep.exists():
            continue
        for module, names in get_imports_from_file(ep):
            mod_file = module_to_filepath(module)
            if mod_file is None:
                continue
            seed_module_level(mod_file)
            for name in names:
                queue.append((mod_file, name))

    # Process queue: for each reachable symbol, find what it references
    while queue:
        filepath, name = queue.pop()
        if (filepath, name) in reachable:
            continue
        reachable.add((filepath, name))

        if not filepath.exists():
            continue

        tree = parse(filepath)
        imap = import_map(filepath)

        # Find all names and attribute-access pairs inside this function/class body
        names_used, attr_pairs = get_names_used_in_function(tree, name)

        # For each name used, check if it's an import from another cru_to_cmr module
        for used_name in names_used:
            if used_name in imap:
                mod_file = module_to_filepath(imap[used_name])
                if mod_file is not None:
                    seed_module_level(mod_file)
                    queue.append((mod_file, used_name))

            # Also check if it's a name defined in the same file
            defs = get_definitions(filepath)
            if used_name in defs:
                queue.append((filepath, used_name))

        # For attribute access like TemporalContext.init — if the object
        # is imported from a cru_to_cmr module, the attribute is reachable there
        for obj_name, attr_name in attr_pairs:
            if obj_name in imap:
                mod_file = module_to_filepath(imap[obj_name])
                if mod_file is not None:
                    queue.append((mod_file, attr_name))

    return reachable


def main():
    reachable = trace_reachable()

    # Build set of reachable names per file
    reachable_by_file: dict[Path, set[str]] = {}
    for filepath, name in reachable:
        if filepath not in reachable_by_file:
            reachable_by_file[filepath] = set()
        reachable_by_file[filepath].add(name)

    total_unused = 0
    total_defs = 0

    for pyfile in sorted(PACKAGE_DIR.rglob("*.py")):
        if pyfile.name == "__init__.py":
            continue

        defs = get_definitions(pyfile)
        if not defs:
            continue

        total_defs += len(defs)
        file_reachable = reachable_by_file.get(pyfile, set())

        unused = []
        for name, lineno in sorted(defs.items(), key=lambda x: x[1]):
            if name.startswith("_"):
                continue
            if name not in file_reachable:
                unused.append((name, lineno))

        if unused:
            total_unused += len(unused)
            print(f"\n{pyfile}:")
            for name, lineno in unused:
                print(f"  line {lineno}: {name}")

    print(f"\n--- Summary ---")
    print(f"Total definitions checked: {total_defs}")
    print(f"Unused definitions found: {total_unused}")

    if total_unused > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
