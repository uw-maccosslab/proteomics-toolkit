"""List what downstream notebooks and scripts use that ``tests/downstream_contract.py`` does not pin yet.

Usage::

    python tests/scan_downstream_usage.py <folder> [<folder> ...]

Walks the folders for ``.py`` and ``.ipynb`` files that use ``proteomics_toolkit``, reads every call
into the package (how many arguments it passes by position, which keywords it passes), every
attribute set on a config object, and every name read, and prints what the contract tables do not
cover yet. New calls are printed as ``DOWNSTREAM_CALLS`` entries ready to paste. Calls to names the
package no longer has are listed separately: that code is already broken.

A notebook's cells share their imports and variables, as they do when the notebook runs. The scan
reads source only, so a call whose arguments come from ``*args`` or ``**kwargs`` is counted by what
it spells out.
"""

import argparse
import ast
import collections
import importlib
import inspect
import json
import os
import sys

from downstream_contract import CONFIG_ATTRIBUTES, DOWNSTREAM_ATTRIBUTES, DOWNSTREAM_CALLS, PUBLIC_NAMES

# Virtual environments, caches, worktree copies and build output hold no notebooks of their own.
SKIP_DIRS = {
    ".git",
    ".venv",
    ".venv-win",
    "node_modules",
    "site-packages",
    "__pycache__",
    ".ipynb_checkpoints",
    ".claude",
    "bin",
    "obj",
}
PACKAGE = "proteomics_toolkit"


def _cell_sources(path):
    """Return the code cells of a notebook, or a script's whole source as one cell."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return []
    if not path.endswith(".ipynb"):
        return [text]
    try:
        cells = json.loads(text).get("cells", [])
    except json.JSONDecodeError:
        return []
    sources = []
    for cell in cells:
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            sources.append("".join(source) if isinstance(source, list) else source)
    return sources


def _without_magics(source):
    """Blank out IPython magics and shell escapes, which are not Python."""
    return "\n".join("" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines())


class Usage:
    """What the scanned code uses, keyed by path relative to the package."""

    def __init__(self):
        self.max_positional = collections.defaultdict(int)
        self.keywords = collections.defaultdict(set)
        self.names = set()
        self.attributes = collections.defaultdict(set)  # config class path -> attributes set or read
        self.files = 0

    def scan_file(self, path):
        sources = _cell_sources(path)
        if not any(PACKAGE in source for source in sources):
            return
        aliases = {}  # local name -> path in the package ("" for the package itself)
        bound = {}  # local variable -> path of the call that produced it
        used = False
        for source in sources:
            try:
                tree = ast.parse(_without_magics(source))
            except SyntaxError:
                continue
            self._read_imports(tree, aliases)
            if not aliases:
                continue
            used = True
            self._read_bindings(tree, aliases, bound)
            self._read_uses(tree, aliases, bound)
        self.files += used

    def _read_imports(self, tree, aliases):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == PACKAGE or alias.name.startswith(PACKAGE + "."):
                        if alias.asname:
                            aliases[alias.asname] = alias.name[len(PACKAGE) + 1 :]
                        else:
                            aliases[PACKAGE] = ""
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == PACKAGE:
                module = node.module[len(PACKAGE) + 1 :]
                for alias in node.names:
                    path = f"{module}.{alias.name}" if module else alias.name
                    self.names.add(path)
                    aliases[alias.asname or alias.name] = path

    @staticmethod
    def _path(expr, aliases):
        """Return the package path an expression such as ``ptk.normalization.median_normalize`` names."""
        chain = []
        while isinstance(expr, ast.Attribute):
            chain.append(expr.attr)
            expr = expr.value
        if not (isinstance(expr, ast.Name) and expr.id in aliases):
            return None
        parts = [aliases[expr.id], *reversed(chain)]
        return ".".join(part for part in parts if part)

    def _read_bindings(self, tree, aliases, bound):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                path = self._path(node.value.func, aliases)
                if path:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            bound[target.id] = path

    def _read_uses(self, tree, aliases, bound):
        # Inside a function, a parameter named like a bound variable (pytest's ``config``, say) is
        # a different object.
        shadowed = set()
        for function in ast.walk(tree):
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                params = {arg.arg for arg in ast.walk(function.args) if isinstance(arg, ast.arg)}
                shadowed.update(
                    id(node)
                    for node in ast.walk(function)
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in params
                )
        for node in ast.walk(tree):
            if id(node) in shadowed:
                continue
            if isinstance(node, ast.Call):
                path = self._path(node.func, aliases)
                if path:
                    positional = sum(1 for arg in node.args if not isinstance(arg, ast.Starred))
                    self.max_positional[path] = max(self.max_positional[path], positional)
                    self.keywords[path].update(keyword.arg for keyword in node.keywords if keyword.arg)
            elif isinstance(node, ast.Attribute):
                path = self._path(node, aliases)
                if path:
                    self.names.add(path)
                elif isinstance(node.value, ast.Name) and bound.get(node.value.id) in CONFIG_PATHS:
                    self.attributes[bound[node.value.id]].add(node.attr)


# Every path a pinned config class is reached by: its module path and its top-level alias.
CONFIG_PATHS = set(CONFIG_ATTRIBUTES) | {path.rsplit(".", 1)[1] for path in CONFIG_ATTRIBUTES}


def _resolve(path):
    module, _, name = path.rpartition(".")
    owner = importlib.import_module(f"{PACKAGE}.{module}" if module else PACKAGE)
    return getattr(owner, name)


def _exists(path):
    try:
        _resolve(path)
    except (ImportError, AttributeError):
        return False
    return True


def _positional_names(path, count):
    """Name the first ``count`` parameters of the callable at ``path``, as the contract pins them."""
    names = []
    try:
        params = inspect.signature(_resolve(path)).parameters.values()
    except (TypeError, ValueError):  # not a callable with a signature
        return ["?"] * count
    for param in params:
        if len(names) == count:
            break
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            names.append(param.name)
        elif param.kind is param.VAR_POSITIONAL:
            names += [f"*{param.name}"] * (count - len(names))
        else:
            break
    return names


def _pinned_names():
    pinned = set(DOWNSTREAM_CALLS)
    for table in (PUBLIC_NAMES, DOWNSTREAM_ATTRIBUTES):
        for module, names in table.items():
            pinned.update(f"{module}.{name}" if module else name for name in names.split())
    return pinned


def report(usage):
    """Return the lines describing what ``usage`` has that the contract does not pin."""
    lines = []
    broken = sorted(path for path in usage.names | set(usage.max_positional) if not _exists(path))
    if broken:
        lines.append("Already broken: these names no longer exist")
        lines += [f"    {path}" for path in broken]

    new_entries, changed = [], []
    for path in sorted(set(usage.max_positional) - set(broken)):
        keywords = sorted(usage.keywords[path])
        if path not in DOWNSTREAM_CALLS:
            positional = " ".join(_positional_names(path, usage.max_positional[path]))
            new_entries.append(f'    "{path}": ("{positional}", "{" ".join(keywords)}"),')
            continue
        pinned_positional, pinned_keywords = (part.split() for part in DOWNSTREAM_CALLS[path])
        new_keywords = sorted(set(keywords) - set(pinned_keywords))
        if usage.max_positional[path] > len(pinned_positional) or new_keywords:
            positional = " ".join(_positional_names(path, max(usage.max_positional[path], len(pinned_positional))))
            merged = " ".join(sorted(set(keywords) | set(pinned_keywords)))
            changed.append(f'    "{path}": ("{positional}", "{merged}"),')
    if new_entries:
        lines.append("Calls not pinned yet (add to DOWNSTREAM_CALLS)")
        lines += new_entries
    if changed:
        lines.append("Calls with new keywords or more positional arguments (replace in DOWNSTREAM_CALLS)")
        lines += changed

    unpinned = sorted(usage.names - _pinned_names() - set(broken))
    if unpinned:
        lines.append("Names read but not pinned (add to DOWNSTREAM_ATTRIBUTES)")
        lines += [f"    {path}" for path in unpinned]

    for config in sorted(CONFIG_ATTRIBUTES):
        seen = usage.attributes.get(config, set()) | usage.attributes.get(config.rsplit(".", 1)[1], set())
        instance = _resolve(config)()
        new = sorted(name for name in seen - set(CONFIG_ATTRIBUTES[config].split()) if hasattr(instance, name))
        ignored = sorted(name for name in seen if not hasattr(instance, name))
        if new:
            lines.append(f"{config} attributes not pinned yet: {' '.join(new)}")
        if ignored:
            lines.append(
                f"{config} attributes callers set that it does not have (their settings do nothing): "
                f"{' '.join(ignored)}"
            )
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", nargs="+", help="folders holding the repositories to scan")
    args = parser.parse_args(argv)

    usage = Usage()
    for folder in args.folders:
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith("proteomics-toolkit")]
            for filename in filenames:
                if filename.endswith((".py", ".ipynb")):
                    usage.scan_file(os.path.join(dirpath, filename))

    lines = report(usage)
    print(f"Scanned {usage.files} files that use {PACKAGE}.")
    print("\n".join(lines) if lines else "Everything in use is pinned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
