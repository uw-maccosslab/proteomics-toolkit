"""List what downstream notebooks and scripts use that ``tests/downstream_contract.py`` does not pin yet.

Usage::

    python tests/scan_downstream_usage.py <folder> [<folder> ...]

Walks the folders for ``.py`` and ``.ipynb`` files that use ``proteomics_toolkit``, reads every call
into the package (how many arguments it passes by position, which keywords it passes), every
attribute set on a config object, and every name read, and prints what the contract tables do not
cover yet. New calls are printed as ``DOWNSTREAM_CALLS`` entries ready to paste. Calls to names the
package no longer has are listed separately: that code is already broken.

The code is read in source order, a scope at a time, so a name means whatever it was last bound to
where it is used: a function's parameter hides a variable of the same name outside it, and a
variable reused for a second config is attributed to the second from then on. A notebook's cells
share their imports and variables, as they do when the notebook runs. The scan reads source only, so
a call whose arguments come from ``*args`` or ``**kwargs`` is counted by what it spells out.
"""

import argparse
import ast
import collections
import importlib
import inspect
import json
import os
import sys
import warnings

try:
    from downstream_contract import CONFIG_ATTRIBUTES, DOWNSTREAM_ATTRIBUTES, DOWNSTREAM_CALLS, PUBLIC_NAMES
except ImportError:  # imported by its tests, from the repository root
    from tests.downstream_contract import CONFIG_ATTRIBUTES, DOWNSTREAM_ATTRIBUTES, DOWNSTREAM_CALLS, PUBLIC_NAMES

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
        reader = _Reader(self)  # one module scope for all of a notebook's cells
        for source in sources:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)  # invalid escapes in notebook strings
                    tree = ast.parse(_without_magics(source))
            except SyntaxError:
                continue
            reader.visit(tree)
        self.files += reader.imported


class _Reader(ast.NodeVisitor):
    """Read code in source order into a ``Usage``, tracking what each name is bound to in each scope.

    A binding is ``("package", path)`` for a name that refers to the package or something in it,
    ``("result", path)`` for a variable holding what a call into the package returned, or None for a
    name bound to something else, which hides any binding of that name in an outer scope.
    """

    def __init__(self, usage):
        self.usage = usage
        self.scopes = [{}]
        self.imported = False

    def _lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def _bind(self, name, binding=None):
        self.scopes[-1][name] = binding

    def _path(self, expr):
        """Return the package path an expression such as ``ptk.normalization.median_normalize`` names."""
        chain = []
        while isinstance(expr, ast.Attribute):
            chain.append(expr.attr)
            expr = expr.value
        binding = self._lookup(expr.id) if isinstance(expr, ast.Name) else None
        if not binding or binding[0] != "package":
            return None
        return ".".join(part for part in [binding[1], *reversed(chain)] if part)

    # Imports

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == PACKAGE or alias.name.startswith(PACKAGE + "."):
                self.imported = True
                if alias.asname:
                    self._bind(alias.asname, ("package", alias.name[len(PACKAGE) + 1 :]))
                else:
                    self._bind(PACKAGE, ("package", ""))
            else:
                self._bind(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node):
        if node.level or not node.module or node.module.split(".")[0] != PACKAGE:
            for alias in node.names:
                if alias.name != "*":
                    self._bind(alias.asname or alias.name)
            return
        self.imported = True
        module = node.module[len(PACKAGE) + 1 :]
        for alias in node.names:
            if alias.name == "*":
                names = _star_names(module)
                if names is None:
                    self.usage.names.add(module)  # the module is gone, so the report lists it as broken
                for name in names or []:
                    self._bind(name, ("package", f"{module}.{name}" if module else name))
                continue
            path = f"{module}.{alias.name}" if module else alias.name
            self.usage.names.add(path)
            self._bind(alias.asname or alias.name, ("package", path))

    # Scopes. Defaults, decorators and annotations are evaluated where the definition is; the body
    # gets its own scope, in which the parameters hide any outer name they share (pytest's ``config``
    # is not a config).

    def _visit_all(self, nodes):
        for node in nodes:
            if node is not None:
                self.visit(node)

    def _visit_in_scope(self, body, args=None):
        scope = {}
        if args is not None:
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                if arg is not None:
                    scope[arg.arg] = None
        self.scopes.append(scope)
        self._visit_all(body)
        self.scopes.pop()

    def _visit_signature(self, args):
        self._visit_all([*args.defaults, *args.kw_defaults])
        arguments = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        self._visit_all(arg.annotation for arg in arguments if arg is not None)

    def visit_FunctionDef(self, node):
        self._visit_all(node.decorator_list)
        self._visit_signature(node.args)
        self._visit_all([node.returns])
        self._bind(node.name)
        self._visit_in_scope(node.body, node.args)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._visit_signature(node.args)
        self._visit_in_scope([node.body], node.args)

    def visit_ClassDef(self, node):
        self._visit_all([*node.decorator_list, *node.bases, *(keyword.value for keyword in node.keywords)])
        self._bind(node.name)
        self._visit_in_scope(node.body)

    def _visit_comprehension(self, node, results):
        # The first iterable is evaluated outside; the loop variables are the comprehension's own.
        self.visit(node.generators[0].iter)
        self.scopes.append({})
        for i, generator in enumerate(node.generators):
            if i:
                self.visit(generator.iter)
            self.visit(generator.target)
            self._visit_all(generator.ifs)
        self._visit_all(results)
        self.scopes.pop()

    def visit_ListComp(self, node):
        self._visit_comprehension(node, [node.elt])

    visit_SetComp = visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        self._visit_comprehension(node, [node.key, node.value])

    # Bindings. The value is read before the names it is assigned to are bound.

    def _bind_result(self, targets, value):
        path = self._path(value.func) if isinstance(value, ast.Call) else None
        if path:
            for target in targets:
                if isinstance(target, ast.Name):
                    self._bind(target.id, ("result", path))

    def visit_Assign(self, node):
        self.visit(node.value)
        self._visit_all(node.targets)
        self._bind_result(node.targets, node.value)

    def visit_AnnAssign(self, node):
        self.visit(node.annotation)
        if node.value is not None:  # a bare annotation binds nothing
            self.visit(node.value)
            self.visit(node.target)
            self._bind_result([node.target], node.value)

    def visit_ExceptHandler(self, node):
        self._visit_all([node.type])
        if node.name:
            self._bind(node.name)
        self._visit_all(node.body)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._bind(node.id)
            return
        path = self._path(node)
        if path:  # a name a star import bound, used bare
            self.usage.names.add(path)

    # Uses

    def visit_Call(self, node):
        path = self._path(node.func)
        if path:
            positional = sum(1 for arg in node.args if not isinstance(arg, ast.Starred))
            self.usage.max_positional[path] = max(self.usage.max_positional[path], positional)
            self.usage.keywords[path].update(keyword.arg for keyword in node.keywords if keyword.arg)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        path = self._path(node)
        if path:
            self.usage.names.add(path)
        elif isinstance(node.value, ast.Name):
            binding = self._lookup(node.value.id)
            if binding and binding[0] == "result" and binding[1] in CONFIG_PATHS:
                self.usage.attributes[binding[1]].add(node.attr)
        self.generic_visit(node)


# Every path a pinned config class is reached by: its module path and its top-level alias.
CONFIG_PATHS = set(CONFIG_ATTRIBUTES) | {path.rsplit(".", 1)[1] for path in CONFIG_ATTRIBUTES}


def _star_names(module):
    """Return the names ``from proteomics_toolkit[.module] import *`` binds, or None if the module is gone."""
    try:
        owner = importlib.import_module(f"{PACKAGE}.{module}" if module else PACKAGE)
    except ImportError:
        return None
    if hasattr(owner, "__all__"):
        return list(owner.__all__)
    # Without ``__all__`` the import binds every public name, including what the module itself
    # imported, such as ``np``. Keep what the package defines: its own modules, functions and
    # classes, and plain constants.
    names = []
    for name, value in vars(owner).items():
        if name.startswith("_"):
            continue
        if inspect.ismodule(value):
            origin = value.__name__
        elif inspect.isclass(value) or inspect.isroutine(value):
            origin = getattr(value, "__module__", None) or ""
        else:
            origin = PACKAGE if type(value).__module__ == "builtins" else ""
        if origin.split(".")[0] == PACKAGE:
            names.append(name)
    return names


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


def _required_names(path):
    """Name the parameters of the callable at ``path`` that have no default, in signature order."""
    try:
        params = inspect.signature(_resolve(path)).parameters.values()
    except (TypeError, ValueError):  # not a callable with a signature
        return []
    return [p.name for p in params if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]


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
            required = " ".join(_required_names(path))
            new_entries.append(f'    "{path}": ("{positional}", "{" ".join(keywords)}", "{required}"),')
            continue
        pinned_positional, pinned_keywords, pinned_required = (part.split() for part in DOWNSTREAM_CALLS[path])
        new_keywords = sorted(set(keywords) - set(pinned_keywords))
        if usage.max_positional[path] > len(pinned_positional) or new_keywords:
            positional = " ".join(_positional_names(path, max(usage.max_positional[path], len(pinned_positional))))
            merged = " ".join(sorted(set(keywords) | set(pinned_keywords)))
            changed.append(f'    "{path}": ("{positional}", "{merged}", "{" ".join(pinned_required)}"),')
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
