"""Conservative Python lexical bindings for live, hash-verified source.

This supplements structural import edges with parameters and assignments, which
are not declaration rows. Dynamic constructs explicitly make a scope uncertain.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class _Scope:
    parent: _Scope | None
    kind: str
    bindings: dict[str, list[ast.AST]] = field(default_factory=dict)
    unsupported: bool = False


class PythonBindings(ast.NodeVisitor):
    def __init__(self, source: bytes) -> None:
        self.lines = [0]
        for line in source.splitlines(keepends=True):
            self.lines.append(self.lines[-1] + len(line))
        self.receiver_parameters: set[int] = set()
        self.attribute_writes: set[str] = set()
        self.qualified: dict[int, str] = {}
        self.owners: list[str] = []
        self.scope = _Scope(None, "module")
        self.module = self.scope
        self.positions: dict[int, _Scope] = {}
        self.nodes: dict[int, ast.AST] = {}
        try:
            self.visit(ast.parse(source))
        except (SyntaxError, ValueError, RecursionError):
            self.module.unsupported = True

    def start(self, node: ast.AST) -> int:
        return self.lines[getattr(node, "lineno", 1) - 1] + getattr(node, "col_offset", 0)

    def end(self, node: ast.AST) -> int:
        return self.lines[(getattr(node, "end_lineno", None) or 1) - 1] + (
            getattr(node, "end_col_offset", None) or 0
        )

    def visit(self, node: ast.AST) -> None:
        if hasattr(node, "lineno"):
            self.positions[self.start(node)] = self.scope
            self.nodes[self.start(node)] = node
        super().visit(node)

    def _bind(self, name: str, node: ast.AST) -> None:
        self.scope.bindings.setdefault(name, []).append(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._bind(node.id, node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.attribute_writes.add(node.attr)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self._bind(node.arg, node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.visit(alias)
            self._bind(alias.asname or alias.name.split(".")[0], alias)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.visit(alias)
            self._bind(alias.asname or alias.name, alias)
            if alias.name == "*":
                self.scope.unsupported = True

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        if not isinstance(node, ast.Lambda):
            if node.type_params:
                # PEP 695 introduces annotation scopes with distinct binding rules.
                self.scope.unsupported = True
            positional = [*node.args.posonlyargs, *node.args.args]
            if (
                self.scope.kind == "class"
                and positional
                and all(
                    isinstance(decorator, ast.Name) and decorator.id == "classmethod"
                    for decorator in node.decorator_list
                )
            ):
                self.receiver_parameters.add(id(positional[0]))
            self._bind(node.name, node)
            self.qualified[id(node)] = ".".join([*self.owners, node.name])
            for decorator in node.decorator_list:
                self.visit(decorator)
            if node.returns:
                self.visit(node.returns)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default:
                self.visit(default)
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg:
            args.append(node.args.vararg)
        if node.args.kwarg:
            args.append(node.args.kwarg)
        for arg in args:
            if arg.annotation:
                self.visit(arg.annotation)
        parent = self.scope
        self.scope = _Scope(parent, "function")
        self.owners.append(node.name if not isinstance(node, ast.Lambda) else "<lambda>")
        for arg in args:
            self.visit(arg)
        if isinstance(node.body, list):
            for statement in node.body:
                self.visit(statement)
        else:
            self.visit(node.body)
        self.scope = parent
        self.owners.pop()

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function
    visit_Lambda = _function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.type_params:
            self.scope.unsupported = True
        self._bind(node.name, node)
        self.qualified[id(node)] = ".".join([*self.owners, node.name])
        for expression in [*node.bases, *node.decorator_list, *node.keywords]:
            self.visit(expression)
        parent = self.scope
        self.scope = _Scope(parent, "class")
        self.owners.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope = parent
        self.owners.pop()

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._bind(node.name, node)
        self.generic_visit(node)

    def _unsupported(self, node: ast.AST) -> None:
        self.scope.unsupported = True
        self.generic_visit(node)

    visit_Global = _unsupported
    visit_Nonlocal = _unsupported

    def _comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        # The initial iterable belongs to the enclosing scope; iteration
        # targets and the body have their own scope and cannot shadow the
        # enclosing call's callee. Walrus targets can escape that scope.
        if any(isinstance(child, ast.NamedExpr) for child in ast.walk(node)):
            self.scope.unsupported = True
        first, *rest = node.generators
        self.visit(first.iter)
        parent = self.scope
        self.scope = _Scope(parent, "comprehension", unsupported=True)
        self.visit(first.target)
        for condition in first.ifs:
            self.visit(condition)
        for generator in rest:
            self.visit(generator)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.scope = parent

    visit_ListComp = _comprehension
    visit_SetComp = _comprehension
    visit_DictComp = _comprehension
    visit_GeneratorExp = _comprehension
    visit_NamedExpr = _unsupported
    visit_Match = _unsupported

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in {
            "exec",
            "eval",
            "globals",
            "locals",
            "vars",
        }:
            self.scope.unsupported = True
        self.generic_visit(node)

    def lookup(self, position: int, name: str) -> tuple[list[ast.AST], bool]:
        scope = self.positions.get(position)
        if scope is None:
            return [], True
        while scope is not None:
            if scope.unsupported:
                return [], True
            if name in scope.bindings:
                return scope.bindings[name], False
            scope = scope.parent
            # A class namespace is visible in its own body, but never forms
            # a closure for a nested function or nested class body.
            while scope is not None and scope.kind == "class":
                scope = scope.parent
        return [], False

    def collision(self, position: int, name: str, *, local_only: bool = False) -> bool:
        if local_only:
            scope = self.positions.get(position)
            return scope is not None and name in scope.bindings
        bindings, _ = self.lookup(position, name)
        return bool(bindings)
