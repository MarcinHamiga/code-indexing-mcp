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
    # Names whose binding here can change behind the scope's back (`global` /
    # `nonlocal` writes). Only these lookups are uncertain; the rest of the
    # scope stays resolvable.
    poisoned: set[str] = field(default_factory=set)


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

    @staticmethod
    def _is_staticmethod(decorator: ast.expr) -> bool:
        if isinstance(decorator, ast.Name):
            return decorator.id == "staticmethod"
        return isinstance(decorator, ast.Attribute) and decorator.attr == "staticmethod"

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        if not isinstance(node, ast.Lambda):
            if node.type_params:
                # PEP 695 introduces annotation scopes with distinct binding rules.
                self.scope.unsupported = True
            positional = [*node.args.posonlyargs, *node.args.args]
            # The first parameter of any method is its receiver unless the method
            # is a `staticmethod`. Other decorators (`property`, `cache`,
            # `setter`, `abstractmethod`, `wraps`-style wrappers) are assumed
            # to forward the instance (or class, for `classmethod`) as the first
            # argument; the caller additionally requires the parameter to be
            # spelled `self`/`cls` and the enclosing type to match the owner.
            if (
                self.scope.kind == "class"
                and positional
                and not any(self._is_staticmethod(decorator) for decorator in node.decorator_list)
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

    def visit_Global(self, node: ast.Global) -> None:
        # `global x` makes this scope's `x` the module's `x`, and any write
        # through it rebinds what every other scope reads. Only the named
        # bindings are uncertain, in this scope and at module level.
        self.scope.poisoned.update(node.names)
        self.module.poisoned.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        # `nonlocal x` rebinds `x` in whichever enclosing function owns it.
        scope: _Scope | None = self.scope
        while scope is not None and scope.kind != "module":
            scope.poisoned.update(node.names)
            scope = scope.parent

    def _binding_scope(self) -> _Scope:
        # A walrus target escapes comprehension scopes into the enclosing one.
        scope = self.scope
        while scope.kind == "comprehension" and scope.parent is not None:
            scope = scope.parent
        return scope

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._binding_scope().bindings.setdefault(node.target.id, []).append(node)

    def _capture(self, name: str | None, node: ast.AST) -> None:
        if name:
            self._bind(name, node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        # Capture patterns bind exactly the names they spell; `match` opens no
        # scope of its own.
        self._capture(node.name, node)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        self._capture(node.name, node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self._capture(node.rest, node)
        self.generic_visit(node)

    def _comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        # The initial iterable belongs to the enclosing scope; iteration
        # targets and the body have their own scope, whose bindings are exactly
        # the iteration targets. Walrus targets escape to the enclosing scope
        # (`_binding_scope`), so the comprehension scope itself is resolvable.
        # A walrus inside a comprehension still withholds the enclosing scope:
        # `test_comprehension_walrus_withholds_outer_binding_edits` pins that
        # conservative behaviour for rename patches.
        if any(isinstance(child, ast.NamedExpr) for child in ast.walk(node)):
            self.scope.unsupported = True
        first, *rest = node.generators
        self.visit(first.iter)
        parent = self.scope
        self.scope = _Scope(parent, "comprehension")
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
            if scope.unsupported or name in scope.poisoned:
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
