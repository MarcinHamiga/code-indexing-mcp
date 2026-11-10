"""Regression tests for REF-11 (Python reference recall, Track L).

Every positive test pins a use that used to stay `likely`/`unresolved` and must
now be `exact`; every negative test pins a look-alike that must NOT become
`exact`, because REF-11 is a recall fix that may not cost precision.
"""

from pathlib import Path

import pytest

from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.models import DeclarationSelector, ReferenceHit, RenameOperation
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.python_bindings import PythonBindings
from code_indexing_mcp.reference_service import ReferenceService
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.storage import LanceStore


class TinyEmbedder:
    model_id = "test/python-recall"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


def _indexed(tmp_path: Path, files: dict[str, str]) -> tuple[ReferenceService, str]:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    for path, source in files.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source, encoding="utf-8")
    project = initialize_project(root)
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=TinyEmbedder(),
        lock_directory=tmp_path / "locks",
    ).index(project)
    return ReferenceService(store), project.id


def _hits(
    tmp_path: Path, files: dict[str, str], path: str, symbol: str, *, in_file: str
) -> list[ReferenceHit]:
    service, project_id = _indexed(tmp_path, files)
    response = service.find_references(
        DeclarationSelector(project=project_id, path=path, qualified_symbol=symbol)
    )
    return [hit for hit in response.hits if hit.path == in_file and hit.kind != "import"]


def _resolutions(hits: list[ReferenceHit]) -> set[str]:
    return {hit.resolution for hit in hits}


# --- cause 1: absolute import from a non-package directory into a src/ layout


@pytest.mark.parametrize("package_init", [False, True])
def test_absolute_import_into_a_src_layout_from_tests_is_exact(
    tmp_path: Path, package_init: bool
) -> None:
    files = {
        "src/pkg/mod.py": "def apply_edits():\n    return 1\n",
        "tests/test_mod.py": (
            "from pkg.mod import apply_edits\n\n\ndef test_it():\n    return apply_edits()\n"
        ),
    }
    if package_init:
        files["src/pkg/__init__.py"] = ""
    hits = _hits(tmp_path, files, "src/pkg/mod.py", "apply_edits", in_file="tests/test_mod.py")
    assert hits
    assert _resolutions(hits) == {"exact"}


def test_single_segment_import_of_a_src_layout_package_initializer_stays_likely(
    tmp_path: Path,
) -> None:
    hits = _hits(
        tmp_path,
        {
            "src/pkg/__init__.py": "def run():\n    return 1\n",
            "scripts/use.py": "from pkg import run\n\n\ndef go():\n    return run()\n",
        },
        "src/pkg/__init__.py",
        "run",
        in_file="scripts/use.py",
    )
    assert hits
    # Single-segment `pkg` is not placed by the unique-path fallback (REF-11).
    assert _resolutions(hits) == {"likely"}


def test_the_pinned_src_layout_module_is_now_placed_uniquely(tmp_path: Path) -> None:
    # The scenario of `test_a_project_module_the_resolver_cannot_place_stays_a_likely_reference`:
    # one `src/pkg/lib.py`, so the import resolves to a single path and is exact.
    hits = _hits(
        tmp_path,
        {
            "src/pkg/lib.py": "def run():\n    return 1\n",
            "scripts/use.py": "from pkg.lib import run\n\n\ndef go():\n    return run()\n",
        },
        "src/pkg/lib.py",
        "run",
        in_file="scripts/use.py",
    )
    assert _resolutions(hits) == {"exact"}


def test_two_candidate_paths_keep_a_src_layout_import_likely(tmp_path: Path) -> None:
    files = {
        "src/pkg/lib.py": "def run():\n    return 1\n",
        "vendor/pkg/lib.py": "def run():\n    return 2\n",
        "scripts/use.py": "from pkg.lib import run\n\n\ndef go():\n    return run()\n",
    }
    for selected in ("src/pkg/lib.py", "vendor/pkg/lib.py"):
        hits = _hits(
            tmp_path / selected.split("/")[0], files, selected, "run", in_file="scripts/use.py"
        )
        assert hits
        assert "exact" not in _resolutions(hits)


def test_a_single_segment_import_never_uses_the_unique_path_fallback(tmp_path: Path) -> None:
    # `import json` / `from json import dumps` must not bind to a project
    # `tools/json.py`: it may be the standard library or a third-party package.
    files = {
        "tools/json.py": "def dumps(value):\n    return str(value)\n",
        "app.py": (
            "import json\nfrom json import dumps\n\n\n"
            "def go(x):\n    json.dumps(x)\n    return dumps(x)\n"
        ),
    }
    service, project_id = _indexed(tmp_path, files)
    selector = DeclarationSelector(
        project=project_id, path="tools/json.py", qualified_symbol="dumps"
    )
    response = service.find_references(selector)
    uses = [hit for hit in response.hits if hit.path == "app.py" and hit.kind != "import"]
    assert uses
    assert "exact" not in _resolutions(uses)
    assert {"likely"} == _resolutions(uses)
    patch = service.emit_refactor_patch(selector, RenameOperation(new_name="encode"))
    assert "app.py" not in patch.patch


def test_a_single_segment_module_at_a_recognised_root_is_still_exact(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {
            "utils.py": "def f():\n    return 1\n",
            "main.py": "from utils import f\n\n\ndef go():\n    return f()\n",
        },
        "utils.py",
        "f",
        in_file="main.py",
    )
    assert _resolutions(hits) == {"exact"}


def test_a_nested_package_is_not_an_import_root(tmp_path: Path) -> None:
    # `src/app` is itself a package, so `src/app/pkg/lib.py` needs `app.pkg.lib`;
    # `from pkg.lib import run` does not reach it.
    hits = _hits(
        tmp_path,
        {
            "src/app/__init__.py": "",
            "src/app/pkg/lib.py": "def run():\n    return 1\n",
            "scripts/use.py": "from pkg.lib import run\n\n\ndef go():\n    return run()\n",
        },
        "src/app/pkg/lib.py",
        "run",
        in_file="scripts/use.py",
    )
    assert "exact" not in _resolutions(hits)


def test_a_src_layout_import_of_a_different_name_is_not_a_use(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {
            "src/pkg/mod.py": "def f():\n    return 1\n\n\ndef g():\n    return 2\n",
            "tests/test_mod.py": "from pkg.mod import g\n\n\ndef test():\n    return g()\n",
        },
        "src/pkg/mod.py",
        "f",
        in_file="tests/test_mod.py",
    )
    assert not hits


# --- cause 2: submodule imports


def test_from_package_import_submodule_member_call_is_exact(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "def f():\n    return 1\n",
            "main.py": "from pkg import mod\n\n\ndef go():\n    return mod.f()\n",
        },
        "pkg/mod.py",
        "f",
        in_file="main.py",
    )
    assert [hit.resolution for hit in hits] == ["exact"]


def test_from_dot_import_submodule_member_call_is_exact(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "def f():\n    return 1\n",
            "pkg/user.py": "from . import mod\n\n\ndef go():\n    return mod.f()\n",
        },
        "pkg/mod.py",
        "f",
        in_file="pkg/user.py",
    )
    assert [hit.resolution for hit in hits] == ["exact"]


def test_submodule_alias_and_src_layout_namespace_package_are_exact(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {
            "src/pkg/mod.py": "def f():\n    return 1\n",
            "tests/test_mod.py": "from pkg import mod as m\n\n\ndef test():\n    return m.f()\n",
        },
        "src/pkg/mod.py",
        "f",
        in_file="tests/test_mod.py",
    )
    assert [hit.resolution for hit in hits] == ["exact"]


def test_a_package_attribute_shadowing_the_submodule_is_not_exact(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {
            "pkg/__init__.py": "mod = object()\n",
            "pkg/mod.py": "def f():\n    return 1\n",
            "main.py": "from pkg import mod\n\n\ndef go():\n    return mod.f()\n",
        },
        "pkg/mod.py",
        "f",
        in_file="main.py",
    )
    assert "exact" not in _resolutions(hits)


def test_a_submodule_member_of_another_name_or_another_module_is_not_exact(
    tmp_path: Path,
) -> None:
    files = {
        "pkg/__init__.py": "",
        "pkg/mod.py": "def f():\n    return 1\n",
        "pkg/other.py": "def f():\n    return 2\n",
        "main.py": "from pkg import other\n\n\ndef go():\n    return other.f()\n",
    }
    hits = _hits(tmp_path, files, "pkg/mod.py", "f", in_file="main.py")
    assert "exact" not in _resolutions(hits)


def test_a_local_shadowing_a_submodule_import_is_not_exact(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "def f():\n    return 1\n",
            "main.py": "from pkg import mod\n\n\ndef go(mod):\n    return mod.f()\n",
        },
        "pkg/mod.py",
        "f",
        in_file="main.py",
    )
    assert "exact" not in _resolutions(hits)


# --- cause 3: dotted namespace imports


def test_dotted_namespace_import_member_call_is_exact(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "def f():\n    return 1\n",
            "pkg/other.py": "def g():\n    return 2\n",
            "main.py": (
                "import pkg.mod\nimport pkg.other\n\n\n"
                "def go():\n    return pkg.mod.f() + pkg.other.g()\n"
            ),
        },
        "pkg/mod.py",
        "f",
        in_file="main.py",
    )
    assert [hit.resolution for hit in hits] == ["exact"]


def test_dotted_receiver_without_its_own_import_is_not_exact(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "def f():\n    return 1\n",
            "pkg/other.py": "def g():\n    return 2\n",
            "main.py": "import pkg.other\n\n\ndef go():\n    return pkg.mod.f()\n",
        },
        "pkg/mod.py",
        "f",
        in_file="main.py",
    )
    assert "exact" not in _resolutions(hits)


def test_a_parameter_named_like_the_package_is_not_exact(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "def f():\n    return 1\n",
            "main.py": "import pkg.mod\n\n\ndef go(pkg):\n    return pkg.mod.f()\n",
        },
        "pkg/mod.py",
        "f",
        in_file="main.py",
    )
    assert "exact" not in _resolutions(hits)


def test_a_rebound_package_name_is_not_exact(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "def f():\n    return 1\n",
            "main.py": "import pkg.mod\n\npkg = object()\n\n\ndef go():\n    return pkg.mod.f()\n",
        },
        "pkg/mod.py",
        "f",
        in_file="main.py",
    )
    assert "exact" not in _resolutions(hits)


# --- cause 4: per-name unsupported state

_LIB = {"lib.py": "def f(value=None):\n    return value\n"}


def _call_resolutions(tmp_path: Path, body: str, header: str = "from lib import f\n") -> set[str]:
    hits = _hits(
        tmp_path,
        {**_LIB, "main.py": header + "\n\n" + body},
        "lib.py",
        "f",
        in_file="main.py",
    )
    assert hits, "the call should still be reported"
    return _resolutions(hits)


@pytest.mark.parametrize(
    "body",
    [
        "def go(items):\n    if (n := len(items)):\n        return f(n)\n",
        "def go(items):\n    return [f(x) for x in items]\n",
        "def go(items):\n    return {x: f(x) for x in items}\n",
        (
            "def go(command):\n    match command:\n        case [first, *rest]:\n"
            "            return f(first)\n"
        ),
        (
            "def go(event):\n    match event:\n        case {'k': v, **extra}:\n"
            "            return f(v)\n"
        ),
        "def other():\n    global counter\n    counter = 1\n\n\ndef go():\n    return f()\n",
        (
            "def outer():\n    n = 0\n\n    def inner():\n        nonlocal n\n"
            "        n += 1\n        return f()\n    return inner\n"
        ),
    ],
)
def test_constructs_that_cannot_rebind_the_callee_keep_it_exact(tmp_path: Path, body: str) -> None:
    assert _call_resolutions(tmp_path, body) == {"exact"}


@pytest.mark.parametrize(
    "body",
    [
        "def go():\n    if (f := len):\n        return f([1])\n",
        "def go(items):\n    return [f(f) for f in items]\n",
        "def go(items):\n    return [y for x in items if (y := x)] + [f()]\n",
        "def go(items):\n    return [y for x in items if (f := x)] and f()\n",
        "def go(command):\n    match command:\n        case f:\n            return f()\n",
        "def go(command):\n    match command:\n        case [*f]:\n            return f()\n",
        "def reset():\n    global f\n    f = None\n\n\ndef go():\n    return f()\n",
        "def go():\n    global f\n    return f()\n",
        (
            "def outer():\n    f = 1\n\n    def inner():\n        nonlocal f\n"
            "        f = 2\n        return f()\n    return inner\n"
        ),
        "def go():\n    eval('1')\n    return f()\n",
        "def go():\n    exec('1')\n    return f()\n",
        "def go():\n    locals()\n    return f()\n",
        "def go():\n    return vars()\n\n\ndef again():\n    vars()\n    return f()\n",
    ],
)
def test_constructs_that_can_rebind_the_callee_stay_non_exact(tmp_path: Path, body: str) -> None:
    hits = _hits(
        tmp_path,
        {**_LIB, "main.py": "from lib import f\n\n\n" + body},
        "lib.py",
        "f",
        in_file="main.py",
    )
    assert "exact" not in _resolutions(hits)


def test_star_import_still_poisons_the_whole_scope(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {
            **_LIB,
            "main.py": "from lib import f\nfrom os.path import *\n\n\ndef go():\n    return f()\n",
        },
        "lib.py",
        "f",
        in_file="main.py",
    )
    assert "exact" not in _resolutions(hits)


def test_bindings_poison_is_per_name() -> None:
    source = (
        b"def go(items):\n"
        b"    if (n := len(items)):\n"
        b"        match n:\n"
        b"            case [a, *b]:\n"
        b"                pass\n"
        b"    global g\n"
        b"    return [x for x in items]\n"
    )
    bindings = PythonBindings(source)
    inside = source.index(b"return [x")
    assert bindings.lookup(inside, "n")[1] is False
    assert len(bindings.lookup(inside, "n")[0]) == 1
    assert len(bindings.lookup(inside, "a")[0]) == 1
    assert len(bindings.lookup(inside, "b")[0]) == 1
    assert bindings.lookup(inside, "g")[1] is True
    assert bindings.lookup(inside, "unrelated") == ([], False)


def test_walrus_in_a_comprehension_still_withholds_the_enclosing_scope() -> None:
    source = b"def go(items):\n    return [y for x in items if (y := x)], y\n"
    bindings = PythonBindings(source)
    outside = source.index(b", y") + 2
    assert bindings.lookup(outside, "y")[1] is True
    # The comprehension's own iteration variable is still its own binding.
    plain = b"def go(items):\n    return [x for x in items]\n"
    assert PythonBindings(plain).lookup(plain.index(b"x for"), "x")[1] is False


def test_exec_family_stays_scope_wide() -> None:
    for call in (b"eval('1')", b"exec('1')", b"globals()", b"locals()", b"vars()"):
        source = b"def go():\n    " + call + b"\n    return 1\n"
        bindings = PythonBindings(source)
        assert bindings.lookup(source.index(b"return"), "anything")[1] is True


# --- cause 5: decorated methods

_DECORATED = """
import functools


class Service:
    def helper(self):
        return 1

    @property
    def prop(self):
        return self.helper()

    @functools.cache
    def cached(self):
        return self.helper()

    @classmethod
    @functools.lru_cache(maxsize=None)
    def build(cls):
        return cls.make()

    @classmethod
    def make(cls):
        return cls()

    @staticmethod
    def static(self):
        return self.helper()
"""


def test_decorated_methods_resolve_self_to_the_owner(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {"service.py": _DECORATED},
        "service.py",
        "Service.helper",
        in_file="service.py",
    )
    exact = {hit.start_line for hit in hits if hit.resolution == "exact"}
    lines = _DECORATED.splitlines()
    prop_call = next(i for i, line in enumerate(lines, 1) if "self.helper()" in line)
    cached_call = next(
        i for i, line in enumerate(lines, 1) if "self.helper()" in line and i > prop_call
    )
    assert {prop_call, cached_call} <= exact


def test_decorated_classmethod_cls_resolves_to_the_owner(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path, {"service.py": _DECORATED}, "service.py", "Service.make", in_file="service.py"
    )
    assert [hit.resolution for hit in hits] == ["exact"]


def test_a_staticmethod_first_parameter_is_not_a_receiver(tmp_path: Path) -> None:
    hits = _hits(
        tmp_path,
        {"service.py": _DECORATED},
        "service.py",
        "Service.helper",
        in_file="service.py",
    )
    lines = _DECORATED.splitlines()
    static_call = next(i for i, line in enumerate(lines, 1) if "self.helper()" in line and i > 25)
    assert all(hit.resolution != "exact" for hit in hits if hit.start_line == static_call)


def test_a_decorated_method_of_a_same_named_class_in_another_file_is_not_exact(
    tmp_path: Path,
) -> None:
    files = {
        "a.py": "class Service:\n    def helper(self):\n        return 1\n",
        "b.py": (
            "import functools\n\n\nclass Service:\n    def helper(self):\n        return 2\n\n"
            "    @functools.cache\n    def run(self):\n        return self.helper()\n"
        ),
    }
    hits = _hits(tmp_path, files, "a.py", "Service.helper", in_file="b.py")
    assert "exact" not in _resolutions(hits)


# --- Track D: dead_code_report and find_references agree on the new import shapes


@pytest.mark.parametrize(
    ("files", "path", "use_file"),
    [
        (
            {
                "src/pkg/mod.py": "def f():\n    return 1\n",
                "tests/test_mod.py": "from pkg.mod import f\n\n\ndef test():\n    return f()\n",
            },
            "src/pkg/mod.py",
            "tests/test_mod.py",
        ),
        (
            {
                "pkg/__init__.py": "",
                "pkg/mod.py": "def f():\n    return 1\n",
                "main.py": "from pkg import mod\n\n\ndef go():\n    return mod.f()\n",
            },
            "pkg/mod.py",
            "main.py",
        ),
        (
            {
                "pkg/__init__.py": "",
                "pkg/mod.py": "def f():\n    return 1\n",
                "main.py": "import pkg.mod\n\n\ndef go():\n    return pkg.mod.f()\n",
            },
            "pkg/mod.py",
            "main.py",
        ),
    ],
)
def test_dead_code_report_agrees_with_find_references(
    tmp_path: Path, files: dict[str, str], path: str, use_file: str
) -> None:
    service, project_id = _indexed(tmp_path, files)
    found = service.find_references(
        DeclarationSelector(project=project_id, path=path, qualified_symbol="f")
    )
    assert any(
        hit.path == use_file and hit.kind == "call" and hit.resolution == "exact"
        for hit in found.hits
    )
    report = service.dead_code_report(project_id)
    flagged = {
        finding.declaration.qualified_symbol
        for finding in report.review
        if finding.declaration.path == path
    }
    assert "f" not in flagged
