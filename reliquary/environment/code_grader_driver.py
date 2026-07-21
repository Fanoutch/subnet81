"""Local structured-case grading driver (run as a subprocess).

Replicates the validator's grader worker semantics VERBATIM so our local
passed/total matches the validator's by construction (same sandbox, same entry
resolution, same value comparison). Reads {"code", "cases"} as JSON on stdin and
prints {"passed": N, "total": M}.

The execution core (`evaluate_call` + helpers + the restricted `_safe_builtins`
sandbox) is copied verbatim from reliquary/environment/grader/worker.py on the
validator (origin/main). The value comparison (`_json_equal` / `_outputs_match`)
is copied from reliquary/environment/grader/server.py (de-classmethod'd). Keeping
the SAME sandbox matters: a completion importing a module the validator forbids
must score 0 here too, or our sigma would diverge from the validator's.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import inspect
import io
import json
import math
import sys
from typing import Any


# ===== VERBATIM from validator grader/worker.py =====

_CRITICAL_BUILTINS = {
    name: getattr(builtins, name)
    for name in ("__import__", "compile", "eval", "exec", "open", "input")
}

_ALLOWED_IMPORT_ROOTS = {
    "abc", "array", "bisect", "collections", "copy", "dataclasses", "decimal",
    "enum", "functools", "heapq", "itertools", "math", "operator", "re",
    "statistics", "string", "typing",
}

_DENIED_BUILTINS = {
    "breakpoint", "compile", "dir", "eval", "exec", "globals", "help", "input",
    "locals", "open", "vars",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".", 1)[0]
    if level != 0 or root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"module {name!r} is not available in the grader sandbox")
    return _CRITICAL_BUILTINS["__import__"](name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    safe = {
        name: value
        for name, value in builtins.__dict__.items()
        if name not in _DENIED_BUILTINS
    }
    safe["__import__"] = _safe_import
    return safe


def _critical_builtins_intact() -> bool:
    return all(
        getattr(builtins, name) is original
        for name, original in _CRITICAL_BUILTINS.items()
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError("dict key is not a string")
            out[k] = _json_safe(v)
        return out
    raise TypeError(f"unsupported output type: {type(value).__name__}")


def _user_defined_names(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _accepts_arity(fn: Any, nargs: int) -> bool:
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return True
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required = sum(1 for p in positional if p.default is p.empty)
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
    upper = float("inf") if has_varargs else len(positional)
    return required <= nargs <= upper


def _defined_functions_in_order(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _call_graph_roots(code: str, fn_names: set[str]) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(fn_names)
    called_by_others: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in fn_names
                and sub.func.id != node.name
            ):
                called_by_others.add(sub.func.id)
    return set(fn_names) - called_by_others


def _returns_a_value(code: str, name: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            visitor = _ValueReturnVisitor()
            for stmt in node.body:
                visitor.visit(stmt)
            return visitor.has_value_return
    return True


class _ValueReturnVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.has_value_return = False

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and not (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ):
            self.has_value_return = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None


def _resolve_function(ns: dict[str, Any], code: str, nargs: int) -> Any | None:
    order = [
        name for name in _defined_functions_in_order(code)
        if callable(ns.get(name)) and not isinstance(ns.get(name), type)
    ]
    candidates = [name for name in order if _accepts_arity(ns[name], nargs)]
    if not candidates:
        return None
    valued = [name for name in candidates if _returns_a_value(code, name)]
    if valued:
        candidates = valued
    if len(candidates) == 1:
        return ns[candidates[0]]
    roots = _call_graph_roots(code, set(order))
    root_candidates = [name for name in candidates if name in roots]
    if len(root_candidates) == 1:
        return ns[root_candidates[0]]
    pool = root_candidates or candidates
    return ns[pool[-1]]


def _resolve_class(ns: dict[str, Any], defined: set[str]) -> Any | None:
    classes = [ns[name] for name in defined if isinstance(ns.get(name), type)]
    return classes[0] if len(classes) == 1 else None


def evaluate_call(
    code: str,
    entry: dict[str, Any],
    args: list[Any],
    kwargs: dict[str, Any],
    timeout_s: float,
) -> tuple[Any | None, str]:
    del timeout_s
    if not code or not code.strip():
        return None, "runtime_error"
    if not isinstance(entry, dict):
        return None, "bad_entry"
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        return None, "bad_request"

    ns: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "__name__": "<miner_code>",
    }
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            exec(compile(code, "<miner_code>", "exec"), ns)
        except ImportError as e:
            if "not available in the grader sandbox" in str(e):
                return None, "forbidden_import"
            return None, "runtime_error"
        except BaseException:
            return None, "runtime_error"
        if not _critical_builtins_intact():
            return None, "tampered"

        try:
            kind = entry.get("kind")
            if kind == "function":
                fn = ns.get(entry["name"])
                if not callable(fn):
                    fn = _resolve_function(ns, code, len(args))
                if not callable(fn):
                    return None, "runtime_error"
            elif kind == "method":
                cls = ns.get(entry["class_name"])
                if not isinstance(cls, type):
                    cls = _resolve_class(ns, _user_defined_names(code))
                if cls is None:
                    return None, "runtime_error"
                fn = getattr(cls(), entry["method"])
            else:
                return None, "bad_entry"
            output = fn(*args, **kwargs)
            if not _critical_builtins_intact():
                return None, "tampered"
            return _json_safe(output), "ok"
        except ImportError as e:
            if "not available in the grader sandbox" in str(e):
                return None, "forbidden_import"
            return None, "runtime_error"
        except TypeError as e:
            if "unsupported output type" in str(e) or "dict key" in str(e) or "non-finite" in str(e):
                return None, "bad_output"
            return None, "runtime_error"
        except BaseException:
            return None, "runtime_error"


# ===== VERBATIM from validator grader/server.py (de-classmethod'd) =====

def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) or isinstance(right, float):
            return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
        return left == right
    if left is None or right is None or isinstance(left, str) or isinstance(right, str):
        return type(left) is type(right) and left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return (
            set(left.keys()) == set(right.keys())
            and all(_json_equal(left[k], right[k]) for k in left)
        )
    return False


def _outputs_match(output: Any, expected: Any, compare: str) -> bool:
    if compare != "exact":
        return False
    return _json_equal(output, expected)


# ===== Driver entrypoint =====

def main() -> None:
    req = json.loads(sys.stdin.read())
    code = req.get("code", "")
    cases = req.get("cases", [])
    passed = 0
    for c in cases:
        output, status = evaluate_call(
            code, c.get("entry", {}), c.get("args", []), c.get("kwargs", {}), 5.0,
        )
        if status == "ok" and _outputs_match(output, c.get("expected"), c.get("compare", "exact")):
            passed += 1
    sys.__stdout__.write(json.dumps({"passed": passed, "total": len(cases)}) + "\n")
    sys.__stdout__.flush()


if __name__ == "__main__":
    main()
