"""Python front end for `agentcost scan`.

Same question as the TypeScript scanner — does each Anthropic call site's cache
breakpoint sit on a prefix big enough to cache — but answered from the stdlib
`ast` parse tree rather than regexes. Python's grammar is available for free, so
call sites, keyword arguments and constant bindings are resolved structurally:
`model=os.environ.get("M", DEFAULT)`, `system=[{..., "cache_control": ...}]`,
`self.MODEL`, `messages.create(**params)` with `params` a dict literal.

Still static analysis only: files are parsed, never imported or executed.

What it adds over the TypeScript side, because Python agents build requests this
way more often:

  - `tools=` counts toward the cached prefix (the API orders the prefix tools ->
    system -> messages), and any request with tools is treated as a lower bound:
    the API adds its own tool-use preamble that no source file shows. A short
    measured prefix on such a request is reported as *possibly* dead (medium),
    never as a CI-failing high.
  - A request assembled at runtime (`create(**kwargs)` from a dict built in
    another function) is reported once as `runtime-built-request` instead of a
    stream of unresolved guesses.
"""

from __future__ import annotations

import ast
import os
import re

try:
    from .scan import CallSite, Finding, analyze_site, normalize_model
except ImportError:  # run as a loose script next to scan.py
    from scan import CallSite, Finding, analyze_site, normalize_model  # type: ignore

# Expressions whose value differs per request. Inside a cached prefix any one of
# them changes the prefix bytes on every call, so the cache never hits.
PY_VOLATILE = re.compile(
    r"\bdatetime\b|\bnow\s*\(|utcnow|\btoday\s*\(|\btime\.time\b|\btime_ns\b|"
    r"monotonic|perf_counter|\buuid|\brandom\b|\bsecrets\.|token_hex|strftime|isoformat",
    re.I,
)

# Interpolating one of these only changes the prefix on deploy.
STABLE_EXPR = re.compile(r"[A-Z][A-Z0-9_]*|[\w.]+\.[A-Z][A-Z0-9_]*")

PASS_THROUGH_METHODS = {"strip", "lstrip", "rstrip"}
MAX_DEPTH = 8


class _Sized:
    """Character count of a string expression, and whether it is exact."""

    __slots__ = ("chars", "exact", "interps")

    def __init__(self, chars: int = 0, exact: bool = True, interps: list | None = None):
        self.chars = chars
        self.exact = exact
        # (line, expression source, is_format_call) for every interpolation seen
        self.interps: list[tuple[int, str, bool]] = interps or []

    def __add__(self, other: "_Sized") -> "_Sized":
        return _Sized(self.chars + other.chars, self.exact and other.exact, self.interps + other.interps)


class _Module:
    """One parsed file: parent links and per-scope assignment tables."""

    def __init__(self, tree: ast.Module):
        self.tree = tree
        self.parent: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parent[child] = node
        self._assigns: dict[ast.AST, dict[str, list[tuple[int, ast.expr]]]] = {}
        self.functions: dict[str, ast.AST] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.setdefault(node.name, node)

    # -- scopes -------------------------------------------------------------

    def scopes(self, node: ast.AST) -> list[ast.AST]:
        """Enclosing scopes, innermost first, ending with the module."""
        out = []
        cur = self.parent.get(node)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                out.append(cur)
            cur = self.parent.get(cur)
        return out

    def assigns(self, scope: ast.AST) -> dict[str, list[tuple[int, ast.expr]]]:
        """Simple `name = value` bindings made directly in `scope` (not nested defs)."""
        if scope in self._assigns:
            return self._assigns[scope]
        table: dict[str, list[tuple[int, ast.expr]]] = {}
        stack = list(getattr(scope, "body", []))
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        table.setdefault(target.id, []).append((node.lineno, node.value))
            elif isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
                table.setdefault(node.target.id, []).append((node.lineno, node.value))
            stack.extend(ast.iter_child_nodes(node))
        for entries in table.values():
            entries.sort(key=lambda e: e[0])
        self._assigns[scope] = table
        return table

    def lookup(self, name: str, at: ast.AST) -> ast.expr | None:
        """The value most recently bound to `name` as seen from `at`.

        Function scopes need the binding to precede the use. Class bodies are not
        visible from their methods, so they are skipped unless innermost. The module
        scope is visible from any function regardless of line order, since the
        function runs after the module has finished executing.
        """
        line = getattr(at, "lineno", 0)
        chain = self.scopes(at)
        in_function = any(isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) for s in chain)
        for i, scope in enumerate(chain):
            if isinstance(scope, ast.ClassDef) and i != 0:
                continue
            entries = self.assigns(scope).get(name)
            if not entries:
                continue
            if isinstance(scope, ast.Module) and in_function:
                return entries[-1][1]
            before = [v for ln, v in entries if ln < line]
            if before:
                return before[-1]
        return None

    def class_attr(self, name: str, at: ast.AST) -> ast.expr | None:
        """`self.NAME` / `cls.NAME` -> the class-body binding of NAME."""
        for scope in self.scopes(at):
            if isinstance(scope, ast.ClassDef):
                entries = self.assigns(scope).get(name)
                return entries[-1][1] if entries else None
        return None

    def single_return(self, fname: str) -> ast.expr | None:
        """The returned expression of a zero-argument helper with one return."""
        fn = self.functions.get(fname)
        if fn is None or fn.args.args or fn.args.posonlyargs or fn.args.kwonlyargs:
            return None
        rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
        return rets[0].value if len(rets) == 1 else None


# ---------------------------------------------------------------------------
# Expression resolution
# ---------------------------------------------------------------------------


def _is_env_read(node: ast.AST) -> bool:
    src = ast.unparse(node)
    return "environ" in src or "getenv" in src


def resolve_model(node: ast.AST, mod: _Module, depth: int = 0) -> tuple[str | None, str]:
    if depth > MAX_DEPTH:
        return None, "unresolved"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return normalize_model(node.value), "literal"
    if isinstance(node, ast.Name):
        value = mod.lookup(node.id, node)
        if value is None:
            return None, "unresolved"
        model, source = resolve_model(value, mod, depth + 1)
        return model, ("const" if source == "literal" else source)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
        value = mod.class_attr(node.attr, node)
        if value is None:
            return None, "unresolved"
        model, source = resolve_model(value, mod, depth + 1)
        return model, ("class-attr" if source == "literal" else source)
    if isinstance(node, ast.Call):
        # os.environ.get("X", DEFAULT) / os.getenv("X", DEFAULT)
        if _is_env_read(node.func):
            default = node.args[1] if len(node.args) >= 2 else next(
                (k.value for k in node.keywords if k.arg == "default"), None)
            if default is None:
                return None, "unresolved"
            model, _ = resolve_model(default, mod, depth + 1)
            return model, "env-default"
        if isinstance(node.func, ast.Name) and not node.args and not node.keywords:
            ret = mod.single_return(node.func.id)
            if ret is not None:
                return resolve_model(ret, mod, depth + 1)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        # os.environ.get("X") or DEFAULT  ->  the fallback is what ships by default
        model, _ = resolve_model(node.values[-1], mod, depth + 1)
        source = "env-default" if any(_is_env_read(v) for v in node.values[:-1]) else "default"
        return model, source
    return None, "unresolved"


def resolve_str(node: ast.AST, mod: _Module, depth: int = 0) -> _Sized | None:
    """Size a string-valued expression; None when its shape is unknown."""
    if depth > MAX_DEPTH:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _Sized(len(node.value))
    if isinstance(node, ast.JoinedStr):
        out = _Sized()
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out = out + _Sized(len(part.value))
            elif isinstance(part, ast.FormattedValue):
                inner = resolve_str(part.value, mod, depth + 1)
                out = out + (inner if inner is not None else _Sized(0, False))
                out.interps.append((node.lineno, ast.unparse(part.value), False))
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = resolve_str(node.left, mod, depth + 1), resolve_str(node.right, mod, depth + 1)
        if left is None and right is None:
            return None
        return (left or _Sized(0, False)) + (right or _Sized(0, False))
    if isinstance(node, ast.Name):
        value = mod.lookup(node.id, node)
        return resolve_str(value, mod, depth + 1) if value is not None else None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
        value = mod.class_attr(node.attr, node)
        return resolve_str(value, mod, depth + 1) if value is not None else None
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in PASS_THROUGH_METHODS and not node.args:
            return resolve_str(func.value, mod, depth + 1)
        if isinstance(func, ast.Attribute) and func.attr == "format":
            base = resolve_str(func.value, mod, depth + 1)
            if base is None:
                return None
            base.exact = False
            base.interps.append((node.lineno, ast.unparse(node), True))
            return base
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "dedent" and len(node.args) == 1:
            return resolve_str(node.args[0], mod, depth + 1)
        if isinstance(func, ast.Name) and not node.args and not node.keywords:
            ret = mod.single_return(func.id)
            if ret is not None:
                return resolve_str(ret, mod, depth + 1)
    return None


def _deref(node: ast.AST | None, mod: _Module, depth: int = 0) -> ast.AST | None:
    """Follow `name` / `self.NAME` bindings to the expression they hold."""
    while depth <= MAX_DEPTH and node is not None:
        if isinstance(node, ast.Name):
            node = mod.lookup(node.id, node)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
            node = mod.class_attr(node.attr, node)
        else:
            return node
        depth += 1
    return None


def _dict_items(node: ast.AST) -> dict[str, ast.expr] | None:
    """String-keyed items of a dict literal or a `dict(k=v)` call."""
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                out[k.value] = v
        return out
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        if node.args or any(k.arg is None for k in node.keywords):
            return None  # dict(other) / dict(**other) copies a runtime mapping
        return {k.arg: k.value for k in node.keywords}
    return None


# ---------------------------------------------------------------------------
# Call sites
# ---------------------------------------------------------------------------


def _getattr_of(node: ast.AST, names: tuple[str, ...]) -> ast.AST | None:
    """`getattr(obj, "name", ...)` with name in `names` -> obj."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
            and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value in names):
        return node.args[0]
    return None


def _is_messages_obj(node: ast.AST, mod: _Module, depth: int = 0) -> bool:
    """`<x>.messages`, `getattr(<x>, "messages")`, or a local name bound to either."""
    if depth > MAX_DEPTH:
        return False
    if isinstance(node, ast.Attribute):
        return node.attr == "messages"
    if _getattr_of(node, ("messages",)) is not None:
        return True
    if isinstance(node, ast.Name):
        value = mod.lookup(node.id, node)
        return value is not None and _is_messages_obj(value, mod, depth + 1)
    return False


def _is_messages_call(node: ast.Call, mod: _Module) -> bool:
    """An Anthropic `messages.create(...)` / `messages.stream(...)` call.

    Also follows local aliases — `api = client.messages; api.create(...)`,
    `api = getattr(client, "messages")`, `fn = api.stream; fn(...)`. A callable
    passed in as a parameter is not followed: its binding is in the caller.
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in ("create", "stream") and _is_messages_obj(func.value, mod)
    if isinstance(func, ast.Name):
        value = mod.lookup(func.id, func)
        if isinstance(value, ast.Attribute):
            return value.attr in ("create", "stream") and _is_messages_obj(value.value, mod)
        owner = _getattr_of(value, ("create", "stream")) if value is not None else None
        return owner is not None and _is_messages_obj(owner, mod)
    return False


class _System:
    def __init__(self):
        self.resolved = False
        self.plain = False
        self.n_blocks = 0
        self.cached_idx: int | None = None
        self.has_cc = False
        self.cached: _Sized | None = None
        self.total: _Sized | None = None
        self.refs: list[str] = []


def analyze_system(node: ast.AST | None, mod: _Module) -> _System:
    sysinfo = _System()
    if node is None:
        sysinfo.resolved = True
        return sysinfo
    if isinstance(node, ast.Name):
        sysinfo.refs.append(node.id)
    target = _deref(node, mod)
    if target is None:
        return sysinfo

    # A helper that builds the block list: follow it when it is zero-arg with
    # one return; otherwise at least learn whether it applies cache_control.
    if isinstance(target, ast.Call) and isinstance(target.func, ast.Name):
        fname = target.func.id
        sysinfo.refs.append(fname)
        ret = mod.single_return(fname) if not target.args and not target.keywords else None
        if ret is not None:
            return _merge_refs(analyze_system(ret, mod), sysinfo.refs)
        fn = mod.functions.get(fname)
        if fn is not None and "cache_control" in ast.unparse(fn):
            sysinfo.resolved = True
            sysinfo.has_cc = True
            sysinfo.cached_idx = 0
            sysinfo.n_blocks = 1
            return sysinfo

    if isinstance(target, (ast.List, ast.Tuple)):
        sysinfo.resolved = True
        sysinfo.n_blocks = len(target.elts)
        sizes: list[_Sized] = []
        for i, elt in enumerate(target.elts):
            items = _dict_items(_deref(elt, mod) or elt)
            if items is None:
                sizes.append(_Sized(0, False))
                continue
            if "cache_control" in items and sysinfo.cached_idx is None:
                sysinfo.cached_idx = i
            text = items.get("text")
            if isinstance(text, ast.Name):
                sysinfo.refs.append(text.id)
            sized = resolve_str(text, mod) if text is not None else None
            sizes.append(sized if sized is not None else _Sized(0, False))
        sysinfo.has_cc = sysinfo.cached_idx is not None
        total = _Sized()
        for s in sizes:
            total = total + s
        sysinfo.total = total
        if sysinfo.has_cc:
            cached = _Sized()
            for s in sizes[: sysinfo.cached_idx + 1]:
                cached = cached + s
            sysinfo.cached = cached
        return sysinfo

    sized = resolve_str(target, mod)
    if sized is not None:
        sysinfo.resolved = True
        sysinfo.plain = True
        sysinfo.n_blocks = 1
        sysinfo.total = sized
    return sysinfo


def _merge_refs(sysinfo: _System, refs: list[str]) -> _System:
    sysinfo.refs = refs + sysinfo.refs
    return sysinfo


def _interp_findings(sized: _Sized, rel: str) -> list[Finding]:
    """Flag interpolations that sit inside a cached prefix."""
    out = []
    seen = set()
    for line, src, is_format in sized.interps:
        if (line, src) in seen:
            continue
        seen.add((line, src))
        short = src if len(src) <= 60 else src[:57] + "..."
        if PY_VOLATILE.search(src):
            out.append(Finding(
                "high", "cache-invalidator", rel, line,
                f"Cached prefix interpolates a per-request value: {short}",
                "A value that changes every request changes the prefix bytes, so the cache never "
                "hits and every call pays full input price. Move it into a block placed after the "
                "cache_control breakpoint.",
            ))
        elif not is_format and STABLE_EXPR.fullmatch(src.strip()):
            out.append(Finding(
                "info", "prefix-interpolation", rel, line,
                f"Cached prefix interpolates {short}",
                "Resolves to a build-time constant, so the prefix only changes on deploy — one cold "
                "write, then normal cache hits.",
            ))
        else:
            out.append(Finding(
                "medium", "prefix-interpolation", rel, line,
                f"Cached prefix interpolates {short}",
                "Could not confirm this is a build-time constant. If it varies per request the "
                "prefix never caches; verify with a runtime usage log.",
            ))
    return out


def scan_python_file(path: str, root: str) -> tuple[list[CallSite], list[Finding]]:
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return [], []
    if "messages" not in text:
        return [], []
    try:
        tree = ast.parse(text, filename=path)
    except (SyntaxError, ValueError, RecursionError):
        return [], []

    rel = os.path.relpath(path, root)
    mod = _Module(tree)
    sites: list[CallSite] = []
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_messages_call(node, mod)):
            continue

        kwargs: dict[str, ast.expr] = {}
        dynamic = False
        for kw in node.keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = kw.value
                continue
            # **params: a dict literal bound in scope is as good as keywords
            items = _dict_items(_deref(kw.value, mod) or kw.value)
            if items is None:
                dynamic = True
            else:
                for k, v in items.items():
                    kwargs.setdefault(k, v)

        # Same method name, different SDK (Twilio's SMS client is the classic one):
        # a fully visible call with none of the Messages API's own arguments is not ours.
        if not dynamic and not kwargs.keys() & {"model", "messages", "max_tokens", "system"}:
            continue

        model, msource = (None, "unresolved")
        if "model" in kwargs:
            model, msource = resolve_model(kwargs["model"], mod)
        elif dynamic:
            msource = "runtime-kwargs"

        if dynamic and "model" not in kwargs and "system" not in kwargs:
            sites.append(CallSite(file=rel, line=node.lineno, model=None, model_source=msource))
            findings.append(Finding(
                "info", "runtime-built-request", rel, node.lineno,
                "Request is assembled at runtime (**kwargs), so model and cache layout are not visible here",
                "Static analysis cannot size a prefix it cannot see. Log input_tokens / "
                "cache_creation_input_tokens / cache_read_input_tokens per call for this route instead.",
            ))
            continue

        sysinfo = analyze_system(kwargs.get("system"), mod)

        # The API caches tools -> system -> messages, so tools sit inside any
        # system breakpoint's prefix. Requests with tools also carry a tool-use
        # preamble the source never shows, so their measured prefix is a floor.
        tools_node = kwargs.get("tools")
        tools_sized = None
        if tools_node is not None:
            tools_target = _deref(tools_node, mod)
            if isinstance(tools_target, (ast.List, ast.Tuple)):
                tools_sized = _Sized(len(ast.unparse(tools_target)), False)
            else:
                tools_sized = _Sized(0, False)

        cached = sysinfo.cached
        if cached is not None and tools_sized is not None:
            cached = tools_sized + cached

        site = CallSite(
            file=rel,
            line=node.lineno,
            model=model,
            model_source=msource,
            system_refs=sorted(set(sysinfo.refs)),
            cached_block_index=sysinfo.cached_idx,
            n_system_blocks=sysinfo.n_blocks,
            cached_chars=cached.chars if cached is not None else None,
            system_chars=sysinfo.total.chars if sysinfo.total is not None else None,
            system_is_plain_string=sysinfo.plain,
            has_cache_control=sysinfo.has_cc,
            cached_is_lower_bound=cached is not None and not cached.exact,
            system_is_lower_bound=sysinfo.total is not None and not sysinfo.total.exact,
        )
        sites.append(site)

        if "system" in kwargs and not sysinfo.resolved:
            findings.append(Finding(
                "info", "runtime-built-system", rel, node.lineno,
                "system= is built at runtime, so its cache layout is not visible here",
                f"Referenced: {', '.join(sorted(set(sysinfo.refs))) or 'an expression'}. "
                "Verify this route's caching with a runtime usage log.",
            ))
            if model is None:
                findings.extend(analyze_site(site, None))
            continue

        findings.extend(analyze_site(site, model))
        if cached is not None:
            findings.extend(_interp_findings(cached, rel))

    return sites, findings
