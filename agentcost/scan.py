#!/usr/bin/env python3
"""agentcost — static cost analyzer for LLM agent codebases.

Finds Anthropic `messages.create` call sites in TypeScript/JavaScript (regex
front end, below) and Python (`ast` front end, pyscan.py), works out which model
each one uses and how its `system` blocks are laid out, then reports
where prompt caching is silently not working.

The failure this exists to catch: `cache_control` on a prefix shorter than the
model's minimum cacheable length. The API does not error — it just doesn't cache
(`cache_creation_input_tokens: 0`), so the breakpoint is inert and every request
pays full input price forever. Nothing in the response tells you.

Static analysis only: read-only, no API key, no network, never imports or runs
the code it scans. Token counts are char/4 estimates, so treat anything within
~20% of a threshold as "measure it properly". Prompt bodies are never emitted —
only names, sizes and line numbers — so reports are safe to share.

Usage:
    python scan.py <path> [<path> ...] [--json out.json] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Iterable

# ---------------------------------------------------------------------------
# Model reference data
# ---------------------------------------------------------------------------

# Minimum cacheable prefix in tokens. Below this the API silently declines to
# cache. NOT monotonic across generations — newer models are cheaper to cache,
# so a prompt that caches on Opus 5 may not cache on Opus 4.6 or Haiku 4.5.
MODEL_MIN_PREFIX = {
    "claude-opus-5": 512,
    "claude-fable-5": 512,
    "claude-mythos-5": 512,
    "claude-opus-4-8": 1024,
    "claude-sonnet-5": 1024,
    "claude-sonnet-4-6": 1024,
    "claude-sonnet-4-5": 1024,
    "claude-opus-4-1": 1024,
    "claude-opus-4-0": 1024,
    "claude-sonnet-4-0": 1024,
    "claude-opus-4-7": 2048,
    "claude-mythos-preview": 2048,
    "claude-3-5-haiku": 2048,
    "claude-opus-4-6": 4096,
    "claude-opus-4-5": 4096,
    "claude-haiku-4-5": 4096,
}

# USD per million tokens (input, output), first-party list price.
MODEL_PRICING = {
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Tier ranking, for flagging routes that look over-modeled.
MODEL_TIER = {
    "claude-haiku-4-5": 1,
    "claude-sonnet-4-5": 2,
    "claude-sonnet-4-6": 2,
    "claude-sonnet-5": 2,
    "claude-opus-4-5": 3,
    "claude-opus-4-6": 3,
    "claude-opus-4-7": 3,
    "claude-opus-4-8": 3,
    "claude-opus-5": 3,
    "claude-fable-5": 4,
    "claude-mythos-5": 4,
}

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

CHARS_PER_TOKEN = 4  # crude but adequate for threshold comparison

# Interpolation inside a cached prefix only breaks caching if the interpolated
# value varies per request. A build-time constant (an imported roster, a config
# object) changes on deploy, which is a cache miss you take once, not a bug.
VOLATILE_EXPR = re.compile(
    r"\bDate\b|\bnow\s*\(|toISOString|Math\.random|uuid|randomUUID|"
    r"timestamp|\bDate\.now\b|performance\.now|process\.hrtime",
    re.I,
)

SOURCE_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py")
SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", ".next", "coverage", ".claude",
    "__pycache__", ".venv", "venv", ".tox", "site-packages", ".mypy_cache", ".pytest_cache",
}
PROJECT_MARKERS = ("package.json", "pyproject.toml", "setup.py", "setup.cfg")


def normalize_model(raw: str) -> str:
    """Strip date suffixes and quotes so 'claude-haiku-4-5-20251001' matches."""
    m = raw.strip().strip("\"'`")
    m = re.sub(r"-\d{8}$", "", m)
    return m


def min_prefix_for(model: str) -> int | None:
    model = normalize_model(model)
    if model in MODEL_MIN_PREFIX:
        return MODEL_MIN_PREFIX[model]
    # Prefix match for unknown dated/variant ids
    for known, val in MODEL_MIN_PREFIX.items():
        if model.startswith(known):
            return val
    return None


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@dataclass
class Finding:
    severity: str
    category: str
    file: str
    line: int
    summary: str
    detail: str = ""


@dataclass
class CallSite:
    file: str
    line: int
    model: str | None
    model_source: str  # "literal", "const", "env-default", "unresolved"
    system_refs: list[str] = field(default_factory=list)
    cached_block_index: int | None = None
    n_system_blocks: int = 0
    cached_chars: int | None = None
    system_chars: int | None = None
    system_is_plain_string: bool = False
    has_cache_control: bool = False
    # True when part of the prefix could not be sized (tools, the API's tool-use
    # preamble, runtime interpolation): the measured size is a floor, not a value.
    cached_is_lower_bound: bool = False
    system_is_lower_bound: bool = False

    @property
    def cached_tokens(self) -> int | None:
        if self.cached_chars is None:
            return None
        return self.cached_chars // CHARS_PER_TOKEN

    @property
    def system_tokens(self) -> int | None:
        if self.system_chars is None:
            return None
        return self.system_chars // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Source scanning
# ---------------------------------------------------------------------------


def project_label(path: str) -> str:
    """Name a scan root by its nearest package.json / pyproject.toml ancestor.

    Scanning `.../my-agent/src` should report as `my-agent`, not `src`.
    """
    cur = os.path.abspath(path)
    for _ in range(6):
        if any(os.path.exists(os.path.join(cur, m)) for m in PROJECT_MARKERS):
            return os.path.basename(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.basename(os.path.normpath(path))


def iter_source_files(root: str) -> Iterable[str]:
    if os.path.isfile(root):
        if root.endswith(SOURCE_EXTS):
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(SOURCE_EXTS):
                yield os.path.join(dirpath, fn)


def extract_string_consts(text: str) -> dict[str, str]:
    """Map const name -> literal body for template literals and quoted strings.

    Only captures single-expression declarations. A template literal containing
    ${...} interpolation is captured as-is; interpolation is reported separately
    because it is the classic cache invalidator.
    """
    consts: dict[str, str] = {}
    for m in re.finditer(r"(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*`((?:[^`\\]|\\.)*)`", text, re.S):
        consts[m.group(1)] = m.group(2)
    for m in re.finditer(r"(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*\"((?:[^\"\\]|\\.)*)\"", text):
        consts.setdefault(m.group(1), m.group(2))
    for m in re.finditer(r"(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*'((?:[^'\\]|\\.)*)'", text):
        consts.setdefault(m.group(1), m.group(2))
    return consts


def resolve_model_expr(expr: str, consts: dict[str, str], text: str) -> tuple[str | None, str]:
    """Resolve the value of a `model:` expression.

    Handles: string literal, const reference, `process.env.X || DEFAULT`, and a
    helper function whose body returns one of those.
    """
    expr = expr.strip().rstrip(",")

    lit = re.fullmatch(r"[\"'`]([\w.\-]+)[\"'`]", expr)
    if lit:
        return normalize_model(lit.group(1)), "literal"

    # process.env.FOO || "model"  /  process.env.FOO ?? DEFAULT_MODEL
    envm = re.search(r"process\.env\.\w+\s*(?:\|\||\?\?)\s*(.+)$", expr)
    if envm:
        inner, _ = resolve_model_expr(envm.group(1), consts, text)
        return inner, "env-default"

    # bare identifier -> const lookup
    ident = re.fullmatch(r"(\w+)", expr)
    if ident:
        name = ident.group(1)
        if name in consts:
            return normalize_model(consts[name]), "const"
        # a getter function: `function name(): string { ... return X }`.
        # The optional `: ReturnType` annotation between ) and { must be allowed
        # or TypeScript declarations never match.
        fn = re.search(
            r"(?:function\s+%s\s*\([^)]*\)|const\s+%s\s*=\s*\([^)]*\)\s*=>)"
            r"\s*(?::\s*[\w<>\[\]|&.\s]+?)?\s*\{(.{0,400}?)\}"
            % (re.escape(name), re.escape(name)),
            text,
            re.S,
        )
        if fn:
            ret = re.search(r"return\s+(.+?);", fn.group(1), re.S)
            if ret:
                return resolve_model_expr(ret.group(1), consts, text)
        return None, "unresolved"

    # function call, e.g. draftModel()
    call = re.fullmatch(r"(\w+)\s*\(\s*\)", expr)
    if call:
        return resolve_model_expr(call.group(1), consts, text)

    return None, "unresolved"


def nearest_binding(text: str, before: int, name: str) -> str | None:
    """Find the value expression of the closest `const/let name = ...` above `before`."""
    best = None
    for m in re.finditer(rf"(?:const|let|var)\s+{re.escape(name)}\s*(?::[^=]+)?=\s*(.+?);", text):
        if m.start() < before:
            best = m.group(1)
        else:
            break
    return best


def resolve_system_helper(
    name: str, text: str, consts: dict[str, str]
) -> tuple[bool, int | None, list[str]]:
    """Inspect a helper like `systemBlocks(x)` that builds the system[] array.

    Returns (helper applies cache_control, measured cached chars, identifiers seen).
    Without this, any call site delegating to a helper looks like it has no
    breakpoint at all — the single largest source of false positives.
    """
    fn = re.search(
        rf"(?:function\s+{re.escape(name)}\s*\((.*?)\)|"
        rf"const\s+{re.escape(name)}\s*=\s*\((.*?)\)\s*=>)"
        r"\s*(?::\s*[\w<>\[\]|&.\s]+?)?\s*\{",
        text,
        re.S,
    )
    if not fn:
        return False, None, []
    body = balanced_slice(text, text.index("{", fn.end() - 1), "{", "}")
    params = (fn.group(1) or fn.group(2) or "")

    has_cc = "cache_control" in body
    idents: list[str] = []

    # Default parameter value is usually the cached prefix:
    #   function systemBlocks(staticText: string = AGENT_KB)
    dm = re.search(r"=\s*([A-Za-z_]\w*)", params)
    if dm:
        idents.append(dm.group(1))

    # Any ALL_CAPS consts referenced in the body
    idents.extend(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", body))

    return has_cc, measure_ref(idents, consts), sorted(set(idents))


def balanced_slice(text: str, start: int, opener: str = "(", closer: str = ")") -> str:
    """Return the substring from `start` through the matching closer."""
    depth = 0
    i = start
    in_str: str | None = None
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in "\"'`":
            in_str = ch
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    return text[start : start + 4000]


def parse_system_arg(block: str) -> tuple[list[str], int | None, int, bool, bool]:
    """Inspect the `system:` argument.

    Returns (referenced identifiers, index of the cache_control block,
    total block count, is_plain_string, has_cache_control).
    """
    sysm = re.search(r"\bsystem\s*:\s*", block)
    if not sysm:
        return [], None, 0, False, False

    tail = block[sysm.end() :].lstrip()
    if tail.startswith("["):
        arr = balanced_slice(tail, 0, "[", "]")
    else:
        # system: SOME_CONST  or  system: `...`  or  system: fn(...)
        stop = re.search(r",\s*\n|\n\s*\}", tail)
        arr = tail[: stop.start()] if stop else tail[:600]
        idents = re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", arr)
        return idents, None, 1, True, False

    # Split top-level `{...}` entries of the array
    entries: list[str] = []
    depth = 0
    cur = ""
    in_str: str | None = None
    for ch in arr[1:-1]:
        if in_str:
            cur += ch
            if ch == in_str:
                in_str = None
            continue
        if ch in "\"'`":
            in_str = ch
            cur += ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            cur += ch
            if depth == 0:
                entries.append(cur)
                cur = ""
            continue
        if depth > 0:
            cur += ch
    if cur.strip():
        entries.append(cur)

    cached_idx = None
    idents: list[str] = []
    for i, e in enumerate(entries):
        if "cache_control" in e:
            cached_idx = i if cached_idx is None else cached_idx
        tm = re.search(r"\btext\s*:\s*(.+)$", e, re.S)
        if tm:
            idents.extend(re.findall(r"\b([A-Za-z_]\w*)\b", tm.group(1))[:3])
    has_cc = cached_idx is not None
    return idents, cached_idx, len(entries), False, has_cc


def measure_ref(refs: list[str], consts: dict[str, str]) -> int | None:
    """Sum the char length of whichever referenced identifiers resolve to consts."""
    total = 0
    hit = False
    for r in refs:
        if r in consts:
            total += len(consts[r])
            hit = True
    return total if hit else None


def scan_file(path: str, root: str) -> tuple[list[CallSite], list[Finding]]:
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return [], []

    if "messages.create" not in text and "messages.stream" not in text:
        return [], []

    rel = os.path.relpath(path, root)
    consts = extract_string_consts(text)
    sites: list[CallSite] = []
    findings: list[Finding] = []

    for m in re.finditer(r"messages\.(?:create|stream)\s*\(", text):
        start = m.end() - 1
        block = balanced_slice(text, start)
        line = text.count("\n", 0, m.start()) + 1

        model, msource = (None, "unresolved")
        mm = re.search(r"\bmodel\s*:\s*(.+?)(?:,\s*\n|\n)", block)
        if mm:
            model, msource = resolve_model_expr(mm.group(1), consts, text)
        elif re.search(r"^\s*model\s*(?:,|$)", block, re.M):
            # ES shorthand `model,` — resolve the nearest preceding binding of
            # `model` in the enclosing scope (commonly
            # `const model = process.env.X || DEFAULT_MODEL`).
            expr = nearest_binding(text, m.start(), "model")
            if expr:
                model, _ = resolve_model_expr(expr, consts, text)
                msource = "shorthand"
            else:
                msource = "unresolved-shorthand"

        refs, cached_idx, nblocks, plain, has_cc = parse_system_arg(block)
        cached_chars = None

        # `system: buildBlocks(...)` — follow the helper before concluding there
        # is no breakpoint, otherwise every delegated call site is a false alarm.
        helper = re.search(r"\bsystem\s*:\s*([A-Za-z_]\w*)\s*\(", block)
        if helper and not has_cc:
            h_cc, h_chars, h_idents = resolve_system_helper(helper.group(1), text, consts)
            if h_cc:
                has_cc = True
                cached_idx = 0
                cached_chars = h_chars
                refs = h_idents
                plain = False

        if has_cc and refs and cached_chars is None:
            cached_chars = measure_ref(refs, consts)

        # Size the system prompt even when uncached: whether adding a breakpoint
        # would help at all depends on it clearing the model's minimum.
        system_chars = measure_ref(refs, consts)

        site = CallSite(
            file=rel,
            line=line,
            model=model,
            model_source=msource,
            system_refs=sorted(set(refs)),
            cached_block_index=cached_idx,
            n_system_blocks=nblocks,
            cached_chars=cached_chars,
            system_chars=system_chars,
            system_is_plain_string=plain,
            has_cache_control=has_cc,
        )
        sites.append(site)
        findings.extend(analyze_site(site, model))

    # Interpolation inside a const that is used as a cached prefix is the
    # classic silent invalidator — flag the const, not the call site.
    for name, body in consts.items():
        exprs = re.findall(r"\$\{([^}]*)\}", body)
        if not exprs:
            continue
        if not any(name in s.system_refs and s.has_cache_control for s in sites):
            continue
        line = text.count("\n", 0, text.find(name)) + 1

        volatile = [e for e in exprs if VOLATILE_EXPR.search(e)]
        if volatile:
            findings.append(
                Finding(
                    "high",
                    "cache-invalidator",
                    rel,
                    line,
                    f"{name} is a cached prefix and interpolates a per-request value: "
                    f"{', '.join(sorted(set(volatile))[:3])}",
                    "A value that changes every request changes the prefix bytes, so the "
                    "cache never hits and every call pays full input price. Move it into a "
                    "block placed after the cache_control breakpoint.",
                )
            )
            continue

        # Interpolation of a build-time constant only busts the cache on deploy.
        stable = all(re.fullmatch(r"[A-Z][A-Z0-9_]*|\w+\.\w+", e.strip()) for e in exprs)
        findings.append(
            Finding(
                "info" if stable else "medium",
                "prefix-interpolation",
                rel,
                line,
                f"{name} is a cached prefix interpolating {', '.join(sorted(set(exprs))[:3])}",
                "Resolves to a build-time constant, so the prefix only changes on deploy — "
                "one cold write, then normal cache hits."
                if stable
                else "Could not confirm these are build-time constants. If any varies per "
                "request the prefix never caches; verify with a runtime usage log.",
            )
        )

    return sites, findings


def analyze_site(s: CallSite, model: str | None) -> list[Finding]:
    out: list[Finding] = []

    if model is None:
        out.append(
            Finding(
                "info",
                "unresolved-model",
                s.file,
                s.line,
                "Could not statically resolve the model for this call site",
                "Cache thresholds are model-specific, so this site was not checked. "
                "Pass the model as a literal or a simple const to make it analyzable.",
            )
        )
        return out

    minpfx = min_prefix_for(model)

    if not s.has_cache_control:
        if not (s.system_is_plain_string or s.n_system_blocks):
            return out
        tok = s.system_tokens
        if minpfx and tok is not None and tok < minpfx and s.system_is_lower_bound:
            out.append(
                Finding(
                    "info",
                    "no-cache-breakpoint",
                    s.file,
                    s.line,
                    f"No cache_control on a {model} call; system prompt is at least ~{tok:,} tok",
                    "Part of the system prompt could not be sized, so whether a breakpoint "
                    "would clear the model's minimum is unknown. Measure with count_tokens.",
                )
            )
            return out
        # A breakpoint below the model minimum is inert, so telling someone to add
        # one on a small prompt is bad advice rather than a finding.
        if minpfx and tok is not None and tok < minpfx:
            out.append(
                Finding(
                    "info",
                    "too-small-to-cache",
                    s.file,
                    s.line,
                    f"System prompt ~{tok:,} tok is below {model}'s {minpfx:,}-tok minimum",
                    "Correctly has no breakpoint — adding one would do nothing. Making this "
                    "route cacheable would mean sharing a common preamble with other routes "
                    "so the combined prefix clears the minimum.",
                )
            )
        else:
            size = f" (~{tok:,} tok)" if tok is not None else ""
            out.append(
                Finding(
                    "medium",
                    "no-cache-breakpoint",
                    s.file,
                    s.line,
                    f"No cache_control on a {model} call with a system prompt{size}",
                    "If this route is called repeatedly with a stable system prompt, a "
                    "breakpoint on the last stable block bills the prefix at 0.1x on "
                    "every subsequent call. " + savings_note(model, tok or 0),
                )
            )
        return out

    if s.cached_tokens is None:
        out.append(
            Finding(
                "info",
                "unmeasured-prefix",
                s.file,
                s.line,
                "cache_control present but prefix size could not be measured",
                f"Referenced identifiers: {', '.join(s.system_refs) or 'none resolved'}. "
                "Likely built at runtime; verify with a runtime usage log.",
            )
        )
        return out

    tok = s.cached_tokens
    if minpfx and tok < minpfx and s.cached_is_lower_bound:
        # A floor below the minimum proves nothing: the unsized part may clear it.
        # Never fail CI on it.
        out.append(
            Finding(
                "medium",
                "possibly-dead-breakpoint",
                s.file,
                s.line,
                f"Measured part of the cached prefix ~{tok:,} tok is below {model}'s "
                f"{minpfx:,}-tok minimum",
                "Some of the prefix could not be sized statically (tools, the API's "
                "tool-use preamble, or runtime interpolation), so the real prefix may "
                "still clear the minimum. Confirm with count_tokens or a usage log: "
                "cache_creation_input_tokens stuck at 0 means the breakpoint is dead.",
            )
        )
    elif minpfx and tok < minpfx:
        out.append(
            Finding(
                "high",
                "dead-breakpoint",
                s.file,
                s.line,
                f"Cached prefix ~{tok:,} tok is below {model}'s {minpfx:,}-tok minimum",
                "The API will silently decline to cache this prefix — no error, "
                "cache_creation_input_tokens stays 0, and every call pays full input "
                "price. Either consolidate this prompt with a shared cached preamble to "
                "clear the minimum, or drop the breakpoint as dead code.",
            )
        )
    elif minpfx and tok < minpfx * 1.2:
        out.append(
            Finding(
                "medium",
                "near-minimum-prefix",
                s.file,
                s.line,
                f"Cached prefix ~{tok:,} tok is within 20% of {model}'s {minpfx:,}-tok minimum",
                "Char/4 estimation has roughly this much error, so this may already be "
                "failing to cache. Confirm with count_tokens or a runtime usage log "
                "before trusting it.",
            )
        )

    if s.cached_block_index is not None and s.n_system_blocks > 1:
        if s.cached_block_index != 0:
            out.append(
                Finding(
                    "medium",
                    "breakpoint-ordering",
                    s.file,
                    s.line,
                    f"cache_control sits on block {s.cached_block_index} of {s.n_system_blocks}",
                    "Everything before the breakpoint is cached with it. Stable content "
                    "should come first and the breakpoint should sit on the last stable "
                    "block, with volatile content after it.",
                )
            )

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def savings_note(model: str, tokens: int) -> str:
    p = MODEL_PRICING.get(normalize_model(model))
    if not p or not tokens:
        return ""
    inp = p[0]
    full = tokens * inp / 1_000_000
    cached = tokens * inp * CACHE_READ_MULTIPLIER / 1_000_000
    return f"${full:.4f} -> ${cached:.4f} per call once cached"


def report(sites: list[CallSite], findings: list[Finding], quiet: bool = False) -> None:
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.file, f.line))

    print()
    print("=" * 78)
    print(" agentcost — static prompt-cache & model-tier scan")
    print("=" * 78)

    by_model: dict[str, int] = {}
    for s in sites:
        by_model[s.model or "unresolved"] = by_model.get(s.model or "unresolved", 0) + 1

    print(f"\n{len(sites)} call site(s) across {len({s.file for s in sites})} file(s)")
    for mdl, n in sorted(by_model.items(), key=lambda kv: -kv[1]):
        pricing = MODEL_PRICING.get(mdl)
        px = f"  (${pricing[0]:g}/${pricing[1]:g} per MTok)" if pricing else ""
        mn = min_prefix_for(mdl)
        mnx = f"  min cacheable prefix {mn:,} tok" if mn else ""
        print(f"  {n:3}x  {mdl}{px}{mnx}")

    cached = [s for s in sites if s.has_cache_control]
    print(f"\ncache_control present on {len(cached)}/{len(sites)} call sites")

    # Per-project model tiers. Static analysis can't tell you a route is
    # over-modeled — only measured pass rates can — but showing which project
    # runs which tier surfaces inversions worth a look.
    per_project: dict[str, dict[str, int]] = {}
    for s in sites:
        proj = s.file.split("/", 1)[0]
        per_project.setdefault(proj, {})
        key = s.model or "unresolved"
        per_project[proj][key] = per_project[proj].get(key, 0) + 1

    if len(per_project) > 1:
        print("\nmodel tier by project:")
        for proj, models in sorted(per_project.items()):
            parts = []
            for mdl, n in sorted(models.items(), key=lambda kv: -kv[1]):
                tier = MODEL_TIER.get(mdl)
                tname = {1: "haiku", 2: "sonnet", 3: "opus", 4: "frontier"}.get(tier, "?")
                parts.append(f"{n}x {mdl} [{tname}]")
            print(f"  {proj:22} {', '.join(parts)}")

    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    order = ["high", "medium", "low", "info"]
    print("findings: " + ", ".join(f"{counts.get(k,0)} {k}" for k in order if counts.get(k)))

    if not findings:
        print("\nNo cache or tier issues detected.")
        return

    print("\n" + "-" * 78)
    for f in findings:
        if quiet and f.severity == "info":
            continue
        tag = f.severity.upper()
        print(f"\n[{tag}] {f.category}  —  {f.file}:{f.line}")
        print(f"  {f.summary}")
        if f.detail:
            for chunk in wrap(f.detail, 74):
                print(f"    {chunk}")

    print("\n" + "-" * 78)
    print(
        "\nStatic analysis with char/4 token estimates. It cannot tell you whether a\n"
        "well-formed breakpoint actually fires at runtime — only a usage log can.\n"
        "Log input_tokens / output_tokens / cache_creation_input_tokens /\n"
        "cache_read_input_tokens per call and compare against these findings."
    )


def wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Static prompt-cache analyzer for agent codebases")
    ap.add_argument("paths", nargs="+", help="project roots or files to scan")
    ap.add_argument("--json", dest="json_out", help="write findings to a JSON file")
    ap.add_argument("--quiet", action="store_true", help="suppress info-level findings")
    args = ap.parse_args(argv)

    try:
        from .pyscan import scan_python_file
    except ImportError:  # run as a loose script
        from pyscan import scan_python_file  # type: ignore

    all_sites: list[CallSite] = []
    all_findings: list[Finding] = []

    for p in args.paths:
        if not os.path.exists(p):
            print(f"skip (not found): {p}", file=sys.stderr)
            continue
        root = p if os.path.isdir(p) else os.path.dirname(p) or "."
        label = project_label(root)
        for f in iter_source_files(p):
            scanner = scan_python_file if f.endswith(".py") else scan_file
            sites, findings = scanner(f, root)
            for s in sites:
                s.file = f"{label}/{s.file}".replace("\\", "/")
            for fd in findings:
                fd.file = f"{label}/{fd.file}".replace("\\", "/")
            all_sites.extend(sites)
            all_findings.extend(findings)

    report(all_sites, all_findings, quiet=args.quiet)

    if args.json_out:
        payload = {
            "call_sites": [asdict(s) for s in all_sites],
            "findings": [asdict(f) for f in all_findings],
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nJSON written to {args.json_out}")

    return 1 if any(f.severity == "high" for f in all_findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
