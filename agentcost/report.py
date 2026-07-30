#!/usr/bin/env python3
"""agentcost — aggregate a runtime usage log into per-route cache economics.

Answers the question static analysis cannot: is the cache actually being hit,
per route, in production.

Two input sources:

  collector.ts logs   Routes group by `prefix_hash`, so a route showing many
                      distinct hashes is a route whose prefix is varying —
                      the invalidator itself, visible without reading a prompt.

  --claude-code       Claude Code writes the same four usage counters to
                      ~/.claude/projects/**/*.jsonl already. No instrumentation
                      needed, so this is the zero-setup way to see real cache
                      economics before wiring anything up. Records are
                      deduplicated by message id: resumed sessions replay
                      earlier turns into later transcripts, and counting those
                      twice inflates every total.

Usage:
    python report.py <log.jsonl> [--since 2026-07-01] [--json out.json]
    python report.py --claude-code [DIR] [--since 2026-07-01]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

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
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25


def price(model: str | None) -> tuple[float, float]:
    if not model:
        return (0.0, 0.0)
    base = model
    for known in MODEL_PRICING:
        if base.startswith(known):
            return MODEL_PRICING[known]
    return (0.0, 0.0)


def cost_of(rec: dict) -> tuple[float, float]:
    """Return (actual cost, hypothetical cost with no caching at all)."""
    pin, pout = price(rec.get("model"))
    if not pin:
        return (0.0, 0.0)
    fresh = rec.get("input_tokens", 0) or 0
    created = rec.get("cache_creation_input_tokens", 0) or 0
    read = rec.get("cache_read_input_tokens", 0) or 0
    out = rec.get("output_tokens", 0) or 0

    actual = (
        fresh * pin
        + created * pin * CACHE_WRITE_MULT
        + read * pin * CACHE_READ_MULT
        + out * pout
    ) / 1_000_000
    uncached = ((fresh + created + read) * pin + out * pout) / 1_000_000
    return (actual, uncached)


def default_claude_code_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".claude", "projects")


def iter_claude_code(root: str, since: str | None = None):
    """Yield collector-shaped records from Claude Code transcripts.

    Deduplicated by message id. A resumed session replays earlier assistant
    turns into the new transcript, so the same message can appear in several
    files; counting each occurrence overstates both tokens and cost.
    """
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if not fn.endswith(".jsonl"):
                continue
            path = os.path.join(dirpath, fn)
            # Attribute to the top-level project directory, not the immediate
            # parent: Claude Code nests subagent transcripts in hash-named
            # subdirectories, which would otherwise each look like a project.
            rel = os.path.relpath(dirpath, root)
            project = rel.split(os.sep)[0] if rel != "." else os.path.basename(root)
            try:
                fh = open(path, encoding="utf-8", errors="ignore")
            except OSError:
                continue
            with fh:
                for line in fh:
                    if '"usage"' not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = rec.get("message")
                    if not isinstance(msg, dict):
                        continue
                    usage = msg.get("usage")
                    if not isinstance(usage, dict):
                        continue

                    key = msg.get("id") or rec.get("requestId")
                    if not key or key in seen:
                        continue
                    seen.add(key)

                    ts = rec.get("timestamp") or ""
                    if since and ts and ts < since:
                        continue

                    yield {
                        "ts": ts,
                        "model": msg.get("model"),
                        # No prefix hash in Claude Code logs — the useful unit
                        # is the project, which is one working context.
                        "prefix_hash": shorten_project(project),
                        "prefix_chars": 0,
                        "input_tokens": usage.get("input_tokens", 0) or 0,
                        "output_tokens": usage.get("output_tokens", 0) or 0,
                        "cache_creation_input_tokens": usage.get(
                            "cache_creation_input_tokens", 0
                        )
                        or 0,
                        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0)
                        or 0,
                    }


def shorten_project(name: str) -> str:
    """Claude Code encodes the full cwd as the directory name; keep the tail.

    A worktree path ends in `<project>--claude-worktrees-<branch>`; report it as
    `project:branch` so worktree sessions don't collapse into their parent.
    """
    parts = [p for p in name.replace("--", "-").split("-") if p]
    if not parts:
        return name[:22]
    if "worktrees" in parts:
        i = parts.index("worktrees")
        branch = "-".join(parts[i + 1 :]) or "wt"
        base = parts[i - 2] if i >= 2 else "?"
        return f"{base}:{branch}"[:22]
    return parts[-1][:22]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate agentcost runtime logs")
    ap.add_argument("log", nargs="?", help="JSONL written by collector.ts")
    ap.add_argument(
        "--claude-code",
        nargs="?",
        const=default_claude_code_dir(),
        metavar="DIR",
        help="read Claude Code transcripts instead (default ~/.claude/projects)",
    )
    ap.add_argument("--since", help="ISO date lower bound, e.g. 2026-07-01")
    ap.add_argument("--json", dest="json_out", help="write aggregate to JSON")
    args = ap.parse_args(argv)

    if not args.log and not args.claude_code:
        ap.error("pass a collector log path, or --claude-code")

    if args.log and not os.path.exists(args.log):
        print(f"no such log: {args.log}", file=sys.stderr)
        return 2
    if args.claude_code and not os.path.isdir(args.claude_code):
        print(f"no such directory: {args.claude_code}", file=sys.stderr)
        return 2

    routes: dict[str, dict] = defaultdict(
        lambda: {
            "calls": 0,
            "model": None,
            "prefix_chars": 0,
            "read": 0,
            "created": 0,
            "fresh": 0,
            "out": 0,
            "actual": 0.0,
            "uncached": 0.0,
        }
    )
    total_calls = 0

    def iter_collector_log(path: str):
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if args.since and (rec.get("ts") or "") < args.since:
                    continue
                yield rec

    if args.claude_code:
        source = iter_claude_code(args.claude_code, args.since)
        source_label = f"Claude Code transcripts ({args.claude_code})"
    else:
        source = iter_collector_log(args.log)
        source_label = args.log

    for rec in source:
        key = rec.get("prefix_hash") or "no-cached-prefix"
        r = routes[key]
        r["calls"] += 1
        # Track model by call count and report the dominant one. Taking the last
        # value lets Claude Code's "<synthetic>" placeholder records — which
        # carry no real usage — overwrite the real model name.
        mdl = rec.get("model")
        if mdl and not mdl.startswith("<"):
            r.setdefault("models", {})
            r["models"][mdl] = r["models"].get(mdl, 0) + 1
            r["model"] = max(r["models"].items(), key=lambda kv: kv[1])[0]
        r["prefix_chars"] = max(r["prefix_chars"], rec.get("prefix_chars", 0) or 0)
        r["read"] += rec.get("cache_read_input_tokens", 0) or 0
        r["created"] += rec.get("cache_creation_input_tokens", 0) or 0
        r["fresh"] += rec.get("input_tokens", 0) or 0
        r["out"] += rec.get("output_tokens", 0) or 0
        a, u = cost_of(rec)
        r["actual"] += a
        r["uncached"] += u
        total_calls += 1

    if not total_calls:
        print("no usable records found")
        return 1

    print()
    print("=" * 82)
    print(" agentcost — runtime cache economics")
    print("=" * 82)
    unit = "projects" if args.claude_code else "distinct cached prefixes"
    print(f"\nsource: {source_label}")
    print(f"{total_calls:,} calls across {len(routes)} {unit}\n")

    label = "project" if args.claude_code else "prefix"
    hdr = f"{label:18} {'model':20} {'calls':>6} {'hit%':>6} {'cost':>10} {'saved':>10}"
    print(hdr)
    print("-" * len(hdr))

    findings: list[str] = []
    grand_actual = grand_uncached = 0.0

    for key, r in sorted(routes.items(), key=lambda kv: -kv[1]["actual"]):
        prompt = r["read"] + r["created"] + r["fresh"]
        hit = 100 * r["read"] / prompt if prompt else 0.0
        saved = r["uncached"] - r["actual"]
        grand_actual += r["actual"]
        grand_uncached += r["uncached"]
        # Cost is computed per record, so a mixed-model route is priced
        # correctly — but showing one model name would imply it was uniform.
        others = len(r.get("models", {})) - 1
        mdl_label = (r["model"] or "?") + (f" +{others}" if others > 0 else "")
        print(
            f"{key[:18]:18} {mdl_label[:20]:20} {r['calls']:6,} "
            f"{hit:5.1f}% ${r['actual']:9.4f} ${saved:9.4f}"
        )

        # A route called repeatedly that never reads cache is the whole point.
        if args.claude_code:
            if r["calls"] >= 20 and hit < 40:
                findings.append(
                    f"{key}: {r['calls']:,} calls at {hit:.1f}% hit rate — low for an "
                    f"interactive workload. Usually many short one-shot sessions that "
                    f"never reach a second turn, so the prefix is written and never read."
                )
        elif r["calls"] >= 3 and hit < 5:
            findings.append(
                f"{key[:12]}: {r['calls']} calls, {hit:.1f}% hit rate — prefix "
                f"({r['prefix_chars']:,} chars) is not caching. Either it is below the "
                f"model minimum or its bytes vary per request."
            )
        elif r["calls"] >= 3 and hit < 50:
            findings.append(
                f"{key[:12]}: {r['calls']} calls, only {hit:.1f}% hit rate — partial "
                f"caching, prefix likely changing part-way through the run."
            )

    print("-" * len(hdr))
    total_saved = grand_uncached - grand_actual
    mult = (grand_uncached / grand_actual) if grand_actual else 0
    print(
        f"{'TOTAL':18} {'':20} {total_calls:6,} {'':6} "
        f"${grand_actual:9.4f} ${total_saved:9.4f}"
    )
    if mult:
        print(f"\nCaching is making this workload {mult:.2f}x cheaper than no caching at all.")

    if findings:
        print("\nfindings:")
        for f in findings:
            print(f"  - {f}")
    else:
        print("\nNo cache regressions detected.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "total_calls": total_calls,
                    "actual_usd": grand_actual,
                    "uncached_usd": grand_uncached,
                    "routes": {k: v for k, v in routes.items()},
                    "findings": findings,
                },
                fh,
                indent=2,
            )
        print(f"\nJSON written to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
