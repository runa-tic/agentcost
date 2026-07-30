"""agentcost command line entry point.

Two subcommands mirroring the two halves of the tool:

    agentcost scan <path>...      static analysis of agent source
    agentcost report [log]        runtime cache economics
"""

from __future__ import annotations

import sys


USAGE = """agentcost — cost observability for LLM agent code

usage:
  agentcost scan <path>... [--json OUT] [--quiet]
      Static analysis. Finds Anthropic call sites, resolves the model and
      system-block layout, and flags prompt-cache breakpoints that cannot fire.

  agentcost report [log.jsonl] [--claude-code [DIR]] [--since ISO] [--json OUT]
      Runtime cache economics, per route. With --claude-code, reads transcripts
      Claude Code already writes — no instrumentation required.

examples:
  agentcost scan ./src
  agentcost report --claude-code
  agentcost report ./logs/agentcost.jsonl --since 2026-07-01
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    cmd, rest = argv[0], argv[1:]

    if cmd == "scan":
        from . import scan

        return scan.main(rest)
    if cmd == "report":
        from . import report

        return report.main(rest)

    print(f"unknown command: {cmd}\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
