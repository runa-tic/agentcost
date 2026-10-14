# agentcost

Cost observability for LLM agent code. Answers the question an invoice cannot:
**which code path generated this bill, and is its prompt cache actually working?**

No dependencies. No API key. Nothing leaves your machine.

## The problem

Prompt caching fails silently. Put `cache_control` on a prefix shorter than the
model's minimum cacheable length and the API does not error — it just declines to
cache, returns `cache_creation_input_tokens: 0`, and every call pays full input
price forever. Same story if a timestamp, a UUID, or a per-user field ends up
inside the cached prefix: the bytes change, the cache never hits, and nothing in
the response says so.

Your output stays correct. Only the bill changes — and the bill is an aggregate,
so a 10x regression on one route hides inside it for months.

Cache reads bill at 0.1x and writes at 1.25x, so a working prefix is roughly 5x
cheaper than none. No provider price cut will ever be a 5x cut. That is the whole
argument for instrumenting this.

## Install

```bash
pip install -e .        # or: pipx install .
```

Python 3.10+. That is the entire dependency list.

## Use

### See your real numbers first (no instrumentation)

If you use Claude Code, the data already exists on disk — it writes the same four
usage counters to `~/.claude/projects/**/*.jsonl`:

```bash
agentcost report --claude-code
```

```
project            model                 calls   hit%       cost      saved
---------------------------------------------------------------------------
backend-api        claude-sonnet-4-6    14,785  94.5% $3558.5536 $16769.8408
data-pipeline      claude-opus-4-8 +1      652  98.2% $ 193.9223 $1270.3312
scratch            claude-opus-5            83  97.5% $  26.4877 $  165.8295
---------------------------------------------------------------------------
TOTAL                                   15,520        $3778.9637 $18206.0015

Caching is making this workload 5.82x cheaper than no caching at all.
```

Records are deduplicated by message id. Resumed sessions replay earlier turns into
later transcripts, and counting those twice inflates every total — by ~65% on the
corpus this was built against.

### Scan your agent source

```bash
agentcost scan ./src ./services/worker/src --quiet
```

Finds Anthropic `messages.create` / `messages.stream` call sites in TypeScript,
JavaScript and Python, resolves the model and system-block layout, estimates the
cached prefix size, and compares it against that model's minimum:

```
[HIGH] dead-breakpoint  —  src/scheduler.ts:735
  Cached prefix ~377 tok is below claude-sonnet-4-6's 1,024-tok minimum
    The API will silently decline to cache this prefix — no error,
    cache_creation_input_tokens stays 0, and every call pays full input price.
```

Python is read through the stdlib `ast` parser, so constants, `os.environ.get`
defaults, `self.MODEL` class attributes, `**params` dict literals and local aliases
(`api = getattr(client, "messages")`) resolve structurally. TypeScript and
JavaScript go through a lighter regex front end.

Read-only. Never imports or executes the code it scans. Exits 1 when any
high-severity finding is present, so it drops into CI as a gate.

### Instrument your own agent

Static analysis proves a breakpoint is well-formed. It cannot prove it fires.
Wrap your client:

```ts
import Anthropic from "@anthropic-ai/sdk";
import { withCostLogging } from "./js/collector";

const client = withCostLogging(new Anthropic(), {
  path: "./logs/agentcost.jsonl",
  service: "reply-drafter",
});
```

Then aggregate:

```bash
agentcost report ./logs/agentcost.jsonl
```

The wrapper fails soft — a logging error can never break a live call.

## Route identity

The collector keys routes by a **hash of the cached prefix**, not a label you
write by hand. Two consequences worth the design:

- No call-site changes. Wrap the client once and every route is covered.
- The grouping key is exactly what determines caching. A route that shows many
  distinct hashes is a route whose prefix is varying — that *is* the invalidator,
  visible without reading a single prompt.

A hand-written label can claim two calls are "the same route" when their bytes
differ. A hash cannot.

## What it will not tell you

- **Whether a route is over-modeled.** Only measured pass rates on a cheaper tier
  can answer that. The tier table in `scan` output is informational.
- **Exact token counts.** Static estimates are `chars / 4`. Anything within ~20%
  of a threshold is reported as `near-minimum` rather than asserted — confirm with
  the `count_tokens` endpoint.
- **Anything about routes it cannot resolve statically.** Models set from runtime
  config report as `unresolved` and are skipped rather than guessed at. A request
  assembled at runtime (`create(**kwargs)` from a dict built elsewhere) is reported
  once as `runtime-built-request` — for those routes the usage log is the only
  instrument.
- **An exact prefix when tools are involved.** The API caches tools before system,
  and adds a tool-use preamble no source file shows. A Python call with `tools=`
  is therefore measured as a floor: a short floor is reported as
  `possibly-dead-breakpoint` (medium), never as a CI-failing high.

A static finding on a code path that never executes is noise, not a bug. Check
findings against runtime traffic before acting on them.

## Privacy

Neither half emits prompt text. The collector logs a hash, sizes, and counters;
the scanner reports identifier names, sizes, and line numbers. Reports are safe to
paste into an issue.

## License

MIT
