/**
 * agentcost runtime collector — drop-in usage logging for an Anthropic client.
 *
 * Static analysis proves a breakpoint is well-formed. It cannot prove the
 * breakpoint fires. Only `cache_read_input_tokens` coming back non-zero on the
 * second and subsequent calls proves that, so this wraps the client and records
 * the four usage counters per request.
 *
 * Routes are keyed by a hash of the cached prefix rather than a hand-written
 * label. That means zero call-site changes, and the grouping key is exactly the
 * thing that determines cache behaviour: if two calls hash the same they should
 * share a cache entry, and if they don't, the prefix is varying and that is the
 * bug. A label you assign by hand can lie about that; the hash cannot.
 *
 * Never logs prompt text — only a hash, sizes, and counters — so the output is
 * safe to hand to someone else.
 *
 *   import { withCostLogging } from "./agentcost/collector";
 *   const client = withCostLogging(new Anthropic(), { path: "./logs/agentcost.jsonl" });
 *
 * Then aggregate with:  python _tools/agentcost/report.py logs/agentcost.jsonl
 */

import { createHash } from "node:crypto";
import { appendFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";

export interface CollectorOptions {
  /** JSONL destination. Appended to; one object per request. */
  path: string;
  /** Optional service name, for when several agents share one log. */
  service?: string;
  /** Swallow logging errors (default true) — never break a live call. */
  swallowErrors?: boolean;
}

interface UsageLike {
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

/** Text of every system block up to and including the cache_control one. */
function cachedPrefixOf(system: unknown): { text: string; blocks: number; cachedIdx: number } {
  if (typeof system === "string") return { text: system, blocks: 1, cachedIdx: -1 };
  if (!Array.isArray(system)) return { text: "", blocks: 0, cachedIdx: -1 };

  let cachedIdx = -1;
  for (let i = 0; i < system.length; i++) {
    if (system[i] && typeof system[i] === "object" && "cache_control" in system[i]) {
      cachedIdx = i;
      break;
    }
  }
  const upto = cachedIdx >= 0 ? system.slice(0, cachedIdx + 1) : system;
  const text = upto
    .map((b: unknown) =>
      b && typeof b === "object" && typeof (b as { text?: unknown }).text === "string"
        ? (b as { text: string }).text
        : "",
    )
    .join("\n");
  return { text, blocks: system.length, cachedIdx };
}

function sha16(s: string): string {
  return createHash("sha256").update(s).digest("hex").slice(0, 16);
}

let warned = false;

export function withCostLogging<T extends object>(client: T, opts: CollectorOptions): T {
  const swallow = opts.swallowErrors !== false;

  const write = async (record: Record<string, unknown>) => {
    try {
      await mkdir(dirname(opts.path), { recursive: true });
      await appendFile(opts.path, JSON.stringify(record) + "\n", "utf8");
    } catch (err) {
      if (!swallow) throw err;
      if (!warned) {
        warned = true;
        console.warn("[agentcost] logging disabled:", (err as Error).message);
      }
    }
  };

  return new Proxy(client, {
    get(target, prop, receiver) {
      const value = Reflect.get(target, prop, receiver);
      if (prop !== "messages" || !value || typeof value !== "object") return value;

      return new Proxy(value as object, {
        get(mTarget, mProp, mReceiver) {
          const inner = Reflect.get(mTarget, mProp, mReceiver);
          if (mProp !== "create" || typeof inner !== "function") return inner;

          return async function wrapped(params: Record<string, unknown>, ...rest: unknown[]) {
            const started = Date.now();
            const resp = await (inner as Function).apply(mTarget, [params, ...rest]);

            try {
              const { text, blocks, cachedIdx } = cachedPrefixOf(params?.system);
              const u: UsageLike = (resp?.usage ?? {}) as UsageLike;
              const read = u.cache_read_input_tokens ?? 0;
              const created = u.cache_creation_input_tokens ?? 0;
              const fresh = u.input_tokens ?? 0;
              const prompt = read + created + fresh;

              await write({
                ts: new Date(started).toISOString(),
                service: opts.service ?? null,
                model: params?.model ?? resp?.model ?? null,
                // Route identity = cached prefix identity.
                prefix_hash: text ? sha16(text) : null,
                prefix_chars: text.length,
                system_blocks: blocks,
                cached_block_index: cachedIdx,
                input_tokens: fresh,
                output_tokens: u.output_tokens ?? 0,
                cache_creation_input_tokens: created,
                cache_read_input_tokens: read,
                cache_hit_rate: prompt ? read / prompt : 0,
                latency_ms: Date.now() - started,
                stop_reason: resp?.stop_reason ?? null,
              });
            } catch (err) {
              if (!swallow) throw err;
            }

            return resp;
          };
        },
      });
    },
  }) as T;
}
