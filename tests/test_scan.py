"""Scanner tests: Python front end, the lower-bound rule, and a TypeScript regression.

Run with:  python -m unittest discover -s tests
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import textwrap
import unittest

from agentcost import scan

LONG = "x" * (5000 * 4)  # ~5,000 tokens: clears every model's minimum
SHORT = "You are a helpful assistant."


def run_scan(files: dict[str, str]) -> tuple[int, dict]:
    """Write `files` into a temp project, scan it, return (exit code, JSON payload)."""
    with tempfile.TemporaryDirectory() as root:
        for name, body in files.items():
            path = os.path.join(root, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(textwrap.dedent(body))
        out = os.path.join(root, "findings.json")
        with contextlib.redirect_stdout(io.StringIO()):
            code = scan.main([root, "--json", out])
        with open(out, encoding="utf-8") as fh:
            return code, json.load(fh)


def categories(payload: dict) -> list[str]:
    return sorted(f["category"] for f in payload["findings"])


class PythonFrontEnd(unittest.TestCase):
    def test_dead_breakpoint_on_literal_system_list(self):
        code, out = run_scan({"agent.py": f'''
            import anthropic
            client = anthropic.Anthropic()
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                system=[{{"type": "text", "text": "{SHORT}", "cache_control": {{"type": "ephemeral"}}}}],
                messages=[],
            )
        '''})
        self.assertEqual(code, 1)
        self.assertIn("dead-breakpoint", categories(out))
        site = out["call_sites"][0]
        self.assertEqual(site["model"], "claude-haiku-4-5")
        self.assertEqual(site["model_source"], "literal")

    def test_large_cached_prefix_is_clean(self):
        code, out = run_scan({"agent.py": f'''
            PROMPT = "{LONG}"
            def go(client):
                return client.messages.create(
                    model="claude-haiku-4-5",
                    system=[{{"type": "text", "text": PROMPT, "cache_control": {{"type": "ephemeral"}}}}],
                    messages=[],
                )
        '''})
        self.assertEqual(code, 0)
        self.assertEqual(categories(out), [])
        self.assertEqual(out["call_sites"][0]["system_refs"], ["PROMPT"])

    def test_env_default_and_module_const_defined_after_use(self):
        # The function runs after the module finishes, so a later module const is visible.
        _, out = run_scan({"agent.py": f'''
            import os
            def go(client):
                model = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
                return client.messages.create(model=model, system="{SHORT}", messages=[])
            DEFAULT_MODEL = "claude-sonnet-4-6"
        '''})
        site = out["call_sites"][0]
        self.assertEqual(site["model"], "claude-sonnet-4-6")
        self.assertEqual(site["model_source"], "env-default")
        self.assertIn("too-small-to-cache", categories(out))

    def test_getenv_or_fallback(self):
        _, out = run_scan({"agent.py": '''
            import os
            def go(client):
                return client.messages.create(model=os.getenv("M") or "claude-opus-4-8", messages=[])
        '''})
        self.assertEqual(out["call_sites"][0]["model"], "claude-opus-4-8")
        self.assertEqual(out["call_sites"][0]["model_source"], "env-default")

    def test_class_attribute_model(self):
        _, out = run_scan({"agent.py": '''
            class Drafter:
                MODEL = "claude-haiku-4-5"
                def run(self, client):
                    return client.messages.create(model=self.MODEL, messages=[])
        '''})
        self.assertEqual(out["call_sites"][0]["model"], "claude-haiku-4-5")
        self.assertEqual(out["call_sites"][0]["model_source"], "class-attr")

    def test_volatile_fstring_in_cached_block(self):
        code, out = run_scan({"agent.py": f'''
            from datetime import datetime
            def go(client):
                head = f"Today is {{datetime.now().isoformat()}}. {LONG}"
                return client.messages.create(
                    model="claude-sonnet-4-6",
                    system=[{{"type": "text", "text": head, "cache_control": {{"type": "ephemeral"}}}}],
                    messages=[],
                )
        '''})
        self.assertEqual(code, 1)
        self.assertIn("cache-invalidator", categories(out))

    def test_volatile_value_after_the_breakpoint_is_fine(self):
        code, out = run_scan({"agent.py": f'''
            from datetime import datetime
            def go(client):
                return client.messages.create(
                    model="claude-sonnet-4-6",
                    system=[
                        {{"type": "text", "text": "{LONG}", "cache_control": {{"type": "ephemeral"}}}},
                        {{"type": "text", "text": f"Now: {{datetime.now()}}"}},
                    ],
                    messages=[],
                )
        '''})
        self.assertEqual(code, 0)
        self.assertNotIn("cache-invalidator", categories(out))

    def test_kwargs_from_dict_literal(self):
        code, out = run_scan({"agent.py": f'''
            def go(client):
                params = {{
                    "model": "claude-haiku-4-5",
                    "system": [{{"type": "text", "text": "{SHORT}", "cache_control": {{"type": "ephemeral"}}}}],
                    "messages": [],
                }}
                return client.messages.create(**params)
        '''})
        self.assertEqual(code, 1)
        self.assertIn("dead-breakpoint", categories(out))

    def test_runtime_kwargs_reported_once_not_failed(self):
        code, out = run_scan({"agent.py": '''
            def go(client, api_kwargs):
                return client.messages.create(**api_kwargs)
        '''})
        self.assertEqual(code, 0)
        self.assertEqual(categories(out), ["runtime-built-request"])

    def test_tools_make_a_short_prefix_possibly_dead_not_high(self):
        code, out = run_scan({"agent.py": f'''
            def go(client, tools):
                return client.messages.create(
                    model="claude-haiku-4-5",
                    tools=tools,
                    system=[{{"type": "text", "text": "{SHORT}", "cache_control": {{"type": "ephemeral"}}}}],
                    messages=[],
                )
        '''})
        self.assertEqual(code, 0)
        self.assertEqual(categories(out), ["possibly-dead-breakpoint"])

    def test_helper_built_system_with_cache_control(self):
        _, out = run_scan({"agent.py": f'''
            KB = "{LONG}"
            def blocks():
                return [{{"type": "text", "text": KB, "cache_control": {{"type": "ephemeral"}}}}]
            def go(client):
                return client.messages.create(model="claude-sonnet-4-6", system=blocks(), messages=[])
        '''})
        site = out["call_sites"][0]
        self.assertTrue(site["has_cache_control"])
        self.assertEqual(categories(out), [])

    def test_runtime_system_is_info(self):
        code, out = run_scan({"agent.py": '''
            def go(client, build):
                return client.messages.create(model="claude-sonnet-4-6", system=build(), messages=[])
        '''})
        self.assertEqual(code, 0)
        self.assertEqual(categories(out), ["runtime-built-system"])

    def test_large_uncached_system_suggests_breakpoint(self):
        _, out = run_scan({"agent.py": f'''
            def go(client):
                return client.messages.create(model="claude-sonnet-4-6", system="{LONG}", messages=[])
        '''})
        self.assertEqual(categories(out), ["no-cache-breakpoint"])

    def test_other_sdks_and_broken_files_are_ignored(self):
        code, out = run_scan({
            "openai_agent.py": '''
                def go(client):
                    return client.chat.completions.create(model="gpt-4o", messages=[])
            ''',
            "broken.py": "def oops(:\n    messages.create(\n",
            "venv/lib/vendored.py": '''
                def go(client):
                    return client.messages.create(model="claude-haiku-4-5", system=[{"text": "x", "cache_control": {}}])
            ''',
        })
        self.assertEqual(code, 0)
        self.assertEqual(out["call_sites"], [])


class PythonAliases(unittest.TestCase):
    def test_getattr_messages_alias(self):
        code, out = run_scan({"agent.py": f'''
            def go(client):
                api = getattr(client, "messages", None)
                return api.create(
                    model="claude-haiku-4-5",
                    system=[{{"type": "text", "text": "{SHORT}", "cache_control": {{"type": "ephemeral"}}}}],
                    messages=[],
                )
        '''})
        self.assertEqual(code, 1)
        self.assertIn("dead-breakpoint", categories(out))

    def test_bound_method_alias(self):
        _, out = run_scan({"agent.py": '''
            def go(client):
                stream_fn = client.beta.messages.stream
                return stream_fn(model="claude-sonnet-4-6", messages=[])
        '''})
        self.assertEqual(out["call_sites"][0]["model"], "claude-sonnet-4-6")

    def test_dict_copy_is_runtime(self):
        _, out = run_scan({"agent.py": '''
            def go(client, kwargs):
                final = dict(kwargs)
                return client.messages.stream(**final)
        '''})
        self.assertEqual(categories(out), ["runtime-built-request"])

    def test_twilio_messages_create_is_not_anthropic(self):
        _, out = run_scan({"sms.py": '''
            def send(client):
                return client.messages.create(body="hi", from_="+10000000000", to="+10000000001")
        '''})
        self.assertEqual(out["call_sites"], [])


class TypeScriptRegression(unittest.TestCase):
    def test_ts_dead_breakpoint_still_detected(self):
        code, out = run_scan({"src/agent.ts": f'''
            const SYSTEM = "{SHORT}";
            await client.messages.create({{
              model: "claude-haiku-4-5",
              system: [{{ type: "text", text: SYSTEM, cache_control: {{ type: "ephemeral" }} }}],
              messages: [],
            }});
        '''})
        self.assertEqual(code, 1)
        self.assertIn("dead-breakpoint", categories(out))


if __name__ == "__main__":
    unittest.main()
