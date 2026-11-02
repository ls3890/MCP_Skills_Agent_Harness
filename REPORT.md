# Job-search agent (API version): report

Uses the **Groq** API (`groq==0.37.1`, chat completions). The model is `openai/gpt-oss-120b` by default; set `GROQ_MODEL` to use a different one. The user chose Groq on purpose, which departs from spec S2 (Anthropic Messages API). Checked with offline tests (fake client); the live scenario has not been run yet.

## Checklist

| ID | Requirement | Status | Evidence |
|---|---|---|---|
| S1 | Structure | PASS | `API/main.py`, `API/agent.py`, 8 files in `API/harness/`, `shared/`, `tests/` (5 files), `requirements.txt`, `scenario.json`, `results/`, `REPORT.md`. `main.py` resets `shared/data/`. |
| S2 | Client | DEVIATION (user decision) | Groq client, pinned `groq==0.37.1`. The spec asks for Anthropic. |
| S3 | Tools, OK/ERR output, error flag, dispatch try/except | PASS | `test_tools.py` (11 tests). Groq tool messages have no `is_error` field, so the `ERR|` prefix carries the error, and the turn log records `is_error`. |
| S4 | Loop, cap, plain dicts, immediate writes | PASS (adapted) | `test_loop.py` (6 tests). The loop continues while the response contains `tool_calls`. Groq needs one `tool` message per call; all of them are appended together right after the assistant message. |
| S5 | Append-only history, load_view, memory keys, atomic write, status_history | PASS | `test_persistence.py` (6 tests) |
| S6 | Static/dynamic system, last 6 turns, no orphan tool result | PASS (adapted) | `test_cut_never_starts_with_tool_result`, `test_static_first_dynamic_second` |
| S7 | Tokens from usage only, per-call record, price table | PASS (code) / NOT VERIFIED (prices) | `tokens.py`; `test_usage_read_from_response_only`; `test_cost_calculation`. Check the prices at groq.com/pricing. |
| S8 | Caching | ADAPTED | Groq caches prefixes automatically and has no `cache_control`. The tools and the static system message stay byte-identical, the dynamic message comes after them, and a warning prints when `usage` shows no cached tokens. `test_prefix_is_stable_and_has_no_markers`. Whether caching works depends on the model and is NOT VERIFIED (live). |
| S9 | Compaction | PASS | `test_compaction_after_turn_appends_summary`, `test_no_compaction_below_threshold`, `test_compaction_threshold` |
| S10 | `[file.py]` logging | PASS | Every harness file, `agent.py`, and `main.py` has a `log()` that prints this prefix |
| S11 | Constants in agent.py | PASS | `MODEL`, `EFFORT` (`reasoning_effort`, gpt-oss only), `MAX_TOKENS`, `MAX_ITERATIONS`, `CONTEXT_BUDGET`, `COMPACT_THRESHOLD`, `SYSTEM_PROMPT` |
| S12 | Scenario | PASS (structure) / NOT VERIFIED (live) | `scenario.json`: 8 turns, 2 sessions, a new `Agent` per session |
| S13 | Checks C1-C8 | PASS (logic) / NOT VERIFIED (live) | `test_checks_fail_when_behavior_is_broken`, `test_checks_pass_on_correct_behavior` |
| S14 | results/api.json | NOT VERIFIED (live) | `main.py` writes it after a live run |
| S15 | Offline tests | PASS | 35 passed |

## Run live

```
pip install -r requirements.txt
python -m pytest tests -q          # offline, free
python API/main.py                 # live, needs GROQ_API_KEY in .env at the project root
```

---

# SDK version (Claude Agent SDK): report

Audited on 2026-10-26. `SDK/` did not exist, so every SDK requirement started as FAIL. The fix was to write the three files the spec lists. The offline tests pass against a stand-in SDK module. `claude-agent-sdk==0.2.174` could not be installed (network blocked), so the live scenario has not run.

## How it works

- `SDK/agent.py`: constants and `build_options()`. Each `Agent.ask()` opens a `ClaudeSDKClient` with `resume=<session id from shared/data/session.json>`, calls `query()`, reads `receive_response()`, and saves `ResultMessage.session_id`. Session 2 is a new `Agent` instance, and every turn starts from disk.
- `SDK/tools.py`: 5 `@tool`s on `create_sdk_mcp_server("jobs")`. Domain rules come from `shared/`. `save_preference` writes `shared/data/CLAUDE.md` atomically using the shared `apply_pref` and `validate_pref`.
- The agent's `cwd` is `shared/data/`, so `setting_sources=["project"]` loads that `CLAUDE.md`, and `main.py`'s reset clears it before each run.
- `tools=[]` removes Claude Code's built-in tools. Without it the model could `Read` `applications.json` directly, because `allowed_tools` only auto-approves; it doesn't restrict. `ENABLE_TOOL_SEARCH=false` loads the 5 tool definitions upfront, not deferred.

## What the SDK adds on its own (S6)

With a plain-string `system_prompt`, Claude Code's default prompt is replaced. Claude Code still adds the `CLAUDE.md` memory (as context, not in our system prompt), the MCP tool definitions, and possibly environment context. `results/sdk.json` → `session2_context` records the categories and memory files that `get_context_usage()` reports on turn 6. That list is not reimplemented.

## SDK vs API differences

- **Effort:** `ClaudeAgentOptions.effort="medium"` is set. The API version (Groq) sends `reasoning_effort="medium"` only on gpt-oss models. This is not parity: different providers and models.
- **Tokens:** totals come from `ResultMessage.usage` per turn. `model_calls` counts distinct `AssistantMessage.message_id` values. Per-call token fields are NOT VERIFIED.
- **Caching and compaction:** both are automatic in the SDK, with no markers and no code.
- **Line counts:** `SDK/` is 463 lines (`agent.py` 146, `main.py` 162, `tools.py` 155). The API harness line counts are written to `results/sdk.json` → `api_harness_line_counts`.

## Run live

```
pip install -r requirements.txt   # includes claude-agent-sdk==0.2.174
claude login                      # Claude Code login; ANTHROPIC_API_KEY must NOT be set
python SDK/main.py                # refuses to run if ANTHROPIC_API_KEY is set; writes results/sdk.json
```
