"""Responsibility: summarize the history view into a new summary record when it grows too large."""
import time

from API.harness import history

COMPACT_MAX_TOKENS = 1500
SUMMARY_PROMPT = ("Summarize this job-search conversation for a future session. Keep: user preferences, "
                  "applications and their ids and statuses, open questions, and pending confirmations. "
                  "Plain text, under 200 words.")


def log(msg: str) -> None:
    print(f"[compaction.py] {msg}")


def should_compact(last_input_tokens: int, budget: int, threshold: float) -> bool:
    return last_input_tokens > threshold * budget


def render(summary: str | None, messages: list[dict]) -> str:
    lines = [f"[previous summary] {summary}"] if summary else []
    for m in messages:
        if m["role"] == "tool":
            lines.append(f"tool result: {m['content']}")
            continue
        if m.get("content"):
            lines.append(f"{m['role']}: {m['content']}")
        for call in m.get("tool_calls", []):
            lines.append(f"tool call: {call['function']['name']} {call['function']['arguments']}")
    return "\n".join(lines)


def compact(client, model: str, request_extras: dict):
    """One model call, no tools, small max tokens. Appends a summary record; removes nothing.
    Returns (response, elapsed, compacted)."""
    summary, messages = history.load_view()
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model, max_completion_tokens=COMPACT_MAX_TOKENS,
        messages=[{"role": "system", "content": SUMMARY_PROMPT},
                  {"role": "user", "content": render(summary, messages)}], **request_extras)
    elapsed = time.perf_counter() - started
    text = (response.choices[0].message.content or "").strip()
    if not text:
        log(f"no summary text (finish_reason={response.choices[0].finish_reason}); history left as is")
        return response, elapsed, False
    history.append_summary(text)
    log(f"compacted {len(messages)} messages into a summary of {len(text)} chars")
    return response, elapsed, True
